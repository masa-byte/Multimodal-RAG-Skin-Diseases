from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from pydantic import BaseModel
from transformers import AlignProcessor, AlignModel
import torch
import base64
from typing import List
from PIL import Image
from io import BytesIO
import google.generativeai as genai

app = FastAPI()

# DATABASE PART
qdrant_url = "http://localhost:6333"
client = qdrant_client.QdrantClient(qdrant_url)
collection_name = "skin_diseases_align"
limit = 4

# MODEL PART
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")
model.to(device)

model.eval()

# LLM PART
genai.configure(api_key="AIzaSyBi9GZh-lIpYbYIVsOZ2VD81wGgXj62eAE")
llm = genai.GenerativeModel("gemini-1.5-flash")


# MODEL FUNCTIONS
def align_image(image_data):
    image = Image.open(BytesIO(image_data))

    image = image.resize((224, 224))
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.get_image_features(**inputs)
    return output.cpu().numpy()


def align_text(text):
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.get_text_features(**inputs)
    return output.cpu().numpy()


# API PART
class QueryResponse(BaseModel):
    response_text: List[str]
    response_images: List[bytes]


def voting_system(response_text_labels, response_image_labels):
    # Voting system to decide the final output based on which key has the most votes
    # If there is a tie, all the keys with the maximum votes will be returned
    results = {}
    for label in response_text_labels:
        results[label] = results.get(label, 0) + 1
    for label in response_image_labels:
        results[label] = results.get(label, 0) + 1

    final_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print(final_results)

    max_votes = max(results.values())
    final_results = [
        (key, value) for key, value in results.items() if value == max_votes
    ]
    print(final_results)
    return final_results


def process_text_input(text):
    text_embedding = align_text(text).flatten().tolist()
    return text_embedding


async def process_image_input(file: UploadFile):
    image_data = await file.read()
    image_embedding = align_image(image_data).flatten().tolist()
    return image_embedding


async def get_input_image(file: UploadFile):
    image_data = await file.read()
    return image_data


def process_input_embedding(embedding):
    search_result_text = client.search(
        collection_name=collection_name,
        query_vector=embedding,
        query_filter=Filter(
            must=[FieldCondition(key="type", match=MatchValue(value="text"))]
        ),
        with_payload=True,
        limit=limit,
    )

    search_result_image = client.search(
        collection_name=collection_name,
        query_vector=embedding,
        query_filter=Filter(
            must=[FieldCondition(key="type", match=MatchValue(value="image"))]
        ),
        with_payload=True,
        limit=limit,
    )

    return search_result_text, search_result_image


@app.post("/query", response_model=QueryResponse)
async def query_qdrant(text: str = Form(None), images: List[UploadFile] = File(None)):
    response_texts = []
    response_images = []
    response_image_names = []
    search_results_text = []
    search_results_image = []

    if text:
        text_embedding = process_text_input(text)
        response_t1, response_im1 = process_input_embedding(text_embedding)
        search_results_text.extend(response_t1)
        search_results_image.extend(response_im1)

    if images:
        for image in images:
            image_embedding = await process_image_input(image)
            response_t1, response_im2 = process_input_embedding(image_embedding)
            search_results_text.extend(response_t1)
            search_results_image.extend(response_im2)

    response_text_labels = [result.payload["label"] for result in search_results_text]
    response_image_labels = [result.payload["label"] for result in search_results_image]
    final_labels = voting_system(response_text_labels, response_image_labels)

    for final_label, _ in final_labels:
        for result in search_results_image:
            if (
                result.payload["label"] == final_label
                and result.payload["image_path"].split("\\")[-1]
                not in response_image_names
            ):
                response_image_names.append(
                    result.payload["image_path"].split("\\")[-1]
                )
                with open(result.payload["image_path"], "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    response_images.append(encoded_string)

        for result in search_results_text:
            if (
                result.payload["label"] == final_label
                and result.payload["original_text"] not in response_texts
            ):
                response_texts.append(result.payload["original_text"])

    # If there was a text result from the database, we will use the LLM to generate an additional response
    if len(response_texts) != 0:
        prompt = "Please rate me the severity level of the disease below on a scale of 1 to 10, and tell me about the next steps to take."
        for text in response_texts:
            full_prompt = prompt + text
            llm_response = llm.generate_content(full_prompt)

            index = response_texts.index(text)
            response_texts[index] = (
                response_texts[index] + "\nLLM added part\n" + llm_response.text
            )

    # If there was no text result from the database, and there was an image result, we will use the LLM to generate an additional response based on image label
    elif len(response_texts) == 0 and len(response_images) != 0:
        prompt = "I have the following problem with my health, can you please give me the severity level of disease on a scale from 1 to 10, some advice and next steps?"
        for final_label, _ in final_labels:
            full_prompt = prompt + final_label
            response = llm.generate_content(full_prompt)
            response_texts.append(response.text)

    # If there was no text result from the database and no image result, and there was user query, LLM will generate a response based on user query
    elif len(response_texts) == 0 and len(response_images) == 0 and text:
        prompt = "I have the following problem with my health, can you please give me some advice?"
        full_prompt = prompt + text
        llm_response = llm.generate_content(full_prompt)
        response_texts.append(llm_response.text)

    # If there was no text result from the database and no image result, and there was user image, LLM will try to generate a response based on user image
    elif len(response_texts) == 0 and len(response_images) == 0 and images:
        prompt = "I have the following problem with my skin, can you please tell me what it potentially is and what next steps to take?"
        images_to_send = []
        for image in images:
            image_data = await get_input_image(image)
            image = Image.open(BytesIO(image_data))
            image = image.resize((224, 224))
            image = image.convert("RGB")
            images_to_send.append(image)
        llm_response = llm.generate_content(contents=[prompt, *images])
        response_texts.append(llm_response.text)

    return JSONResponse(
        content={
            "response_texts": response_texts,
            "response_images": response_images,
            "response_image_names": response_image_names,
        }
    )
