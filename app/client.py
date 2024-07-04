import streamlit as st
import requests
from PIL import Image
import io
import base64

backend_url = "http://localhost:8000"

st.title("Skin Disease Demo App")

user_text = st.text_area("Enter your description here:")

uploaded_images = st.file_uploader(
    "Upload an image:", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# Submit button
if st.button("Submit"):

    files = []
    if uploaded_images:
        for img in uploaded_images:
            files.append(("images", (img.name, img.getvalue(), img.type)))
    data = {"text": user_text}

    response = requests.post(f"{backend_url}/query", files=files, data=data)

    if response.status_code == 200:
        response_data = response.json()

        st.subheader("Response Texts:")
        for text in response_data.get("response_texts", []):
            st.write(text)

        st.subheader("Response Images:")
        for img_data in response_data.get("response_images", []):
            index = response_data.get("response_images").index(img_data)
            image = Image.open(io.BytesIO(base64.b64decode(img_data)))
            st.image(image)
            image_name = response_data.get("response_image_names")[index]
            st.write(image_name)
    else:
        st.error("Failed to get a response from the server. Please try again later.")
