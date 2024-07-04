### Multimodal RAG System For Help With Skin Disease Diagnosis ###
This is a multimodal RAG system for help with skin disease diagnosis. The system is composed of a multimodal RAG model and a web application. The multimodal RAG model works with a multimodal dataset, which contains images and rich text descriptions of skin diseases. The web application allows users to upload images of a skin disease, or wrtie textual description of the disease, or both, and get a text description of the disease, with potential causes, severity level, symptoms, risk groups, and next steps. Similar images are also included in the response. The system is designed to be the initial step for people to use at home and get a general idea of the skin disease they have, and then decide whether to see a doctor or not. It is not a replacement for professional medical advice.

### Application type
This is a web application. Users can upload images of a skin disease, or write textual description of the disease, or both, and get a text description of the disease, with potential causes, severity level, symptoms, and next steps. Similar images are also included in the response. The app uses **streamlit** as the front end and **fastapi** as the back end.

### Idea
The idea comes from the cost of seeing a dermatologist, their accessibility, and the time it takes to get an appointment. The system is designed to be the initial step for people to use at home and get a general idea of the skin disease they have, and then decide whether to see a doctor or not. It is not a replacement for professional medical advice.

### Data
Data was collected from four different datasets on Kaggle. The datasets are:
1. https://www.kaggle.com/datasets/haroonalam16/20-skin-diseases-dataset
2. https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset
3. https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset
4. vhttps://www.kaggle.com/dsv/6695743
Images were also collected from DermNet.
After collecting the images, two texts were written for each disease. The first text was a rich and detailed description of the disease, inlcuding causes, skin and general symptoms, potential causes, risk groups, and next steps. The second text contained only skin symptoms of diseases, because that is what similarity search is based on. The texts were written by me through an extensive research on the internet.
Images were augmented in order to increase the size of the dataset. The augmented images are then stored in the vector database, along with the texts.

### Architecture
User text and/or image input is sent to the backend, where the embedding model embeds queries and a similarity search is performed on the vector database. A *limit* number of items is retrieved, separately for texts and images in the daatbase, and then a voting mechanism is implemented to vote for the final (most likely) class. Text and images which correspond ot the final label are collected and then sent to the large language model for final response formatting. The response is then sent back to the user. There are different queries for the llm and outcomes depending on whether the text was found, or image was found, or none of them were found, and whether user provided both text and image, or only one of them.

### Tech stack
The system is built using **fastapi** and **streamlit**. Vector database used is **Qdrant**. Embedding model is **ALIGN**. Large language model is **Google Gemini**.

### How to run
1. Clone the repository
2. Install the requirements
3. Run docker-compose up to start the vector database
4. Optionally run the data-preprocessing script if you want to modify augmented-data
5. Run database.ipynb to create a collection and populate the vector database
6. Run fastapi dev server.py to start the backend
7. Run client.py to start the front end