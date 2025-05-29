import torch
import torch
from scipy.spatial.distance import cosine
import streamlit as st
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tempfile import NamedTemporaryFile
import boto3
import os
from dotenv import load_dotenv


st.set_page_config(page_title="Amigo Flamingo", layout="wide")



load_dotenv()



# AWS S3 credentials (you can set them as environment variables or in the AWS config)
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
bucket_name = os.getenv('AWS_BUCKET_NAME')
region_name = os.getenv('AWS_REGION')

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"



s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region_name
)



@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.spinner('Loading model... please wait!')  # Shows loading spinner
    model = AutoModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor



model, processor = load_model()



def dowloadfile_and_return_temp_path(file_path):
    # Create a temporary directory to store the images
    with NamedTemporaryFile(delete=False) as tmpfile:
        # Download the file from S3 to the temporary file
        s3_client.download_file(bucket_name, file_path, tmpfile.name)
        # print(f"Downloaded {file_path} to {tmpfile.name}")
        return tmpfile.name



def predict_bird_from_description(user_description, processor, model, device="cpu", falsely_predicted_birds=[], top_k=3):
    # list of dictionary --> [{"name": "bird_name", "image_path": "path/to/image", "image_embedding": image_embedding}]
    bird_embeddings_ls = torch.load("bird_embeddings.pt")

    if falsely_predicted_birds:
        # Filter out the falsely predicted birds
        bird_embeddings_ls = [bird for bird in bird_embeddings_ls if bird["name"] not in falsely_predicted_birds]

    # Step 1: Encode the user's description with processor
    text_inputs = processor(text=[user_description], return_tensors="pt", truncation=True, padding=True).to(device)

    # 💡 Use get_text_features directly (avoids pixel_values error)
    with torch.no_grad():
        user_text_embedding = model.get_text_features(**text_inputs).cpu().squeeze(0).numpy()

    # Step 2: Compare to all image embeddings
    similarity_results = []
    for bird in bird_embeddings_ls:
        image_embedding = bird["image_embedding"].squeeze(0).numpy()
        similarity = 1 - cosine(user_text_embedding, image_embedding)
        similarity_results.append((similarity, bird["name"], bird["image_path"]))

    # Step 3: Sort and return top K
    similarity_results.sort(reverse=True)
    top_matches = similarity_results[:top_k]
    return top_matches



def generate_result():
    print('Generating results...')
    if st.session_state.get("results"):
        print('Result:', st.session_state.results)
        with st.spinner("waiting"):
            for result in st.session_state.results:
                name = result[1].replace("_", " ").title()
                img_path = result[2][0]
                img_path = dowloadfile_and_return_temp_path(img_path)
                st.subheader(name)
                st.image(Image.open(img_path), caption=name)
                st.session_state.prediction_log.append(result[1])
            if st.session_state.get("prediction_log"):
                st.session_state.results = predict_bird_from_description(st.session_state.input, processor, model, falsely_predicted_birds=st.session_state.prediction_log)


def home_page():
    st.header("Welcome to the Amigo Flamingo Bird Identification System!")
    st.write("Identify various bird species of Kerala with ease using this system!")


    user_description = st.text_area("Describe the bird you saw:")

    if st.button("Identify Bird") and user_description:
        st.session_state.input = user_description
        st.session_state.results = predict_bird_from_description(user_description, processor, model)
        st.session_state.feedback_visible = True


def main():

    # Set the custom theme using CSS (Flemingo color theme)
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5b0b0; /* Light Flemingo color */
        }
        h1 {
            color: #ff6f61; /* Primary color for the title */
            font-family: 'Arial', sans-serif;
        }
        .stTextInput>div>div>input {
            background-color: #ffe6e6; /* Light background for text input */
        }
        .stButton>button {
            background-color: #ff6f61; /* Button color matching the title */
            color: white;
        }
        .stButton>button:hover {
            background-color: #ff4c38; /* Slightly darker color on hover */
        }
        </style>
    """, unsafe_allow_html=True)



    if "prediction_log" not in st.session_state:
        st.session_state.prediction_log = []

    # Title of the app
    st.title("Amigo Flamingo")

    home_page()


    if st.session_state.get("results") and st.session_state.get("regenerate_results", False) is False:
        generate_result()

    if st.session_state.get("feedback_visible", False):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔁 Try Again"):
                print("Try again")






if __name__ == "__main__":
    # try:
        main()
    # except Exception as e:
    #     st.error("Oops! Something went wrong. Please try again later.")