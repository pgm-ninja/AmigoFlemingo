import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
import torch
from transformers import pipeline, CLIPTokenizer
from sklearn.cluster import DBSCAN
import torch.nn.functional as F


# Load the Hugging Face CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device)

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Path to your dataset folder
dataset_dir = "dataset"

# Store results here


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")




def bird_embeddings():
    bird_embeddings_ls = []
    # Walk through each bird folder
    for bird_folder in tqdm(os.listdir(dataset_dir)):
        print(f"Processing {bird_folder}...")
        bird_path = os.path.join(dataset_dir, bird_folder)
        if not os.path.isdir(bird_path):
            continue

        # Read and truncate description
        desc_file = os.path.join(bird_path, "description.txt")
        if not os.path.exists(desc_file):
            print(f"Missing description.txt in {bird_path}")
            continue

        with open(desc_file, "r", encoding="utf-8") as f:
            description = f.read().strip()
        token_count = len(tokenizer(description, add_special_tokens=False)["input_ids"])
        if token_count > 75:
            description = summarizer(description, max_length=77, min_length=20, do_sample=False)
            description = description[0]['summary_text']
        # Encode each image in the folder

        image_files = []
        embeddings = []
        text_embedding = None

        for file_ in os.listdir(bird_path):
            if file_.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(bird_path, file_)

                # Load and preprocess
                image = Image.open(img_path).convert("RGB")
                inputs = processor(text=[description], images=image, return_tensors="pt", padding=True, truncation=True).to(device)

                # Forward pass
                with torch.no_grad():
                    outputs = model(**inputs)
                    text_embedding = outputs.text_embeds.cpu()
                    image_embedding = outputs.image_embeds.cpu()
                image_files.append(img_path)
                embeddings.append(image_embedding)


            # Stack all embeddings into a tensor
        embeddings = torch.vstack(embeddings)  # shape: [num_images, embedding_dim]

        # Normalize embeddings
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        # DBSCAN clustering

        embeddings_np = normalized_embeddings.numpy()
        dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine')  # you might want to tune eps
        cluster_labels = dbscan.fit_predict(embeddings_np)

        for cluster_id in set(cluster_labels):
            if cluster_id == -1:
                continue  # optional: skip outliers/noise

            indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            group_embeddings = embeddings[indices]
            group_embedding = group_embeddings.mean(dim=0)
            group_embedding /= group_embedding.norm()

            bird_embeddings_ls.append({
                "name": bird_folder,
                "description": description,
                "image_path": [image_files[i] for i in indices],
                "text_embedding": text_embedding,
                "image_embedding": group_embedding
            })

    print(f"Processed {len(bird_embeddings_ls)} bird images.")
    torch.save(bird_embeddings_ls, "bird_embeddings.pt")
    model.save_pretrained("saved_model/")
    processor.save_pretrained("saved_model/")
    return bird_embeddings_ls



bird_embeddings()