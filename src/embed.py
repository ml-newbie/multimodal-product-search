import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
import os

# ------------------------------
# Config
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
DATA_CSV = "data/products_clean.csv"
IMAGE_EMB_PATH = "data/embeddings/image_embeddings.npy"
TEXT_EMB_PATH = "data/embeddings/text_embeddings.npy"

# Ensure embeddings folder exists
os.makedirs("embeddings", exist_ok=True)

# ------------------------------
# Load CLIP model
# ------------------------------
print(f"Loading CLIP model on {DEVICE}...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv(DATA_CSV)
num_samples = len(df)
print(f"Dataset loaded: {num_samples} products")

# ------------------------------
# Helper functions
# ------------------------------
def batchify(lst, batch_size):
    """Yield successive batches from a list"""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# ------------------------------
# Image Embeddings
# ------------------------------

print("Generating image embeddings...")

# This list will store embeddings for each batch before stacking them together at the end
image_embeddings = []

# Loop over images in batches (e.g., 32 at a time)
for batch_num, batch_indices in enumerate(batchify(range(num_samples), BATCH_SIZE), start=1):
    # Progress display
    print(f"Processing image batch {batch_num} / {len(range(num_samples)) // BATCH_SIZE + 1}...")

    # ---------------------------------------------------------
    # Load the actual images for this batch
    # df.iloc[i] accesses row 'i' from the DataFrame
    # 'image_path' holds the path to the resized image file
    # convert("RGB") ensures the image has 3 channels for CLIP
    # ---------------------------------------------------------
    images = [Image.open(df.iloc[i]['image_path']).convert("RGB") for i in batch_indices]

    # ---------------------------------------------------------
    # CLIP processor:
    # - Resizes images as needed
    # - Normalizes pixels
    # - Converts them into PyTorch tensors
    # - Adds batch dimension automatically
    # Then we move the tensors to DEVICE (CPU or GPU)
    # ---------------------------------------------------------
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)

    # ---------------------------------------------------------
    # Disable gradient tracking since we are not training CLIP.
    # This makes inference faster and reduces memory usage.
    # ---------------------------------------------------------
    with torch.no_grad():
        # Pass batch through CLIP's image encoder
        embs = model.get_image_features(**inputs)   # shape: (batch_size, 512)

    # ---------------------------------------------------------
    # Move embeddings to CPU and convert to NumPy arrays
    # because FAISS and np.save require NumPy format
    # ---------------------------------------------------------
    embs = embs.cpu().numpy()

    # Append this batch's embeddings to our list
    image_embeddings.append(embs)

# ---------------------------------------------------------
# Combine all batches into one big embedding matrix:
# Final shape should be (num_samples, embedding_dim)
# Example: (44441 images, 512 dimensions)
# ---------------------------------------------------------
image_embeddings = np.vstack(image_embeddings)

# ---------------------------------------------------------
# Normalize embeddings so each vector has unit length.
# This makes cosine similarity equal to dot product,
# which improves FAISS retrieval quality.
# ---------------------------------------------------------
image_embeddings = normalize(image_embeddings)

# ---------------------------------------------------------
# Save the full embedding matrix to disk.
# This will be loaded later by the FAISS search module.
# ---------------------------------------------------------
np.save(IMAGE_EMB_PATH, image_embeddings)
print(f"Image embeddings saved: {IMAGE_EMB_PATH}")




# ------------------------------
# Text Embeddings
# ------------------------------

print("\n===== TEXT EMBEDDING STARTED =====")

# -------------------------------------------------------------
# 1. CLEAN / FIX TEXT COLUMN
# -------------------------------------------------------------

# Make sure everything is a string
df['title'] = df['title'].astype(str)

# Identify rows where title is invalid: 'nan', 'none', empty, whitespace
mask_invalid = (
    df['title'].str.lower().isin(["nan", "none"]) |
    (df['title'].str.strip().str.len() == 0)
)

print(f"Found {mask_invalid.sum()} invalid titles. Fixing them...")

# Replace missing / invalid titles with the product 'id'
df.loc[mask_invalid, 'title'] = df.loc[mask_invalid, 'product_id'].astype(str)

# Verify cleanup
print("Verification — remaining invalid titles:", 
      df['title'].str.strip().str.len().eq(0).sum())

num_samples = len(df)
print(f"Total rows after cleaning titles: {num_samples}")

# -------------------------------------------------------------
# 2. GENERATE TEXT EMBEDDINGS
# -------------------------------------------------------------

print("Generating text embeddings...")
text_embeddings = []

total_batches = (num_samples // BATCH_SIZE) + 1

for batch_num, batch_indices in enumerate(batchify(range(num_samples), BATCH_SIZE), start=1):
    print(f"Processing text batch {batch_num} / {total_batches}...")

    # Extract batch titles
    texts = df.iloc[batch_indices]['title'].tolist()

    # Tokenize with CLIP's processor
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    # Compute CLIP text embeddings
    with torch.no_grad():
        embs = model.get_text_features(**inputs)

    # Move to CPU + convert to numpy
    embs = embs.cpu().numpy()

    # Append this batch
    text_embeddings.append(embs)

# -------------------------------------------------------------
# 3. STACK + NORMALIZE + SAVE
# -------------------------------------------------------------

text_embeddings = np.vstack(text_embeddings)
text_embeddings = normalize(text_embeddings)   # L2-normalize for cosine similarity

np.save(TEXT_EMB_PATH, text_embeddings)

print(f"Text embeddings saved to: {TEXT_EMB_PATH}")
print("===== TEXT EMBEDDING COMPLETE ✅ =====\n")
