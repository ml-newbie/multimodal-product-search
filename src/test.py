import os
import faiss
import numpy as np
import pandas as pd

# ---------------------------------------
# CONFIG
# ---------------------------------------
IMAGE_EMB_PATH = "data/embeddings/image_embeddings.npy"
TEXT_EMB_PATH = "data/embeddings/text_embeddings.npy"
CLEAN_CSV = "data/products_clean.csv"

IMAGE_INDEX_PATH = "data/indexes/image_index.faiss"
TEXT_INDEX_PATH = "data/indexes/text_index.faiss"

METADATA_PATH = "data/indexes/metadata.parquet"

# ---------------------------------------
# ENSURE OUTPUT DIRECTORIES EXIST
# ---------------------------------------
os.makedirs("data/indexes", exist_ok=True)

# ---------------------------------------
# LOAD DATA
# ---------------------------------------
print("Loading embeddings...")
image_embs = np.load(IMAGE_EMB_PATH)
text_embs = np.load(TEXT_EMB_PATH)

print(f"Image embeddings: {image_embs.shape}")
print(f"Text embeddings:  {text_embs.shape}")
print(image_embs.shape[1])
