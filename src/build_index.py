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

# Load metadata CSV — use same order as embeddings
df = pd.read_csv(CLEAN_CSV)

if len(df) != len(image_embs):
    raise ValueError(f"Row count mismatch! metadata={len(df)} images={len(image_embs)}")

if len(df) != len(text_embs):
    raise ValueError(f"Row count mismatch! metadata={len(df)} text={len(text_embs)}")

print("Metadata loaded:", df.shape)

# -------------------------------------------------
# BUILD FAISS INDEX (cosine → use inner product)
# -------------------------------------------------
dim = image_embs.shape[1]

def build_faiss_index(vectors: np.ndarray):
    """
    For cosine similarity, vectors MUST be normalized
    and FAISS index should use IndexFlatIP.
    """
    print("Normalizing vectors...")
    faiss.normalize_L2(vectors)

    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity if normalized

    print("Adding vectors to index...")
    index.add(vectors)

    print("Total vectors indexed:", index.ntotal)
    return index

# ---------------------------------------
# BUILD BOTH INDEXES
# ---------------------------------------
print("\nBuilding IMAGE index...")
image_index = build_faiss_index(image_embs)

print("\nBuilding TEXT index...")
text_index = build_faiss_index(text_embs)

# ---------------------------------------
# SAVE INDEXES
# ---------------------------------------
print("\nSaving indexes...")

faiss.write_index(image_index, IMAGE_INDEX_PATH)
faiss.write_index(text_index, TEXT_INDEX_PATH)

print("Indexes saved:")
print(" -", IMAGE_INDEX_PATH)
print(" -", TEXT_INDEX_PATH)

# ---------------------------------------
# SAVE METADATA (for search results)
# ---------------------------------------
print("\nSaving metadata...")

df.to_parquet(METADATA_PATH, index=False)

print("Metadata saved:", METADATA_PATH)

print("\nFAISS index building complete! ✅")
