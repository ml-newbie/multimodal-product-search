import streamlit as st
import faiss
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import requests
from io import BytesIO

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
IMAGE_INDEX_PATH = "data/indexes/image_index.faiss"
TEXT_INDEX_PATH = "data/indexes/text_index.faiss"
METADATA_PATH = "data/indexes/metadata.parquet"
LOCAL_IMAGE_DIR = "data/e-commerce/images"
GDRIVE_MAPPING_PATH = "data/gdrive_mapping.csv"
DEVICE = "cpu"

# ---------------------------------------------------------
# DYNAMIC IMAGE LOADING CONTROL (Local-First Logic)
# ---------------------------------------------------------
FORCE_GDRIVE_IMAGES = False  # Set to False to enable local-first dynamic loading
if FORCE_GDRIVE_IMAGES:
    st.info("⚠️ Using Google Drive images locally for testing. Local images are ignored.")

# Detect if running on Streamlit Cloud
ON_STREAMLIT_CLOUD = "CLOUD_ENV" in os.environ
USE_GDRIVE = FORCE_GDRIVE_IMAGES or ON_STREAMLIT_CLOUD

# ---------------------------------------------------------
# LOAD MODEL & INDEXES
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_resource
def load_indexes():
    image_index = faiss.read_index(IMAGE_INDEX_PATH)
    text_index = faiss.read_index(TEXT_INDEX_PATH)
    return image_index, text_index

@st.cache_resource
def load_metadata():
    return pd.read_parquet(METADATA_PATH)

@st.cache_data
def load_gdrive_mapping():
    if os.path.exists(GDRIVE_MAPPING_PATH):
        return pd.read_csv(GDRIVE_MAPPING_PATH)
    else:
        return pd.DataFrame(columns=["product_id", "gdrive_url"])

model, processor = load_model()
image_index, text_index = load_indexes()
df = load_metadata()
gdrive_mapping_df = load_gdrive_mapping()

# ---------------------------------------------------------
# HELPER: Convert Google Drive share link to direct URL
# ---------------------------------------------------------
def gdrive_direct_url(share_url):
    """Converts a Google Drive share link to a direct download URL."""
    if "drive.google.com" not in share_url:
        return share_url  # Already a direct link
    try:
        file_id = share_url.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=view&id={file_id}"
    except IndexError:
        return share_url

# ---------------------------------------------------------
# RESOLVE IMAGE PATHS
# ---------------------------------------------------------
def resolve_image_path(product_id):
    # 1. Try local image first
    local_path = os.path.join(LOCAL_IMAGE_DIR, f"{product_id}.jpg")
    if os.path.exists(local_path):
        return local_path

    # 2. If not found locally, use Google Drive URL as a fallback
    row = gdrive_mapping_df.loc[gdrive_mapping_df["product_id"] == int(product_id)]
    if not row.empty:
        share_url = row.iloc[0]["gdrive_url"]
        return gdrive_direct_url(share_url)

    return None

df['image_path'] = df['product_id'].apply(resolve_image_path)

# Graceful exit if no images
if df['image_path'].isna().all():
    st.error("No product images found! Please check local or Google Drive image paths.")
    st.stop()

# ---------------------------------------------------------
# IMAGE LOADER WITH CACHING (Handles both Local Path and URL)
# ---------------------------------------------------------
@st.cache_resource
def load_image(img_path):
    try:
        if img_path.startswith("http"):
            # Load from Google Drive URL (or any direct URL)
            response = requests.get(img_path, timeout=10)
            response.raise_for_status() 
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # Load from local file path
            img = Image.open(img_path).convert("RGB")
        return img
    except Exception as e:
        return None

# ---------------------------------------------------------
# EMBEDDING FUNCTIONS
# ---------------------------------------------------------
def embed_text(query: str):
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb.cpu().numpy()
    emb /= np.linalg.norm(emb)
    return emb

def embed_image(img: Image.Image):
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb.cpu().numpy()
    emb /= np.linalg.norm(emb)
    return emb

# ---------------------------------------------------------
# SEARCH FUNCTIONS
# ---------------------------------------------------------
def search_by_text(query: str, k=12):
    query_emb = embed_text(query)
    scores, indices = text_index.search(query_emb, k)
    return scores[0], indices[0]

def search_by_image(img: Image.Image, k=12):
    query_emb = embed_image(img)
    scores, indices = image_index.search(query_emb, k)
    return scores[0], indices[0]

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
st.set_page_config(page_title="Multimodal Product Search", layout="wide")
st.title("🛍️ Multimodal Product Search Engine")
st.markdown(
    '<p style="font-size:10px; color:gray; text-align:left;">Based on OpenAI CLIP Pre-trained Model - © 2025 John Merwin. All rights reserved.</p>',
    unsafe_allow_html=True
)
st.write("Search products using **text** or **image upload**")

tab1, tab2 = st.tabs(["🔤 Text Search", "🖼️ Image Search"])

# ---------------------------------------------------------
# TEXT SEARCH TAB (Spinner Added)
# ---------------------------------------------------------
with tab1:
    text_query = st.text_input("Enter product description (ex: 'red cotton men's shirt')")
    if st.button("Search", type="primary"):
        if text_query.strip() == "":
            st.warning("Please enter a search query.")
        else:
            # --- START SPINNER for search operation ---
            with st.spinner('Searching for products...'):
                scores, idxs = search_by_text(text_query)
                
            st.subheader("Results")
            
            num_cols = min(4, len(idxs)) 
            cols = st.columns(num_cols) 

            for j, (i, score) in enumerate(zip(idxs, scores)):
                item = df.iloc[i]
                img_path = item["image_path"]
                
                # We can load the image inside the loop, but since load_image is cached,
                # the result is fast unless it's a new URL/file.
                img = load_image(img_path) if img_path else None
                
                col = cols[j % num_cols] 

                with col:
                    if img:
                        st.image(img, img.width) 
                    else:
                        st.warning(f"Image for product {item['product_id']} failed to load.")

                    st.write(f"**{item['title']}**")
                    st.caption(f"Score: {float(score):.4f}")

# ---------------------------------------------------------
# IMAGE SEARCH TAB (Spinner Added)
# ---------------------------------------------------------
with tab2:
    uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Query Image", width=200)
        
        if st.button("Search Similar Products"):
            # --- START SPINNER for search operation ---
            with st.spinner('Searching for similar products...'):
                scores, idxs = search_by_image(img)
            
            st.subheader("Results")
            
            num_cols = min(4, len(idxs))
            cols = st.columns(num_cols)

            for j, (i, score) in enumerate(zip(idxs, scores)):
                item = df.iloc[i]
                img_path = item["image_path"]
                result_img = load_image(img_path) if img_path else None
                
                col = cols[j % num_cols]

                with col:
                    if result_img:
                        st.image(result_img, result_img.width) 
                    else:
                        st.warning(f"Image for product {item['product_id']} failed to load.")

                    st.write(f"**{item['title']}**")
                    st.caption(f"Score: {float(score):.4f}")