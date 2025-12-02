# src/preprocess.py

import pandas as pd
from PIL import Image
import os

def clean_csv(input_csv, images_dir, output_csv):
    df = pd.read_csv(input_csv)
    df['image_path'] = df['id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
    df_clean = df[['id', 'productDisplayName', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'image_path']]
    df_clean.rename(columns={
        'id':'product_id',
        'productDisplayName':'title',
        'masterCategory':'category',
        'subCategory':'subcategory'
    }, inplace=True)
    df_clean.to_csv(output_csv, index=False)
    print(f"Cleaned CSV saved to {output_csv}")
    return df_clean

def resize_images(df, input_dir, output_dir, size=(224,224)):
    os.makedirs(output_dir, exist_ok=True)
    for idx, row in df.iterrows():
        img_file = f"{row['product_id']}.jpg"
        img_path = os.path.join(input_dir, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(size)
            img.save(os.path.join(output_dir, img_file))
        except Exception as e:
            print(f"Skipping {img_file}: {e}")
    print(f"Images resized to {size} and saved in {output_dir}")
