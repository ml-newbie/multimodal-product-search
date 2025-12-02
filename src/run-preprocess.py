from preprocess import clean_csv, resize_images

if __name__ == "__main__":
    # Step 1: Clean the CSV
    df_clean = clean_csv(
        input_csv="data/styles.csv",
        images_dir="data/e-commerce/images",
        output_csv="data/products_clean.csv"
    )

    # Step 2: Resize images
    resize_images(
        df=df_clean,
        input_dir="data/e-commerce/images",
        output_dir="data/images_resized"
    )
