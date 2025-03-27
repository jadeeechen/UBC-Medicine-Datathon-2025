from PIL import Image
import pandas as pd
import os

# Load the CSV with bounding box information
bbox_df = pd.read_csv('data/processed/02_bbox_filtered.csv')

input_path = "images/processed/images_002_resized/"
output_path = "images/processed/images_002_cropped/"

def crop_images():
    # List all files in the input directory
    dirs = os.listdir(input_path)

    # Loop through each file in the directory
    for item in dirs:
        
        # Check if image filename exists in the CSV bounding box list
        bbox_row = bbox_df[bbox_df['Image Index'] == item]
        
        # Proceed only if bounding box data exists for the current image
        if not bbox_row.empty:
            # Extract bounding box coordinates from the CSV
            bbox_x = bbox_row['Bbox [x'].values[0]
            bbox_y = bbox_row['y'].values[0]
            bbox_w = bbox_row['w'].values[0]
            bbox_h = bbox_row['h]'].values[0]

            # Construct the path to the image file
            img_path = os.path.join(input_path, item)

            # Open the image
            img = Image.open(img_path)

            # Crop the image using the bounding box coordinates
            img_cropped = img.crop((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))

            # Save the cropped image to the output directory
            output_filename = f"cropped_{item}"
            img_cropped.save(os.path.join(output_path, output_filename))  # Save with a modified name

            print(f'Cropped and saved: {output_filename}')
        else:
            print(f'No bounding box data for {item}')

crop_images()




