from PIL import Image
import numpy as np
import os

# Define input and output paths
input_path = "images/processed/images_002_cropped/"
output_path = "images/processed/images_002_normalized/"

def normalize_images():
    # List all files in the input directory
    dirs = os.listdir(input_path)

    # Loop through each file in the directory
    for item in dirs:
        if item.lower().endswith((".png", ".jpg", ".jpeg")):
            # Construct the path to the image file
            img_path = os.path.join(input_path, item)
            
            # Open the image
            img = Image.open(img_path)

            # Convert the image to a numpy array for easier manipulation
            img_array = np.array(img)

            # Normalize the pixel values to the range [0, 1] by dividing by 255
            img_normalized = img_array / 255.0

            # Convert back to an image object
            img_normalized = Image.fromarray((img_normalized * 255).astype(np.uint8))  # Scaling back to 0-255

            # Save the normalized image to the output directory
            output_filename = f"normalized_{item}"
            img_normalized.save(os.path.join(output_path, output_filename))  # Save with a modified name

            print(f'Normalized and saved: {output_filename}')
        else:
            print(f'Skipping non-image file: {item}')

normalize_images()
