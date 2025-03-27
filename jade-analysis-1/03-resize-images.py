from PIL import Image
import os

# Define the paths
input_path = "images/original/images_002/"
output_path = "images/processed/images_002_resized/"

def resize():
    # List all files in the input directory
    dirs = os.listdir(input_path)
    #TODO - should add a check for whether dir exists
    # Loop through each file
    for item in dirs:
        if item.lower().endswith((".png")):
            # Open the image
            im = Image.open(os.path.join(input_path, item))
            
            # Resize the image to 128x128
            imResize = im.resize((128, 128))
            
            # Save the resized image to the output directory
            imResize.save(os.path.join(output_path, item))  # Save with the same name in the new folder

resize()