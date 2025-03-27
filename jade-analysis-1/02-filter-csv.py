import os
import pandas as pd

# Path to the folder containing your images
images_dir = 'images/original/images_002'

# Get a list of all image filenames in the images 002 folder
images_filenames = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Load the bounding box dataframe
entry_df = pd.read_csv('data/processed/01_entry_drop_unnamed.csv')

bbox_df = pd.read_csv('data/processed/01_bbox_drop_unnamed.csv')

# Filter bbox_df to only include rows where the Image Index is in the list of filenames
filtered_entry_df = entry_df[entry_df['Image Index'].isin(images_filenames)]

filtered_bbox_df = bbox_df[bbox_df['Image Index'].isin(images_filenames)]

# drop all rows that contain 'No Finding'
drop_filtered_entry_df = filtered_entry_df.drop(filtered_entry_df[filtered_entry_df['Finding Labels'] == 'No Finding'].index)

# Compare the (1) og vs (2) prefiltered vs (3) prefiltered and dropped shape
print(entry_df.shape)
print(filtered_entry_df.shape)
print(drop_filtered_entry_df.shape)

print(bbox_df.shape)
print(filtered_bbox_df.shape)

# Save the cleaned datasets to new CSV files
drop_filtered_entry_df.to_csv('data/processed/02_entry_filtered.csv', index=False)
filtered_bbox_df.to_csv('data/processed/02_bbox_filtered.csv', index=False)