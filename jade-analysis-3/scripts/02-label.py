import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('sample_labels.csv')

label_encoder = LabelEncoder()
df['Finding Labels Encoded'] = label_encoder.fit_transform(df['Finding Labels'])

import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# Define a function to resize images
def resize_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    return img

# Randomly select a few image indices from the 'Image Index' column of your DataFrame
sample_image_indices = random.sample(list(df['Image Index']), 5)

# Displaying the resized sample images
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    img_resized = resize_image(f'images/{sample_image_indices[i]}')
    ax.imshow(img_resized)
    ax.set_title(df.loc[df['Image Index'] == sample_image_indices[i], 'Finding Labels'].values[0])
    ax.axis('off')  # Hide axes
plt.show()

# Identify multi-label and single-label cases
# The pipe | is a special character in regular expressions, 
# which causes it to match any string containing the character "or".
# Need to escape the pipe symbol by using \| to match it literally.
df['Is Multi-label'] = df['Finding Labels'].str.contains('\|') 

# Count the number of single-label and multi-label cases
single_label_count = df[~df['Is Multi-label']].shape[0]
multi_label_count = df[df['Is Multi-label']].shape[0]

# Calculate proportions
total_count = df.shape[0]
single_label_proportion = single_label_count / total_count
multi_label_proportion = multi_label_count / total_count

# Print the results
print(f"Single-label count: {single_label_count}")
print(f"Multi-label count: {multi_label_count}")
print(f"Proportion of single-label cases: {single_label_proportion:.2f}")
print(f"Proportion of multi-label cases: {multi_label_proportion:.2f}")

# Results: multilables only use a small proportion
# Single-label count: 4626
# Multi-label count: 980
# Proportion of single-label cases: 0.83
# Proportion of multi-label cases: 0.17

# Filter out the multi-label rows
df_filtered = df[~df['Is Multi-label']]

# Save the filtered DataFrame to a CSV file
df_filtered.to_csv('02_filtered_single_labels.csv', index=False)

# Optional: Confirm that the file has been saved
print("Filtered dataset saved as 'filtered_single_labels.csv'")


# TODO: try not to delete ALL of the multilabels, use the following to try to delete only classes with 1 sample
# TODO: that would mean my model needs to be able to handle multiple labels (perhaps split them into individual row?)

# # Remove classes with only 1 sample
# class_counts = df['Finding Labels Encoded'].value_counts()
# classes_to_keep = class_counts[class_counts > 1].index
# df_filtered = df[df['Finding Labels Encoded'].isin(classes_to_keep)]