import pandas as pd
import matplotlib.pyplot as plt

#load datasets
#note: manually deleted rows onwards from 00029479_003.png (broken image)
df = pd.read_csv('sample_labels.csv')

#quick check of datasets
df.head()

# check column names and count of each unique value, drop the unnamed(empty) columns)
df.columns
df.nunique()
df['Finding Labels'].unique()

# Check for missing values in each column
df.isnull().sum()  #RESULT: No missing values


# Split the 'Finding Labels' into individual diseases by separating on the '|' character
diseases = df['Finding Labels'].str.split('|', expand=True).stack()

# Get the count of each disease
disease_counts = diseases.value_counts()


# Plot the bar plot for disease occurances
diseases = df['Finding Labels'].str.split('|', expand=True).stack() # Split the 'Finding Labels' into individual diseases by separating on the '|' character
disease_counts = diseases.value_counts() # Get the count of each disease
plt.figure(figsize=(10, 5))
disease_counts.plot(kind='bar', color='skyblue')
plt.title('Disease Occurance in Dataset')
plt.xlabel('Disease')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/eda/disease_occurance_bar.png', bbox_inches='tight') # Saves as PNG
plt.show() # Show the plot

# Visualize Sample Images: It’s useful to visualize a few sample images to get a sense of the data.

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # Displaying random sample of images
# sample_image_indices = df['Image Index'].sample(5).values  # Take random sample of 5 images
# fig, axes = plt.subplots(1, 5, figsize=(15, 5))
# for i, ax in enumerate(axes):
#     img = mpimg.imread(f'images/{sample_image_indices[i]}')
#     ax.imshow(img)
#     ax.set_title(df.loc[df['Image Index'] == sample_image_indices[i], 'Finding Labels'].values[0])
#     ax.axis('off')
# plt.show()