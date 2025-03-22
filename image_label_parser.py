import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)

df = pd.read_csv('./data/original/BBox_List_2017.csv')
print(df.head(1))

image_folder = 'images/original/images'
image_files = set(os.listdir(image_folder))

image_files = {f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))}

df_filtered = df[df['Image Index'].isin(image_files)]

df_filtered.to_csv('filtered_labels.csv', index=False)