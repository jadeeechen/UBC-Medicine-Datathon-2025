import pandas as pd

#load datasets
df_entry = pd.read_csv('data/original/Data_Entry_2017.csv')

df_bbox = pd.read_csv('data/original/BBox_List_2017.csv')

#quick check of datasets
df_entry.head()

df_bbox.head()

# check column names and count of each unique value, drop the unnamed(empty) columns)
df_entry.columns
df_entry.nunique()
df_entry_drop_unnamed = df_entry.drop(columns=['Unnamed: 11'])

df_bbox.columns
df_bbox.nunique()
df_bbox_drop_unnamed = df_bbox.drop(columns=['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'])


# column_of_interest = 'Finding Labels'
# df_entry[column_of_interest].unique()
# df_entry[column_of_interest].value_counts()
# # drop all rows that contain 'No Finding'
# df_entry = df_entry.drop(df_entry[df_entry['Finding Labels'] == 'No Finding'].index)

# Save the cleaned datasets to new CSV files
df_entry_drop_unnamed.to_csv('data/processed/01_entry_drop_unnamed.csv', index=False)
df_bbox_drop_unnamed.to_csv('data/processed/01_bbox_drop_unnamed.csv', index=False)