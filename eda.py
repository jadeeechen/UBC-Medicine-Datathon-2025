import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PATH = "./data/original/"
FILE = "Data_Entry_2017.csv"
df = pd.read_csv(PATH + FILE)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df.head())
print(df.describe())
# Data Preprocessing
df.info()
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

df = df[df["Patient Age"] <= 120]
df = df[df["Follow-up #"] <= 20]

df = df.drop(columns="Image Index")


# # Split the labels and count occurrences
# all_labels = df["Finding Labels"].str.split('|').explode()  # Splitting and flattening
# label_counts = all_labels.value_counts()

# # Plot the findings distribution
# plt.figure(figsize=(14, 10))
# sns.barplot(x=label_counts.index, y=label_counts.values)
# plt.xticks(rotation=45, ha="right")
# plt.xlabel("Finding Labels")
# plt.ylabel("Count")
# plt.title("Distribution of Finding Labels")
# plt.show()


# Define pathologies
pathologies = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

# Expand "Finding Labels" (split multiple labels per row)
df_exploded = df.assign(Finding_Labels=df["Finding Labels"].str.split("|")).explode("Finding_Labels")

# Keep only the relevant pathologies
df_exploded = df_exploded[df_exploded["Finding_Labels"].isin(pathologies)]

# # ---------------------------
# # Create Individual Histograms for Age Distribution
# fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))  # Adjust grid size
# axes = axes.flatten()  # Flatten for easier indexing
#
# for i, pathology in enumerate(pathologies):
#     ax = axes[i]
#     subset = df_exploded[df_exploded["Finding_Labels"] == pathology]
#
#     ax.hist(subset["Patient Age"], bins=20, color="blue", alpha=0.7, edgecolor="black")
#     ax.set_title(f"Age Distribution: {pathology}")
#     ax.set_xlabel("Age")
#     ax.set_ylabel("Frequency")
#
# plt.tight_layout()
# plt.show()


# # ---------------------------
# # Create Individual Bar Charts for Gender Distribution
# fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))  # Adjust grid size
# axes = axes.flatten()
#
# for i, pathology in enumerate(pathologies):
#     ax = axes[i]
#     subset = df_exploded[df_exploded["Finding_Labels"] == pathology]
#
#     gender_counts = subset["Patient Gender"].value_counts()
#     ax.bar(gender_counts.index, gender_counts.values, color=["blue", "pink"], alpha=0.7)
#     ax.set_title(f"Gender Distribution: {pathology}")
#     ax.set_xlabel("Gender")
#     ax.set_ylabel("Count")
#
# plt.tight_layout()
# plt.show()


# # ---------------------------
# # Create Individual Bar Charts for Gender Distribution
# fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))  # Adjust grid size
# axes = axes.flatten()
#
# for i, pathology in enumerate(pathologies):
#     ax = axes[i]
#     subset = df_exploded[df_exploded["Finding Labels"] == pathology]
#
#     follow_up_counts = subset["Follow-up #"].value_counts()
#     ax.bar(follow_up_counts.index, follow_up_counts.values, color=["blue", "pink"], alpha=0.7)
#     ax.set_title(f"Follow up Distribution: {pathology}")
#     ax.set_xlabel("Number of follow ups")
#     ax.set_ylabel("Count")
#
# plt.tight_layout()
# plt.show()


# X = df.drop(columns="Finding Labels")
# y = df["Finding Labels"]

# if row contains no findings drop?!?!

