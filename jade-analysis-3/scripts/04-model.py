import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE  # Import SMOTE
from tensorflow.keras.callbacks import EarlyStopping

# Define constants
IMG_SIZE = 128
BATCH_SIZE = 32
DATASET_CSV = "02_filtered_single_labels.csv"
IMAGE_DIR = "images"


# Load dataset
labels_df = pd.read_csv("02_filtered_single_labels.csv")

# Filter for only Infiltration and No Finding
labels_df = labels_df[labels_df['Finding Labels'].isin(['Infiltration', 'No Finding'])]

# Update target column
labels_df["target-infiltration"] = labels_df['Finding Labels'].apply(lambda x: 1 if x == "Infiltration" else 0)

# Check balance of classes
print(labels_df["target-infiltration"].value_counts())

# Function to load images
def load_images(df, img_dir):
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row["Image Index"])
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img / 255.0)  # Normalize
            labels.append(row["target-infiltration"])
    return np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1), np.array(labels)


# # Load dataset
# labels_df = pd.read_csv("02_filtered_single_labels.csv")
# #labels_df["target-infiltration"] = labels_df['Finding Labels'].apply(lambda x: 1 if x == "Infiltration" else 0)
# labels_df["target-any-disease"] = labels_df['Finding Labels'].apply(lambda x: 0 if x == "No Finding" else 1)

# labels_df.head(3)

# # Function to load images
# def load_images(df, img_dir):
#     images = []
#     labels = []
#     for _, row in df.iterrows():
#         img_path = os.path.join(img_dir, row["Image Index"])
#         if os.path.exists(img_path):
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#             images.append(img / 255.0)  # Normalize
#             #labels.append(row["target-infiltration"])
#             labels.append(row["target-any-disease"])
#     return np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1), np.array(labels)

# Load data
X, y = load_images(labels_df, IMAGE_DIR)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X.reshape(X.shape[0], -1), y)  # Reshape X to 2D for SMOTE

# Reshape X back to 4D after SMOTE
X_resampled = X_resampled.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Check class distribution before and after SMOTE
print(f"Original class distribution: {np.unique(y, return_counts=True)}\nResampled class distribution after SMOTE: {np.unique(y_resampled, return_counts=True)}")

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Perform the stratified split
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2,2)),

    #increased filters in the second convolutional layer from 64 to 128
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# improve model attempt: data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Define class weights
class_weight = {0: 1., 1: 5.}  # Assign higher weight to the minority class (Infiltration)

early_stopping = EarlyStopping(
    monitor='val_loss',           # Monitor validation loss
    patience=3,                   # Stop after 3 epochs of no improvement
    restore_best_weights=True     # Restore weights from the best epoch
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=BATCH_SIZE,
    class_weight=class_weight, #Add class weights
    callbacks=[early_stopping]  # Add early stopping here
)

# Save model
model.save('infiltration_model.keras')

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to 0 or 1

# Calculate precision, recall, and F1 score, and generate confusion matrix
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nConfusion Matrix:\n{cm}")

