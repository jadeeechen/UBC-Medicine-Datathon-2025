import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd

model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = torch.nn.Linear(num_ftrs, 1)
model = torch.nn.Sequential(model, torch.nn.Sigmoid()) # Used for binary classification

# Load trained weights (replace with actual CheXNet weights if available)
# Example: model.load_state_dict(torch.load("chexnet.pth"))

model.eval()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_hernia(image_tensor):
  with torch.no_grad():
    output = model(image_tensor)
    prob = output.item()
    prediction = 1 if prob >= 0.5 else 0
    return prediction, prob

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os

def evaluate_model(model, df, image_folder, target_disease="Infiltration"):
    model.eval()
    y_true = []
    y_pred = []

    for i, row in df.iterrows():
        image_path = os.path.join(image_folder, row["Image Index"])

        # Skip if image is missing
        if not os.path.exists(image_path):
            continue

        try:
            # Preprocess image
            image_tensor = preprocess_image(image_path)

            # Get model prediction
            with torch.no_grad():
                output = model(image_tensor)
                prob = output.item()
                pred = 1 if prob >= 0.5 else 0
        except Exception as e:
            print(f"Skipping {image_path} due to error: {e}")
            continue

        # Ground truth: 1 if 'Hernia' in label list
        labels = row["Finding Labels"]
        true_label = 1 if target_disease in labels else 0

        y_true.append(true_label)
        y_pred.append(pred)

    # Evaluation metrics
    print("\nEvaluation Metrics for Infiltration Detection:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, target_names=["No Infiltration", "Infiltration"]))

# Assume your dataframe is already loaded and cleaned
df = pd.read_csv("Data_Entry_2017.csv")
df["Finding Labels"] = df["Finding Labels"].apply(lambda x: x.split('|') if x != "No Finding" else [])
df["image_path"] = df["Image Index"].apply(lambda x: os.path.join("images", x))
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

# Use a sample or full set (e.g., df.head(500))
evaluate_model(model, df.head(500), image_folder="images")

# Best Labels: Infiltration, Effusion