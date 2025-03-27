import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Load pre-trained MobileNetV2
mobilenet = models.mobilenet_v2(pretrained=True)

# Modify the classifier for 14 classes (14 diseases)
num_classes = 14
mobilenet.classifier = nn.Sequential(
    nn.Dropout(0.2),  # Helps prevent overfitting
    nn.Linear(mobilenet.last_channel, num_classes)
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet = mobilenet.to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer (Adam works well for fine-tuning)
optimizer = optim.Adam(mobilenet.parameters(), lr=0.0001)

# Learning rate scheduler (optional)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

class CTScanDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['Image Index'])
        image = Image.open(img_name).convert("RGB")  # Ensure 3-channel RGB
        label = self.data.iloc[idx]['Finding Labels']

        # Convert label to index (if not already one-hot encoded)
        label_dict = {disease: i for i, disease in enumerate([
            "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", 
            "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia", 
            "Pleural_thickening", "Cardiomegaly", "Nodule Mass", "Hernia"
        ])}
        label_idx = label_dict.get(label, 14)  # Default to "No findings" if not listed

        if self.transform:
            image = self.transform(image)

        return image, label_idx

# Define DataLoader
train_dataset = CTScanDataset(csv_file="data/processed/02_entry_filtered.csv", img_dir="images/processed/images_002_normalized", transform=None)  # Add transform
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

num_epochs = 10  # Adjust based on dataset size

for epoch in range(num_epochs):
    mobilenet.train()  # Set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Clear gradients

        outputs = mobilenet(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    scheduler.step()  # Update learning rate

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
