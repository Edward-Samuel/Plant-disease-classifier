import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torchvision import transforms
import pandas as pd
from PIL import Image

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Class Labels
class_labels = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
]

# Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(224, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_paths = self.data['image_path'].values
        self.labels = self.data['folder_name'].values
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(class_labels)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        folder_name = self.labels[idx]
        label = self.label_map[folder_name]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the model and modify the classifier
model = AutoModelForImageClassification.from_pretrained("facebook/deit-tiny-patch16-224", trust_remote_code=True)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier.in_features, len(class_labels))
)
model.to(device)

# Load dataset
csv_file = "/mnt/c/intern_project/train_images.csv"  
full_dataset = CustomImageDataset(csv_file=csv_file, transform=transform)

# Balanced sampling to handle class imbalance
class_counts = [len([label for label in full_dataset.labels if label == cls]) for cls in class_labels]
class_weights = [1.0 / count for count in class_counts]
sample_weights = [class_weights[full_dataset.label_map[label]] for label in full_dataset.labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(full_dataset, batch_size=32, sampler=sampler)

# Define loss, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    scheduler.step(running_loss)

# Save the model
model_save_path = "enhanced_vit.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
