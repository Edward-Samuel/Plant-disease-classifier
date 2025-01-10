import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.image_paths = self.data['image_path'].values
        self.labels = self.data['folder_name'].values
        self.transform = transform
        self.class_labels = [
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

        self.label_map = {label: idx for idx, label in enumerate(self.class_labels)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        folder_name = self.labels[idx]
        label = self._get_label_from_folder_name(folder_name)
        
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_label_from_folder_name(self, folder_name):
       
        label_vector = [0] * len(self.class_labels)
        if folder_name in self.label_map:
            label_vector[self.label_map[folder_name]] = 1
        return torch.tensor(label_vector, dtype=torch.float32)

processor = AutoImageProcessor.from_pretrained("facebook/deit-tiny-patch16-224", use_fast=True, trust_remote_code=True)
model = AutoModelForImageClassification.from_pretrained("facebook/deit-tiny-patch16-224", trust_remote_code=True)
model.classifier = nn.Linear(model.classifier.in_features, len(class_labels))
model.to(device)


csv_file = "/mnt/c/intern_project/train_images.csv"  

full_dataset = CustomImageDataset(csv_file=csv_file, transform=transform)

def get_random_subset(dataset, num_samples_per_class):
    class_indices = {cls: [] for cls in class_labels}
    for idx, (_, label) in enumerate(dataset):
        cls_index = torch.argmax(label).item()
        class_name = class_labels[cls_index]
        class_indices[class_name].append(idx)

    subset_indices = []
    for cls, indices in class_indices.items():
        subset_indices.extend(random.sample(indices, min(len(indices), num_samples_per_class)))

    return Subset(dataset, subset_indices)

subset_dataset = get_random_subset(full_dataset, num_samples_per_class=250)

train_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
num_epochs = 30

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

        predicted = torch.sigmoid(outputs.logits) > 0.5
        correct += (predicted == labels).all(dim=1).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

model_save_path = "vit2.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
