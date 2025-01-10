import os
import random
import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torchvision import transforms
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

test_csv = "/mnt/c/intern_project/test_images.csv"  
test_csv =  "/mnt/c/intern_project/validation_images.csv"  

model = AutoModelForImageClassification.from_pretrained("facebook/deit-tiny-patch16-224", trust_remote_code=True)
model.classifier = nn.Linear(model.classifier.in_features, len(class_labels))  
model.load_state_dict(torch.load("deit_tiny_multilabel.pth",weights_only=True))  
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def get_random_images_from_csv(csv_file, num_images=5):
    df = pd.read_csv(csv_file)
    random_rows = df.sample(n=num_images)
    images = []
    for _, row in random_rows.iterrows():
        image_path = row['image_path']
        label = row['folder_name']
        images.append((image_path, label))
    return images

num_images_to_test = 20  

random_images = get_random_images_from_csv(test_csv, num_images=num_images_to_test)

cols = 4
rows = (num_images_to_test // cols) + (1 if num_images_to_test % cols != 0 else 0)
fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
axes = axes.flatten()

correct_predictions = 0
total_predictions = 0


for idx, (image_path, true_label) in enumerate(random_images):
    image_tensor = load_image(image_path, transform)
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    predicted_label = class_labels[predicted_class]
    total_predictions += 1
    if predicted_label == true_label:
        correct_predictions += 1


    image = Image.open(image_path)
    axes[idx].imshow(image)
    axes[idx].set_title(f"Pred: {predicted_label}\nTrue: {true_label}")
    axes[idx].axis('off')

for i in range(idx + 1, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()

accuracy = (correct_predictions / total_predictions) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Optionally, run tests multiple times without visualization for better accuracy estimation
# num_tests = 20  # Run the test multiple times
# accuracies = []
# for _ in range(num_tests):
#     random_images = get_random_images_from_csv(test_csv, num_images=num_images_to_test)
#     correct_predictions = 0
#     total_predictions = 0
#     for _, (image_path, true_label) in enumerate(random_images):
#         image_tensor = load_image(image_path, transform)
#         with torch.no_grad():
#             outputs = model(pixel_values=image_tensor)
#             logits = outputs.logits
#             predicted_class = torch.argmax(logits, dim=1).item()
#         predicted_label = class_labels[predicted_class]
#         total_predictions += 1
#         if predicted_label == true_label:
#             correct_predictions += 1
#     accuracy = (correct_predictions / total_predictions) * 100
#     accuracies.append(accuracy)
#     print(accuracy)
# average_accuracy = sum(accuracies) / len(accuracies)
# print(f"Average Accuracy over {num_tests} runs: {average_accuracy:.2f}%")
