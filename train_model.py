import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "animal_skin_scanner_dataset"

batch_size = 16
image_size = 224

train_transform = transforms.Compose([  # Data augmentation for training   
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
valid_data = ImageFolder(os.path.join(data_dir, "valid"), transform=valid_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)

print("Classes:", train_data.classes)
print("Train size:", len(train_data))
print("Valid size:", len(valid_data))

print("Loading model...")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

epochs = 20

print("Starting training...")

for epoch in range(epochs): # Training loop
    model.train()
    train_correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_data)

    model.eval()
    val_correct = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(valid_data)

    print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

torch.save(model.state_dict(), "animal_disease_weights.pt")
print("Training complete")