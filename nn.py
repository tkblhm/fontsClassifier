import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from image_loader import *

class FontCNN(nn.Module):
    def __init__(self):
        super(FontCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        # x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        return x

# Instantiate model
model = FontCNN()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            # labels = labels.to("cpu", dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_acc = evaluate_model(model, val_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    print("Training Complete.")


# Validation function
def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():  # No gradients needed for validation
        for images, labels in dataloader:
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# Split dataset
dataset = FontDataset(DATASET_PATH, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Training Samples: {train_size}, Validation Samples: {val_size}")
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)
#
# def predict_font(image_path, model):
#     model.eval()
#     transform = transforms.Compose([
#         transforms.Grayscale(),
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#
#     img = Image.open(image_path)
#     img = transform(img).unsqueeze(0)
#     output = model(img).item()
#
#     if output > 0.5:
#         print(f"Predicted Font: {FONT_CLASSES[1]} (Courier New)")
#     else:
#         print(f"Predicted Font: {FONT_CLASSES[0]} (Comic Sans MS)")
#
# # Example usage
# predict_font(r"C:\Users\hxtx1\Pictures\Screenshots\屏幕截图 2025-02-07 141325.png", model)
