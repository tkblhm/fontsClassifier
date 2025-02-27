import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from image_loader import *
import matplotlib.pyplot as plt
import numpy as np

class FontCNN(nn.Module):
    def __init__(self, num_fonts):
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
        self.fc3 = nn.Linear(64, num_fonts)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, criterion, optimizer):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer

        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    def load_model(self, file):
        self.model.load_state_dict(torch.load(file, map_location=self.device))
        self.model.to(device)

    def save_model(self, file):
        torch.save(self.model.state_dict(), file)

    def plot(self):
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def predict(self, image_path):
        self.model.eval()

        img = Image.open(image_path).convert("L")
        img = self.transform(img)

        print("img:", img)
        np.savetxt("D:gyt\\array2.txt", img.view(-1).numpy())
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = model(img)

        # Get the predicted class
        predicted_class = torch.argmax(output, dim=1).item()
        print("distribution:", output)
        print(f"Predicted Font: {predicted_class}")
        return predicted_class


        
    def train_model(self, epochs=10, plot=True):
        if plot:
            self.train_losses = []
            self.val_losses = []
            self.train_acc = []
            self.val_acc = []
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct, total = 0, 0

            for images, labels in self.train_loader:
                # labels = labels.to("cpu", dtype=torch.float)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

            train_acc = 100 * correct / total
            val_loss, val_acc = self.evaluate_model()
            if plot:
                self.train_losses.append(running_loss)
                self.train_acc.append(train_acc)
                self.val_losses.append(val_loss)
                self.val_acc.append(val_acc)
            print(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        print("Training Complete.")


    # Validation function
    def evaluate_model(self):
        self.model.eval()
        correct, total = 0, 0
        loss = 0
        with torch.no_grad():  # No gradients needed for validation
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        return loss, 100 * correct / total

# Instantiate model
model = FontCNN(len(FONT_CLASSES))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define loss and optimizer
criterion = nn.CrossEntropyLoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)
