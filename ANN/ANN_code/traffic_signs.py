import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from datetime import datetime
import time

# Function to get categories from a text file
def get_categories(file_location):
    try:
        with open(file_location, 'r') as file:
            lines = file.readlines()
            first_words = [line.split()[0] for line in lines]
            return first_words
    except FileNotFoundError:
        print(f"Error: The file '{file_location}' was not found.")
        return []

# Set data directory and load categories
datadir = '../ANN_data/100x/'
Categories = get_categories(datadir + "description.txt")
print("Categories read from description file:")
print(Categories)

# Load and preprocess data
def load_data_from_folder(path, categories):
    data = []
    labels = []
    for i in categories:
        folder_path = os.path.join(path, i)
        for img in os.listdir(folder_path):
            if img.endswith(".ppm"):
                img_array = imread(os.path.join(folder_path, img))
                img_resized = resize(img_array, (15, 15, 3))
                data.append(img_resized.flatten())
                labels.append(categories.index(i))
    return np.array(data), np.array(labels)

# Load training, validation, and test data
train_path = os.path.join(datadir, "Training")
x_train, y_train = load_data_from_folder(train_path, Categories)
print("Training set already read")

validation_path = os.path.join(datadir, "Validation")
x_validation, y_validation = load_data_from_folder(validation_path, Categories)
print("Validation set already read")

test_path = os.path.join(datadir, "Test")
x_test, y_test = load_data_from_folder(test_path, Categories)
print("Testing set already read")

# Convert data to PyTorch tensors
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).long()
x_validation = torch.tensor(x_validation).float()
y_validation = torch.tensor(y_validation).long()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).long()

# === ANN DataLoaders ===
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

validation_dataset = TensorDataset(x_validation, y_validation)
validation_loader = DataLoader(validation_dataset, batch_size=32)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32)

# === CNN Input Reshaping and DataLoaders ===
x_train_cnn = x_train.view(-1, 3, 15, 15)
x_validation_cnn = x_validation.view(-1, 3, 15, 15)
x_test_cnn = x_test.view(-1, 3, 15, 15)

train_dataset_cnn = TensorDataset(x_train_cnn, y_train)
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=32, shuffle=True)

validation_dataset_cnn = TensorDataset(x_validation_cnn, y_validation)
validation_loader_cnn = DataLoader(validation_dataset_cnn, batch_size=32)

test_dataset_cnn = TensorDataset(x_test_cnn, y_test)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=32)

# === ANN Model ===
class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        self.fc1 = nn.Linear(15 * 15 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(Categories))

    def forward(self, x):
        x = x.view(-1, 15 * 15 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# === CNN Model ===
class TrafficSignCNN(nn.Module):
    def __init__(self):
        super(TrafficSignCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, len(Categories))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # -> [batch, 32, 7, 7]
        x = self.pool(torch.relu(self.conv2(x)))  # -> [batch, 64, 3, 3]
        x = x.view(-1, 64 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# === Training and Evaluation Functions ===
def train_model(model, train_loader, criterion, optimizer, val_loader=None, num_epochs=10, patience=3, name="Model"):
    model.train()
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        val_acc = None
        if val_loader is not None:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_outputs = model(val_inputs)
                    _, val_predicted = torch.max(val_outputs, 1)
                    total += val_labels.size(0)
                    correct += (val_predicted == val_labels).sum().item()
            val_acc = correct / total

            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        # Print training and validation stats
        if val_acc is not None:
            print(f"{name} Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"{name} Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

        # Early stopping condition
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
            break

def evaluate_model(model, val_loader, name="Validation"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"{name} Accuracy: {accuracy:.4f}")

# === Run ANN ===
model = TrafficSignNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nTraining ANN model with early stopping...")
train_model(model, train_loader, criterion, optimizer, val_loader=validation_loader, num_epochs=50, patience=3, name="ANN")
evaluate_model(model, validation_loader, name="ANN Validation")
evaluate_model(model, test_loader, name="ANN Test")

# === Run CNN ===
cnn_model = TrafficSignCNN()
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

print("\nTraining CNN model with early stopping...")
train_model(cnn_model, train_loader_cnn, cnn_criterion, cnn_optimizer, val_loader=validation_loader_cnn, num_epochs=50, patience=3, name="CNN")
evaluate_model(cnn_model, validation_loader_cnn, name="CNN Validation")
evaluate_model(cnn_model, test_loader_cnn, name="CNN Test")

# === Final Comparison ===
print("\n==== FINAL ACCURACY COMPARISON ====")
evaluate_model(model, test_loader, name="ANN Test")
evaluate_model(cnn_model, test_loader_cnn, name="CNN Test")
