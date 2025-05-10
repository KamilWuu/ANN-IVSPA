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
datadir = '../ANN_data/database_1000_photos/'
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

# Load training data
train_path = os.path.join(datadir, "Training")
x_train, y_train = load_data_from_folder(train_path, Categories)
print("Training set already read")

# Load validation data
validation_path = os.path.join(datadir, "Validation")
x_validation, y_validation = load_data_from_folder(validation_path, Categories)
print("Validation set already read")

# Load testing data
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

# Create DataLoaders
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

validation_dataset = TensorDataset(x_validation, y_validation)
validation_loader = DataLoader(validation_dataset, batch_size=32)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define ANN model
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

# Initialize model, loss function and optimizer
model = TrafficSignNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Evaluation function
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

# Run training and evaluation
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
evaluate_model(model, validation_loader, name="Validation")
evaluate_model(model, test_loader, name="Test")
