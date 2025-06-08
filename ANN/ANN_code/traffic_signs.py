import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product

from skimage.io import imread
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load category labels
def get_categories(file_location):
    try:
        with open(file_location, 'r') as file:
            lines = file.readlines()
            first_words = [line.split()[0] for line in lines]
            return first_words
    except FileNotFoundError:
        print(f"Error: The file '{file_location}' was not found.")
        return []

# Load image data
def load_data_from_folder(path, categories, cnn=False):
    data, labels = [], []
    for i in categories:
        folder_path = os.path.join(path, i)
        for img in os.listdir(folder_path):
            if img.endswith(".ppm"):
                img_array = imread(os.path.join(folder_path, img))
                img_resized = resize(img_array, (15, 15, 3))
                if cnn:
                    data.append(np.transpose(img_resized, (2, 0, 1)))  # NCHW for CNN
                else:
                    data.append(img_resized.flatten())
                labels.append(categories.index(i))
    return np.array(data), np.array(labels)

# Dataset setup
datadir = '../ANN_data/100x/'
Categories = get_categories(datadir + "description.txt")

train_path = os.path.join(datadir, "Training")
x_train, y_train = load_data_from_folder(train_path, Categories)
x_train_cnn, _ = load_data_from_folder(train_path, Categories, cnn=True)

validation_path = os.path.join(datadir, "Validation")
x_validation, y_validation = load_data_from_folder(validation_path, Categories)
x_validation_cnn, _ = load_data_from_folder(validation_path, Categories, cnn=True)

test_path = os.path.join(datadir, "Test")
x_test, y_test = load_data_from_folder(test_path, Categories)
x_test_cnn, _ = load_data_from_folder(test_path, Categories, cnn=True)

# Convert to tensors
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).long()
x_validation = torch.tensor(x_validation).float()
y_validation = torch.tensor(y_validation).long()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).long()

x_train_cnn = torch.tensor(x_train_cnn).float()
x_validation_cnn = torch.tensor(x_validation_cnn).float()
x_test_cnn = torch.tensor(x_test_cnn).float()

# DataLoaders
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
validation_loader = DataLoader(TensorDataset(x_validation, y_validation), batch_size=32)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32)

train_loader_cnn = DataLoader(TensorDataset(x_train_cnn, y_train), batch_size=32, shuffle=True)
validation_loader_cnn = DataLoader(TensorDataset(x_validation_cnn, y_validation), batch_size=32)
test_loader_cnn = DataLoader(TensorDataset(x_test_cnn, y_test), batch_size=32)

# ANN model
class TrafficSignNet(nn.Module):
    def __init__(self, hidden1=128, hidden2=64):
        super(TrafficSignNet, self).__init__()
        self.fc1 = nn.Linear(15 * 15 * 3, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, len(Categories))

    def forward(self, x):
        x = x.view(-1, 15 * 15 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# CNN model
class TrafficSignCNN(nn.Module):
    def __init__(self, conv1_out=32, conv2_out=64, fc_size=128):
        super(TrafficSignCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, conv1_out, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(conv2_out * 3 * 3, fc_size)
        self.fc2 = nn.Linear(fc_size, len(Categories))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training with early stopping
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
        if val_loader:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_outputs = model(val_inputs)
                    _, val_predicted = torch.max(val_outputs, 1)
                    total += val_labels.size(0)
                    correct += (val_predicted == val_labels).sum().item()
            val_acc = correct / total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if val_acc:
            print(f"{name} Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"{name} Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
            break

# Evaluation
def evaluate_model(model, loader, name="Test"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f"{name} Accuracy: {acc:.4f}")
    return acc

# Parameter grid
def get_param_combinations(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, v)) for v in product(*values)]

# ANN grid search
ann_param_grid = {
    'lr': [0.001, 0.0005],
    'hidden1': [128, 256],
    'hidden2': [64, 128]
}

best_ann_acc, best_ann_params = 0, None
for params in get_param_combinations(ann_param_grid):
    print(f"\nTraining ANN with {params}")
    model = TrafficSignNet(hidden1=params['hidden1'], hidden2=params['hidden2'])
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, criterion, optimizer, val_loader=validation_loader, num_epochs=15, patience=3, name="ANN")
    acc = evaluate_model(model, test_loader, name="ANN Test")
    if acc > best_ann_acc:
        best_ann_acc = acc
        best_ann_params = params

print(f"\n Best ANN Accuracy: {best_ann_acc:.4f} with params: {best_ann_params}")

# CNN grid search
cnn_param_grid = {
    'lr': [0.001, 0.0005],
    'conv1_out': [32, 64],
    'conv2_out': [64, 128],
    'fc_size': [128, 256]
}

best_cnn_acc, best_cnn_params = 0, None
for params in get_param_combinations(cnn_param_grid):
    print(f"\nTraining CNN with {params}")
    model = TrafficSignCNN(conv1_out=params['conv1_out'], conv2_out=params['conv2_out'], fc_size=params['fc_size'])
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    cnn_criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader_cnn, cnn_criterion, optimizer, val_loader=validation_loader_cnn, num_epochs=15, patience=3, name="CNN")
    acc = evaluate_model(model, test_loader_cnn, name="CNN Test")
    if acc > best_cnn_acc:
        best_cnn_acc = acc
        best_cnn_params = params

print(f"\n Best CNN Accuracy: {best_cnn_acc:.4f} with params: {best_cnn_params}")
