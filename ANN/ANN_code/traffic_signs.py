import pandas as pd 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 

import time
from datetime import datetime

def get_categories(file_location):
    try:
        with open(file_location, 'r') as file:
            lines = file.readlines()
            first_words = [line.split()[0] for line in lines]
            return first_words
    except FileNotFoundError:
        print(f"Error: The file '{file_location}' was not found.")
        return []

datadir='../ANN_data/database_1000_photos/'
Categories=get_categories(datadir+"description.txt") 
print("Categories read from description file:")
print(Categories)

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