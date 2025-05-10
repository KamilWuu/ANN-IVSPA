import pandas as pd 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

datadir='../IVSPA-database/database_1000_photos/'
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

# Load validation data
validation_path = os.path.join(datadir, "Validation")
x_validation, y_validation = load_data_from_folder(validation_path, Categories)

# Load testing data
test_path = os.path.join(datadir, "Test")
x_test, y_test = load_data_from_folder(test_path, Categories)

#           Grid Search -- disabled for now
# Defining the parameters grid for GridSearchCV
# param_grid={'C':[0.1,1,10,100], 
#             'gamma':[0.0001,0.001,0.1,1], 
#             'kernel':['linear']} 
#             # 'kernel':['rbf','poly']} 

# Creating a support vector classifier 
# svc=svm.SVC(probability=True) 

# Creating a model using GridSearchCV with the parameters grid 
# model=GridSearchCV(svc,param_grid, n_jobs=-1)

model = svm.SVC(C=0.1, gamma=0.0001, kernel='linear', probability=True)

print("\n     Starting training!")
start_time = time.time()


# Training the model using the training data 
model.fit(x_train,y_train)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training finished! Time elapsed in training: {round(elapsed_time, 1)} seconds")

# Print the best parameters found by GridSearchCV
# print("Best parameters found:")
# print(model.best_params_)

print("\n     Starting testing!")
start_time = time.time()

# Testing the model using the testing data 
y_pred = model.predict(x_test) 

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Testing finished! Time elapsed in testing: {round(elapsed_time, 1)} seconds\n")

# Calculating the accuracy of the model 
accuracy = accuracy_score(y_pred, y_test) 

# Print the accuracy of the model 
print(f"The model is {accuracy*100}% accurate")


report = classification_report(y_test, y_pred, target_names=Categories)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"reports/classification_report_{timestamp}.txt"
with open(filename, "w") as file:
    file.write(report)
print(f"Classification report saved as: {filename}")
print("Contents of report:")

print(report)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Categories)
disp.plot(cmap='Greens')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
