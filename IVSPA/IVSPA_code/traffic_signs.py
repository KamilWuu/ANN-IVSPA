import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from joblib import Parallel, delayed

import time
from datetime import datetime
import cpuinfo
import itertools

photo_dimension_after_resizing = 20

def get_categories(file_location):
    try:
        with open(file_location, 'r') as file:
            lines = file.readlines()
            first_words = [line.split()[0] for line in lines]
            return first_words
    except FileNotFoundError:
        print(f"Error: The file '{file_location}' was not found.")
        return []

datadir = '../IVSPA-database/nowa baza 20 znakow maks 480/database_480_photos/'
Categories = get_categories("../IVSPA-database/nowa baza 20 znakow maks 480/new_description.txt")
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
                img_resized = resize(img_array, (photo_dimension_after_resizing,
                                                 photo_dimension_after_resizing, 3))
                data.append(img_resized.flatten())
                labels.append(categories.index(i))
    return np.array(data), np.array(labels)

# Load data
x_train, y_train = load_data_from_folder(os.path.join(datadir, "Training"), Categories)
x_validation, y_validation = load_data_from_folder(os.path.join(datadir, "Validation"), Categories)
x_test, y_test = load_data_from_folder(os.path.join(datadir, "Test"), Categories)

# Parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# param_grid={
#     'C':[0.03,0.1,0.3,0.6,1,3,6,10,30,60,100], 
#     'gamma': [0.0001, 0.0003, 0.006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
# }

# Setup
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_directory = f"reports/{timestamp}"
os.makedirs(report_directory, exist_ok=True)
grid_search_filename = f"{report_directory}/grid_search_{timestamp}.txt"

# Define the function for training and testing
def train_and_test_model(C, gamma, kernel):
    print(f"Training with C={C}, gamma={gamma}, kernel={kernel}...")
    model = svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=True)

    start_time = time.time()
    model.fit(x_train, y_train)
    learn_time = time.time() - start_time
    start_time = time.time()
    y_pred = model.predict(x_test)
    test_time = time.time() - start_time

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return {
        'C': C,
        'gamma': gamma,
        'kernel': kernel,
        'accuracy': acc,
        'f1': f1,
        'learn_time': learn_time,
        'test_time': test_time,
        'model': model
    }

# Log header
with open(grid_search_filename, "w") as f:
    f.write(f"Processor used to perform tests: {cpuinfo.get_cpu_info()['brand_raw']}\n")
    f.write("C,gamma,kernel,test_accuracy,test_f1_weighted,learn_time,test_time\n")

print("\n     Starting parallel grid search on TEST SET")

# Run grid search in parallel
results = Parallel(n_jobs=-1)(
    delayed(train_and_test_model)(C, gamma, kernel)
    for C, gamma, kernel in itertools.product(param_grid['C'], param_grid['gamma'], param_grid['kernel'])
)

# Save results and find best model
best_result = max(results, key=lambda r: r['accuracy'])
best_model = best_result['model']
best_params = {k: best_result[k] for k in ['C', 'gamma', 'kernel']}

# Log results to file
with open(grid_search_filename, "a") as f:
    for r in results:
        f.write(f"{r['C']},{r['gamma']},{r['kernel']},{r['accuracy']:.4f},{r['f1']:.4f},{r['learn_time']:.2f},{r['test_time']:.2f}\n")

print(f"Grid search results saved as: {grid_search_filename}")
print("Best parameters found based on test set:")
print(best_params)

# Final evaluation
print("\n     Starting final evaluation with best model")
start_time = time.time()
y_pred = best_model.predict(x_test)
testing_elapsed_time = time.time() - start_time
accuracy = accuracy_score(y_pred, y_test)

# Save classification report
report = classification_report(y_test, y_pred, target_names=Categories)
report_filename = f"{report_directory}/classification_report_{timestamp}.txt"
with open(report_filename, "w") as file:
    file.write(f"Width and heights of the images after resizing: {photo_dimension_after_resizing}px\n")
    file.write(f"SVM parameters: {best_params}\n")
    file.write(f"Model run on processor: {cpuinfo.get_cpu_info()['brand_raw']}\n")
    file.write(f"Time elapsed in final test prediction: {round(testing_elapsed_time, 1)} seconds\n\n")
    file.write(report)

print(f"Classification report saved as: {report_filename}")
print("Contents of report:")
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Categories)
disp.plot(cmap='Greens')
plt.title("Confusion Matrix")
plt.tight_layout()

conf_matrix_path = f"{report_directory}/confusion_matrix_{timestamp}.png"
plt.savefig(conf_matrix_path)
plt.show()
