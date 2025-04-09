import pandas as pd 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

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

datadir='../IVSPA-database/'
Categories=get_categories(datadir+"description.txt") 
print(Categories)
flat_data_arr=[] #input array 
target_arr=[] #output array 
#path which contains all the categories of images 
for i in Categories: 
    print(f'loading... category : {i}') 
    path=os.path.join(datadir,i) 
    print(f"image path: {path}")
    for img in os.listdir(path):
        if img.endswith(".ppm"): 
            img_array=imread(os.path.join(path,img)) 
            img_resized=resize(img_array,(50,50,3)) 
            flat_data_arr.append(img_resized.flatten()) 
            target_arr.append(Categories.index(i)) 
    print(f'loaded category:{i} successfully') 
flat_data=np.array(flat_data_arr) 
target=np.array(target_arr)


#dataframe 
df=pd.DataFrame(flat_data) 
df['Target']=target 
df.shape


#input data 
x=df.iloc[:,:-1] 
#output data 
y=df.iloc[:,-1]

# Splitting the data into training and testing sets 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, 
                                            random_state=77, 
                                            stratify=y) 

# Defining the parameters grid for GridSearchCV 
param_grid={'C':[0.1,1,10,100], 
            'gamma':[0.0001,0.001,0.1,1], 
            'kernel':['linear']} 
            # 'kernel':['rbf','poly']} 

# Creating a support vector classifier 
svc=svm.SVC(probability=True) 

# Creating a model using GridSearchCV with the parameters grid 
model=GridSearchCV(svc,param_grid, n_jobs=-1)

print("\n     Starting training!")
start_time = time.time()


# Training the model using the training data 
model.fit(x_train,y_train)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training finished! Time elapsed in training: {round(elapsed_time, 1)} seconds")

# Print the best parameters found by GridSearchCV
print("Best parameters found:")
print(model.best_params_)

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
