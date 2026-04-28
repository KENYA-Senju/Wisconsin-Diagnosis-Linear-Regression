import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
import kagglehub as kh 
import os 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

path = kh.dataset_download("uciml/breast-cancer-wisconsin-data")

csv_files = [file for file in os.listdir(path) if file.endswith(".csv")]


file_path = os.path.join(path,csv_files[0])
df = pd.read_csv(file_path)

X = df[["perimeter_worst", "concave points_worst", "radius_worst"]]

#M: Malignant Diagnosis B: Benign
Y = df["diagnosis"].map({"M": 1, "B": 0})

from sklearn.model_selection import train_test_split
X_train,X_test, Y_train,Y_test = train_test_split(X,Y, test_size = .3, shuffle = True, random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter = 1000) 
model.fit(X_train, Y_train) 

from sklearn.metrics import accuracy_score,classification_report 
y_pred = model.predict(X_test) 

score = accuracy_score(Y_test, y_pred)
print("Accuracy: ", score) 


print(classification_report(Y_test, y_pred))


print() 
inp  = [0,0,0]
inp[0]  = float(input("Worst Perimeter: "))
inp[1]  = float(input("Concave Worst Points: "))
inp[2] = float(input("Worst Radius: "))

ucs = scaler.transform([[inp[0], inp[1], inp[2]]])
prediction = model.predict(ucs)

print("Prediction: ", ("Malignant" if prediction[0] == 1 else "Benign"))

