import kagglehub as kh 
import os 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np


from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
path = kh.dataset_download("uciml/pima-indians-diabetes-database")

scaler = StandardScaler()

csv_files = [file for file in os.listdir(path) if file.endswith(".csv")]


file_path = os.path.join(path,csv_files[0])
df = pd.read_csv(file_path)


print(df.head())

from sklearn.linear_model import LogisticRegression 
model = LogisticRegression(max_iter = 1000)

#V17
pos = df.drop(columns = ["Outcome"])
X, y  = df[["Glucose"]], df["Outcome"]

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test =train_test_split(X,y, test_size= 0.3,random_state= 42, shuffle = True)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train) 


pred = model.predict(X_test_scaled)

import numpy as np
import matplotlib.pyplot as plt

# Assume you've already trained:
# model and scaler

# Pick a feature index (e.g., Glucose = column 1)
feature_idx = 0

# Create a range of values for that feature
x_vals = np.linspace(X["Glucose"].min(), X["Glucose"].max(), 100)

# Create a baseline input (mean of training data)
X_mean = X_train.mean(axis=0)

# Repeat baseline and replace one feature
X_plot = np.tile(X_mean, (100, 1))
X_plot[:, feature_idx] = x_vals

# Scale it
X_plot_scaled = scaler.transform(X_plot)

# Predict probabilities
y_probs = model.predict_proba(X_plot_scaled)[:, 1]

# Plot
plt.plot(x_vals, y_probs)
plt.xlabel("Glucose")
plt.ylabel("P(Diabetes)")
plt.title("Sigmoid Curve (Glucose vs Probability)")
plt.show()

