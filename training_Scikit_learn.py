import os 
os.chdir(r"C:\Users\kidem\OneDrive\Documents\Pictures\cfTraining\PYProjets") 
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np





def fit_linear_regression(x,y,n): 
    # y = x + b (univariate, so there is no need for multiple variables) 
    numerator  = (n*np.sum(x * y) - (np.sum(x) * np.sum(y)))
    denominator = (n * np.sum(x**2) - (np.sum(x) ** 2))

    m = numerator/denominator 
    b = (np.sum(y)  - m *np.sum(x))/n
    return m,b

def normalEquation(X,y): 
    X_T = np.transpose(X)
    X_T_X = np.dot(X_T, X)
    X_T_Y = np.dot(X_T, y)
    
    theta = np.linalg.solve(X_T_X, X_T_Y) 
    return theta

from sklearn.linear_model import LinearRegression
import pandas as pd

URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
data = pd.read_csv(URL)
model = LinearRegression()
X,y = data[["total_rooms"]], data["households"]


from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, shuffle = True)
model.fit(X_train,y_train)

prediction = model.predict(X_test)
from sklearn.metrics import r2_score 
r2 =r2_score(y_test, prediction)

print(f"{round(r2*100,2)}% of housing numbers in CA (1997) \ncan be explained by the number of rooms")
plt.scatter(X,y, label = "Trained Data", color = "blue", edgecolors = "face")
plt.plot(X, X*model.coef_  + model.intercept_, color = "red", label = "Regression Line")
plt.grid()
plt.legend()
plt.show()



def linRegreshFitting():
    data = pd.DataFrame({
        "Experience": [1,2,3,5,7,10],
        "Education_Level": [1,1,2,2,3,3],
        "Hours_Week": [40,45,40,50,55,60],
        "Salary": [40000,45000,50000,65000,80000,95000]
    })

    X = data[["Experience", "Education_Level", "Hours_Week"]]
    y = data["Salary"]

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict([[6, 2, 50]])

    print("Predicted salary:", round(prediction[0], 2))


# plotting the points





# 1: We will first see how a single variable effects the dependent variable  (charge) : classic Univariate linear regression model 