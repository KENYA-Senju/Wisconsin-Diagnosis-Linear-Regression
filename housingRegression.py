
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

PROMPT = int(input("Total Number of Rooms in household area: "))
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
ans = PROMPT*model.coef_  + model.intercept_


print(f"{round(r2*100,2)}% of housing numbers in CA (1997) \ncan be explained by the number of rooms")
print("Median Housing Price:") 
print(f"${round(ans[0],2)}")
plt.scatter(X,y, label = "Trained Data", color = "blue", edgecolors = "face")
plt.plot(X, X*model.coef_  + model.intercept_, color = "red", label = "Regression Line")
plt.scatter(int(PROMPT), ans[0], color = "green",marker = 'x')
plt.grid()
plt.legend()
plt.show()