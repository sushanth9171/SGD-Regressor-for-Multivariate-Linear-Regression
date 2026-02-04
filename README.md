# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: G Sushanth
RegisterNumber:  25011663
*/



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
data = fetch_california_housing()
X = data.data[:, :3]
y_price = data.target         
y_occup = data.data[:, 5]     
Y = np.column_stack((y_price, y_occup))
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model = MultiOutputRegressor(sgd)

model.fit(X_train_scaled, Y_train_scaled)
mse_price = mean_squared_error(Y_test_original[:, 0], Y_pred[:, 0])
mse_occup = mean_squared_error(Y_test_original[:, 1], Y_pred[:, 1])

print("Mean Squared Error (House Price):", mse_price)
print("Mean Squared Error (Average Occupants):", mse_occup)

print("\nSample Predictions (Price, Occupants):")
for i in range(5):
    print("Predicted:", Y_pred[i], " | Actual:", Y_test_original[i])
```
## Output:
<img width="930" height="285" alt="Screenshot 2026-02-04 085942" src="https://github.com/user-attachments/assets/3fd11c4f-a2c8-4ae7-9d84-45179afaf868" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
