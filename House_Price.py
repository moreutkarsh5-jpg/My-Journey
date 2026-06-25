import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=2)

df = pd.read_csv(r"C:\Users\dell\OneDrive\Desktop\BorntoCode\Machine Learning\House Price Prediction Dataset.csv")

df = df.drop(['Location', 'Condition', 'Garage'], axis=1)
print(df.head())
X = df.drop(['Price', 'Id'], axis=1).values
Y = df['Price'].values

print(X.shape)
print(Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)    
X_test_norm = scaler.transform(X_test)

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_train_norm, y_train)
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters= w: {w_norm}, b:{b_norm}")

y_pred = sgdr.predict(X_test_norm)
for actual, pred in zip(y_test[:5], y_pred[:5]):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(y_test, y_pred)    
# print(f"Mean Squared Error: {mse:.2f}")

my_house = np.array([[2000, 2, 1, 2, 2000]])
my_house_norm = scaler.transform(my_house)
predicted_price = sgdr.predict(my_house_norm)
print(f"Predicted Price: Rs. {predicted_price[0]:.2f}")