import numpy as np
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X, y)

y_pred = lr_model.predict([X[0]])  # Predicting the class for the first sample in the dataset
print("Prediction on training set:", y_pred)

print("Accuracy on training set:", lr_model.score(X, y))
b_norm = lr_model.intercept_
w_norm = lr_model.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")