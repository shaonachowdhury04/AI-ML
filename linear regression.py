import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
boston = load_boston()

# Create a DataFrame
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target  # Adding the target column

# Features and target variable
X = data.drop('MEDV', axis=1)  # Independent variables
y = data['MEDV']  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2 Score):", r2)

# Coefficients and Intercept
print("\nCoefficients:")
for feature, coef in zip(boston.feature_names, model.coef_):
    print(f"{feature}: {coef:.4f}")
print("\nIntercept:", model.intercept_)
