import numpy as np
import matplotlib.pyplot as plt

# Given data: Study hours (x) and Exam scores (y)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 6, 8])

# Step 1: Compute means of x and y
x_mean = np.mean(x)
y_mean = np.mean(y)

# Step 2: Compute slope (m)
numerator = np.sum((x - x_mean) * (y - y_mean))  # Σ(xi - x̄)(yi - ȳ)
denominator = np.sum((x - x_mean) ** 2)          # Σ(xi - x̄)^2
m = numerator / denominator

# Step 3: Compute intercept (b)
b = y_mean - m * x_mean

# Step 4: Create regression line
y_pred = m * x + b  # Predicted y values

# Step 5: Plot data points and regression line
plt.scatter(x, y, color='blue', label='Actual Data')  # Scatter plot of original data
plt.plot(x, y_pred, color='red', label='Regression Line')  # Regression line
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Linear Regression: Study Hours vs Exam Score")
plt.legend()
plt.show()

# Step 6: Predict for a new value (e.g., 6 hours)
x_new = 6
y_new = m * x_new + b
print(f"Predicted score for {x_new} study hours: {y_new:.2f}")
