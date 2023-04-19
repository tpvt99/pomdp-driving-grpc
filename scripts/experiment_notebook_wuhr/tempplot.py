import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example data
X = np.array([1, 2, 3, 4, 5]) # Independent variable
y = np.array([2, 4, 4, 5, 5]) # Dependent variable
# Perform linear regression
slope, intercept = np.polyfit(X, y, 1)
# Create scatter plot
plt.scatter(X, y)

# Plot regression line
plt.plot(X, slope*X + intercept, color='red', label='Linear Regression')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Show plot
plt.show()

plt.savefig('hihi.png')