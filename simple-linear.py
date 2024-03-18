import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
# X = data features
X = wine_quality.data.features
# Y = data targets
y = wine_quality.data.targets

# Extracting the alcohol content and quality data
alcohol_content = X['alcohol'].values.reshape(-1, 1)  # Reshape to create a 2D array
quality = y.values

# Creating a linear regression model
model = LinearRegression()

# Fitting the model
model.fit(alcohol_content, quality)

# Getting the coefficients and intercept
coefficient = model.coef_[0]
intercept = model.intercept_

# Predicting wine quality based on the alcohol content
predicted_quality = model.predict(alcohol_content)

# Plotting the data and regression line
plt.scatter(alcohol_content, quality, label='Data Points')
plt.plot(alcohol_content, predicted_quality, color='red', label='Regression Line')

# Adding labels and title
plt.xlabel('Alcohol Content')
plt.ylabel('Wine Quality')
plt.title('Relationship between Alcohol Content and Wine Quality')

# Adding legend
plt.legend()

# Display the plot
plt.show()

# Printing the coefficient (impact of alcohol content on quality)
print("Coefficient (Impact of alcohol content on quality):", coefficient)
