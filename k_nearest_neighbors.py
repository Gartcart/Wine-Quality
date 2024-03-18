from data_set import *

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

quality = y['quality']
X_density = X['density'].values.reshape(-1, 1)
X_residual_sugar = X['residual_sugar'].values.reshape(-1, 1)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X_density, X_residual_sugar, c=quality, cmap='viridis', s=100, alpha=0.8)
plt.colorbar(label='Quality')
plt.xlabel('Density')
plt.ylabel('Residual Sugar')
plt.title('Relationship between Quality, Density, and Residual Sugar')
plt.grid(True)
plt.show()

X_density_res_sugar = X[['density', 'residual_sugar']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_density_res_sugar, quality, test_size=0.2, random_state=42)

numberOfNeighbors = [5, 25, 100]
density = 1
residual_sugar = 10
wine_to_predict = [[density, residual_sugar]]

for i in numberOfNeighbors:
    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=i)  # You can adjust the number of neighbors
    knn.fit(X_train, y_train)

    # Make predictions for the wine with specified density and residual sugar level
    prediction = knn.predict(wine_to_predict)

    print(f"When k = {i} the predicted quality level is {prediction[0]}")

print("\nAverage Quality of wine for reference is: ", y['quality'].mean())