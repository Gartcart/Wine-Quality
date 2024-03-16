from data_set import *

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# Only use the volatile acidity and quality columns to answer this question
X_volatile_acidity = X['volatile_acidity'].values.reshape(-1, 1)
y_quality = (y['quality'] >= 5).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_volatile_acidity, y_quality, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
naive_bayes = GaussianNB()

# Train the classifier on the training data
naive_bayes.fit(X_train, y_train)

# Make predictions for the wine with fixed volatile acidity of 7.5
wine_to_predict = [[0.2]]  # Input features for fixed volatile acidity
prediction = naive_bayes.predict(wine_to_predict)

# Make predictions for the wine with given fixed volatile acidity
prediction_values = [[[0.5]], [[0.7]], [[1.0]]]
for prediction in prediction_values:
    p = naive_bayes.predict(prediction)
    if p[0] == 1:
        quality = "high"
    else:
        quality = "low"
    print(f"A wine with the volatile acidity of {prediction} would be classified as {quality}")

# Create a scatter plot of volatile acidity vs. quality
plt.scatter(X_volatile_acidity, y['quality'])
plt.xlabel('Volatile Acidity')
plt.ylabel('Quality')
plt.title('Scatter Plot of Volatile Acidity vs. Wine Quality')
plt.show()
