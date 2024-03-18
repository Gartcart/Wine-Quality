from data_set import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pandas as pd

# Only use the volatile acidity and quality columns to answer this question
X_volatile_acidity = X['volatile_acidity'].values.reshape(-1, 1)
y_quality = pd.cut(y['quality'], bins=[0, 4, 6, 10], labels=['low', 'mid', 'high'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_volatile_acidity, y_quality, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
naive_bayes = GaussianNB()

# Train the classifier on the training data
naive_bayes.fit(X_train, y_train)

# Make predictions for the wines with fixed volatile acidity
prediction_values = [[0.1], [0.5], [1.0]]
for prediction in prediction_values:
    p = naive_bayes.predict([prediction])
    print(f"A wine with a volatile acidity of {prediction[0]} would be classified as {p[0]}-quality")

# Create a scatter plot of volatile acidity vs. quality
plt.scatter(X_volatile_acidity, y_quality.cat.codes)
plt.xlabel('Volatile Acidity')
plt.ylabel('Quality')
plt.title('Scatter Plot of Volatile Acidity vs. Wine Quality')
plt.show()
