import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import matplotlib.pyplot as plt

# In this example, the dataset is fetched directly instead of being imported from the ucirepo website
dataset = pd.read_csv("winequality-red.csv", delimiter=';')

# Select 'volatile acidity' as the predictor and 'quality' as the target variable
X = dataset[['volatile acidity']].values
y = dataset['quality'].values

# Split the data into test and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Instantiate the logistic regression model
logreg = LogisticRegression(max_iter=1000, solver='liblinear')
warnings.filterwarnings("ignore", category=UserWarning)

# Fit the model to the training data
logreg.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# After fitting the model, let's grab some coefficients to determine their impact
feature_names = ['volatile acidity']
coefficients = logreg.coef_[0]

# Combine feature names and coefficients into a DataFrame for easier interpretation
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
print(feature_importance)

# THE FOLLOWING RESETS THE RESPONSE INTO A BINARY CATEGORY OF HIGH OR LOW QUALITY FOR PREDICTION
# Define thresholds for categorizing wine quality
low_threshold = 4
high_threshold = 7

# Categorize wine quality based on thresholds
dataset['quality_category'] = pd.cut(dataset['quality'], bins=[-np.inf, low_threshold, high_threshold, np.inf], labels=['low', 'medium', 'high'])

# Select 'volatile acidity' as the predictor and 'quality_category' as the target variable
X = dataset[['volatile acidity']].values
y = dataset['quality_category'].values

# Split the data into test and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Instantiate the logistic regression model
logreg = LogisticRegression(max_iter=1000, solver='liblinear')
warnings.filterwarnings("ignore", category=UserWarning)

# Fit the model to the training data
logreg.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# After fitting the model, let's grab some coefficients to determine their impact
feature_names = ['volatile acidity']
coefficients = logreg.coef_[0]

# Combine feature names and coefficients into a DataFrame for easier interpretation
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
print(feature_importance)
# Get the actual labels of the test set
y_test_labels = np.where(y_test == 'high', 1, 0)  # Convert 'high' to 1 and 'low' to 0

# Plot the scatter plot
plt.figure(figsize=(10, 6))

# Scatter plot of test data
plt.scatter(X_test[y_test_labels == 1], logreg.predict_proba(X_test[y_test_labels == 1])[:, 2], color='blue', label='High Quality Predictions')
plt.scatter(X_test[y_test_labels == 0], logreg.predict_proba(X_test[y_test_labels == 0])[:, 0], color='red', label='Low Quality Predictions')

# Plot accurate predictions
plt.scatter(X_test[y_test == y_pred], logreg.predict_proba(X_test[y_test == y_pred])[:, 1], color='green', label='Correct Predictions')

# Plot formatting
plt.title('Predicted Probabilities of High and Low-Quality Wines with Accurate Predictions')
plt.xlabel('Volatile Acidity')
plt.ylabel('Predicted Probability')
plt.legend()
plt.grid(True)
plt.show()