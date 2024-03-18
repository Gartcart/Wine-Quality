import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# In this example, the dataset is fetched directly instead of being imported from the ucirepo website
dataset = pd.read_csv("winequality-red.csv", delimiter=';')

print(dataset.columns)
print(dataset.head())

# Here we grab each predictor and response and assign them x and y

X = dataset[['fixed acidity', 'volatile acidity', 'citric acid',
             'residual sugar', 'chlorides', 'free sulfur dioxide',
             'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values

y = dataset['quality'].values

# Split the data into test and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Instantiate the logistic regression model
logreg = LogisticRegression(max_iter=1000, solver='liblinear')
# Suppress warnings about precision being ill-defined
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

# After fitting the model, lets grab some coefficients to determine which have the highest impact
feature_names = dataset.columns[:-1]  # Exclude the target variable
coefficients = logreg.coef_[0]

# Combine feature names and coefficients into a DataFrame for easier interpretation
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
print(feature_importance)