import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# In this example, the dataset is fetched directly instead of being imported from the ucirepo website
dataset = pd.read_csv("winequality-red.csv", delimiter=';')

# Define thresholds for quality categories
low_threshold = 5
high_threshold = 6

# Categorize quality into 'low', 'medium', and 'high'
dataset['quality_category'] = pd.cut(dataset['quality'], bins=[-np.inf, low_threshold, high_threshold, np.inf], labels=['low', 'medium', 'high'])

# Select predictors and target variable
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid',
             'residual sugar', 'chlorides', 'free sulfur dioxide',
             'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
y = dataset['quality_category'].values

# Split the data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Instantiate the multinomial logistic regression model
logreg_multinomial = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')

# Suppress warnings about precision being ill-defined
warnings.filterwarnings("ignore", category=UserWarning)

# Fit the model to the training data
logreg_multinomial.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = logreg_multinomial.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Create a confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=logreg_multinomial.classes_, yticklabels=logreg_multinomial.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

# After fitting the model, let's grab some coefficients to determine their impact
feature_names = dataset.columns[:-2]  # Exclude the target variable and quality category
coefficients = logreg_multinomial.coef_

# Combine feature names and coefficients into a DataFrame for easier interpretation
feature_importance = pd.DataFrame({'Feature': feature_names})
for idx, category in enumerate(logreg_multinomial.classes_):
    feature_importance[category] = coefficients[idx]
print(feature_importance)
