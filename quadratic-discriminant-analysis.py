from data_set import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

y = y.values.ravel()

# Convert y to a pandas DataFrame
y_df = pd.DataFrame(y, columns=['Quality'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_df, test_size=0.2, random_state=42)

# Initialize the Quadratic Discriminant Analysis classifier
qda = QuadraticDiscriminantAnalysis()

# Train the classifier on the training data
qda.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = qda.predict(X_test)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()