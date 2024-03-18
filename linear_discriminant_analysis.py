from data_set import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

y = y.values.ravel()

# Convert y to a pandas DataFrame
y_df = pd.DataFrame(y, columns=['Quality'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_df, test_size=0.2, random_state=42)

# Initialize the Linear Discriminant Analysis classifier
lda = LinearDiscriminantAnalysis()

# Train the classifier on the training data
lda.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_lda = lda.predict(X_test)

# Compute confusion matrix
conf_matrix_lda = confusion_matrix(y_test, y_pred_lda)

# Print classification report
print("Classification Report (LDA):")
print(classification_report(y_test, y_pred_lda, zero_division=0))

# Generate tick labels from 3 to 10
tick_labels = [str(i) for i in range(3, 11)]

# Plot confusion matrix heatmap for LDA with quality range from 3 to 10
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_lda, annot=True, fmt='d', cmap='Blues',
            xticklabels=tick_labels, yticklabels=tick_labels)
plt.title('Confusion Matrix (LDA)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
