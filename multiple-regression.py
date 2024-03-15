import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
# X = data features
X = wine_quality.data.features
# Y = data targets
y = wine_quality.data.targets

# Creating pandas dataframe from features and target
df = pd.concat([X, y], axis=1)

# Splitting the dataset into features (X) and target variable (y)
X = df.drop('quality', axis=1)
y = df['quality']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the testing set
y_pred = model.predict(X_test)

# Calculating mean squared error to evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Coefficients of the linear regression model
coefficients = dict(zip(X.columns, model.coef_))
print("Coefficients:")
for feature, coef in coefficients.items():
    print(f"{feature}: {coef}")

