from ucimlrepo import fetch_ucirepo

# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
# X = data features
X = wine_quality.data.features
# Y = data targets
y = wine_quality.data.targets
<<<<<<< HEAD
=======

# metadata
# print(wine_quality.metadata)

# variable information
# print(wine_quality.variables)
>>>>>>> 4c0786561c8ea86417a3e735b8c16daa61a5b0f0
