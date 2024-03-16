from ucimlrepo import fetch_ucirepo

# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
# X = data features
X = wine_quality.data.features
# Y = data targets
y = wine_quality.data.targets
