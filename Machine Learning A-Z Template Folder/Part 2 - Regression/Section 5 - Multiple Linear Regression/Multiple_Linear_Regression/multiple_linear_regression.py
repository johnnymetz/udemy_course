# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])  # str category --> numerical encoding
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()  # add new 0, 1 columns

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling not required

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())
# # x2 (NY - index 2) had the highest p-value (0.990) so we remove it (Adj R^2=0.945)
# X_opt = X[:, [0, 1, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())
# # x1 (Florida - index 1) had the highest p-value (0.940) so we remove it (Adj R^2=0.946)
# X_opt = X[:, [0, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())
# # x2 (admin - index 4) had the highest p-value (0.602) so we remove it (Adj R^2=0.9475)
# X_opt = X[:, [0, 3, 5]]
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())
# # x2 (marketing - index 5) had the highest p-value (0.06) so we remove it (Adj R^2=0.9483)
# X_opt = X[:, [0, 3]]
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())
# # All p-values are below the "significant level" BUT but Adj R^2 decreased (Adj R^2=0.9454) so we add marketing back in
X_opt = X[:, [0, 3, 5]]

# # DONE
# # The three variables which best determine profit are: R&D spending + Marketing spending + located in CA