# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # (10, 1)
y = dataset.iloc[:, 2].values  # (10,)

# Dataset is too small to split

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)  # (10, 1)
# reshape(-1, 1) --> convert to one column matrix (-1 means make compatible): (10, 1)
# ravel --> convert to 1d array which is required for regressor.fit(): (10,)
y = np.ravel(sc_y.fit_transform(y.reshape(-1, 1)))  # (10,)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result
x_fs = sc_X.transform(np.array([[6.5]]))
y_fs = regressor.predict(x_fs)
y_pred = sc_y.inverse_transform(y_fs)
print(x_fs)
print(y_fs)
print(y_pred)  # ~170k

# # Visualising the SVR results
# plt.scatter(X, y, color='red')
# plt.plot(X, regressor.predict(X), color='blue')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()