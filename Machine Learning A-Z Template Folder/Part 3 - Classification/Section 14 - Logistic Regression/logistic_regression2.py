# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')  # (400, 4)
# print(dataset['Gender'].value_counts())
X = dataset.iloc[:, [1, 3]].values  # (400, 2) Gender, Age, Salary
y = dataset.iloc[:, 4].values  # (400,)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])  # str category --> numerical encoding (convert in array)
mapping = {value: i for i, value in enumerate(labelencoder.classes_)}  # cat --> encoding mapping
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()  # add new 0, 1 columns (400, 3)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]  # (400, 2) --> remove female which is included in constant

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)  # 400 --> 300 + 100

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# print(classifier.score(X_test, y_test))  # 0.80

# Predicting the Test set results
y_pred = classifier.predict(X_test)  # (100,)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)
