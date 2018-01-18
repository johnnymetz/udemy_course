# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords  # removes common words ('this', 'is', etc.)
from nltk.stem.porter import PorterStemmer  # get root of word ('loved' --> 'love')
corpus = []  # means collection of text
reviews = dataset['Review'].tolist()
for review in reviews:
    review = re.sub('[^a-zA-Z]', ' ', review)  # substitute non-letters with spaces
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)  # max features removes uncommon words (1565 --> 1500)
# can remove stopwords + lowercase above as well
X = cv.fit_transform(corpus).toarray()
y = dataset['Liked'].values  # CHANGED

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))  # 0.73

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)