# Random Selection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000  # number of customers
d = 10  # number of different ads
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)  # randomly select ad (0-9)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]  # get customer selection (0 or 1)
    total_reward = total_reward + reward  # add selection to total
# how often randomly selecting an ad for each customer was correct
print(total_reward)  # ~1200

# Visualising the results
plt.hist(ads_selected)  # how many times each ad was selected RANDOMLY
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()