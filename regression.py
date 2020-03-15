'''
regression.py

Calculate the beta parameters for regret calculation using a batch linear regression
for each arm. 

To run with augmented features, use data/augmented_features.csv and the full range of
X values.
To run with initial features, use data/features.csv and comment out X values after 'amiodarone'.
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('data/augmented_features.csv')
X = dataset[['age', 'height', 'weight', 'asian', 'black', 'unknown_race', 'enzyme_inducer', 'amiodarone', 'gender', 'smoker', 'aspirin', 'cyp2c9', 'vkorc1']]
X_vals = X.values
Y = dataset['dose']
Y_vals = Y.values
arm_vals = [np.full(Y_vals.shape, -1), np.full(Y_vals.shape, -1), np.full(Y_vals.shape, -1)]
for i in range(len(Y_vals)):
    if Y_vals[i] < 21: #correct dosage is low
        arm_vals[0][i] = 0 #reward for giving low dosage is best
    elif Y_vals[i] >= 21 and Y_vals[i] <= 49:
        arm_vals[1][i] = 0
    elif Y_vals[i] > 49:
        arm_vals[2][i] = 0
regressorLow = LinearRegression()
regressorLow.fit(X_vals, arm_vals[0])
regressorMed = LinearRegression()
regressorMed.fit(X_vals, arm_vals[1])
regressorHigh = LinearRegression()
regressorHigh.fit(X_vals, arm_vals[2])

def findArmReward(test, armIndex):
    if armIndex == 0:
        return regressorLow.predict(test)
    elif armIndex == 1:
        return regressorMed.predict(test)
    elif armIndex == 2:
        return regressorHigh.predict(test)

def findBestArmReward(test):
    y_pred_low = regressorLow.predict(test)
    y_pred_med = regressorMed.predict(test)
    y_pred_high = regressorHigh.predict(test)
    return max(y_pred_low, y_pred_med, y_pred_high)[0]


