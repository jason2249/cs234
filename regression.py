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
        #arm_vals[2][i] = -1
    elif Y_vals[i] >= 21 and Y_vals[i] <= 49:
        arm_vals[1][i] = 0
    elif Y_vals[i] > 49:
        arm_vals[2][i] = 0
        #arm_vals[0][i] = -1
regressorLow = LinearRegression()
regressorLow.fit(X_vals, arm_vals[0])
#coeff_df = pd.DataFrame(regressorLow.coef_, X.columns, columns=['Coefficient'])
#print(coeff_df)
#print(regressorLow.intercept_)
regressorMed = LinearRegression()
regressorMed.fit(X_vals, arm_vals[1])
regressorHigh = LinearRegression()
regressorHigh.fit(X_vals, arm_vals[2])
#test = [[9, 166.37, 79.4,0,0,0,0,1]]

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
    #print(y_pred_low, y_pred_med, y_pred_high)
    return max(y_pred_low, y_pred_med, y_pred_high)[0]


