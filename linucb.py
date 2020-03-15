'''
linucb.py

Run 20 permutations of the LinUCB algorithm on the given dataset, calculate the
upper and lower confidence bounds of the results, and plot them as needed.

To run the augmented version, use data/augmented_features.csv, and change d=13
To run the initial version, use data/features.csv, and change d=8
'''

import numpy as np
import baseline
import regression
import matplotlib.pyplot as plt
import csv
import math

alpha = 1.0
d = 13 #num features
linucb_correct_fractions = np.zeros((20,4374)) #20 permutations, 4374 patients 
cum_regret = np.zeros((20,4374))
pct_severe_mistakes = np.zeros((20,4374))
with open('data/augmented_features.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    all_rows = []
    first = True
    for row in reader:
        if first:
            first = False
            continue
        all_rows.append([float(x) for x in row])
    all_permuted_rows = []
    for perm_num in range(20):
        np.random.seed(perm_num)
        permuted_rows = np.random.permutation(all_rows)
        all_permuted_rows.append(permuted_rows)
    for iteration, permuted_rows in enumerate(all_permuted_rows): #20 permutations
        # low, medium, high dose
        As = [np.eye(d), np.eye(d), np.eye(d)]
        bs = [np.zeros((d,1)), np.zeros((d,1)), np.zeros((d,1))]
        num_correct = 0
        num_total = 0
        cum_regret_total = 0
        num_severe_mistakes = 0
        for patient_num in range(len(permuted_rows)): #i is patient number
            row = permuted_rows[patient_num]
            num_total += 1
            dose = row[0]
            xt = np.array(row[1:]).reshape((d,1))
            xtt = np.transpose(xt)
            max_payoff = float('-inf')
            best_arm = None
            for i, A in enumerate(As):
                Ainv = np.linalg.inv(A)
                theta_hat = np.dot(Ainv, bs[i]) #xt dx1, theta is dx1, A is dxd
                payoff = np.dot(xtt, theta_hat) + alpha * math.sqrt(np.dot(np.dot(xtt, Ainv), xt))
                if payoff > max_payoff:
                    max_payoff = payoff
                    best_arm = i
            reward = -1
            if dose < 21:
                if best_arm == 0:
                    num_correct += 1
                    reward = 0
                elif best_arm == 2:
                    num_severe_mistakes += 1
                    reward = -4
            elif dose >= 21 and dose <= 49:
                if best_arm == 1:
                    num_correct += 1
                    reward = 0
            elif dose > 49:
                if best_arm == 2:
                    num_correct += 1
                    reward = 0
                elif best_arm == 0:
                    num_severe_mistakes += 1
                    reward = -4
            xtreshaped = xt.reshape((1,-1))
            regret = regression.findBestArmReward(xtreshaped) - regression.findArmReward(xtreshaped, best_arm)
            cum_regret_total += regret
            cum_regret[iteration, patient_num] = cum_regret_total
            linucb_correct_fractions[iteration, patient_num] = num_correct/num_total
            pct_severe_mistakes[iteration, patient_num] = num_severe_mistakes/num_total
            As[best_arm] += np.dot(xt, xtt)
            bs[best_arm] += (reward * xt)
        print("permutation " + str(iteration) + ": " + str(linucb_correct_fractions[iteration, -1]))

linucb_correct_stds = np.std(linucb_correct_fractions, axis=0)
linucb_correct_fractions = np.mean(linucb_correct_fractions, axis=0)
cum_regret_stds = np.std(cum_regret, axis=0)
cum_regret = np.mean(cum_regret, axis=0)
severe_stds = np.std(pct_severe_mistakes, axis=0)
pct_severe_mistakes = np.mean(pct_severe_mistakes, axis=0)
t_val = 2.093
sqrt_sample_size = math.sqrt(20)
correct_conf_intervals = np.divide(linucb_correct_stds, sqrt_sample_size)
correct_conf_intervals = np.multiply(correct_conf_intervals, t_val)
correct_lower_bounds = np.subtract(linucb_correct_fractions, correct_conf_intervals)
correct_upper_bounds = np.add(linucb_correct_fractions, correct_conf_intervals)
regret_conf_intervals = np.divide(cum_regret, sqrt_sample_size)
regret_conf_intervals = np.multiply(regret_conf_intervals, t_val)
regret_lower_bounds = np.subtract(cum_regret, regret_conf_intervals)
regret_upper_bounds = np.add(cum_regret, regret_conf_intervals)
severe_conf_intervals = np.divide(severe_stds, sqrt_sample_size)
severe_conf_intervals = np.multiply(severe_conf_intervals, t_val)
severe_lower_bounds = np.subtract(pct_severe_mistakes, severe_conf_intervals)
severe_upper_bounds = np.add(pct_severe_mistakes, severe_conf_intervals)
print("average linucb performance:", linucb_correct_fractions[-1])
print("average linucb regret:", cum_regret[-1])
print("average linucb pct severe mistakes:", pct_severe_mistakes[-1])
linucb1_correct_pct = np.load("data/linucb1_correct_pct.npy")
linucb1_regret = np.load("data/linucb1_regret.npy")
linucb1_severe_pct = np.load("data/linucb1_severe_pct.npy")

plt.plot(baseline.fixed_correct_fractions, label="fixed")
plt.plot(baseline.linear_correct_fractions, label="linear")
plt.plot(linucb1_correct_pct, label="initial LinUCB")
plt.plot(linucb_correct_fractions, label="augmented LinUCB -1, -4 rewards")
plt.xlabel('Number of Patients Served')
plt.ylabel('Percentage of Correct Diagnoses')
plt.title('Percentage of Correct Diagnoses of Models Over Time')
plt.legend()
plt.show()

plt.plot(baseline.fixed_cum_regret, label="fixed")
plt.plot(baseline.linear_cum_regret, label="linear")
plt.plot(linucb1_regret, label="initial LinUCB")
plt.plot(cum_regret, label="augmented LinUCB -1, -4 rewards")
plt.xlabel('Number of Patients Served')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret of Models Over Time')
plt.legend()
plt.show()

plt.plot(baseline.fixed_pct_severe_mistakes, label="fixed")
plt.plot(baseline.linear_pct_severe_mistakes, label="linear")
plt.plot(linucb1_severe_pct, label="initial LinUCB")
plt.plot(pct_severe_mistakes, label="augmented LinUCB -1, -4 rewards")
plt.ylim(top=.05)
plt.xlabel('Number of Patients Served')
plt.ylabel('Percentage of Severe Mistakes')
plt.title('Percentage of Severe Mistakes of Models Over Time')
plt.legend()
plt.show()

plt.plot(pct_severe_mistakes, label="mean")
plt.plot(severe_lower_bounds, label="lower bound")
plt.plot(severe_upper_bounds, label="upper bound")
plt.ylim(top=.05)
plt.xlabel('Number of Patients Served')
plt.ylabel('Percentage of Severe Mistakes')
plt.title('Percentage of Severe Mistakes of Augmented LinUCB Over Time')
plt.legend()
plt.show()

plt.plot(linucb_correct_fractions, label="mean")
plt.plot(correct_lower_bounds, label="lower bound")
plt.plot(correct_upper_bounds, label="upper bound")
plt.xlabel('Number of Patients Served')
plt.ylabel('Correct Percentage')
plt.title('Correct Percentage of Augmented LinUCB Over Time')
plt.legend()
plt.show()

plt.plot(cum_regret, label="mean")
plt.plot(regret_lower_bounds, label="lower bound")
plt.plot(regret_upper_bounds, label="upper bound")
plt.xlabel('Number of Patients Served')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret of Augmented LinUCB Over Time')
plt.legend()
plt.show()

