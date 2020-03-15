import csv
import regression
import matplotlib.pyplot as plt

outrows = []
fixed_correct_fractions = []
fixed_cum_regret = []
fixed_pct_severe_mistakes = [0 for i in range(4374)]
fixed_regret_total = 0
linear_correct_fractions = []
linear_cum_regret = []
linear_pct_severe_mistakes = []
linear_regret_total = 0
num_severe = 0
with open('data/warfarin.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    fixed_correct = 0
    fixed_total = 0
    linear_correct = 0
    linear_total = 0
    for row in reader:
        actual_dose = row['Therapeutic Dose of Warfarin']
        try:
            actual_dose = float(actual_dose)
        except:
            continue
        if actual_dose >= 21 and actual_dose <= 49:
            fixed_correct += 1
        fixed_total += 1
        if len(fixed_correct_fractions) < 4374:
            fixed_correct_fractions.append(fixed_correct/fixed_total) 
        age = row['Age']
        if age == '0 - 9':
            age = 0
        elif age == '10 - 19':
            age = 1
        elif age == '20 - 29':
            age = 2
        elif age == '30 - 39':
            age = 3
        elif age == '40 - 49':
            age = 4
        elif age == '50 - 59':
            age = 5
        elif age == '60 - 69':
            age = 6
        elif age == '70 - 79':
            age = 7
        elif age == '80 - 89':
            age = 8
        elif age == '90+':
            age = 9
        else:
            continue
        height = row['Height (cm)']
        try:
            height = float(height)
        except:
            continue
        weight = row['Weight (kg)']
        try:
            weight = float(weight)
        except:
            continue
        asian = 0
        if row['Race'] == 'Asian':
            asian = 1
        black = 0
        if row['Race'] == 'Black or African American':
            black = 1
        unknown_race = 0
        if row['Race'] == 'Unknown':
            unknown_race = 1
        enzyme_inducer = 0
        if row['Carbamazepine (Tegretol)'] == '1' or row['Phenytoin (Dilantin)'] == '1' or row['Rifampin or Rifampicin'] == '1':
            enzyme_inducer = 1
        amiodarone = 0
        if row['Amiodarone (Cordarone)'] == '1':
            amiodarone = 1
        gender = 0
        if row['Gender'] == 'male':
            gender = 1
        elif row['Gender'] == 'female':
            gender = 2
        smoker = 0
        if row['Current Smoker'] == '0':
            smoker = 1
        elif row['Current Smoker'] == '1':
            smoker = 2
        aspirin = 0
        if row['Aspirin'] == '0':
            aspirin = 1
        elif row['Aspirin'] == '1':
            aspirin = 2
        cyp2c9 = 0
        if row['Cyp2C9 genotypes'] == '*1/*1':
            cyp2c9 = 1
        elif row['Cyp2C9 genotypes'] == '*1/*2':
            cyp2c9 = 2
        elif row['Cyp2C9 genotypes'] == '*1/*3':
            cyp2c9 = 3
        elif row['Cyp2C9 genotypes'] == '*2/*2':
            cyp2c9 = 4
        elif row['Cyp2C9 genotypes'] == '*2/*3':
            cyp2c9 = 5
        elif row['Cyp2C9 genotypes'] == '*3/*3':
            cyp2c9 = 6
        vkorc1 = 0
        if row['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] == 'A/G':
            vkorc1 = 1
        elif row['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] == 'A/A':
            vkorc1 = 2
        outrow = {'dose':actual_dose, 'age':age, 'height':height, 'weight':weight, 'asian':asian, 'black':black, 'unknown_race':unknown_race, 'enzyme_inducer':enzyme_inducer, 'amiodarone':amiodarone, 'gender':gender, 'smoker':smoker, 'aspirin':aspirin, 'cyp2c9':cyp2c9, 'vkorc1':vkorc1}
        outrows.append(outrow)
        linear_est = 4.0376 - (0.2546 * age) + (0.0118 * height) + (0.0134 * weight) - (0.6752 * asian) + (0.04060 * black) + (.0443 * unknown_race) + (1.2799 * enzyme_inducer) - (0.5695 * amiodarone)
        linear_est = linear_est**2
        best_linear_arm = 2
        if linear_est < 21:
            best_linear_arm = 0
        elif linear_est >= 21 and linear_est <= 49:
            best_linear_arm = 1
        if actual_dose < 21 and best_linear_arm == 2:
            num_severe += 1
        elif actual_dose > 49 and best_linear_arm == 0:
            num_severe += 1
        if (actual_dose < 21 and linear_est < 21):
            linear_correct += 1
        elif (actual_dose >=21 and actual_dose <= 49) and (linear_est >= 21 and linear_est <= 49):
            linear_correct += 1
        elif (actual_dose > 49 and linear_est > 49):
            linear_correct += 1
        linear_total += 1
        linear_correct_fractions.append(linear_correct/linear_total)
        linear_pct_severe_mistakes.append(num_severe/linear_total)
        xt = [[age, height, weight, asian, black, unknown_race, enzyme_inducer, amiodarone, gender, smoker, aspirin, cyp2c9, vkorc1]] 
        fixed_regret = regression.findBestArmReward(xt) - regression.findArmReward(xt, 1)
        fixed_regret_total += fixed_regret[0]
        fixed_cum_regret.append(fixed_regret_total)
        linear_regret = regression.findBestArmReward(xt) - regression.findArmReward(xt, best_linear_arm)
        linear_regret_total += linear_regret[0]
        linear_cum_regret.append(linear_regret_total)
    print("Fixed performance =", fixed_correct / fixed_total)
    print("Fixed cum regret =", fixed_regret_total)
    print("Fixed pct severe mistakes =", 0.0 / linear_total)
    print("Warfarin clinical dosing performance =", linear_correct / linear_total)
    print("Linear cum regret = ", linear_regret_total)
    print("Linear pct severe mistakes =", num_severe / linear_total)


with open('data/augmented_features.csv', 'w', newline='') as outfile:
    fieldnames = ['dose', 'age', 'height', 'weight', 'asian', 'black', 'unknown_race', 'enzyme_inducer', 'amiodarone', 'gender', 'smoker', 'aspirin', 'cyp2c9', 'vkorc1']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(outrows)

'''
plt.plot(linear_correct_fractions)
plt.xlabel('Number of Patients Served')
plt.ylabel('Correct Percentage')
plt.title('Correct Percentage of Warfarin Clinical Dosing Algorithm Over Time')
plt.show()

plt.plot(fixed_cum_regret, label="fixed")
plt.plot(linear_cum_regret, label="linear")
plt.xlabel('Number of Patients Served')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret of Baseline Dosing Algorithms Over Time')
plt.legend()
plt.show()
'''






