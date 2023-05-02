import numpy as np
from scipy import stats
from scipy.stats import normaltest
import pandas as pd

# Accuracies of model 1 over 10 random iterations
model1 = pd.read_csv('scores_gru.csv')
model2 = pd.read_csv('scores_lstm.csv')

model1_acc = model1['Accuracy']
model2_acc = model2['Accuracy']

# Calculate means and standard deviations of accuracies
mean1 = np.mean(model1_acc)
mean2 = np.mean(model2_acc)
std1 = np.std(model1_acc)
std2 = np.std(model2_acc)

# Calculate t-value and degrees of freedom
t, p = stats.ttest_ind(model1_acc, model2_acc)
df = len(model1_acc) + len(model2_acc) - 2

# Calculate critical t-value at desired level of significance
alpha = 0.01
t_crit = stats.t.ppf(1 - alpha/2, df)

print(f"t-value: {t}")
print(f"p-value: {p}")
print(f"Degrees of freedom: {df}")
print(f"Critical t-value: {t_crit}")

# Compare calculated t-value to critical t-value
if abs(t) > t_crit:
    print(f"The difference in accuracies between the two models is statistically significant at alpha = {alpha}.")
else:
    print(f"The difference in accuracies between the two models is not statistically significant at alpha = {alpha}.")