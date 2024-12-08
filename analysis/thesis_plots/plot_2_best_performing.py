import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the Plots directory exists
os.makedirs('../Plots', exist_ok=True)

# Define the width for the bars
width = 0.2

# Data for layer 14
layer_index_14 = 5  # Index for layer 14

conceptor_mean_centered_layer14 = [
    95.14, 46.08, 91.24, 47.96, 79.36, 95.48
]
conceptor_layer14 = [
    91.52, 43.22, 91.72, 37.98, 73.64, 96.28
]
additive_mean_centered_layer14 = [
    87.06, 24.3, 82.9, 5.64, 57.42, 94.62
]
additive_layer14 = [
    74.34, 15.26, 72, 7.7, 27.94, 90.62
]

conceptor_mean_centered_std_layer14 = [
    0.821218607, 0.453431, 0.752596, 1.105622, 1.187603, 0.318748
]
conceptor_std_layer14 = [
    1.041921302, 1.108873, 0.348712, 1.509172, 1.022937, 0.800999
]
additive_mean_centered_std_layer14 = [
    1.237093368, 0.987927, 1.228007, 1.429126, 1.208967, 0.919565
]
additive_std_layer14 = [
    1.843474979, 0.449889, 1.892089, 0.878635, 1.573023, 0.775629
]

avg_conceptor_mean_centered_14 = np.mean(conceptor_mean_centered_layer14)
avg_conceptor_14 = np.mean(conceptor_layer14)
avg_additive_mean_centered_14 = np.mean(additive_mean_centered_layer14)
avg_additive_14 = np.mean(additive_layer14)

avg_conceptor_mean_centered_std_14 = np.sqrt(np.sum(np.array(conceptor_mean_centered_std_layer14) ** 2) / len(conceptor_mean_centered_std_layer14))
avg_conceptor_std_14 = np.sqrt(np.sum(np.array(conceptor_std_layer14) ** 2) / len(conceptor_std_layer14))
avg_additive_mean_centered_std_14 = np.sqrt(np.sum(np.array(additive_mean_centered_std_layer14) ** 2) / len(additive_mean_centered_std_layer14))
avg_additive_std_14 = np.sqrt(np.sum(np.array(additive_std_layer14) ** 2) / len(additive_std_layer14))

tasks = ['Singular-Plural', 'Antonyms', 'Present-Past', 'English-French', 'Country-Capital', 'Capitalize', 'Average']
conceptor_mean_centered_values_14 = conceptor_mean_centered_layer14 + [avg_conceptor_mean_centered_14]
conceptor_values_14 = conceptor_layer14 + [avg_conceptor_14]
additive_mean_centered_values_14 = additive_mean_centered_layer14 + [avg_additive_mean_centered_14]
additive_values_14 = additive_layer14 + [avg_additive_14]

conceptor_mean_centered_std_values_14 = conceptor_mean_centered_std_layer14 + [avg_conceptor_mean_centered_std_14]
conceptor_std_values_14 = conceptor_std_layer14 + [avg_conceptor_std_14]
additive_mean_centered_std_values_14 = additive_mean_centered_std_layer14 + [avg_additive_mean_centered_std_14]
additive_std_values_14 = additive_std_layer14 + [avg_additive_std_14]

# Data for layer 15
layer_index_15 = 6  # Index for layer 15

conceptor_mean_centered_layer15 = [
    92.08, 40.4, 90.66, 44.94, 75.94, 95.4
]
conceptor_layer15 = [
    87.36, 34.1, 89.56, 28.58, 65, 94.22
]
additive_mean_centered_layer15 = [
    73.26, 18.06, 76.48, 10.46, 54.38, 87.46
]
additive_layer15 = [
    59.6, 10, 61.08, 3.16, 17.7, 78.12
]

conceptor_mean_centered_std_layer15 = [
    1.34669967, 0.834266, 0.656049, 1.209297, 0.652993, 0.695701
]
conceptor_std_layer15 = [
    1.096540013, 1.279062, 0.755248, 1.041921, 1.285302, 0.783326
]
additive_mean_centered_std_layer15 = [
    0.8822698, 0.578273, 1.222129, 1.510761, 2.04098, 1.284679
]
additive_std_layer15 = [
    1.700588134, 0.209762, 2.093227, 0.826075, 1.740115, 1.307517
]

avg_conceptor_mean_centered_15 = np.mean(conceptor_mean_centered_layer15)
avg_conceptor_15 = np.mean(conceptor_layer15)
avg_additive_mean_centered_15 = np.mean(additive_mean_centered_layer15)
avg_additive_15 = np.mean(additive_layer15)

avg_conceptor_mean_centered_std_15 = np.sqrt(np.sum(np.array(conceptor_mean_centered_std_layer15) ** 2) / len(conceptor_mean_centered_std_layer15))
avg_conceptor_std_15 = np.sqrt(np.sum(np.array(conceptor_std_layer15) ** 2) / len(conceptor_std_layer15))
avg_additive_mean_centered_std_15 = np.sqrt(np.sum(np.array(additive_mean_centered_std_layer15) ** 2) / len(additive_mean_centered_std_layer15))
avg_additive_std_15 = np.sqrt(np.sum(np.array(additive_std_layer15) ** 2) / len(additive_std_layer15))

conceptor_mean_centered_values_15 = conceptor_mean_centered_layer15 + [avg_conceptor_mean_centered_15]
conceptor_values_15 = conceptor_layer15 + [avg_conceptor_15]
additive_mean_centered_values_15 = additive_mean_centered_layer15 + [avg_additive_mean_centered_15]
additive_values_15 = additive_layer15 + [avg_additive_15]

conceptor_mean_centered_std_values_15 = conceptor_mean_centered_std_layer15 + [avg_conceptor_mean_centered_std_15]
conceptor_std_values_15 = conceptor_std_layer15 + [avg_conceptor_std_15]
additive_mean_centered_std_values_15 = additive_mean_centered_std_layer15 + [avg_additive_mean_centered_std_15]
additive_std_values_15 = additive_std_layer15 + [avg_additive_std_15]

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot for layer 14
ax1.set_facecolor('#f0f0f0')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

rects1_14 = ax1.bar(np.arange(len(tasks)) - 1.5*width, conceptor_mean_centered_values_14, width, yerr=conceptor_mean_centered_std_values_14, label='Conceptor Mean Centered', color='green', capsize=5)
rects2_14 = ax1.bar(np.arange(len(tasks)) - 0.5*width, conceptor_values_14, width, yerr=conceptor_std_values_14, label='Conceptor', color='mediumseagreen', capsize=5)
rects3_14 = ax1.bar(np.arange(len(tasks)) + 0.5*width, additive_mean_centered_values_14, width, yerr=additive_mean_centered_std_values_14, label='Additive Mean Centered', color='orange', capsize=5)
rects4_14 = ax1.bar(np.arange(len(tasks)) + 1.5*width, additive_values_14, width, yerr=additive_std_values_14, label='Additive', color='indianred', capsize=5)

ax1.set_xlabel('Tasks', fontsize=16, labelpad=10)  # Adjust labelpad to move label down
ax1.set_ylabel('Top-1 Accuracy (%) ± Std Dev', fontsize=16)
ax1.set_title('Performance at Layer 14', fontsize=20)
ax1.set_xticks(np.arange(len(tasks)))
ax1.set_xticklabels(tasks)
ax1.legend(facecolor='white')

# Plot for layer 15
ax2.set_facecolor('#f0f0f0')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

rects1_15 = ax2.bar(np.arange(len(tasks)) - 1.5*width, conceptor_mean_centered_values_15, width, yerr=conceptor_mean_centered_std_values_15, label='Conceptor Mean Centered', color='green', capsize=5)
rects2_15 = ax2.bar(np.arange(len(tasks)) - 0.5*width, conceptor_values_15, width, yerr=conceptor_std_values_15, label='Conceptor', color='mediumseagreen', capsize=5)
rects3_15 = ax2.bar(np.arange(len(tasks)) + 0.5*width, additive_mean_centered_values_15, width, yerr=additive_mean_centered_std_values_15, label='Additive Mean Centered', color='orange', capsize=5)
rects4_15 = ax2.bar(np.arange(len(tasks)) + 1.5*width, additive_values_15, width, yerr=additive_std_values_15, label='Additive', color='indianred', capsize=5)

ax2.set_xlabel('Tasks', fontsize=16, labelpad=10)  # Adjust labelpad to move label down
ax2.set_ylabel('Top-1 Accuracy (%) ± Std Dev', fontsize=16)
ax2.set_title('Performance at Layer 15', fontsize=20)
ax2.set_xticks(np.arange(len(tasks)))
ax2.set_xticklabels(tasks)
ax2.legend(facecolor='white')

fig.tight_layout(h_pad=2)  # Add vertical space between plots

plt.savefig('../Plots/2_best_perfoming.pdf', format='pdf')
plt.show()
