import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the Plots directory exists
os.makedirs('../Plots', exist_ok=True)

# Data for layer 14
layer_index = 5  # Index for layer 14

# Data for all plots at layer 14
conceptor_mean_centered_layer14 = [
    95.14,  # Singular-Plural
    46.08,  # Antonyms
    91.24,  # Present-Past
    47.96,  # English-French
    79.36,  # Country-Capital
    95.48   # Capitalize
]

conceptor_layer14 = [
    91.52,  # Singular-Plural
    43.22,  # Antonyms
    91.72,  # Present-Past
    37.98,  # English-French
    73.64,  # Country-Capital
    96.28   # Capitalize
]

additive_mean_centered_layer14 = [
    87.06,  # Singular-Plural
    24.3,   # Antonyms
    82.9,   # Present-Past
    5.64,   # English-French
    57.42,  # Country-Capital
    94.62   # Capitalize
]

additive_layer14 = [
    74.34,  # Singular-Plural
    15.26,  # Antonyms
    72,     # Present-Past
    7.7,    # English-French
    27.94,  # Country-Capital
    90.62   # Capitalize
]

# Standard deviations for layer 14
conceptor_mean_centered_std_layer14 = [
    0.821218607,  # Singular-Plural
    0.453431,     # Antonyms
    0.752596,     # Present-Past
    1.105622,     # English-French
    1.187603,     # Country-Capital
    0.318748      # Capitalize
]

conceptor_std_layer14 = [
    1.041921302,  # Singular-Plural
    1.108873,     # Antonyms
    0.348712,     # Present-Past
    1.509172,     # English-French
    1.022937,     # Country-Capital
    0.800999      # Capitalize
]

additive_mean_centered_std_layer14 = [
    1.237093368,  # Singular-Plural
    0.987927,     # Antonyms
    1.228007,     # Present-Past
    1.429126,     # English-French
    1.208967,     # Country-Capital
    0.919565      # Capitalize
]

additive_std_layer14 = [
    1.843474979,  # Singular-Plural
    0.449889,     # Antonyms
    1.892089,     # Present-Past
    0.878635,     # English-French
    1.573023,     # Country-Capital
    0.775629      # Capitalize
]

# Calculate the average performance and std for each mechanism at layer 14
avg_conceptor_mean_centered = np.mean(conceptor_mean_centered_layer14)
avg_conceptor = np.mean(conceptor_layer14)
avg_additive_mean_centered = np.mean(additive_mean_centered_layer14)
avg_additive = np.mean(additive_layer14)

avg_conceptor_mean_centered_std = np.sqrt(np.sum(np.array(conceptor_mean_centered_std_layer14) ** 2) / len(conceptor_mean_centered_std_layer14))
avg_conceptor_std = np.sqrt(np.sum(np.array(conceptor_std_layer14) ** 2) / len(conceptor_std_layer14))
avg_additive_mean_centered_std = np.sqrt(np.sum(np.array(additive_mean_centered_std_layer14) ** 2) / len(additive_mean_centered_std_layer14))
avg_additive_std = np.sqrt(np.sum(np.array(additive_std_layer14) ** 2) / len(additive_std_layer14))

# Combine data for plotting
tasks = ['Singular-Plural', 'Antonyms', 'Present-Past', 'English-French', 'Country-Capital', 'Capitalize', 'Average']
conceptor_mean_centered_values = conceptor_mean_centered_layer14 + [avg_conceptor_mean_centered]
conceptor_values = conceptor_layer14 + [avg_conceptor]
additive_mean_centered_values = additive_mean_centered_layer14 + [avg_additive_mean_centered]
additive_values = additive_layer14 + [avg_additive]

conceptor_mean_centered_std_values = conceptor_mean_centered_std_layer14 + [avg_conceptor_mean_centered_std]
conceptor_std_values = conceptor_std_layer14 + [avg_conceptor_std]
additive_mean_centered_std_values = additive_mean_centered_std_layer14 + [avg_additive_mean_centered_std]
additive_std_values = additive_std_layer14 + [avg_additive_std]

# Bar plot
x = np.arange(len(tasks))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

# Set plot area background color
ax.set_facecolor('#e0e0e0')

# Remove border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

rects1 = ax.bar(x - 1.5*width, conceptor_mean_centered_values, width, yerr=conceptor_mean_centered_std_values, label='Conceptor Mean Centered', color='green', capsize=5)
rects2 = ax.bar(x - 0.5*width, conceptor_values, width, yerr=conceptor_std_values, label='Conceptor', color='mediumseagreen', capsize=5)
rects3 = ax.bar(x + 0.5*width, additive_mean_centered_values, width, yerr=additive_mean_centered_std_values, label='Additive Mean Centered', color='orange', capsize=5)
rects4 = ax.bar(x + 1.5*width, additive_values, width, yerr=additive_std_values, label='Additive', color='indianred', capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Tasks', fontsize=14, labelpad=10)
ax.set_ylabel('Top-1 Accuracy (%) ± Std Dev', fontsize=14)
ax.set_title('Performance at Layer 14', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(tasks)

# Customizing legend background color
legend = ax.legend(facecolor='white')

fig.tight_layout()

# Save the plot
plt.savefig('../Plots/best_perfoming_layer_14.pdf', format='pdf')
plt.show()
