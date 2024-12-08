import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the Plots directory exists
os.makedirs('../Plots', exist_ok=True)

# Data for layer 15
layer_index = 6  # Index for layer 15

# Data for all plots at layer 15
conceptor_mean_centered_layer15 = [
    92.08,  # Singular-Plural
    40.4,   # Antonyms
    90.66,  # Present-Past
    44.94,  # English-French
    75.94,  # Country-Capital
    95.4    # Capitalize
]

conceptor_layer15 = [
    87.36,  # Singular-Plural
    34.1,   # Antonyms
    89.56,  # Present-Past
    28.58,  # English-French
    65,     # Country-Capital
    94.22   # Capitalize
]

additive_mean_centered_layer15 = [
    73.26,  # Singular-Plural
    18.06,  # Antonyms
    76.48,  # Present-Past
    10.46,  # English-French
    54.38,  # Country-Capital
    87.46   # Capitalize
]

additive_layer15 = [
    59.6,   # Singular-Plural
    10,     # Antonyms
    61.08,  # Present-Past
    3.16,   # English-French
    17.7,   # Country-Capital
    78.12   # Capitalize
]

# Standard deviations for layer 15
conceptor_mean_centered_std_layer15 = [
    1.34669967,  # Singular-Plural
    0.834266,    # Antonyms
    0.656049,    # Present-Past
    1.209297,    # English-French
    0.652993,    # Country-Capital
    0.695701     # Capitalize
]

conceptor_std_layer15 = [
    1.096540013,  # Singular-Plural
    1.279062,     # Antonyms
    0.755248,     # Present-Past
    1.041921,     # English-French
    1.285302,     # Country-Capital
    0.783326      # Capitalize
]

additive_mean_centered_std_layer15 = [
    0.8822698,  # Singular-Plural
    0.578273,   # Antonyms
    1.222129,   # Present-Past
    1.510761,   # English-French
    2.04098,    # Country-Capital
    1.284679    # Capitalize
]

additive_std_layer15 = [
    1.700588134,  # Singular-Plural
    0.209762,     # Antonyms
    2.093227,     # Present-Past
    0.826075,     # English-French
    1.740115,     # Country-Capital
    1.307517      # Capitalize
]

# Calculate the average performance and std for each mechanism at layer 15
avg_conceptor_mean_centered = np.mean(conceptor_mean_centered_layer15)
avg_conceptor = np.mean(conceptor_layer15)
avg_additive_mean_centered = np.mean(additive_mean_centered_layer15)
avg_additive = np.mean(additive_layer15)

avg_conceptor_mean_centered_std = np.sqrt(np.sum(np.array(conceptor_mean_centered_std_layer15) ** 2) / len(conceptor_mean_centered_std_layer15))
avg_conceptor_std = np.sqrt(np.sum(np.array(conceptor_std_layer15) ** 2) / len(conceptor_std_layer15))
avg_additive_mean_centered_std = np.sqrt(np.sum(np.array(additive_mean_centered_std_layer15) ** 2) / len(additive_mean_centered_std_layer15))
avg_additive_std = np.sqrt(np.sum(np.array(additive_std_layer15) ** 2) / len(additive_std_layer15))

# Combine data for plotting
tasks = ['Singular-Plural', 'Antonyms', 'Present-Past', 'English-French', 'Country-Capital', 'Capitalize', 'Average']
conceptor_mean_centered_values = conceptor_mean_centered_layer15 + [avg_conceptor_mean_centered]
conceptor_values = conceptor_layer15 + [avg_conceptor]
additive_mean_centered_values = additive_mean_centered_layer15 + [avg_additive_mean_centered]
additive_values = additive_layer15 + [avg_additive]

conceptor_mean_centered_std_values = conceptor_mean_centered_std_layer15 + [avg_conceptor_mean_centered_std]
conceptor_std_values = conceptor_std_layer15 + [avg_conceptor_std]
additive_mean_centered_std_values = additive_mean_centered_std_layer15 + [avg_additive_mean_centered_std]
additive_std_values = additive_std_layer15 + [avg_additive_std]

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
ax.set_title('Performance at Layer 15', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(tasks)

# Customizing legend background color
legend = ax.legend(facecolor='white')

fig.tight_layout()

# Save the plot
plt.savefig('../Plots/best_perfoming_layer_15.pdf', format='pdf')
plt.show()
