import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the Plots directory exists
os.makedirs('../Plots', exist_ok=True)

# Data
baseline = [0, 0, 0, 0, 0]
singular_plural_conceptor = [96.5, 94.3, 95.6, 94.9, 94.4]
capitalize_conceptor = [96.6, 95.9, 95.9, 95.5, 95.2]
singular_plural_capitalize_conceptor = [59.7, 60.3, 65, 60.9, 60.8]
singular_plural_additive = [89.2, 85.9, 87.7, 86.2, 86.3]
capitalize_additive = [94.9, 95, 93.1, 94.2, 93.3]
singular_plural_capitalize_additive = [30.1, 25.6, 29.4, 22.9, 26.6]

# Calculate means and standard deviations
means = [
    np.mean(baseline),
    np.mean(singular_plural_conceptor),
    np.mean(capitalize_conceptor),
    np.mean(singular_plural_capitalize_conceptor),
    np.mean(singular_plural_additive),
    np.mean(capitalize_additive),
    np.mean(singular_plural_capitalize_additive)
]

std_devs = [
    np.std(baseline),
    np.std(singular_plural_conceptor),
    np.std(capitalize_conceptor),
    np.std(singular_plural_capitalize_conceptor),
    np.std(singular_plural_additive),
    np.std(capitalize_additive),
    np.std(singular_plural_capitalize_additive)
]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

# Define group widths and spacing
initial_width = 0.2
width = initial_width * (1/4)  # Reduce width to 2/3
gap_between_groups = 0.075  # Reduced gap
group_size = 3 * width  # 3 bars with no gaps

# Adjusted x positions to create gaps between groups
x_conceptor = [0 + i * width for i in range(3)]
x_additive = [group_size + gap_between_groups + i * width for i in range(3)]
x_positions = x_conceptor + x_additive

colors = ['darkgreen', 'green', 'mediumseagreen', 'indianred', 'orange', 'gold']
labels = [
    'Singular-Plural Conceptor',
    'Capitalize Conceptor',
    'Singular-Plural + Capitalize Conceptor',
    'Singular-Plural Additive',
    'Capitalize Additive',
    'Singular-Plural + Capitalize Additive'
]

# Transparency values for the two groups
alphas = [1.0, 1.0, 1.0, 0.7, 0.7, 0.7]  # Right side bars more transparent

# Bar plot with error bars
for i in range(6):
    ax.bar(x_positions[i], means[i + 1], width, yerr=std_devs[i + 1], capsize=5, color=colors[i], alpha=alphas[i], label=labels[i])

# Add baseline as a dotted line
ax.axhline(y=0.4, color='grey', linestyle='--', label='Baseline')

# Title and labels with increased font sizes
ax.set_title('Singular-Plural and Capitalize Combined Steering (Layer 14)', fontsize=16)
ax.set_xlabel('Steering Mechanisms', fontsize=14)
ax.set_ylabel('Top-1 Accuracy (%) ± Std Dev', fontsize=14)
ax.set_xticks([group_size / 2, group_size + gap_between_groups + group_size / 2])
ax.set_xticklabels(['Conceptor (MC)', 'Additive (MC)'], rotation=45, ha='right')

# Add legend for all groups including baseline
handles = [
    plt.Line2D([0], [0], color='darkgreen', lw=4),
    plt.Line2D([0], [0], color='green', lw=4),
    plt.Line2D([0], [0], color='mediumseagreen', lw=4),
    plt.Line2D([0], [0], color='indianred', lw=4, alpha=0.2),
    plt.Line2D([0], [0], color='orange', lw=4, alpha=0.2),
    plt.Line2D([0], [0], color='gold', lw=4, alpha=0.2),
    plt.Line2D([0], [0], color='grey', linestyle='--', lw=2)
]
group_labels = [
    'Singular-Plural Conceptor',
    'Capitalize Conceptor',
    'Singular-Plural ∧ Capitalize Conceptor',
    'Singular-Plural Additive',
    'Capitalize Additive',
    'Singular-Plural + Capitalize Additive',
    'Baseline'
]
ax.legend(handles, group_labels, loc='upper left', bbox_to_anchor=(1, 1))

fig.tight_layout()
plt.savefig('../Plots/boolean_conceptors.pdf', format='pdf')
plt.show()
