import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the Plots directory exists
os.makedirs('../Plots', exist_ok=True)

# Function to plot the data
def plot_data(layers, additive, additive_std, additive_mean_centered, additive_mean_centered_std, baseline, baseline_std, conceptor, conceptor_std, conceptor_mean_centered, conceptor_mean_centered_std, title):
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')

    # Plotting the data
    line1 = plt.errorbar(layers, conceptor_mean_centered, yerr=conceptor_mean_centered_std, label='Conceptor Mean Centered', color='green', marker='o', capsize=3)
    line2 = plt.errorbar(layers, conceptor, yerr=conceptor_std, label='Conceptor', color='mediumseagreen', marker='o', capsize=3)
    line3 = plt.errorbar(layers, additive_mean_centered, yerr=additive_mean_centered_std, label='Additive Mean Centered', color='orange', marker='o', capsize=3)
    line4 = plt.errorbar(layers, additive, yerr=additive_std, label='Additive', color='indianred', marker='o', capsize=3)
    baseline_line = plt.plot(layers, baseline, color='darkgray', linestyle='--', label='Baseline')

    # Styling
    plt.xlabel('Layer Modified', color='black')
    plt.ylabel('Top-1 Accuracy (%) ± Std Dev', color='black')
    plt.title(title, fontsize=16)
    plt.tick_params(axis='x', colors='black')
    plt.tick_params(axis='y', colors='black')
    handles = [line1, line2, line3, line4, baseline_line[0]]
    labels = ['Conceptor Mean Centered', 'Conceptor', 'Additive Mean Centered', 'Additive', 'Baseline']
    plt.legend(handles=handles, labels=labels)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().patch.set_facecolor('#f0f0f0')

    # Save the plot
    filename = f'../Plots/{title}.pdf'
    plt.savefig(filename, format='pdf')
    plt.close()

# Data and titles for all plots
plots = [
    {
        "layers": np.array([9, 10, 11, 12, 13, 14, 15, 16]),
        "additive": np.array([64.16, 65.92, 64.64, 64.32, 67.2, 74.34, 59.6, 55.1]),
        "additive_std": np.array([1.535708306, 1.503861696, 1.580632785, 1.522366579, 1.468332387, 1.843474979, 1.700588134, 1.64195006]),
        "additive_mean_centered": np.array([65.1, 71.38, 70.48, 63.7, 73.92, 87.06, 73.26, 85.5]),
        "additive_mean_centered_std": np.array([2.0376457, 1.747455293, 1.482430437, 1.666133248, 2.322412539, 1.237093368, 0.8822698, 1.573531061]),
        "baseline": np.array([0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44]),
        "baseline_std": np.array([0.29664794, 0.29664794, 0.29664794, 0.29664794, 0.29664794, 0.29664794, 0.29664794, 0.29664794]),
        "conceptor": np.array([55.7, 62.4, 63, 65.02, 86.64, 91.52, 87.36, 82.82]),
        "conceptor_std": np.array([0.678232998, 0.864869932, 0.855569985, 0.73593478, 0.928654941, 1.041921302, 1.096540013, 0.928224111]),
        "conceptor_mean_centered": np.array([63.72, 63.24, 63.72, 64.64, 85.38, 95.14, 92.08, 91.5]),
        "conceptor_mean_centered_std": np.array([1.349666626, 1.298614646, 1.349666626, 1.360294086, 1.643654465, 0.821218607, 1.34669967, 0.772010363]),
        "title": 'Singular-Plural'
    },
    {
        "layers": np.array([9, 10, 11, 12, 13, 14, 15, 16]),
        "additive": np.array([10.88, 18.3, 19.28, 17.74, 19.28, 15.26, 10, 5.92]),
        "additive_std": np.array([0.730479, 0.83666, 0.872697, 0.722772, 0.538145, 0.449889, 0.209762, 0.53066]),
        "additive_mean_centered": np.array([17.2, 20.78, 31.26, 24.86, 31.12, 24.3, 18.06, 16.9]),
        "additive_mean_centered_std": np.array([1.224745, 0.775629, 1.048046, 0.909065, 0.851822, 0.987927, 0.578273, 0.987927]),
        "baseline": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        "baseline_std": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        "conceptor": np.array([25.5, 33.2, 33.64, 52.26, 52.34, 43.22, 34.1, 21.72]),
        "conceptor_std": np.array([0.687023, 0.469042, 1.125344, 1.269015, 1.149957, 1.108873, 1.279062, 0.515364]),
        "conceptor_mean_centered": np.array([29.64, 40.56, 38.78, 51.62, 52.08, 46.08, 40.4, 32.12]),
        "conceptor_mean_centered_std": np.array([0.861626, 0.427083, 0.974474, 0.627375, 0.248193, 0.453431, 0.834266, 1.633891]),
        "title": 'Antonyms'
    },
    {
        "layers": np.array([9, 10, 11, 12, 13, 14, 15, 16]),
        "additive": np.array([43.48, 43.2, 43.1, 42.16, 49.78, 72, 61.08, 55.84]),
        "additive_std": np.array([1.582909, 1.39714, 1.606238, 1.300154, 1.95182, 1.892089, 2.093227, 2.169424]),
        "additive_mean_centered": np.array([41.22, 43.34, 43.64, 45.82, 49.9, 82.9, 76.48, 84.28]),
        "additive_mean_centered_std": np.array([1.399143, 1.3544, 1.179152, 1.259206, 1.374045, 1.228007, 1.222129, 0.594643]),
        "baseline": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        "baseline_std": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        "conceptor": np.array([35.28, 38, 42.64, 43.48, 46.58, 91.72, 89.56, 88.56]),
        "conceptor_std": np.array([1.324236, 1.161034, 1.641463, 1.582909, 1.276558, 0.348712, 0.755248, 1.385063]),
        "conceptor_mean_centered": np.array([42.8, 43.48, 43.48, 43.24, 44.92, 91.24, 90.66, 86.08]),
        "conceptor_mean_centered_std": np.array([1.720465, 1.582909, 1.582909, 1.540909, 1.48243, 0.752596, 0.656049, 1.587955]),
        "title": 'Present-Past'
    },
    {
        "layers": np.array([9, 10, 11, 12, 13, 14, 15, 16]),
        "additive": np.array([17.96, 18.72, 19.76, 17.7, 18.54, 7.7, 3.16, 2.3]),
        "additive_std": np.array([1.132431, 1.064707, 1.07443, 0.789937, 0.891291, 0.878635, 0.826075, 0.209762]),
        "additive_mean_centered": np.array([16.38, 19.94, 28.44, 25.62, 35.74, 5.64, 10.46, 24.7]),
        "additive_mean_centered_std": np.array([1.041921, 0.564269, 1.441666, 1.330263, 1.423517, 1.429126, 1.510761, 1.240967]),
        "baseline": np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]),
        "baseline_std": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        "conceptor": np.array([13.66, 22, 34.92, 46.78, 59.84, 37.98, 28.58, 10.2]),
        "conceptor_std": np.array([0.615142, 1.193315, 1.803774, 1.74631, 0.811419, 1.509172, 1.041921, 1.106345]),
        "conceptor_mean_centered": np.array([17.42, 27.98, 43.68, 47.3, 60.84, 47.96, 44.94, 33.02]),
        "conceptor_mean_centered_std": np.array([0.858836, 1.066583, 2.339573, 1.964688, 0.926499, 1.105622, 1.209297, 0.890842]),
        "title": 'English-French'
    },
    {
        "layers": np.array([9, 10, 11, 12, 13, 14, 15, 16]),
        "additive": np.array([7.74, 19.08, 14.2, 17.22, 20.1, 27.94, 17.7, 13.78]),
        "additive_std": np.array([0.402989, 0.957914, 1.356466, 0.835225, 0.681175, 1.573023, 1.740115, 1.205653]),
        "additive_mean_centered": np.array([14.04, 37.78, 41.14, 43.08, 65.8, 57.42, 54.38, 44.94]),
        "additive_mean_centered_std": np.array([1.415062, 2.314649, 2.632565, 2.605686, 2.115656, 1.208967, 2.04098, 1.888491]),
        "baseline": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        "baseline_std": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        "conceptor": np.array([9.06, 30.28, 55.52, 73.64, 77.58, 73.64, 65, 47.9]),
        "conceptor_std": np.array([1.07629, 1.343726, 0.667533, 1.551258, 0.719444, 1.022937, 1.285302, 2.37571]),
        "conceptor_mean_centered": np.array([40.48, 22.62, 71.1, 80.7, 76.32, 79.36, 75.94, 72.54]),
        "conceptor_mean_centered_std": np.array([3.20025, 1.062826, 2.061068, 1.063955, 3.140955, 1.187603, 0.652993, 1.46506]),
        "title": 'Country-Capital'
    },
    {
        "layers": np.array([9, 10, 11, 12, 13, 14, 15, 16]),
        "additive": np.array([79.44, 80.08, 85.78, 86.3, 91.78, 90.62, 78.12, 69.34]),
        "additive_std": np.array([2.488855, 2.478225, 1.526303, 1.449138, 0.643117, 0.775629, 1.307517, 1.718837]),
        "additive_mean_centered": np.array([79.1, 88.92, 92.18, 64.94, 90.48, 94.62, 87.46, 94.92]),
        "additive_mean_centered_std": np.array([2.366432, 1.295222, 0.994786, 2.216845, 0.752064, 0.919565, 1.284679, 0.574108]),
        "baseline": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        "baseline_std": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        "conceptor": np.array([15.18, 21.78, 55.38, 70.04, 91.5, 96.28, 94.22, 93.74]),
        "conceptor_std": np.array([0.968297, 1.07963, 1.884038, 2.791845, 0.83666, 0.800999, 0.783326, 0.703136]),
        "conceptor_mean_centered": np.array([77.24, 74.68, 83.58, 79.8, 94.56, 95.48, 95.4, 95]),
        "conceptor_mean_centered_std": np.array([1.46506, 1.712776, 1.795996, 3.267415, 0.417612, 0.318748, 0.695701, 0.357771]),
        "title": 'Capitalize'
    }
]

# Plot all datasets
for plot in plots:
    plot_data(
        plot["layers"],
        plot["additive"],
        plot["additive_std"],
        plot["additive_mean_centered"],
        plot["additive_mean_centered_std"],
        plot["baseline"],
        plot["baseline_std"],
        plot["conceptor"],
        plot["conceptor_std"],
        plot["conceptor_mean_centered"],
        plot["conceptor_mean_centered_std"],
        plot["title"]
    )
