import os
import matplotlib.pyplot as plt
import pandas as pd


plt.style.use('seaborn-darkgrid')
plt.rc_context({"context": "paper"})


def plot_over_runtime():
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    data = {
        'Runtime': [
            # BreastCancer
            23.97, 14.94, 191.22, 11.93,
            24.00, 14.80, 209.34, 23.06,
            23.50, 15.14, 247.67, 32.46,
            23.71, 15.28, 270.60, 44.41,
            23.22, 15.62, 304.16, 58.67,
            25.13, 16.41, 325.21, 74.35,
            26.40, 17.52, 391.79, 83.35,
            23.89, 16.74, 341.068, 87.28,
            23.23, 15.84, 344.80, 95.85,
            23.01, 16.19, 453.24, 99.91,
            # Gisette
            # 23.97, 14.94, 191.22, 11.93,
            # 24.00, 14.80, 209.34, 23.06,
            # 23.50, 15.14, 247.67, 32.46,
            # 23.71, 15.28, 270.60, 44.41,
            # 23.22, 15.62, 304.16, 58.67,
            # 25.13, 16.41, 325.21, 74.35,
            # 26.40, 17.52, 391.79, 83.35,
            # 23.89, 16.74, 341.068, 87.28,
            # 23.23, 15.84, 344.80, 95.85,
            # 23.01, 16.19, 453.24, 99.91
        ],
        'Dataset': [
            # BreastCancer
            'BreastCancer', 'BreastCancer', 'BreastCancer', 'BreastCancer',
            'BreastCancer', 'BreastCancer', 'BreastCancer', 'BreastCancer',
            'BreastCancer', 'BreastCancer', 'BreastCancer', 'BreastCancer',
            'BreastCancer', 'BreastCancer', 'BreastCancer', 'BreastCancer',
            'BreastCancer', 'BreastCancer', 'BreastCancer', 'BreastCancer',
            'BreastCancer', 'BreastCancer', 'BreastCancer', 'BreastCancer',
            'BreastCancer', 'BreastCancer', 'BreastCancer', 'BreastCancer',
            'BreastCancer', 'BreastCancer', 'BreastCancer', 'BreastCancer',
            'BreastCancer', 'BreastCancer', 'BreastCancer', 'BreastCancer',
            'BreastCancer', 'BreastCancer', 'BreastCancer', 'BreastCancer',
            # Gisette
            # 'Gisette', 'Gisette', 'Gisette', 'Gisette',
            # 'Gisette', 'Gisette', 'Gisette', 'Gisette',
            # 'Gisette', 'Gisette', 'Gisette', 'Gisette',
            # 'Gisette', 'Gisette', 'Gisette', 'Gisette',
            # 'Gisette', 'Gisette', 'Gisette', 'Gisette',
            # 'Gisette', 'Gisette', 'Gisette', 'Gisette',
            # 'Gisette', 'Gisette', 'Gisette', 'Gisette',
            # 'Gisette', 'Gisette', 'Gisette', 'Gisette',
            # 'Gisette', 'Gisette', 'Gisette', 'Gisette',
            # 'Gisette', 'Gisette', 'Gisette', 'Gisette',
        ],
        'Method': [
            # BreastCancer
            'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty'
            # Gisette
            # 'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            # 'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            # 'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            # 'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            # 'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            # 'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            # 'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            # 'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            # 'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty',
            # 'Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty'
        ]
    }
    df = pd.DataFrame(data)

    custom_palette = ["#10A5D6", "#C6209B", "#BF9000", "#2EB835"]
    g = sns.catplot(
        data=df, x="Dataset", y="Runtime", hue="Method",
        capsize=.2, palette=custom_palette, errorbar="se",
        kind="point", height=6, aspect=.75,
    )
    g.despine(left=True)
    g.set(xlabel="Database")
    g.set(ylabel="Runtime (milliseconds)")

    # Set background
    ax = plt.gca()
    ax.set_facecolor('#F0F0F0')
    ax.grid(color='white')

    # Create the directory if it doesn't exist
    directory = "./results_runtime2"
    os.makedirs(directory, exist_ok=True)
    # Save the figure to folder
    plt.savefig(f'./results_runtime2/result.png')
    # plt.show()
    plt.clf()
