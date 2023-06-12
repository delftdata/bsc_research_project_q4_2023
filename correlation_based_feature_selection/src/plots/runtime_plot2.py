import os
import matplotlib.pyplot as plt
import pandas as pd


plt.style.use('seaborn-darkgrid')
plt.rc_context({"context": "paper"})


def plot_over_runtime():
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    data = {
        'Duration': [
            # BreastCancer
            2, 3, 5, 6,
            2, 3, 5, 12,
            2, 1, 4, 6,
            1, 4, 6, 2,
            1, 4, 2, 9],
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
