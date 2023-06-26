import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.rc_context({"context": "paper"})

def parse_results(directory="./results_runtime2/txt_files"):
    runtimes = []
    methods = []
    datasets = []
    percentanges = []

    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue

        # File path
        file_path = os.path.join(directory, filename)

        # Extract the file name without the extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, 'r') as file:
            data = file.read()

        # Split the data by newline character
        lines = data.strip().split('\n\n')

        # Iterate over each section of data and extract the values
        for line in lines:
            # Split the lines by newline character and remove any leading/trailing spaces
            linesplit = [x.strip() for x in line.split('\n')]
            percentage = int(linesplit[0].split(':')[-1])

            # Extract the values
            for l in linesplit[1:]:
                label, value = l.split(':')
                label = label.strip()
                value = float(value.strip())

                if label == "PEARSON RUNTIME":
                    runtimes.append(value)
                    methods.append('Pearson')
                    datasets.append(file_name)
                    percentanges.append(percentage)
                elif label == "SPEARMAN RUNTIME":
                    runtimes.append(value)
                    methods.append('Spearman')
                    datasets.append(file_name)
                    percentanges.append(percentage)
                elif label == "CRAMER'S V RUNTIME":
                    runtimes.append(value)
                    methods.append('Cramér\'s V')
                    datasets.append(file_name)
                    percentanges.append(percentage)
                elif label == "SYMMETRIC UNCERTAINTY RUNTIME":
                    runtimes.append(value)
                    methods.append('Symmetric Uncertainty')
                    datasets.append(file_name)
                    percentanges.append(percentage)

    dataframe = pd.DataFrame({
        "Runtime": runtimes,
        "Dataset": datasets,
        "Method": methods,
        "Percentage of samples": percentanges
    })

    return dataframe

def plot_over_runtime():
    dataframe = parse_results(directory="./results_runtime2/txt_files")

    sns.set(font_scale=2.6)
    sns.set_style("whitegrid", {"grid.color": "0.9", "grid.linestyle": "-", "grid.linewidth": "0.2"})
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 12), dpi=1200, gridspec_kw={'hspace': 0.5})

    custom_palette = ["#10A5D6", "#7030A0", "#BF9000", "#c51b8a"]
    sns.scatterplot(
        data=dataframe[dataframe['Dataset'] == 'BikeSharing'], x="Percentage of samples", y="Runtime", hue="Method",
        palette=custom_palette, ax=axes[0][0], legend=False, s=350
    )
    sns.scatterplot(
        data=dataframe[dataframe['Dataset'] == 'BreastCancer'], x="Percentage of samples", y="Runtime", hue="Method",
        palette=custom_palette, ax=axes[0][1], legend=False, s=350
    )
    sns.scatterplot(
        data=dataframe[dataframe['Dataset'] == 'Connect4'], x="Percentage of samples", y="Runtime", hue="Method",
        palette=custom_palette, ax=axes[0][2], legend=False, s=350
    )
    sns.scatterplot(
        data=dataframe[dataframe['Dataset'] == 'Arrhythmia'], x="Percentage of samples", y="Runtime", hue="Method",
        palette=custom_palette, ax=axes[1][0], legend=False, s=350
    )
    sns.scatterplot(
        data=dataframe[dataframe['Dataset'] == 'InternetAds'], x="Percentage of samples", y="Runtime", hue="Method",
        palette=custom_palette, ax=axes[1][1], legend='full', s=350
    )
    sns.scatterplot(
        data=dataframe[dataframe['Dataset'] == 'Gisette'], x="Percentage of samples", y="Runtime", hue="Method",
        palette=custom_palette, ax=axes[1][2], legend=False, s=350
    )

    axes[0][0].set_title('BikeSharing (17,379x16)')
    axes[0][1].set_title('BreastCancer (569x31)')
    axes[0][2].set_title('Connect-4 (67,557x42)')
    axes[1][0].set_title('Arrhythmia (452x279)')
    axes[1][1].set_title('InternetAds (3,279x1,558)')
    axes[1][2].set_title('Gisette (6,000x5,000)')

    #g.set(xlabel="Dataset")
    axes[0][0].set_ylabel(ylabel="Execution time (sec.)")
    axes[1][0].set_ylabel(ylabel="Execution time (sec.)")
    axes[0][1].set_ylabel(ylabel="")
    axes[0][2].set_ylabel(ylabel="")
    axes[1][1].set_ylabel(ylabel="")
    axes[1][2].set_ylabel(ylabel="")

    axes[0][1].set_xlabel(xlabel='Percentage of samples')
    axes[1][1].set_xlabel(xlabel='Percentage of samples')
    axes[0][0].set_xlabel(xlabel='')
    axes[0][2].set_xlabel(xlabel='')
    axes[1][0].set_xlabel(xlabel='')
    axes[1][2].set_xlabel(xlabel='')

    for i in [0, 1]:
        for j in [0, 1, 2]:
            axes[i][j].spines['top'].set_linewidth(3)
            axes[i][j].spines['bottom'].set_linewidth(3)
            axes[i][j].spines['left'].set_linewidth(3)
            axes[i][j].spines['right'].set_linewidth(3)

    lgnd = axes[1][1].legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.7),
               frameon=True, facecolor='white', framealpha=1)

    for handle in lgnd.legendHandles:
        handle._sizes = [250]

    directory = "./results_runtime2"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'./results_runtime2/fs_runtime.pdf', dpi=1200,
                bbox_inches='tight')
    plt.clf()

# def plot_over_runtime_large_datasets():
#     dataframe = parse_results(directory="./results_runtime2/txt_files2")
#
#     sns.set_theme(style="whitegrid")
#     sns.set_style("whitegrid")
#     custom_palette = ["#10A5D6", "#045A8D", "#BF9000", "#CA0020"]
#     g = sns.catplot(
#         data=dataframe, x="Dataset", y="Runtime", hue="Method",
#         capsize=.2, palette=custom_palette, errorbar="sd", estimator="median",
#         kind="point", height=7, aspect=.75, legend=False,
#         order=["Connect-4 (42)", "HousingPrices (80)", "Arrhythmia (279)",
#                "InternetAds (1558)", "Gisette (5000)"]
#     )
#     g.despine(left=True)
#     g.set(xlabel="Data")
#     g.set(ylabel="Runtime (seconds)")
#
#     plt.xticks(rotation=45)
#
#     plt.gcf().set_size_inches(8, 8)
#     plt.subplots_adjust(bottom=0.2)
#
#     ax = plt.gca()
#     ax.grid(color='white')
#
#     plt.legend(loc='upper left', title='Correlation technique')
#     plt.yticks([0.1, 5, 10, 15, 20, 25, 30, 35])
#     plt.text(0.84, 0.30, '10% of rows', transform=ax.transAxes, fontsize=9,
#              verticalalignment='top')
#     plt.text(0.84, 0.99, '100% of rows', transform=ax.transAxes, fontsize=9,
#              verticalalignment='top')
#
#     directory = "./results_runtime2"
#     os.makedirs(directory, exist_ok=True)
#     plt.savefig(f'./results_runtime2/result_large_datasets.png')
#     plt.clf()