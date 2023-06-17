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
        "Percentage": percentanges
    })

    print(dataframe)
    return dataframe

def plot_over_runtime():
    dataframe = parse_results(directory="./results_runtime2/txt_files")
    methods = ['Pearson', 'Spearman', "Cramér's V", 'Symmetric Uncertainty']

    sns.set_theme(style="whitegrid")
    custom_palette = ["#E4EBF1", "#D8E9F9", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b",
                      "000000"]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.tight_layout(pad=4.0)

    for i, method in enumerate(methods):
        subplot_ax = axs[i]
        method_dataframe = dataframe[dataframe['Method'] == method]
        g = sns.barplot(
            data=method_dataframe, x="Dataset", y="Runtime", hue="Percentage",
            order=['Bike Sharing (16)', 'Breast Cancer (31)', 'Steel Plates Faults (33)', 'Connect4 (42)',
                   'Housing Prices (80)', 'Arrhythmia (279)'],
            palette=custom_palette,
            ax=subplot_ax
        )
        g.set(xlabel="Database")
        g.set(ylabel="Runtime (seconds)")
        g.set(title=method)
        g.set_xticklabels(rotation=45, labels=['Bike Sharing (16)', 'Breast Cancer (31)', 'Steel Plates Faults (33)',
                                               'Connect4 (42)',
                                               'Housing Prices (80)', 'Arrhythmia (279)'])
        g.grid(color='white')

        if i != 0:
            g.legend_.remove()

    # Create a single legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels))

    directory = "./results_runtime2"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'./results_runtime2/result.pdf')
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