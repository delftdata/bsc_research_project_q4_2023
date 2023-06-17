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
        "Percentage of rows": percentanges
    })

    return dataframe

def plot_over_runtime():
    dataframe = parse_results(directory="./results_runtime2/txt_files")
    #dataset = 'Breast Cancer (31)'
    #dataset = 'Gisette (5000)'
    #dataset = 'Housing Prices (80)'
    #dataset = 'Internet Ads (1558)'
    #dataset = 'Arrhythmia (279)'
    dataset = 'Bike Sharing (16)'
    breast_cancer_dataframe = dataframe[dataframe['Dataset'] == dataset]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6), dpi=1200)

    # custom_palette = ["#E4EBF1", "#D8E9F9", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b",
    #                   "000000"]
    custom_palette = ["#00005A", "#10A5D6", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61",
                      "#f46d43", "#CA0020", "#9A031B"]
    # custom_palette = ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15",
    #                   "#67000d", "000000"]

    g = sns.barplot(
        data=breast_cancer_dataframe, x="Method", y="Runtime", hue="Percentage of rows",
        palette=custom_palette
    )
    g.set(xlabel="Database")
    g.set(ylabel="Runtime (seconds)")
    g.set_title("Data table with 16 columns")
    g.set_xticklabels(['Pearson', 'Spearman', 'Cramér\'s V', 'SU'])

    plt.xticks(rotation=45)
    sns.set(font_scale=1.9)

    ax = plt.gca()
    ax.set_facecolor('white')
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['top'].set_edgecolor('black')
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['left'].set_edgecolor('black')
    ax.spines['right'].set_edgecolor('black')

    # [0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    plt.yticks([0.1, 0.5, 1, 2, 3, 4])
    #plt.yticks([0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
    plt.tick_params(axis='x', which='both', pad=0)

    directory = "./results_runtime2"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'./results_runtime2/result_{dataset}.pdf')
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