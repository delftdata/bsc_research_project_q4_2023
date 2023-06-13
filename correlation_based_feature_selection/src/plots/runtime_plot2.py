import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.style.use('seaborn-darkgrid')
plt.rc_context({"context": "paper"})


def parse_results(directory="./results_runtime2/txt_files"):
    runtimes = []
    methods = []
    datasets = []

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
                elif label == "SPEARMAN RUNTIME":
                    runtimes.append(value)
                    methods.append('Spearman')
                    datasets.append(file_name)
                elif label == "CRAMER'S V RUNTIME":
                    runtimes.append(value)
                    methods.append('Cramér\'s V')
                    datasets.append(file_name)
                elif label == "SYMMETRIC UNCERTAINTY RUNTIME":
                    runtimes.append(value)
                    methods.append('Symmetric Uncertainty')
                    datasets.append(file_name)

    dataframe = pd.DataFrame({
        "Runtime": runtimes,
        "Dataset": datasets,
        "Method": methods
    })

    return dataframe


def plot_over_runtime():
    sns.set_theme(style="whitegrid")

    dataframe = parse_results(directory="./results_runtime2/txt_files")

    custom_palette = ["#10A5D6", "#045a8d", "#BF9000", "#ca0020"]
    g = sns.catplot(
        data=dataframe, x="Dataset", y="Runtime", hue="Method",
        capsize=.2, palette=custom_palette, errorbar="sd",
        kind="point", height=7, aspect=.75, legend=False
    )
    g.despine(left=True)
    g.set(xlabel="Data")
    g.set(ylabel="Runtime (milliseconds)")

    plt.xticks(rotation=45)

    plt.gcf().set_size_inches(7, 7)
    plt.subplots_adjust(bottom=0.2)

    ax = plt.gca()
    ax.set_facecolor('#F0F0F0')
    ax.grid(color='white')

    plt.legend(loc='upper left', title='Correlation technique')
    plt.yticks([100, 1000, 2000, 3000, 4000, 5000, 6000])
    plt.text(0.44, 0.38, '10% of rows', transform=ax.transAxes, fontsize=9,
             verticalalignment='top')
    plt.text(0.43, 0.92, '100% of rows', transform=ax.transAxes, fontsize=9,
             verticalalignment='top')

    directory = "./results_runtime2"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'./results_runtime2/result.png')
    plt.clf()


def plot_over_runtime_large_datasets():
    sns.set_theme(style="whitegrid")

    dataframe = parse_results(directory="./results_runtime2/txt_files2")

    custom_palette = ["#10A5D6", "#045a8d", "#BF9000", "#ca0020"]
    g = sns.catplot(
        data=dataframe, x="Dataset", y="Runtime", hue="Method",
        capsize=.2, palette=custom_palette, errorbar="sd",
        kind="point", height=7, aspect=.75, legend=False
    )
    g.despine(left=True)
    g.set(xlabel="Data")
    g.set(ylabel="Runtime (milliseconds)")

    plt.xticks(rotation=45)

    plt.gcf().set_size_inches(7, 7)
    plt.subplots_adjust(bottom=0.2)

    ax = plt.gca()
    ax.set_facecolor('#F0F0F0')
    ax.grid(color='white')

    plt.legend(loc='upper left', title='Correlation technique')
    plt.yticks([100, 1000, 2000, 3000, 4000, 5000, 6000])
    # plt.text(0.44, 0.38, '10% of rows', transform=ax.transAxes, fontsize=9,
    #          verticalalignment='top')
    # plt.text(0.43, 0.92, '100% of rows', transform=ax.transAxes, fontsize=9,
    #          verticalalignment='top')

    directory = "./results_runtime2"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'./results_runtime2/result_large_datasets.png')
    plt.clf()