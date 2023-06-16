import numpy as np
import os
import re
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


evaluation_metrics_options = {
    'accuracy': 'Accuracy (%)',
    'rmse': 'Root mean square error',
}
current_algorithm = 'LinearModel'
current_dataset = 'SteelPlatesFaults'

def parse_data(dataset=current_dataset, algorithm=current_algorithm):
    pearson_performance = []
    spearman_performance = []
    cramersv_performance = []
    su_performance = []

    file_path_pearson = f'./{dataset}/{dataset}_1_{algorithm}_Pearson.txt'
    with open(file_path_pearson, 'r') as file:
        for line in file:
            match = re.search(r'CURRENT PERFORMANCE: (\d+\.\d+)', line)
            if match:
                pearson_value = float(match.group(1))
                pearson_performance.append(pearson_value)

    file_path_spearman = f'./{dataset}/{dataset}_1_{algorithm}_Spearman.txt'
    with open(file_path_spearman, 'r') as file:
        for line in file:
            match = re.search(r'CURRENT PERFORMANCE: (\d+\.\d+)', line)
            if match:
                spearman_value = float(match.group(1))
                spearman_performance.append(spearman_value)

    file_path_cramer = f'./{dataset}/{dataset}_1_{algorithm}_Cramer.txt'
    with open(file_path_cramer, 'r') as file:
        for line in file:
            match = re.search(r'CURRENT PERFORMANCE: (\d+\.\d+)', line)
            if match:
                cramer_value = float(match.group(1))
                cramersv_performance.append(cramer_value)

    file_path_su = f'./{dataset}/{dataset}_1_{algorithm}_SU.txt'
    with open(file_path_su, 'r') as file:
        for line in file:
            match = re.search(r'CURRENT PERFORMANCE: (\d+\.\d+)', line)
            if match:
                su_value = float(match.group(1))
                su_performance.append(su_value)

    return pearson_performance, spearman_performance, cramersv_performance, su_performance


def get_dataset_details():
    dataset_name = current_dataset
    algorithm = current_algorithm
    number_of_features = 33

    return dataset_name, algorithm, number_of_features


def plot_over_number_of_features(dataset_type=1, evaluation_metric='accuracy'):
    dataset_name, algorithm, number_of_features = get_dataset_details()
    pearson_performance, spearman_performance, cramersv_performance, su_performance = parse_data()

    evaluation_metric_name = evaluation_metrics_options.get(evaluation_metric)
    number_of_features_iteration = list(range(1, number_of_features + 1))

    sns.set(style='whitegrid')

    plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    sns.lineplot(x=number_of_features_iteration, y=np.array(pearson_performance) * 100,
                 marker='D', color='#10A5D6', label='Pearson')
    sns.lineplot(x=number_of_features_iteration, y=np.array(spearman_performance) * 100,
                 marker='o', color='#00005A', label='Spearman')
    sns.lineplot(x=number_of_features_iteration, y=np.array(cramersv_performance) * 100,
                 marker='>', color='#BF9000', label='Cramer\'s V')
    sns.lineplot(x=number_of_features_iteration, y=np.array(su_performance) * 100,
                 marker='s', color='#CA0020', label='Symmetric Uncertainty')
    # sns.lineplot(x=number_of_features_iteration[-1], y=baseline_performance * 100,
    #              marker='o', color='#000000')

    plt.xlabel('Number of features')
    plt.ylabel(str(evaluation_metric_name))
    # plt.title('Change of ' + str(evaluation_metric_name)
    #           + f' for {algorithm} on {dataset_name} dataset ({dataset_type}) '
    #           + 'with the increase of selected features')

    # Add maximum value as a tick on the y-axis
    max_value = max(np.max(spearman_performance), np.max(pearson_performance),
                    np.max(cramersv_performance), np.max(su_performance))
    min_value = min(np.min(spearman_performance), np.min(pearson_performance),
                    np.min(cramersv_performance), np.min(su_performance))
    sns.set(font_scale=1.9)
    y_ticks = list(plt.yticks()[0])
    if max_value * 100 not in y_ticks:
        y_ticks.append(max_value * 100)
        y_ticks.append(min_value * 100)
    #y_ticks.remove(100)
    plt.yticks(y_ticks)
    plt.xticks([2, 5, 10, 15, 20, 25, 32])
    plt.xlim(0, number_of_features + 1)
    plt.ylim(60, 101)

    ax.set_facecolor('white')

    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)

    # Create the directory if it doesn't exist
    directory = "./results_performance"
    os.makedirs(directory, exist_ok=True)
    # Save the figure to folder
    plt.savefig(f'./results_performance/result_{dataset_name}_{dataset_type}_{algorithm}.pdf')
    # plt.show()
    plt.clf()
