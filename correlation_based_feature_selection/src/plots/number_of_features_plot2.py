import numpy as np
import os
import re
import random
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


current_algorithm = 'XGBoost'
current_dataset = 'InternetAds'
current_number_of_features = 200
evaluation_metrics_options = {
    'accuracy': 'Accuracy (%)',
    'rmse': 'Root mean square error',
}


def parse_data(dataset=current_dataset, algorithm=current_algorithm):
    pearson_performance = []
    spearman_performance = []
    cramersv_performance = []
    su_performance = []
    baseline_performance = 0

    file_path_pearson = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Pearson.txt'
    with open(file_path_pearson, 'r') as file:
        for line in file:
            match = re.search(r'CURRENT PERFORMANCE: (\d+\.\d+)', line)
            if match:
                pearson_value = float(match.group(1))
                pearson_performance.append(pearson_value)

    file_path_spearman = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Spearman.txt'
    with open(file_path_spearman, 'r') as file:
        for line in file:
            match = re.search(r'CURRENT PERFORMANCE: (\d+\.\d+)', line)
            if match:
                spearman_value = float(match.group(1))
                spearman_performance.append(spearman_value)

    file_path_cramer = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Cramer.txt'
    with open(file_path_cramer, 'r') as file:
        for line in file:
            match = re.search(r'CURRENT PERFORMANCE: (\d+\.\d+)', line)
            if match:
                cramer_value = float(match.group(1))
                cramersv_performance.append(cramer_value)

    file_path_su = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_SU.txt'
    with open(file_path_su, 'r') as file:
        for line in file:
            match = re.search(r'CURRENT PERFORMANCE: (\d+\.\d+)', line)
            if match:
                su_value = float(match.group(1))
                su_performance.append(su_value)
            match = re.search(r'BASELINE PERFORMANCE: (\d+\.\d+)', line)
            if match:
                baseline_value = float(match.group(1))
                baseline_performance = baseline_value

    return pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance


def plot_over_number_of_features(dataset_type=1, evaluation_metric='accuracy'):
    dataset_name = current_dataset
    algorithm = current_algorithm
    number_of_features = current_number_of_features
    pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance = parse_data()

    evaluation_metric_name = evaluation_metrics_options.get(evaluation_metric)
    number_of_features_iteration = list(range(1, number_of_features + 1))

    sns.set_style("whitegrid", {"grid.color": "0.9", "grid.linestyle": "-", "grid.linewidth": "0.2"})
    plt.figure(figsize=(8, 6), dpi=1200)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    # performance_values = np.concatenate([
    #     np.array(pearson_performance) * 100,
    #     np.array(spearman_performance) * 100,
    #     np.array(cramersv_performance) * 100,
    #     np.array(su_performance) * 100
    # ])
    # unique_values = np.unique(performance_values)
    # ordered_values = np.sort(unique_values)
    # print(list(ordered_values))

    sns.lineplot(x=number_of_features_iteration, y=np.array(pearson_performance) * 100,
                 marker='D', color='#10A5D6', label='Pearson', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(spearman_performance) * 100,
                 marker='o', color='#00005A', label='Spearman', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(cramersv_performance) * 100,
                 marker='>', color='#BF9000', label='Cramér\'s V', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(su_performance) * 100,
                 marker='s', color='#CA0020', label='Symmetric Uncertainty', linewidth=1.5)
    sns.lineplot(x=[number_of_features_iteration[-1]], y=[baseline_performance * 100],
                 marker='p', color='#000000', label='Baseline')

    plt.xlabel('Number of features')
    plt.ylabel(str(evaluation_metric_name))
    max_value = round(max(np.max(np.array(pearson_performance) * 100), np.max(np.array(spearman_performance) * 100),
                    np.max(np.array(cramersv_performance) * 100), np.max(np.array(su_performance) * 100)), 2)
    min_value = round(min(np.min(np.array(pearson_performance) * 100), np.min(np.array(spearman_performance) * 100),
                    np.min(np.array(cramersv_performance) * 100), np.min(np.array(su_performance) * 100)), 2)
    sns.set(font_scale=1.9)

    # THIS VARIES PER DATASET
    # y_ticks = [70, 72, 74, 76, 78, 80, 82, 84] # CI-RF
    # y_ticks = [77, 78, 79, 80, 81, 82, 83] # CI-LR
    y_ticks = [66, 68, 70, 72, 74, 78, 80, 82, 84, 86, min_value, max_value] #CI-LG, CI-XB
    # y_ticks = [64, 68, 72, 76, 80, 84, 88, 92, 96]
    # y_ticks = [54, 64, 68, min_value, 72, 76, 80, 84, 88, 92, 96, max_value, 100]
    plt.yticks(y_ticks)
    plt.xticks([1, 2, 4, 6, 8, 10, 12, 14])
    print(plt.gca().get_yticklabels())
    plt.gca().get_yticklabels()[11].set_color('#CA0020')
    plt.gca().get_yticklabels()[10].set_color('#CA0020')
    plt.xlim(0, number_of_features + 1)
    plt.ylim(50, 74)

    ax.set_facecolor('white')
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['top'].set_edgecolor('black')
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['left'].set_edgecolor('black')
    ax.spines['right'].set_edgecolor('black')

    # Create the directory if it doesn't exist
    directory = "./results_performance"
    os.makedirs(directory, exist_ok=True)
    # Save the figure to folder
    plt.savefig(f'./results_performance/result_{dataset_name}_{dataset_type}_{algorithm}.pdf')
    # plt.show()
    plt.clf()


def parse_data_custom(dataset=current_dataset, algorithm=current_algorithm):
    pearson_performance = []
    spearman_performance = []
    cramersv_performance = []
    su_performance = []
    baseline_performance = 0
    good_features = list(range(10, 271, 10))
    good_features.append(279)

    current_performance = None
    current_num_features = None
    file_path_pearson = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Pearson.txt'
    with open(file_path_pearson, 'r') as file:
        for line in file:
            performance_match = re.search(r'CURRENT PERFORMANCE: (\d+\.\d+)', line)
            features_match = re.search(r'SUBSET OF FEATURES: (\d+)', line)

            if features_match:
                current_num_features = int(features_match.group(1))
            if performance_match:
                current_performance = float(performance_match.group(1))

            if current_performance is not None and current_num_features is not None:
                if current_num_features in good_features:
                    # print("Matching block:")
                    # print("Current Performance:", current_performance)
                    # print("Number of Features:", current_num_features)
                    # print("-------------------")
                    pearson_performance.append(current_performance)

                current_performance = None
                current_num_features = None

    file_path_spearman = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Spearman.txt'
    with open(file_path_spearman, 'r') as file:
        for line in file:
            performance_match = re.search(r'CURRENT PERFORMANCE: (\d+\.\d+)', line)
            features_match = re.search(r'SUBSET OF FEATURES: (\d+)', line)

            if features_match:
                current_num_features = int(features_match.group(1))
            if performance_match:
                current_performance = float(performance_match.group(1))

            if current_performance is not None and current_num_features is not None:
                if current_num_features in good_features:
                    # print("Matching block:")
                    # print("Current Performance:", current_performance)
                    # print("Number of Features:", current_num_features)
                    # print("-------------------")
                    spearman_performance.append(current_performance)

                current_performance = None
                current_num_features = None

    file_path_cramer = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Cramer.txt'
    with open(file_path_cramer, 'r') as file:
        for line in file:
            performance_match = re.search(r'CURRENT PERFORMANCE: (\d+\.\d+)', line)
            features_match = re.search(r'SUBSET OF FEATURES: (\d+)', line)

            if features_match:
                current_num_features = int(features_match.group(1))
            if performance_match:
                current_performance = float(performance_match.group(1))

            if current_performance is not None and current_num_features is not None:
                if current_num_features in good_features:
                    # print("Matching block:")
                    # print("Current Performance:", current_performance)
                    # print("Number of Features:", current_num_features)
                    # print("-------------------")
                    cramersv_performance.append(current_performance)

                current_performance = None
                current_num_features = None

    file_path_su = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_SU.txt'
    with open(file_path_su, 'r') as file:
        for line in file:
            performance_match = re.search(r'CURRENT PERFORMANCE: (\d+\.\d+)', line)
            features_match = re.search(r'SUBSET OF FEATURES: (\d+)', line)

            if features_match:
                current_num_features = int(features_match.group(1))
            if performance_match:
                current_performance = float(performance_match.group(1))

            if current_performance is not None and current_num_features is not None:
                if current_num_features in good_features:
                    # print("Matching block:")
                    # print("Current Performance:", current_performance)
                    # print("Number of Features:", current_num_features)
                    # print("-------------------")
                    su_performance.append(current_performance)

                current_performance = None
                current_num_features = None

            match = re.search(r'BASELINE PERFORMANCE: (\d+\.\d+)', line)
            if match:
                baseline_value = float(match.group(1))
                baseline_performance = baseline_value

    return good_features, pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance


def plot_over_number_of_features_custom(dataset_type=1, evaluation_metric='accuracy'):
    dataset_name = current_dataset
    algorithm = current_algorithm
    feature_list, pearson_performance, spearman_performance, cramersv_performance, su_performance, \
        baseline_performance = parse_data_custom()

    evaluation_metric_name = evaluation_metrics_options.get(evaluation_metric)
    number_of_features_iteration = feature_list

    sns.set_style("whitegrid", {"grid.color": "0.9", "grid.linestyle": "-", "grid.linewidth": "0.2"})
    plt.figure(figsize=(8, 6), dpi=1200)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    print(number_of_features_iteration)
    print(pearson_performance)
    sns.lineplot(x=number_of_features_iteration, y=np.array(pearson_performance) * 100,
                 marker='D', color='#10A5D6', label='Pearson', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(spearman_performance) * 100,
                 marker='o', color='#00005A', label='Spearman', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(cramersv_performance) * 100,
                 marker='>', color='#BF9000', label='Cramér\'s V', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(su_performance) * 100,
                 marker='s', color='#CA0020', label='Symmetric Uncertainty', linewidth=1.5)
    print(baseline_performance * 100)
    sns.lineplot(x=[279], y=[baseline_performance * 100],
                 marker='p', color='#000000', label='Baseline')

    plt.xlabel('Number of features')
    plt.ylabel(str(evaluation_metric_name))
    max_value = round(max(np.max(np.array(pearson_performance) * 100), np.max(np.array(spearman_performance) * 100),
                    np.max(np.array(cramersv_performance) * 100), np.max(np.array(su_performance) * 100)), 2)
    min_value = round(min(np.min(np.array(pearson_performance) * 100), np.min(np.array(spearman_performance) * 100),
                    np.min(np.array(cramersv_performance) * 100), np.min(np.array(su_performance) * 100)), 2)
    sns.set(font_scale=1.9)

    # THIS VARIES PER DATASET
    # y_ticks = [70, 72, 74, 76, 78, 80, 82, 84] # CI-RF
    # y_ticks = [77, 78, 79, 80, 81, 82, 83] # CI-LR
    # y_ticks = [94, 95, 96, 97, 98, 99, 100, min_value, max_value] #CI-LG, CI-XB
    # y_ticks = [64, 68, 72, 76, 80, 84, 88, 92, 96]
    # y_ticks = [54, 64, 68, min_value, 72, 76, 80, 84, 88, 92, 96, max_value, 100]
    y_ticks = [46, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 74, max_value, min_value]
    plt.yticks(y_ticks)
    plt.xticks([10, 40, 70, 100, 130, 160, 190, 220, 250, 279])
    print(plt.gca().get_yticklabels())
    plt.gca().get_yticklabels()[13].set_color('#CA0020')
    plt.gca().get_yticklabels()[14].set_color('#CA0020')
    #plt.xlim(0, len(feature_list) + 1)
    plt.ylim(46, 74)
    plt.xlim(1, 283)

    ax.set_facecolor('white')
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['top'].set_edgecolor('black')
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['left'].set_edgecolor('black')
    ax.spines['right'].set_edgecolor('black')

    # Create the directory if it doesn't exist
    directory = "./results_performance"
    os.makedirs(directory, exist_ok=True)
    # Save the figure to folder
    plt.savefig(f'./results_performance/result_{dataset_name}_{dataset_type}_{algorithm}.pdf')
    # plt.show()
    plt.clf()