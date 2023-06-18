import numpy as np
import os
import re
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


current_algorithm = 'RandomForest'
current_dataset = 'BreastCancer'
current_number_of_features = 31


def parse_data(dataset=current_dataset, algorithm=current_algorithm):
    pearson_runtime = []
    spearman_runtime = []
    cramersv_runtime = []
    su_runtime = []
    baseline_runtime = 0

    file_path_pearson = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Pearson.txt'
    with open(file_path_pearson, 'r') as file:
        for line in file:
            match = re.search(r'CURRENT RUNTIME: (\d+\.\d+)', line)
            if match:
                pearson_value = float(match.group(1))
                pearson_runtime.append(pearson_value)

    file_path_spearman = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Spearman.txt'
    with open(file_path_spearman, 'r') as file:
        for line in file:
            match = re.search(r'CURRENT RUNTIME: (\d+\.\d+)', line)
            if match:
                spearman_value = float(match.group(1))
                spearman_runtime.append(spearman_value)

    file_path_cramer = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Cramer.txt'
    with open(file_path_cramer, 'r') as file:
        for line in file:
            match = re.search(r'CURRENT RUNTIME: (\d+\.\d+)', line)
            if match:
                cramer_value = float(match.group(1))
                cramersv_runtime.append(cramer_value)

    file_path_su = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_SU.txt'
    with open(file_path_su, 'r') as file:
        for line in file:
            match = re.search(r'CURRENT RUNTIME: (\d+\.\d+)', line)
            if match:
                su_value = float(match.group(1))
                su_runtime.append(su_value)
            match = re.search(r'BASELINE RUNTIME: (\d+\.\d+)', line)
            if match:
                baseline_value = float(match.group(1))
                baseline_runtime = baseline_value

    return pearson_runtime, spearman_runtime, cramersv_runtime, su_runtime, baseline_runtime


def plot_over_number_of_features_runtime(dataset_type=1):
    dataset_name = current_dataset
    algorithm = current_algorithm
    number_of_features = current_number_of_features
    pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance = parse_data()
    number_of_features_iteration = list(range(1, number_of_features + 1))

    sns.set_style("whitegrid", {"grid.color": "0.9", "grid.linestyle": "-", "grid.linewidth": "0.2"})
    plt.figure(figsize=(8, 6), dpi=1200)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    sns.lineplot(x=number_of_features_iteration, y=np.array(pearson_performance),
                 marker='D', color='#10A5D6', label='Pearson', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(spearman_performance),
                 marker='o', color='#00005A', label='Spearman', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(cramersv_performance),
                 marker='>', color='#BF9000', label='Cramér\'s V', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(su_performance),
                 marker='s', color='#CA0020', label='Symmetric Uncertainty', linewidth=1.5)
    print(baseline_performance)
    sns.lineplot(x=[number_of_features_iteration[-1]], y=[baseline_performance],
                 marker='p', color='#7030A0', label='Baseline')

    plt.xlabel('Number of features')
    plt.ylabel('Runtime (seconds)')
    max_value = round(max(np.max(np.array(pearson_performance)), np.max(np.array(spearman_performance)),
                    np.max(np.array(cramersv_performance)), np.max(np.array(su_performance)),
                          baseline_performance), 2)
    min_value = round(min(np.min(np.array(pearson_performance)), np.min(np.array(spearman_performance)),
                    np.min(np.array(cramersv_performance)), np.min(np.array(su_performance)),
                          baseline_performance), 2)
    sns.set(font_scale=1.9)

    # THIS VARIES PER DATASET
    # y_ticks = [70, 72, 74, 76, 78, 80, 82, 84] # CI-RF
    # y_ticks = [77, 78, 79, 80, 81, 82, 83] # CI-LR
    # y_ticks = [66, 68, 70, 72, 74, 78, 80, 82, 84, 86, min_value, max_value] #CI-LG, CI-XB
    # y_ticks = [64, 68, 72, 76, 80, 84, 88, 92, 96]
    # y_ticks = [54, 64, 68, min_value, 72, 76, 80, 84, 88, 92, 96, max_value, 100]
    # plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, max_value, min_value])
    #plt.xticks([1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 42])
    plt.xticks([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31])
    plt.yticks([0, 0.25, 0.5, 1, 1.25, 1.5, 1.75, 2, min_value, max_value])
    print(plt.gca().get_yticklabels())
    plt.gca().get_yticklabels()[8].set_color('#CA0020')
    plt.gca().get_yticklabels()[9].set_color('#CA0020')
    plt.xlim(0, 32)
    plt.ylim(0, 2)

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
    directory = "./results_runtime"
    os.makedirs(directory, exist_ok=True)
    # Save the figure to folder
    plt.savefig(f'./results_runtime/result_{dataset_name}_{dataset_type}_{algorithm}_runtime.pdf')
    # plt.show()
    plt.clf()


def parse_data_custom(dataset=current_dataset, algorithm=current_algorithm):
    pearson_runtime = []
    spearman_runtime = []
    cramersv_runtime = []
    su_runtime = []
    baseline_runtime = 0
    good_features = list(range(10, 251, 10))

    current_performance = None
    current_num_features = None
    file_path_pearson = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Pearson.txt'
    with open(file_path_pearson, 'r') as file:
        for line in file:
            performance_match = re.search(r'CURRENT RUNTIME: (\d+\.\d+)', line)
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
                    pearson_runtime.append(current_performance)

                current_performance = None
                current_num_features = None

    file_path_spearman = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Spearman.txt'
    with open(file_path_spearman, 'r') as file:
        for line in file:
            performance_match = re.search(r'CURRENT RUNTIME: (\d+\.\d+)', line)
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
                    spearman_runtime.append(current_performance)

                current_performance = None
                current_num_features = None

    file_path_cramer = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Cramer.txt'
    with open(file_path_cramer, 'r') as file:
        for line in file:
            performance_match = re.search(r'CURRENT RUNTIME: (\d+\.\d+)', line)
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
                    cramersv_runtime.append(current_performance)

                current_performance = None
                current_num_features = None

    file_path_su = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_SU.txt'
    with open(file_path_su, 'r') as file:
        for line in file:
            performance_match = re.search(r'CURRENT RUNTIME: (\d+\.\d+)', line)
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
                    su_runtime.append(current_performance)

                current_performance = None
                current_num_features = None

            match = re.search(r'BASELINE RUNTIME: (\d+\.\d+)', line)
            if match:
                baseline_value = float(match.group(1))
                baseline_performance = baseline_value

    return good_features, pearson_runtime, spearman_runtime, cramersv_runtime, su_runtime, \
        baseline_performance


def plot_over_number_of_features_runtime_custom(dataset_type=1, evaluation_metric='accuracy'):
    dataset_name = current_dataset
    algorithm = current_algorithm
    feature_list, pearson_performance, spearman_performance, cramersv_performance, su_performance, \
        baseline_performance = parse_data_custom()
    number_of_features_iteration = feature_list

    sns.set_style("whitegrid", {"grid.color": "0.9", "grid.linestyle": "-", "grid.linewidth": "0.2"})
    plt.figure(figsize=(8, 6), dpi=1200)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    print(number_of_features_iteration)
    print(pearson_performance)
    sns.lineplot(x=number_of_features_iteration, y=np.array(pearson_performance),
                 marker='D', color='#10A5D6', label='Pearson', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(spearman_performance),
                 marker='o', color='#00005A', label='Spearman', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(cramersv_performance),
                 marker='>', color='#BF9000', label='Cramér\'s V', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(su_performance),
                 marker='s', color='#CA0020', label='Symmetric Uncertainty', linewidth=1.5)
    sns.lineplot(x=[279], y=[baseline_performance],
                 marker='p', color='#7030A0', label='Baseline')

    plt.xlabel('Number of features')
    plt.ylabel('Runtime (seconds)')
    max_value = round(max(np.max(np.array(pearson_performance)), np.max(np.array(spearman_performance)),
                    np.max(np.array(cramersv_performance)), np.max(np.array(su_performance))), 2)
    min_value = round(min(np.min(np.array(pearson_performance)), np.min(np.array(spearman_performance)),
                    np.min(np.array(cramersv_performance)), np.min(np.array(su_performance))), 2)
    sns.set(font_scale=1.9)

    # THIS VARIES PER DATASET
    # y_ticks = [70, 72, 74, 76, 78, 80, 82, 84] # CI-RF
    # y_ticks = [77, 78, 79, 80, 81, 82, 83] # CI-LR
    # y_ticks = [94, 95, 96, 97, 98, 99, 100, min_value, max_value] #CI-LG, CI-XB
    # y_ticks = [64, 68, 72, 76, 80, 84, 88, 92, 96]
    # y_ticks = [54, 64, 68, min_value, 72, 76, 80, 84, 88, 92, 96, max_value, 100]
    y_ticks = [0, 3, 4, 5, 6, 7, 8, 9, 10, max_value, min_value]
    plt.yticks(y_ticks)
    plt.xticks([10, 40, 70, 100, 130, 160, 190, 220, 250])
    print(plt.gca().get_yticklabels())
    plt.gca().get_yticklabels()[10].set_color('#CA0020')
    plt.gca().get_yticklabels()[9].set_color('#CA0020')
    #plt.xlim(0, len(feature_list) + 1)
    plt.ylim(0, 20)
    plt.xlim(1, 253)

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
    directory = "./results_runtime"
    os.makedirs(directory, exist_ok=True)
    # Save the figure to folder
    plt.savefig(f'./results_runtime/result_{dataset_name}_{dataset_type}_{algorithm}_runtime.pdf')
    # plt.show()
    plt.clf()
