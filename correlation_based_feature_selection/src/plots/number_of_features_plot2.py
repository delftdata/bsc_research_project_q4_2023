import numpy as np
import os
import re
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


current_algorithm = 'SVM2'
#current_algorithm = 'LinearModel'
current_dataset = 'InternetAds'
current_number_of_features = 250
current_good_features = [1] + list(range(10, 251, 10))
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


def plot_over_number_of_features(dataset_type=1, evaluation_metric='rmse'):
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
                 marker='p', color='#7030A0', label='Baseline')

    plt.xlabel('Number of features')
    plt.ylabel(str(evaluation_metric_name))
    max_value = round(max(np.max(np.array(pearson_performance) * 100), np.max(np.array(spearman_performance) * 100),
                    np.max(np.array(cramersv_performance) * 100), np.max(np.array(su_performance) * 100)), 2)
    min_value = round(min(np.min(np.array(pearson_performance) * 100), np.min(np.array(spearman_performance) * 100),
                    np.min(np.array(cramersv_performance) * 100), np.min(np.array(su_performance) * 100)), 2)
    sns.set(font_scale=1.9)

    # THIS VARIES PER DATASET
    # plt.yticks([54, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, min_value, max_value])
    # plt.yticks([66, 76, 70, 72, 74, 78, 80, 82, 84, 88, min_value, max_value])
    # plt.yticks([54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, min_value, max_value])
    # plt.yticks([55, 60, 65, 75, 80, 85, 90, 95, 100, min_value, max_value])
    #plt.yticks([64, 68, 70, 72, 74, 82, 78, 80, 86, 84, 88, 90, min_value, max_value])

    # print(plt.gca().get_yticklabels())
    #plt.gca().get_yticklabels()[13].set_color('#CA0020')
    #plt.gca().get_yticklabels()[12].set_color('#CA0020')

    # plt.xticks([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31])
    #plt.xticks([1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 42])
    # plt.xticks([1, 2, 4, 6, 8, 10, 12, 14])
    # plt.xticks([1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33])
    # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8])

    #plt.xlim(0, 43)
    #plt.ylim(64, 90)

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
    good_features = current_good_features

    current_performance = None
    current_num_features = None
    file_path_pearson = f'./raw_results/{dataset}/{dataset}_1_{algorithm}_Pearson.txt'
    with open(file_path_pearson, 'r') as file:
        for line in file:
            performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+\.\d+)', line)
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
            performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+\.\d+)', line)
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
            performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+\.\d+)', line)
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
            performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+\.\d+)', line)
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

            match = re.search(r'BASELINE PERFORMANCE: (-?\d+\.\d+)', line)
            if match:
                baseline_value = float(match.group(1))
                baseline_performance = baseline_value

    return good_features, pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance


def plot_over_number_of_features_custom_regression_tasks(dataset_type=1, evaluation_metric='rmse'):
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

    sns.lineplot(x=number_of_features_iteration, y=np.array(pearson_performance),
                 marker='D', color='#10A5D6', label='Pearson', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(spearman_performance),
                 marker='o', color='#00005A', label='Spearman', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(cramersv_performance),
                 marker='>', color='#BF9000', label='Cramér\'s V', linewidth=1.5)
    sns.lineplot(x=number_of_features_iteration, y=np.array(su_performance),
                 marker='s', color='#CA0020', label='Symmetric Uncertainty', linewidth=1.5)
    print(baseline_performance)
    sns.lineplot(x=[current_good_features[-1]], y=[baseline_performance],
                 marker='p', color='#7030A0', label='Baseline')

    plt.xlabel('Number of features')
    plt.ylabel(str(evaluation_metric_name))
    max_value = round(max(np.max(np.array(pearson_performance)), np.max(np.array(spearman_performance)),
                    np.max(np.array(cramersv_performance)), np.max(np.array(su_performance))), 2)
    min_value = round(min(np.min(np.array(pearson_performance)), np.min(np.array(spearman_performance)),
                    np.min(np.array(cramersv_performance)), np.min(np.array(su_performance))), 2)
    sns.set(font_scale=1.9)

    # plt.yticks([-1 * 10**5, -0.75 * 10**5, -0.5 * 10**5, -0* 10**5, 0.25 * 10**5, 0.75 * 10**5, 0.5*10**5,
    #             1 * 10**5, 1.25 * 10**5, 1.5 * 10**5, 1.75*10**5, 2* 10**5, min_value, max_value])
    # plt.gca().get_yticklabels()[12].set_color('#CA0020')
    # plt.gca().get_yticklabels()[13].set_color('#CA0020')
    plt.yticks([-300, -250, -200, -150, -100, -50, 50, 100, 150, 200, 300, min_value, max_value])
    plt.gca().get_yticklabels()[11].set_color('#CA0020')
    plt.gca().get_yticklabels()[12].set_color('#CA0020')

    #plt.xticks([1, 5, 10, 20, 30, 40, 50, 60, 70, 80])
    plt.xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])

    plt.ylim(-300, 300)
    plt.xlim(0, current_good_features[-1] + 1)

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


def plot_over_number_of_features_custom(dataset_type=1, evaluation_metric='accuracy'):
    dataset_name = current_dataset
    algorithm = current_algorithm
    feature_list, pearson_performance, spearman_performance, cramersv_performance, su_performance, \
        baseline_performance = parse_data_custom()
    pearson_performance = np.array(pearson_performance) * 100
    spearman_performance = np.array(spearman_performance) * 100
    cramersv_performance = np.array(cramersv_performance) * 100
    su_performance = np.array(su_performance) * 100
    baseline_performance = baseline_performance * 100

    evaluation_metric_name = evaluation_metrics_options.get(evaluation_metric)
    number_of_features_iteration = feature_list

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
    sns.lineplot(x=[current_good_features[-1]], y=[baseline_performance],
                 marker='p', color='#7030A0', label='Baseline')

    plt.xlabel('Number of features')
    plt.ylabel(str(evaluation_metric_name))
    max_value = round(max(np.max(np.array(pearson_performance)), np.max(np.array(spearman_performance)),
                    np.max(np.array(cramersv_performance)), np.max(np.array(su_performance))), 2)
    min_value = round(min(np.min(np.array(pearson_performance)), np.min(np.array(spearman_performance)),
                    np.min(np.array(cramersv_performance)), np.min(np.array(su_performance))), 2)
    sns.set(font_scale=1.9)

    # THIS VARIES PER DATASET
    # plt.yticks([35, 40, 45, 55, 60, 65, 70, 75, min_value, max_value])
    plt.yticks([87, 89, 90, 91, 93, 92, 94, 95, 96, 98, 99, 100, min_value, max_value])
    plt.gca().get_yticklabels()[12].set_color('#CA0020')
    plt.gca().get_yticklabels()[13].set_color('#CA0020')

    # plt.xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
    plt.xticks([1, 10, 40, 70, 100, 130, 160, 190, 220, 250])

    plt.ylim(87, 100)
    plt.xlim(-3, current_good_features[-1] + 3)

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