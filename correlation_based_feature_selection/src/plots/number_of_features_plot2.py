import numpy as np
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


current_algorithm = 'SVM2'
current_dataset = 'BreastCancer'
current_number_of_features = 31
current_good_features = list(range(1, 32, 1))
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
    pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance = \
        parse_data(dataset=current_dataset, algorithm='LightGBM')

    pearson_performance2, spearman_performance2, cramersv_performance2, su_performance2, baseline_performance2 = \
        parse_data(dataset=current_dataset, algorithm='RandomForest')

    pearson_performance3, spearman_performance3, cramersv_performance3, su_performance3, baseline_performance3 = \
        parse_data(dataset=current_dataset, algorithm='XGBoost')

    pearson_performance4, spearman_performance4, cramersv_performance4, su_performance4, baseline_performance4 = \
        parse_data(dataset=current_dataset, algorithm='LinearModel')

    pearson_performance5, spearman_performance5, cramersv_performance5, su_performance5, baseline_performance5 = \
        parse_data(dataset=current_dataset, algorithm='SVM2')

    sns.set(font_scale=2.6)
    sns.set_style("whitegrid", {"grid.color": "0.9", "grid.linestyle": "-", "grid.linewidth": "0.2"})
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 12), dpi=1200, gridspec_kw={'hspace': 0.5})

    axes[1][1].remove()

    sns.lineplot(x=current_good_features, y=np.array(pearson_performance) * 100, legend=False,
                 marker='D', color='#10A5D6', label='Pearson', linewidth=5, ax=axes[0][0], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(spearman_performance) * 100, legend=False,
                 marker='o', color='#7030A0', label='Spearman', linewidth=5,  ax=axes[0][0], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(cramersv_performance) * 100, legend=False,
                 marker='p', color='#BF9000', label='Cramér\'s V', linewidth=5,  ax=axes[0][0], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(su_performance) * 100, legend=False,
                 marker='s', color='#c51b8a', label='Symmetric Uncertainty', linewidth=5, ax=axes[0][0], markersize=8)
    sns.lineplot(x=[current_good_features[-1]], y=[baseline_performance * 100],
                 marker='P', color='#000000', label='Baseline', ax=axes[0][0], legend=False, markersize=8,
                 linewidth=5)

    sns.lineplot(x=current_good_features, y=np.array(pearson_performance2) * 100,
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[0][1], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(spearman_performance2) * 100,
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[0][1], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(cramersv_performance2) * 100,
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[0][1], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(su_performance2) * 100,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[0][1], markersize=8)
    sns.lineplot(x=[current_good_features[-1]], y=[baseline_performance2 * 100],
                 marker='P', color='#000000',  ax=axes[0][1], linewidth=5, markersize=8)

    sns.lineplot(x=current_good_features, y=np.array(pearson_performance3) * 100,
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[0][2], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(spearman_performance3) * 100,
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[0][2], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(cramersv_performance3) * 100,
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[0][2], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(su_performance3) * 100,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[0][2], markersize=8)
    sns.lineplot(x=[current_good_features[-1]], y=[baseline_performance3 * 100],
                 marker='P', color='#000000', ax=axes[0][2], linewidth=5, markersize=8)

    sns.lineplot(x=current_good_features, y=np.array(pearson_performance4) * 100,
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[1][0], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(spearman_performance4) * 100,
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[1][0], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(cramersv_performance4) * 100,
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[1][0], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(su_performance4) * 100,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[1][0], markersize=8)
    sns.lineplot(x=[current_good_features[-1]], y=[baseline_performance4 * 100],
                 marker='P', color='#000000', ax=axes[1][0], linewidth=5, markersize=8)

    sns.lineplot(x=current_good_features, y=np.array(pearson_performance5) * 100,
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[1][2], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(spearman_performance5) * 100,
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[1][2], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(cramersv_performance5) * 100,
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[1][2], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(su_performance5) * 100,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[1][2], markersize=8)
    sns.lineplot(x=[current_good_features[-1]], y=[baseline_performance5 * 100],
                 marker='P', color='#000000', ax=axes[1][2], linewidth=5, markersize=8)

    axes[0][1].set_xlabel('Number of features')
    axes[1][1].set_xlabel('Number of features')
    axes[0][0].set_ylabel('Accuracy (%)')
    axes[1][0].set_ylabel('Accuracy (%)')
    axes[0][0].set_title('LightGBM')
    axes[0][1].set_title('RandomForest')
    axes[0][2].set_title('XGBoost')
    axes[1][0].set_title('LinearModel')
    axes[1][2].set_title('SVM')

    axes[0][0].set_xticks([0, 10, 20, 31])
    axes[0][1].set_xticks([0, 10, 20, 31])
    axes[0][2].set_xticks([0, 10, 20, 31])
    axes[1][0].set_xticks([0, 10, 20, 31])
    axes[1][2].set_xticks([0, 10, 20, 31])


    for i in [0, 1]:
        for j in [0, 1, 2]:
            axes[i][j].spines['top'].set_linewidth(3)
            axes[i][j].spines['bottom'].set_linewidth(3)
            axes[i][j].spines['left'].set_linewidth(3)
            axes[i][j].spines['right'].set_linewidth(3)

    lines, labels = axes[0][0].get_legend_handles_labels()
    all_lines = lines
    all_labels = labels
    fig.legend(all_lines, all_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1),
               frameon=True, facecolor='white', framealpha=1)

    # Create the directory if it doesn't exist
    directory = "./results_performance"
    os.makedirs(directory, exist_ok=True)
    # Save the figure to folder
    plt.savefig(f'./results_performance/result_{current_dataset}_{dataset_type}.pdf', dpi=1200,
                bbox_inches='tight')
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