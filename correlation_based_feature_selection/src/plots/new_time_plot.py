import numpy as np
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


current_algorithm = 'SVM2'
current_dataset = 'HousingPrices' # h
current_number_of_features = 80 # h
current_good_features = list(range(1, 81, 5)) # h
# for i in [210, 220, 230, 240, 250]:
#     current_good_features.append(i)
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
    good_features = current_good_features

    current_performance = None
    current_num_features = None
    file_path_pearson = f'./results_tables_time/txt_files/{dataset}_1_{algorithm}_Pearson.txt'
    with open(file_path_pearson, 'r') as file:
        for line in file:
            performance_match = re.search(r'CURRENT RUNTIME: (-?\d+\.\d+)', line)
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
            performance_match = re.search(r'CURRENT RUNTIME: (-?\d+\.\d+)', line)
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
            performance_match = re.search(r'CURRENT RUNTIME: (-?\d+\.\d+)', line)
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
            performance_match = re.search(r'CURRENT RUNTIME: (-?\d+\.\d+)', line)
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

            match = re.search(r'BASELINE RUNTIME: (-?\d+\.\d+)', line)
            if match:
                baseline_value = float(match.group(1))
                baseline_performance = baseline_value

    return pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance


def plot_over_number_of_features(dataset_type=1, evaluation_metric='rsme'):
    pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance = \
        parse_data(dataset=current_dataset, algorithm='LightGBM')

    pearson_performance2, spearman_performance2, cramersv_performance2, su_performance2, baseline_performance2 = \
        parse_data(dataset=current_dataset, algorithm='RandomForest')

    pearson_performance3, spearman_performance3, cramersv_performance3, su_performance3, baseline_performance3 = \
        parse_data(dataset=current_dataset, algorithm='XGBoost')

    pearson_performance4, spearman_performance4, cramersv_performance4, su_performance4, baseline_performance4 = \
        parse_data(dataset=current_dataset, algorithm='LinearModel')

    pearson_performance5, spearman_performance5, cramersv_performance5, su_performance5, baseline_performance5 = \
        parse_data(dataset=current_dataset, algorithm='SVM')

    sns.set(font_scale=2.6)
    sns.set_style("whitegrid", {"grid.color": "0.9", "grid.linestyle": "-", "grid.linewidth": "0.2"})
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 12), dpi=1200, gridspec_kw={'hspace': 0.5})

    axes[1][1].remove()

    baseline_performance = min(baseline_performance, max(np.max(pearson_performance), np.max(spearman_performance),
                                np.max(cramersv_performance), np.max(su_performance)) + 3)
    sns.lineplot(x=current_good_features, y=np.array(pearson_performance) , legend=False,
                 marker='D', color='#10A5D6', label='Pearson', linewidth=5, ax=axes[0][0], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(spearman_performance), legend=False,
                 marker='o', color='#7030A0', label='Spearman', linewidth=5,  ax=axes[0][0], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(cramersv_performance), legend=False,
                 marker='p', color='#BF9000', label='Cramér\'s V', linewidth=5,  ax=axes[0][0], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(su_performance) , legend=False,
                 marker='s', color='#c51b8a', label='Symmetric Uncertainty', linewidth=5, ax=axes[0][0], markersize=8)
    sns.lineplot(x=[current_good_features[-1]], y=[baseline_performance],
                 marker='P', color='#000000', label='Baseline', ax=axes[0][0], legend=False, markersize=8,
                 linewidth=5)

    sns.lineplot(x=current_good_features, y=np.array(pearson_performance2),
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[0][1], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(spearman_performance2),
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[0][1], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(cramersv_performance2),
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[0][1], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(su_performance2),
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[0][1], markersize=8)
    sns.lineplot(x=[current_good_features[-1]], y=[baseline_performance2],
                 marker='P', color='#000000',  ax=axes[0][1], linewidth=5, markersize=8)

    sns.lineplot(x=current_good_features, y=np.array(pearson_performance3),
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[0][2], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(spearman_performance3),
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[0][2], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(cramersv_performance3),
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[0][2], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(su_performance3),
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[0][2], markersize=8)
    sns.lineplot(x=[current_good_features[-1]], y=[baseline_performance3],
                 marker='P', color='#000000', ax=axes[0][2], linewidth=5, markersize=8)

    baseline_performance4 = min(baseline_performance4, max(np.max(pearson_performance4), np.max(spearman_performance4),
                                np.max(cramersv_performance4), np.max(su_performance4)) + 3)
    sns.lineplot(x=current_good_features, y=np.array(pearson_performance4),
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[1][0], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(spearman_performance4),
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[1][0], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(cramersv_performance4),
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[1][0], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(su_performance4) ,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[1][0], markersize=8)
    sns.lineplot(x=[current_good_features[-1]], y=[baseline_performance4],
                 marker='P', color='#000000', ax=axes[1][0], linewidth=5, markersize=8)

    sns.lineplot(x=current_good_features, y=np.array(pearson_performance5),
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[1][2], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(spearman_performance5),
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[1][2], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(cramersv_performance5),
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[1][2], markersize=8)
    sns.lineplot(x=current_good_features, y=np.array(su_performance5),
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[1][2], markersize=8)
    sns.lineplot(x=[current_good_features[-1]], y=[baseline_performance5],
                 marker='P', color='#000000', ax=axes[1][2], linewidth=5, markersize=8)

    axes[0][1].set_xlabel('Number of features')
    axes[1][1].set_xlabel('Number of features')
    # axes[0][0].set_ylabel('Accuracy (%)')
    # axes[1][0].set_ylabel('Accuracy (%)')
    # axes[0][0].set_ylabel('RMSE')
    # axes[1][0].set_ylabel('RMSE')
    axes[0][0].set_ylabel('Execution time (sec.)')
    axes[1][0].set_ylabel('Execution time (sec.)')
    axes[0][0].set_title('LightGBM')
    axes[0][1].set_title('RandomForest')
    axes[0][2].set_title('XGBoost')
    axes[1][0].set_title('LinearModel')
    axes[1][2].set_title('SVM')

    axes[0][0].set_xticks([0, 20, 40, 60, 80])
    axes[0][1].set_xticks([0, 20, 40, 60, 80])
    axes[0][2].set_xticks([0, 20, 40, 60, 80])
    axes[1][0].set_xticks([0, 20, 40, 60, 80])
    axes[1][2].set_xticks([0, 20, 40, 60, 80])

    # axes[0][0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # axes[0][1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # axes[0][2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # axes[1][0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # axes[1][2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

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
    plt.savefig(f'./results_performance/result_{current_dataset}_{dataset_type}_time.pdf', dpi=1200,
                bbox_inches='tight')
    # plt.show()
    plt.clf()
