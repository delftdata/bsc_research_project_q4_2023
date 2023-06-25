import numpy as np
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


current_dataset1 = 'Gisette'
current_number_of_features1 = 200
current_good_features1 = list(range(1, 201, 10))

current_dataset2 = 'BreastCancer'
current_number_of_features2 = 31
current_good_features2 = list(range(1, 32, 1))

current_dataset3 = 'SteelPlatesFaults'
current_number_of_features3 = 33
current_good_features3 = list(range(1, 34, 1))

current_dataset4 = 'CensusIncome'
current_number_of_features4 = 14
current_good_features4 = list(range(1, 15, 1))

current_dataset5 = 'Nursery'
current_number_of_features5 = 8
current_good_features5 = list(range(1, 9, 1))

current_dataset6 = 'Arrhythmia'
current_number_of_features6 = 279
current_good_features6 = list(range(1, 201, 10))
for i in [210, 220, 230, 240, 250, 260, 270, 279]:
    current_good_features6.append(i)

threshold_values = [0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1, 0]

evaluation_metrics_options = {
    'accuracy': 'Average accuracy (%)',
    'rmse': 'Average root mean square error',
}


def parse_data_all_threshold(dataset=current_dataset1, num_files=5):
    pearson_performance = []
    spearman_performance = []
    cramersv_performance = []
    su_performance = []
    baseline_performance = 0

    for i in range(1, num_files + 1):
        if i == 1:
            file_path_pearson = f'./results_tables_select_c/txt_files/{dataset}_1_XGBoost_Pearson.txt'
            file_path_spearman = f'./results_tables_select_c/txt_files/{dataset}_1_XGBoost_Spearman.txt'
            file_path_cramer = f'./results_tables_select_c/txt_files/{dataset}_1_XGBoost_Cramer.txt'
            file_path_su = f'./results_tables_select_c/txt_files/{dataset}_1_XGBoost_SU.txt'
        elif i == 2:
            file_path_pearson = f'./results_tables_select_c/txt_files/{dataset}_1_LightGBM_Pearson.txt'
            file_path_spearman = f'./results_tables_select_c/txt_files/{dataset}_1_LightGBM_Spearman.txt'
            file_path_cramer = f'./results_tables_select_c/txt_files/{dataset}_1_LightGBM_Cramer.txt'
            file_path_su = f'./results_tables_select_c/txt_files/{dataset}_1_LightGBM_SU.txt'
        elif i == 3:
            file_path_pearson = f'./results_tables_select_c/txt_files/{dataset}_1_RandomForest_Pearson.txt'
            file_path_spearman = f'./results_tables_select_c/txt_files/{dataset}_1_RandomForest_Spearman.txt'
            file_path_cramer = f'./results_tables_select_c/txt_files/{dataset}_1_RandomForest_Cramer.txt'
            file_path_su = f'./results_tables_select_c/txt_files/{dataset}_1_RandomForest_SU.txt'
        elif i == 4:
            file_path_pearson = f'./results_tables_select_c/txt_files/{dataset}_1_LinearModel_Pearson.txt'
            file_path_spearman = f'./results_tables_select_c/txt_files/{dataset}_1_LinearModel_Spearman.txt'
            file_path_cramer = f'./results_tables_select_c/txt_files/{dataset}_1_LinearModel_Cramer.txt'
            file_path_su = f'./results_tables_select_c/txt_files/{dataset}_1_LinearModel_SU.txt'
        elif i == 5:
            file_path_pearson = f'./results_tables_select_c/txt_files/{dataset}_1_SVM_Pearson.txt'
            file_path_spearman = f'./results_tables_select_c/txt_files/{dataset}_1_SVM_Spearman.txt'
            file_path_cramer = f'./results_tables_select_c/txt_files/{dataset}_1_SVM_Cramer.txt'
            file_path_su = f'./results_tables_select_c/txt_files/{dataset}_1_SVM_SU.txt'
        count_1 = 0
        with open(file_path_pearson, 'r') as file:
            for line in file:
                performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+(\.\d+)?|0)', line)

                if performance_match:
                    current_performance = float(performance_match.group(1))
                    if i >= 2:
                        pearson_performance[count_1] += current_performance
                        count_1 = count_1 + 1
                    else:
                        pearson_performance.append(current_performance)

        count_2 = 0
        with open(file_path_spearman, 'r') as file:
            for line in file:
                performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+(\.\d+)?|0)', line)

                if performance_match:
                    current_performance = float(performance_match.group(1))
                    if i >= 2:
                        spearman_performance[count_2] += current_performance
                        count_2 = count_2 + 1
                    else:
                        spearman_performance.append(current_performance)

        count_3 = 0
        with open(file_path_cramer, 'r') as file:
            for line in file:
                performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+(\.\d+)?|0)', line)

                if performance_match:
                    current_performance = float(performance_match.group(1))
                    if i >= 2:
                        cramersv_performance[count_3] += current_performance
                        count_3 = count_3 + 1
                    else:
                        cramersv_performance.append(current_performance)

        count_4 = 0
        with open(file_path_su, 'r') as file:
            for line in file:
                performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+(\.\d+)?|0)', line)

                if performance_match:
                    current_performance = float(performance_match.group(1))
                    if i >= 2:
                        su_performance[count_4] += current_performance
                        count_4 = count_4 + 1
                    else:
                        su_performance.append(current_performance)

        with open(file_path_su, 'r') as file:
            for line in file:
                match = re.search(r'BASELINE PERFORMANCE: (\d+\.\d+)', line)
                if match:
                    baseline_value = float(match.group(1))
                    baseline_performance += baseline_value
                    break

    pearson_performance = [value / num_files for value in pearson_performance]
    spearman_performance = [value / num_files for value in spearman_performance]
    cramersv_performance = [value / num_files for value in cramersv_performance]
    su_performance = [value / num_files for value in su_performance]
    baseline_performance /= num_files

    # print(pearson_performance)
    # print(spearman_performance)
    # print(cramersv_performance)
    # print(su_performance)
    # print(baseline_value)

    return pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance


def plot_average_over_number_of_features_threshold(dataset_type=1, evaluation_metric='accuracy'):
    pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance = \
        parse_data_all_threshold(dataset=current_dataset1, num_files=5)

    pearson_performance2, spearman_performance2, cramersv_performance2, su_performance2, baseline_performance2 = \
        parse_data_all_threshold(dataset=current_dataset2, num_files=5)

    pearson_performance3, spearman_performance3, cramersv_performance3, su_performance3, baseline_performance3 = \
        parse_data_all_threshold(dataset=current_dataset3, num_files=5)

    pearson_performance4, spearman_performance4, cramersv_performance4, su_performance4, baseline_performance4 = \
        parse_data_all_threshold(dataset=current_dataset4, num_files=5)

    pearson_performance5, spearman_performance5, cramersv_performance5, su_performance5, baseline_performance5 = \
        parse_data_all_threshold(dataset=current_dataset5, num_files=5)

    pearson_performance6, spearman_performance6, cramersv_performance6, su_performance6, baseline_performance6 = \
        parse_data_all_threshold(dataset=current_dataset6, num_files=5)

    sns.set(font_scale=2.6)
    sns.set_style("whitegrid", {"grid.color": "0.9", "grid.linestyle": "-", "grid.linewidth": "0.2"})
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 12), dpi=1200, gridspec_kw={'hspace': 0.5})

    sns.lineplot(x=threshold_values, y=np.array(pearson_performance) * 100, legend=False,
                 marker='D', color='#10A5D6', label='Pearson', linewidth=5, ax=axes[0][0], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(spearman_performance) * 100, legend=False,
                 marker='o', color='#7030A0', label='Spearman', linewidth=5,  ax=axes[0][0], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(cramersv_performance) * 100, legend=False,
                 marker='p', color='#BF9000', label='Cramér\'s V', linewidth=5,  ax=axes[0][0], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(su_performance) * 100, legend=False,
                 marker='s', color='#c51b8a', label='Symmetric Uncertainty', linewidth=5, ax=axes[0][0], markersize=8)
    sns.lineplot(x=[threshold_values[-1]], y=[baseline_performance * 100],
                 marker='P', color='#000000', label='Baseline', ax=axes[0][0], legend=False, markersize=8,
                 linewidth=5)

    sns.lineplot(x=threshold_values, y=np.array(pearson_performance2) * 100,
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[0][1], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(spearman_performance2) * 100,
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[0][1], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(cramersv_performance2) * 100,
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[0][1], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(su_performance2) * 100,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[0][1], markersize=8)
    sns.lineplot(x=[threshold_values[-1]], y=[baseline_performance2 * 100],
                 marker='P', color='#000000',  ax=axes[0][1], linewidth=5, markersize=8)

    sns.lineplot(x=threshold_values, y=np.array(pearson_performance3) * 100,
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[0][2], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(spearman_performance3) * 100,
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[0][2], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(cramersv_performance3) * 100,
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[0][2], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(su_performance3) * 100,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[0][2], markersize=8)
    sns.lineplot(x=[threshold_values[-1]], y=[baseline_performance3 * 100],
                 marker='P', color='#000000', ax=axes[0][2], linewidth=5, markersize=8)

    sns.lineplot(x=threshold_values, y=np.array(pearson_performance4) * 100,
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[1][0], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(spearman_performance4) * 100,
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[1][0], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(cramersv_performance4) * 100,
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[1][0], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(su_performance4) * 100,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[1][0], markersize=8)
    sns.lineplot(x=[threshold_values[-1]], y=[baseline_performance4 * 100],
                 marker='P', color='#000000', ax=axes[1][0], linewidth=5, markersize=8)

    sns.lineplot(x=threshold_values, y=np.array(pearson_performance5) * 100,
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[1][1], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(spearman_performance5) * 100,
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[1][1], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(cramersv_performance5) * 100,
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[1][1], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(su_performance5) * 100,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[1][1], markersize=8)
    sns.lineplot(x=[threshold_values[-1]], y=[baseline_performance5 * 100],
                 marker='P', color='#000000', ax=axes[1][1], linewidth=5, markersize=8)

    sns.lineplot(x=threshold_values, y=np.array(pearson_performance6) * 100,
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[1][2], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(spearman_performance6) * 100,
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[1][2], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(cramersv_performance6) * 100,
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[1][2], markersize=8)
    sns.lineplot(x=threshold_values, y=np.array(su_performance6) * 100,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[1][2], markersize=8)
    sns.lineplot(x=[threshold_values[-1]], y=[baseline_performance6 * 100],
                 marker='P', color='#000000', ax=axes[1][2], linewidth=5, markersize=8)

    axes[0][1].set_xlabel('Number of features')
    axes[1][1].set_xlabel('Number of features')
    axes[0][0].set_ylabel('Accuracy (%)')
    axes[1][0].set_ylabel('Accuracy (%)')
    axes[0][0].set_title(current_dataset1 + ' (disc.)')
    axes[0][1].set_title(current_dataset2 + ' (cont.)')
    axes[0][2].set_title(current_dataset3 + ' (disc., cont.)')
    axes[1][0].set_title(current_dataset4 + ' (cont., nom.)')
    axes[1][1].set_title(current_dataset5 + ' (nom., ord.)')
    axes[1][2].set_title(current_dataset6 + ' (disc., cont., nom.)')

    for i in [0, 1]:
        for j in [0, 1, 2]:
            axes[i][j].spines['top'].set_linewidth(3)
            axes[i][j].spines['bottom'].set_linewidth(3)
            axes[i][j].spines['left'].set_linewidth(3)
            axes[i][j].spines['right'].set_linewidth(3)

    # axes[1].set_facecolor('white')
    # axes[1].spines['top'].set_linewidth(1.2)
    # axes[1].spines['bottom'].set_linewidth(1.2)
    # axes[1].spines['left'].set_linewidth(1.2)
    # axes[1].spines['right'].set_linewidth(1.2)
    # axes[1].spines['top'].set_edgecolor('black')
    # axes[1].spines['bottom'].set_edgecolor('black')
    # axes[1].spines['left'].set_edgecolor('black')
    # axes[1].spines['right'].set_edgecolor('black')

    # axes[2].set_facecolor('white')
    # axes[2].spines['top'].set_linewidth(1.2)
    # axes[2].spines['bottom'].set_linewidth(1.2)
    # axes[2].spines['left'].set_linewidth(1.2)
    # axes[2].spines['right'].set_linewidth(1.2)
    # axes[2].spines['top'].set_edgecolor('black')
    # axes[2].spines['bottom'].set_edgecolor('black')
    # axes[2].spines['left'].set_edgecolor('black')
    # axes[2].spines['right'].set_edgecolor('black')

    lines, labels = axes[0][0].get_legend_handles_labels()
    all_lines = lines
    all_labels = labels
    fig.legend(all_lines, all_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1),
               frameon=True, facecolor='white', framealpha=1)

    # Create the directory if it doesn't exist
    directory = "./results_performance_avg_database"
    os.makedirs(directory, exist_ok=True)
    # Save the figure to folder
    #plt.tight_layout()
    plt.savefig(f'./results_performance_avg_database/average_datasets_effectiveness_select_c_best.pdf', dpi=1200,
                bbox_inches='tight')
    # plt.show()
    plt.clf()
