import numpy as np
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


current_dataset4 = 'CensusIncome'
current_number_of_features4 = 14
current_good_features4 = list(range(1, 15, 1))

current_dataset2 = 'BreastCancer'
current_number_of_features2 = 31
current_good_features2 = list(range(1, 32, 1))

current_dataset3 = 'SteelPlatesFaults'
current_number_of_features3 = 33
current_good_features3 = list(range(1, 34, 1))

current_dataset5 = 'Nursery'
current_number_of_features5 = 8
current_good_features5 = list(range(1, 9, 1))

current_dataset6 = 'Arrhythmia'
current_number_of_features6 = 279
current_good_features6 = list(range(1, 201, 10))
for i in [210, 220, 230, 240, 250, 260, 270, 279]:
    current_good_features6.append(i)

current_dataset1 = 'Gisette'
current_number_of_features1 = 200
current_good_features1 = list(range(1, 201, 10))

evaluation_metrics_options = {
    'accuracy': 'Average accuracy (%)',
    'rmse': 'Average root mean square error',
}


def parse_data_all(dataset=current_dataset1, num_files=5, current_good_features=current_good_features1):
    pearson_performance = []
    spearman_performance = []
    cramersv_performance = []
    su_performance = []
    baseline_performance = 0

    current_performance = None
    current_num_features = None
    for i in range(1, num_files + 1):
        if i == 1:
            file_path_pearson = f'./raw_results/{dataset}/{dataset}_1_XGBoost_Pearson.txt'
            file_path_spearman = f'./raw_results/{dataset}/{dataset}_1_XGBoost_Spearman.txt'
            file_path_cramer = f'./raw_results/{dataset}/{dataset}_1_XGBoost_Cramer.txt'
            file_path_su = f'./raw_results/{dataset}/{dataset}_1_XGBoost_SU.txt'
        elif i == 2:
            file_path_pearson = f'./raw_results/{dataset}/{dataset}_1_LightGBM_Pearson.txt'
            file_path_spearman = f'./raw_results/{dataset}/{dataset}_1_LightGBM_Spearman.txt'
            file_path_cramer = f'./raw_results/{dataset}/{dataset}_1_LightGBM_Cramer.txt'
            file_path_su = f'./raw_results/{dataset}/{dataset}_1_LightGBM_SU.txt'
        elif i == 3:
            file_path_pearson = f'./raw_results/{dataset}/{dataset}_1_RandomForest_Pearson.txt'
            file_path_spearman = f'./raw_results/{dataset}/{dataset}_1_RandomForest_Spearman.txt'
            file_path_cramer = f'./raw_results/{dataset}/{dataset}_1_RandomForest_Cramer.txt'
            file_path_su = f'./raw_results/{dataset}/{dataset}_1_RandomForest_SU.txt'
        elif i == 4:
            file_path_pearson = f'./raw_results/{dataset}/{dataset}_1_LinearModel_Pearson.txt'
            file_path_spearman = f'./raw_results/{dataset}/{dataset}_1_LinearModel_Spearman.txt'
            file_path_cramer = f'./raw_results/{dataset}/{dataset}_1_LinearModel_Cramer.txt'
            file_path_su = f'./raw_results/{dataset}/{dataset}_1_LinearModel_SU.txt'
        elif i == 5:
            file_path_pearson = f'./raw_results/{dataset}/{dataset}_1_SVM2_Pearson.txt'
            file_path_spearman = f'./raw_results/{dataset}/{dataset}_1_SVM2_Spearman.txt'
            file_path_cramer = f'./raw_results/{dataset}/{dataset}_1_SVM2_Cramer.txt'
            file_path_su = f'./raw_results/{dataset}/{dataset}_1_SVM2_SU.txt'
        count_1 = 0
        with open(file_path_pearson, 'r') as file:
            for line in file:
                performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+\.\d+)', line)
                features_match = re.search(r'SUBSET OF FEATURES: (\d+)', line)

                if features_match:
                    current_num_features = int(features_match.group(1))
                if performance_match:
                    current_performance = float(performance_match.group(1))

                if current_performance is not None and current_num_features is not None:
                    if current_num_features in current_good_features:
                        if i >= 2:
                            pearson_performance[count_1] += current_performance
                            count_1 = count_1 + 1
                        else:
                            pearson_performance.append(current_performance)

                    current_performance = None
                    current_num_features = None

        count_2 = 0
        with open(file_path_spearman, 'r') as file:
            for line in file:
                performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+\.\d+)', line)
                features_match = re.search(r'SUBSET OF FEATURES: (\d+)', line)

                if features_match:
                    current_num_features = int(features_match.group(1))
                if performance_match:
                    current_performance = float(performance_match.group(1))

                if current_performance is not None and current_num_features is not None:
                    if current_num_features in current_good_features:
                        if i >= 2:
                            spearman_performance[count_2] += current_performance
                            count_2 = count_2 + 1
                        else:
                            spearman_performance.append(current_performance)

                    current_performance = None
                    current_num_features = None

        count_3 = 0
        with open(file_path_cramer, 'r') as file:
            for line in file:
                performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+\.\d+)', line)
                features_match = re.search(r'SUBSET OF FEATURES: (\d+)', line)

                if features_match:
                    current_num_features = int(features_match.group(1))
                if performance_match:
                    current_performance = float(performance_match.group(1))

                if current_performance is not None and current_num_features is not None:
                    if current_num_features in current_good_features:
                        if i >= 2:
                            cramersv_performance[count_3] += current_performance
                            count_3 = count_3 + 1
                        else:
                            cramersv_performance.append(current_performance)

                    current_performance = None
                    current_num_features = None

        count_4 = 0
        with open(file_path_su, 'r') as file:
            for line in file:
                performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+\.\d+)', line)
                features_match = re.search(r'SUBSET OF FEATURES: (\d+)', line)

                if features_match:
                    current_num_features = int(features_match.group(1))
                if performance_match:
                    current_performance = float(performance_match.group(1))

                if current_performance is not None and current_num_features is not None:
                    if current_num_features in current_good_features:
                        if i >= 2:
                            su_performance[count_4] += current_performance
                            count_4 = count_4 + 1
                        else:
                            su_performance.append(current_performance)

                    current_performance = None
                    current_num_features = None

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

    print(dataset)
    print(stats.ttest_ind(spearman_performance, su_performance))
    print('\n')

    return pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance


def parse_data_all_encoded(dataset=current_dataset1, num_files=5, current_good_features=current_good_features1):
    pearson_performance = []
    spearman_performance = []
    cramersv_performance = []
    su_performance = []
    baseline_performance = 0

    current_performance = None
    current_num_features = None
    for i in range(1, num_files + 1):
        if i == 1:
            file_path_pearson = f'./results_tables_new2/txt_files/{dataset}_3_XGBoost_Pearson.txt'
            file_path_spearman = f'./results_tables_new2/txt_files/{dataset}_3_XGBoost_Spearman.txt'
            file_path_cramer = f'./results_tables_new2/txt_files/{dataset}_3_XGBoost_Cramer.txt'
            file_path_su = f'./results_tables_new2/txt_files/{dataset}_3_XGBoost_SU.txt'
            if dataset == current_dataset3:
                file_path_pearson = f'./results_tables_new2/txt_files/{dataset}_2_XGBoost_Pearson.txt'
                file_path_spearman = f'./results_tables_new2/txt_files/{dataset}_2_XGBoost_Spearman.txt'
                file_path_cramer = f'./results_tables_new2/txt_files/{dataset}_2_XGBoost_Cramer.txt'
                file_path_su = f'./results_tables_new2/txt_files/{dataset}_2_XGBoost_SU.txt'
        elif i == 2:
            file_path_pearson = f'./results_tables_new2/txt_files/{dataset}_3_LightGBM_Pearson.txt'
            file_path_spearman = f'./results_tables_new2/txt_files/{dataset}_3_LightGBM_Spearman.txt'
            file_path_cramer = f'./results_tables_new2/txt_files/{dataset}_3_LightGBM_Cramer.txt'
            file_path_su = f'./results_tables_new2/txt_files/{dataset}_3_LightGBM_SU.txt'
            if dataset == current_dataset3:
                file_path_pearson = f'./results_tables_new2/txt_files/{dataset}_2_LightGBM_Pearson.txt'
                file_path_spearman = f'./results_tables_new2/txt_files/{dataset}_2_LightGBM_Spearman.txt'
                file_path_cramer = f'./results_tables_new2/txt_files/{dataset}_2_LightGBM_Cramer.txt'
                file_path_su = f'./results_tables_new2/txt_files/{dataset}_2_LightGBM_SU.txt'
        elif i == 3:
            file_path_pearson = f'./results_tables_new2/txt_files/{dataset}_3_RandomForest_Pearson.txt'
            file_path_spearman = f'./results_tables_new2/txt_files/{dataset}_3_RandomForest_Spearman.txt'
            file_path_cramer = f'./results_tables_new2/txt_files/{dataset}_3_RandomForest_Cramer.txt'
            file_path_su = f'./results_tables_new2/txt_files/{dataset}_3_RandomForest_SU.txt'
            if dataset == current_dataset3:
                file_path_pearson = f'./results_tables_new2/txt_files/{dataset}_2_RandomForest_Pearson.txt'
                file_path_spearman = f'./results_tables_new2/txt_files/{dataset}_2_RandomForest_Spearman.txt'
                file_path_cramer = f'./results_tables_new2/txt_files/{dataset}_2_RandomForest_Cramer.txt'
                file_path_su = f'./results_tables_new2/txt_files/{dataset}_2_RandomForest_SU.txt'
        elif i == 4:
            file_path_pearson = f'./results_tables_new2/txt_files/{dataset}_3_LinearModel_Pearson.txt'
            file_path_spearman = f'./results_tables_new2/txt_files/{dataset}_3_LinearModel_Spearman.txt'
            file_path_cramer = f'./results_tables_new2/txt_files/{dataset}_3_LinearModel_Cramer.txt'
            file_path_su = f'./results_tables_new2/txt_files/{dataset}_3_LinearModel_SU.txt'
            if dataset == current_dataset3:
                file_path_pearson = f'./results_tables_new2/txt_files/{dataset}_2_LinearModel_Pearson.txt'
                file_path_spearman = f'./results_tables_new2/txt_files/{dataset}_2_LinearModel_Spearman.txt'
                file_path_cramer = f'./results_tables_new2/txt_files/{dataset}_2_LinearModel_Cramer.txt'
                file_path_su = f'./results_tables_new2/txt_files/{dataset}_2_LinearModel_SU.txt'
        elif i == 5:
            file_path_pearson = f'./results_tables_new2/txt_files/{dataset}_3_SVM_Pearson.txt'
            file_path_spearman = f'./results_tables_new2/txt_files/{dataset}_3_SVM_Spearman.txt'
            file_path_cramer = f'./results_tables_new2/txt_files/{dataset}_3_SVM_Cramer.txt'
            file_path_su = f'./results_tables_new2/txt_files/{dataset}_3_SVM_SU.txt'
            if dataset == current_dataset3:
                file_path_pearson = f'./results_tables_new2/txt_files/{dataset}_2_SVM_Pearson.txt'
                file_path_spearman = f'./results_tables_new2/txt_files/{dataset}_2_SVM_Spearman.txt'
                file_path_cramer = f'./results_tables_new2/txt_files/{dataset}_2_SVM_Cramer.txt'
                file_path_su = f'./results_tables_new2/txt_files/{dataset}_2_SVM_SU.txt'
        count_1 = 0
        with open(file_path_pearson, 'r') as file:
            for line in file:
                performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+\.\d+)', line)
                features_match = re.search(r'SUBSET OF FEATURES: (\d+)', line)

                if features_match:
                    current_num_features = int(features_match.group(1))
                if performance_match:
                    current_performance = float(performance_match.group(1))

                if current_performance is not None and current_num_features is not None:
                    if current_num_features in current_good_features:
                        if i >= 2:
                            pearson_performance[count_1] += current_performance
                            count_1 = count_1 + 1
                        else:
                            pearson_performance.append(current_performance)

                    current_performance = None
                    current_num_features = None

        count_2 = 0
        with open(file_path_spearman, 'r') as file:
            for line in file:
                performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+\.\d+)', line)
                features_match = re.search(r'SUBSET OF FEATURES: (\d+)', line)

                if features_match:
                    current_num_features = int(features_match.group(1))
                if performance_match:
                    current_performance = float(performance_match.group(1))

                if current_performance is not None and current_num_features is not None:
                    if current_num_features in current_good_features:
                        if i >= 2:
                            spearman_performance[count_2] += current_performance
                            count_2 = count_2 + 1
                        else:
                            spearman_performance.append(current_performance)

                    current_performance = None
                    current_num_features = None

        count_3 = 0
        with open(file_path_cramer, 'r') as file:
            for line in file:
                performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+\.\d+)', line)
                features_match = re.search(r'SUBSET OF FEATURES: (\d+)', line)

                if features_match:
                    current_num_features = int(features_match.group(1))
                if performance_match:
                    current_performance = float(performance_match.group(1))

                if current_performance is not None and current_num_features is not None:
                    if current_num_features in current_good_features:
                        if i >= 2:
                            cramersv_performance[count_3] += current_performance
                            count_3 = count_3 + 1
                        else:
                            cramersv_performance.append(current_performance)

                    current_performance = None
                    current_num_features = None

        count_4 = 0
        with open(file_path_su, 'r') as file:
            for line in file:
                performance_match = re.search(r'CURRENT PERFORMANCE: (-?\d+\.\d+)', line)
                features_match = re.search(r'SUBSET OF FEATURES: (\d+)', line)

                if features_match:
                    current_num_features = int(features_match.group(1))
                if performance_match:
                    current_performance = float(performance_match.group(1))

                if current_performance is not None and current_num_features is not None:
                    if current_num_features in current_good_features:
                        if i >= 2:
                            su_performance[count_4] += current_performance
                            count_4 = count_4 + 1
                        else:
                            su_performance.append(current_performance)

                    current_performance = None
                    current_num_features = None

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

    print(dataset)
    print(stats.ttest_ind(pearson_performance, su_performance))
    print('\n')

    return pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance


def plot_average_over_number_of_features_encoded(dataset_type=1, evaluation_metric='accuracy'):
    # SPL unencoded
    pearson_performance, spearman_performance, cramersv_performance, su_performance, baseline_performance = \
        parse_data_all(dataset=current_dataset3, num_files=5, current_good_features=current_good_features3)

    # SPL encoded
    pearson_performance2, spearman_performance2, cramersv_performance2, su_performance2, baseline_performance2 = \
        parse_data_all_encoded(dataset=current_dataset3, num_files=5, current_good_features=current_good_features3)

    # NU unencoded
    pearson_performance3, spearman_performance3, cramersv_performance3, su_performance3, baseline_performance3 = \
        parse_data_all(dataset=current_dataset5, num_files=5, current_good_features=current_good_features5)

    # NU encoded
    pearson_performance4, spearman_performance4, cramersv_performance4, su_performance4, baseline_performance4 = \
        parse_data_all_encoded(dataset=current_dataset5, num_files=5, current_good_features=current_good_features5)

    sns.set(font_scale=2.6)
    sns.set_style("whitegrid", {"grid.color": "0.9", "grid.linestyle": "-", "grid.linewidth": "0.2"})
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12), dpi=1200, gridspec_kw={'hspace': 0.5})

    sns.lineplot(x=current_good_features3, y=np.array(pearson_performance) * 100, legend=False,
                 marker='D', color='#10A5D6', label='Pearson', linewidth=5, ax=axes[0][0], markersize=8)
    sns.lineplot(x=current_good_features3, y=np.array(spearman_performance) * 100, legend=False,
                 marker='o', color='#7030A0', label='Spearman', linewidth=5,  ax=axes[0][0], markersize=8)
    sns.lineplot(x=current_good_features3, y=np.array(cramersv_performance) * 100, legend=False,
                 marker='p', color='#BF9000', label='Cramér\'s V', linewidth=5,  ax=axes[0][0], markersize=8)
    sns.lineplot(x=current_good_features3, y=np.array(su_performance) * 100, legend=False,
                 marker='s', color='#c51b8a', label='Symmetric Uncertainty', linewidth=5, ax=axes[0][0], markersize=8)
    sns.lineplot(x=current_good_features3[-1], y=[baseline_performance * 100],
                 marker='P', color='#000000', label='Baseline', ax=axes[0][0], legend=False, markersize=8,
                 linewidth=5)

    sns.lineplot(x=current_good_features3, y=np.array(pearson_performance2) * 100,
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[0][1], markersize=8)
    sns.lineplot(x=current_good_features3, y=np.array(spearman_performance2) * 100,
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[0][1], markersize=8)
    sns.lineplot(x=current_good_features3, y=np.array(cramersv_performance2) * 100,
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[0][1], markersize=8)
    sns.lineplot(x=current_good_features3, y=np.array(su_performance2) * 100,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[0][1], markersize=8)
    sns.lineplot(x=current_good_features3[-1], y=[baseline_performance2 * 100],
                 marker='P', color='#000000',  ax=axes[0][1], linewidth=5, markersize=8)

    sns.lineplot(x=current_good_features5, y=np.array(cramersv_performance3) * 100,
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[1][0], markersize=8)
    sns.lineplot(x=current_good_features5, y=np.array(su_performance3) * 100,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[1][0], markersize=8)
    sns.lineplot(x=current_good_features5, y=np.array(pearson_performance3) * 100,
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[1][0], markersize=8)
    sns.lineplot(x=current_good_features5, y=np.array(spearman_performance3) * 100,
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[1][0], markersize=8)
    sns.lineplot(x=current_good_features5[-1], y=[baseline_performance3 * 100],
                 marker='P', color='#000000', ax=axes[1][0], linewidth=5, markersize=8)

    sns.lineplot(x=current_good_features5, y=np.array(cramersv_performance4) * 100,
                 marker='p', color='#BF9000', linewidth=5,  ax=axes[1][1], markersize=8)
    sns.lineplot(x=current_good_features5, y=np.array(su_performance4) * 100,
                 marker='s', color='#c51b8a', linewidth=5, ax=axes[1][1], markersize=8)
    sns.lineplot(x=current_good_features5, y=np.array(pearson_performance4) * 100,
                 marker='D', color='#10A5D6', linewidth=5, ax=axes[1][1], markersize=8)
    sns.lineplot(x=current_good_features5, y=np.array(spearman_performance4) * 100,
                 marker='o', color='#7030A0', linewidth=5,  ax=axes[1][1], markersize=8)
    sns.lineplot(x=[current_good_features5[-1]], y=[baseline_performance4 * 100],
                 marker='P', color='#000000', ax=axes[1][1], linewidth=5, markersize=8)

    axes[0][1].set_xlabel('Number of features')
    axes[1][1].set_xlabel('Number of features')
    axes[0][0].set_xlabel('Number of features')
    axes[1][0].set_xlabel('Number of features')
    axes[0][0].set_ylabel('Accuracy (%)')
    axes[1][0].set_ylabel('Accuracy (%)')
    axes[0][0].set_title(current_dataset3 + ' (disc., cont.)')
    axes[0][1].set_title(current_dataset3 + ' (cat.)')
    axes[1][0].set_title(current_dataset5 + ' (nom., ord.)')
    axes[1][1].set_title(current_dataset5 + ' (sparse num.)')

    # axes[1][0].set_xticks([5, 10, 14])
    axes[0][0].set_xticks([0, 10, 20, 33])
    axes[0][1].set_xticks([0, 10, 20, 33])


    for i in [0, 1]:
        for j in [0, 1]:
            axes[i][j].spines['top'].set_linewidth(3)
            axes[i][j].spines['bottom'].set_linewidth(3)
            axes[i][j].spines['left'].set_linewidth(3)
            axes[i][j].spines['right'].set_linewidth(3)

            # if i == 0 and j == 2:
            #     axes[i][j].spines['top'].set_edgecolor('black')
            #     axes[i][j].spines['bottom'].set_edgecolor('black')
            #     axes[i][j].spines['left'].set_edgecolor('black')
            #     axes[i][j].spines['right'].set_edgecolor('black')
            # if i == 1 and j == 1:
            #     axes[i][j].spines['top'].set_edgecolor('black')
            #     axes[i][j].spines['bottom'].set_edgecolor('black')
            #     axes[i][j].spines['left'].set_edgecolor('black')
            #     axes[i][j].spines['right'].set_edgecolor('black')

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
    plt.savefig(f'./results_performance_avg_database/encoded_vs_unencoded.pdf', dpi=1200,
                bbox_inches='tight')
    # plt.show()
    plt.clf()