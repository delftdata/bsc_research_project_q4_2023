import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import figure
import seaborn as sns


def plot_over_features_mifs(dataset_name, title, n_features, mifs_000, mifs_025, mifs_050, mifs_075, mifs_100, is_classification, save=True):
    """
    Creates a single plot based on algorithm provided.
    Currently, the method will create plots for SVM model.

    Args:
        dataset_name: name of dataset
        title: title of plot
        n_features: the number of features to plot
        mifs_000: the results for MIFS with beta=0
        mifs_025: the results for MIFS with beta=0.25
        mifs_050: the results for MIFS with beta=0.50
        mifs_075: the results for MIFS with beta=0.75
        mifs_100: the results for MIFS with beta=1
        is_classification: is the dataset classification or regression task
        save: should you save to disk or not
    """
    features = list(range(1, n_features + 1))

    sns.set(font_scale=1.3, palette='bright')
    sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":", 'lines.markersize': 10000})

    k = int(n_features / 10)
    l = 2.2
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(k))
    x_label = 'Number of Features'

    if is_classification:
        y_label = 'Classification Accuracy (%)'
        mifs_000 = [[100 * el for el in model] for model in mifs_000]
        mifs_025 = [[100 * el for el in model] for model in mifs_025]
        mifs_050 = [[100 * el for el in model] for model in mifs_050]
        mifs_075 = [[100 * el for el in model] for model in mifs_075]
        mifs_100 = [[100 * el for el in model] for model in mifs_100]
    else:
        y_label = 'Root Mean Squared Error (RMSE)'

    figure(figsize=(8, 4.5))

    palette_tab10 = sns.color_palette("bright", 10)
    plt.plot(features, mifs_000[2], marker='^', markevery=k, linewidth=l, color=palette_tab10[4])
    plt.plot(features, mifs_025[2], marker='<', markevery=k, linewidth=l, color=palette_tab10[5])
    plt.plot(features, mifs_050[2], marker='s', markevery=k, linewidth=l, color=palette_tab10[0])
    plt.plot(features, mifs_075[2], marker='>', markevery=k, linewidth=l, color=palette_tab10[6])
    plt.plot(features, mifs_100[2], marker='v', markevery=k, linewidth=l, color=palette_tab10[8])
    plt.title(title)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.grid(True, linewidth=1.5, linestyle=':')
    # plt.legend(['MIFS', 'MRMR', 'CIFE', 'JMI', 'IG'],
    #            bbox_to_anchor=(0.8, 0), loc=4, ncol=5)

    if save:
        plt.savefig(f'./results/mifs_300/result_svm_{dataset_name}.png', dpi=300)
    plt.show()


def plot_over_features(dataset_name, title, n_features, mrmr, mifs, jmi, cife, base, is_classification, save=True):
    """
    Creates a single plot based on algorithm provided.
    Currently, the method will create plots for SVM model.

    Args:
        dataset_name: name of dataset
        title: title of plot
        n_features: number of features to plot
        mrmr: the results for MRMR
        mifs: the results for MIFS
        jmi: the results for JMI
        cife: the results for CIFE
        base: the results for IG
        is_classification: classification or regression dataset
        save: should you save to disk or not
    """
    features = list(range(1, n_features + 1))


    sns.set(font_scale=1.3, palette='bright')
    sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":", 'lines.markersize': 10000})

    k = int(n_features / 10)
    l = 2.2
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(k))
    x_label = 'Number of Features'
    # figure(figsize=(8, 6), dpi=100)

    if is_classification:
        y_label = 'Classification Accuracy (%)'
        mrmr = [[100 * el for el in model] for model in mrmr]
        mifs = [[100 * el for el in model] for model in mifs]
        cife = [[100 * el for el in model] for model in cife]
        jmi = [[100 * el for el in model] for model in jmi]
        base = [[100 * el for el in model] for model in base]
    else:
        y_label = 'Root Mean Squared Error (RMSE)'

    figure(figsize=(8, 4.5))

    plt.plot(features, mifs[0], marker='s', markevery=k, linewidth=l)
    plt.plot(features, mrmr[0], marker='o', markevery=k, linewidth=l)
    plt.plot(features, cife[0], marker='D', markevery=k, linewidth=l)
    plt.plot(features, jmi[0], marker='X', markevery=k, linewidth=l)
    plt.plot(features, base[0], marker='^', markevery=k, linewidth=l)
    plt.title(title)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.grid(True, linewidth=1.5, linestyle=':')
    # plt.legend(['MIFS', 'MRMR', 'CIFE', 'JMI', 'IG'],
    #            bbox_to_anchor=(0.8, 0), loc=4, ncol=5)

    if save:
        plt.savefig(f'./results/test_300/result_svm_{dataset_name}.png', dpi=300)
    plt.show()


def plot_over_features_2(dataset_name, title, n_features, mrmr, mifs, jmi, cife, base, save=True):
    """
    Plots the effectiveness difference in entropy estimators.
    For instance, to show that the complex one performs worse than the simple one.

    Args:
        dataset_name: name of dataset
        title: title of plot
        n_features: number of features to plot
        mrmr: the results for MRMR
        mifs: the results for MIFS
        jmi: the results for JMI
        cife: the results for CIFE
        base: the results for IG
        save: should you save to disk or not

    Returns:

    """
    features = list(range(1, n_features+1))

    sns.set(font_scale=1.9, palette='bright')
    sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":"})
    font_color = '#017188'
    # figure(figsize=(14, 6), dpi=100)

    l = 2.2
    k = 10

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(14, 5.5)
    ax1.plot(features, np.array([i[0] for i in mifs[0]]) * 100, marker='s', markevery=k, linewidth=l)
    ax1.plot(features, np.array([i[0] for i in mrmr[0]]) * 100, marker='o', markevery=k, linewidth=l)
    ax1.plot(features, np.array([i[0] for i in cife[0]]) * 100, marker='D', markevery=k, linewidth=l)
    ax1.plot(features, np.array([i[0] for i in jmi[0]]) * 100, marker='X', markevery=k, linewidth=l)
    ax1.plot(features, np.array([i[0] for i in base[0]]) * 100, marker='^', markevery=k, linewidth=l)
    ax1.set_title('Simple entropy estimator')

    ax2.plot(features, np.array([i[0] for i in mifs[1]]) * 100, marker='s', markevery=k, linewidth=l)
    ax2.plot(features, np.array([i[0] for i in mrmr[1]]) * 100, marker='o', markevery=k, linewidth=l)
    ax2.plot(features, np.array([i[0] for i in cife[1]]) * 100, marker='D', markevery=k, linewidth=l)
    ax2.plot(features, np.array([i[0] for i in jmi[1]]) * 100, marker='X', markevery=k, linewidth=l)
    ax2.plot(features, np.array([i[0] for i in base[1]]) * 100, marker='^', markevery=k, linewidth=l)
    ax2.set_title('Complex entropy estimator')

    ax1.set_xlabel('Number of Features')
    ax2.set_xlabel('Number of Features')
    ax1.set_ylabel('Classification Accuracy (%)')
    ax2.set_ylabel('Classification Accuracy (%)')
    ax1.grid(True, linewidth=2, linestyle=':')
    ax2.grid(True, linewidth=2, linestyle=':')
    fig.legend(['MIFS', 'MRMR', 'CIFE', 'JMI', 'IG'],
               bbox_to_anchor=(0.86, 0), loc=4, ncol=5)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    # fig.suptitle(title)

    if save:
        fig.savefig(f'./results/result_entropy_two_{dataset_name}.pdf', dpi=400)
    fig.show()

def plot_mifs_3(dataset_name, title, n_features, mifs_000, mifs_025, mifs_050, mifs_075, mifs_100, is_classification, save=True):
    """
    Plots the effectiveness difference in MIFS hyperparameter tuning.

    Args:
        dataset_name: name of dataset
        title: title of plot
        n_features: the number of features to plot
        mifs_000: the results for MIFS with beta=0
        mifs_025: the results for MIFS with beta=0.25
        mifs_050: the results for MIFS with beta=0.50
        mifs_075: the results for MIFS with beta=0.75
        mifs_100: the results for MIFS with beta=1
        is_classification: is the dataset classification or regression task
        save: should you save to disk or not
    """

    features = list(range(1, n_features + 1))

    sns.set(font_scale=1.9, palette='bright')
    sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":"})
    k = int(n_features / 10)
    l = 2.2
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(k))
    x_label = 'Number of Features'
    # figure(figsize=(8, 6), dpi=100)

    if is_classification:
        y_label = 'Classification Accuracy (%)'
        mifs_000 = [[100 * el for el in model] for model in mifs_000]
        mifs_025 = [[100 * el for el in model] for model in mifs_025]
        mifs_050 = [[100 * el for el in model] for model in mifs_050]
        mifs_075 = [[100 * el for el in model] for model in mifs_075]
        mifs_100 = [[100 * el for el in model] for model in mifs_100]
    else:
        y_label = 'RMSE'

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(14, 5)
    palette_tab10 = sns.color_palette("bright", 10)

    ax1.plot(features, mifs_000[2], marker='^', markevery=k, linewidth=l, color=palette_tab10[4])
    ax1.plot(features, mifs_025[2], marker='<', markevery=k, linewidth=l, color=palette_tab10[5])
    ax1.plot(features, mifs_050[2], marker='s', markevery=k, linewidth=l, color=palette_tab10[0])
    ax1.plot(features, mifs_075[2], marker='>', markevery=k, linewidth=l, color=palette_tab10[6])
    ax1.plot(features, mifs_100[2], marker='v', markevery=k, linewidth=l, color=palette_tab10[8])
    ax1.set_title('Logistic Regression')

    ax2.plot(features, mifs_000[1], marker='^', markevery=k, linewidth=l, color=palette_tab10[4])
    ax2.plot(features, mifs_025[1], marker='<', markevery=k, linewidth=l, color=palette_tab10[5])
    ax2.plot(features, mifs_050[1], marker='s', markevery=k, linewidth=l, color=palette_tab10[0])
    ax2.plot(features, mifs_075[1], marker='>', markevery=k, linewidth=l, color=palette_tab10[6])
    ax2.plot(features, mifs_100[1], marker='v', markevery=k, linewidth=l, color=palette_tab10[8])
    ax2.set_title('XGBoost')

    ax3.plot(features, mifs_000[0], marker='^', markevery=k, linewidth=l, color=palette_tab10[4])
    ax3.plot(features, mifs_025[0], marker='<', markevery=k, linewidth=l, color=palette_tab10[5])
    ax3.plot(features, mifs_050[0], marker='s', markevery=k, linewidth=l, color=palette_tab10[0])
    ax3.plot(features, mifs_075[0], marker='>', markevery=k, linewidth=l, color=palette_tab10[6])
    ax3.plot(features, mifs_100[0], marker='v', markevery=k, linewidth=l, color=palette_tab10[8])
    ax3.set_title('Support Vector Machine')

    ax1.set_xlabel(x_label)
    ax2.set_xlabel(x_label)
    ax3.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax2.set_ylabel(y_label)
    ax3.set_ylabel(y_label)
    ax1.grid(True, linewidth=2, linestyle=':')
    ax2.grid(True, linewidth=2, linestyle=':')
    ax3.grid(True, linewidth=2, linestyle=':')
    fig.legend(['beta=0', 'beta=0.25', 'beta=0.5', 'beta=0.75', 'beta=1.0'],
               bbox_to_anchor=(0.95, 0), loc=4, ncol=5)

    # fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.1, 1, 1])

    if save:
        fig.savefig(f'./results/result_mifs_{dataset_name}.pdf', dpi=400)
    fig.show()


def plot_over_features_3(dataset_name, title, n_features, mrmr, mifs, jmi, cife, base, is_classification, save=True):
    """
    Plots the effectiveness difference in the five feature selection methods.

    Args:
        dataset_name: name of dataset
        title: title of plot
        n_features: number of features to plot
        mrmr: the results for MRMR
        mifs: the results for MIFS
        jmi: the results for JMI
        cife: the results for CIFE
        base: the results for IG
        is_classification: classification or regression dataset
        save: should you save to disk or not
    """
    features = list(range(1, n_features + 1))

    sns.set(font_scale=1.9, palette='bright')
    sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":", 'lines.markersize': 10000})
    k = int(n_features / 10)
    l = 2.2
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(k))
    x_label = 'Number of Features'
    # figure(figsize=(8, 6), dpi=100)

    if is_classification:
        y_label = 'Classification Accuracy (%)'
        mrmr = [[100 * el for el in model] for model in mrmr]
        mifs = [[100 * el for el in model] for model in mifs]
        cife = [[100 * el for el in model] for model in cife]
        jmi = [[100 * el for el in model] for model in jmi]
        base = [[100 * el for el in model] for model in base]
    else:
        y_label = 'RMSE'

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(14, 5)
    ax1.plot(features, mifs[2], marker='s', markevery=k, linewidth=l)
    ax1.plot(features, mrmr[2], marker='o', markevery=k, linewidth=l)
    ax1.plot(features, cife[2], marker='D', markevery=k, linewidth=l)
    ax1.plot(features, jmi[2], marker='X', markevery=k, linewidth=l)
    ax1.plot(features, base[2], marker='^', markevery=k, linewidth=l)
    ax1.set_title('Logistic Regression')

    ax2.plot(features, mifs[1], marker='s', markevery=k, linewidth=l)
    ax2.plot(features, mrmr[1], marker='o', markevery=k, linewidth=l)
    ax2.plot(features, cife[1], marker='D', markevery=k, linewidth=l)
    ax2.plot(features, jmi[1], marker='X', markevery=k, linewidth=l)
    ax2.plot(features, base[1], marker='^', markevery=k, linewidth=l)
    ax2.set_title('XGBoost')

    ax3.plot(features, mifs[0], marker='s', markevery=k, linewidth=l)
    ax3.plot(features, mrmr[0], marker='o', markevery=k, linewidth=l)
    ax3.plot(features, cife[0], marker='D', markevery=k, linewidth=l)
    ax3.plot(features, jmi[0], marker='X', markevery=k, linewidth=l)
    ax3.plot(features, base[0], marker='^', markevery=k, linewidth=l)
    ax3.set_title('Support Vector Machine')

    ax1.set_xlabel(x_label)
    ax2.set_xlabel(x_label)
    ax3.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax2.set_ylabel(y_label)
    ax3.set_ylabel(y_label)
    ax1.grid(True, linewidth=2, linestyle=':')
    ax2.grid(True, linewidth=2, linestyle=':')
    ax3.grid(True, linewidth=2, linestyle=':')
    fig.legend(['MIFS', 'MRMR', 'CIFE', 'JMI', 'IG'],
               bbox_to_anchor=(0.86, 0), loc=4, ncol=5)

    # fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.1, 1, 1])

    if save:
        fig.savefig(f'./results/result_three_{dataset_name}.pdf', dpi=400)
    fig.show()


def plot_performance(dataset_name, n_features, mrmr, mifs, jmi, cife, save=True):
    """
    Plots the runtime difference in the four feature selection methods.

    Args:
        dataset_name: name of dataset
        n_features: number of features to plot
        mrmr: the results for MRMR
        mifs: the results for MIFS
        jmi: the results for JMI
        cife: the results for CIFE
        save: should you save to disk or not
    """
    features = list(range(1, n_features+1))

    font_color = '#017188'
    figure(figsize=(10, 8), dpi=100)
    sns.set(font_scale=2, palette='bright')
    sns.set_style('whitegrid',  {"grid.color": ".6", "grid.linestyle": ":"})

    k = int(np.log(n_features) / np.log(2)) - 1
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(k))

    plt.plot(features, np.array(mifs), marker='s', markevery=k, linestyle='-')
    plt.plot(features, np.array(mrmr), marker='o', markevery=k, linestyle='--')
    plt.plot(features, np.array(cife), marker='D', markevery=k, linestyle='-.')
    plt.plot(features, np.array(jmi), marker='X', markevery=k, linestyle=':')

    plt.xlabel('Number of Features', color=font_color)
    plt.xticks(color=font_color)
    plt.ylabel('Seconds',  color=font_color)
    plt.yticks(color=font_color)
    plt.legend(['MIFS', 'MRMR', 'CIFE', 'JMI'], labelcolor=font_color)
    plt.title(f'{dataset_name} dataset runtime performance', color=font_color)

    # sns.despine()

    if save:
        plt.savefig(f'./results/result_{dataset_name}_runtime.png', dpi=400)
    plt.show()
    plt.clf()


def plot_performance_two(title, n_features, mrmr, mifs, jmi, cife, base, mrmr_complex, mifs_complex, jmi_complex, cife_complex, base_complex, save=True):
    """
    Plots the runtime difference in entropy estimators.

    Args:
        dataset_name: name of dataset
        n_features: number of features to plot
        mrmr: the results for MRMR on simple entropy estimator
        mifs: the results for MIFS on simple entropy estimator
        jmi: the results for JMI on simple entropy estimator
        cife: the results for CIFE on simple entropy estimator
        mrmr_complex: the results for MRMR on complex entropy estimator
        mifs_complex: the results for MIFS on complex entropy estimator
        jmi_complex: the results for JMI on complex entropy estimator
        cife_complex: the results for CIFE on complex entropy estimator
        save: should you save to disk or not
    """

    features = list(range(1, n_features + 1))

    sns.set(font_scale=1.9, palette='bright')
    sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":"})
    k = int(n_features / 10)
    l = 2.2
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(k))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(14, 6)
    ax1.plot(features, np.array(mifs), linestyle='-', linewidth=l)
    ax1.plot(features, np.array(mrmr), linestyle='--', linewidth=l)
    ax1.plot(features, np.array(cife), linestyle='-.', linewidth=l)
    ax1.plot(features, np.array(jmi), linestyle=':', linewidth=l)
    ax1.plot(features, np.array(base), linestyle='dashdot', linewidth=l)
    ax1.plot(features, np.array(mifs_complex), marker='o', linestyle='-', markevery=k, linewidth=l)
    ax1.plot(features, np.array(mrmr_complex), marker='s', linestyle='--', markevery=k, linewidth=l)
    ax1.plot(features, np.array(cife_complex), marker='^', linestyle='-.', markevery=k, linewidth=l)
    ax1.plot(features, np.array(jmi_complex), marker='*', linestyle=':', markevery=k, linewidth=l)
    ax1.plot(features, np.array(base_complex), marker='v', linestyle='dashdot', markevery=k, linewidth=l)
    ax1.set_title('Simple and complex entropy estimator')

    ax2.plot(features, np.array(mifs), linestyle='-', linewidth=l)
    ax2.plot(features, np.array(mrmr), linestyle='--', linewidth=l)
    ax2.plot(features, np.array(cife), linestyle='-.', linewidth=l)
    ax2.plot(features, np.array(jmi), linestyle=':', linewidth=l)
    ax2.plot(features, np.array(base), linestyle='dashdot', linewidth=l)
    ax2.set_title('Enhanced view of simple entropy estimator')

    ax1.set_xlabel('Number of Features')
    ax2.set_xlabel('Number of Features')
    ax1.set_ylabel('Seconds')
    ax2.set_ylabel('Seconds')
    ax1.grid(True, linewidth=2, linestyle=':')
    ax2.grid(True, linewidth=2, linestyle=':')
    fig.legend(['MIFS', 'MRMR', 'CIFE', 'JMI', 'IG', 'MIFS complex', 'MRMR complex', 'CIFE complex', 'JMI complex', 'IG complex'],
               bbox_to_anchor=(1, 0), loc=4, ncol=5)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    # fig.suptitle(title)
    fig.show()

    if save:
        fig.savefig(f'./results/result_side_runtime.pdf', dpi=400)
