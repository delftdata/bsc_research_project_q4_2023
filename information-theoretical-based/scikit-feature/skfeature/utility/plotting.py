import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import figure
import seaborn as sns


def plot_over_features(dataset_name, model_name, n_features, mrmr, mifs, jmi, cife, save=True):
    features = list(range(1, n_features+1))

    sns.set(font_scale=1.4, palette='bright')
    sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":"})
    font_color = '#017188'
    figure(figsize=(8, 6), dpi=100)

    k = int(np.log(n_features) / np.log(2))
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(k))


    plt.plot(features, np.array(mifs) * 100, marker='s', markevery=k)
    plt.plot(features, np.array(mrmr) * 100, marker='o', markevery=k)
    plt.plot(features, np.array(cife) * 100, marker='D', markevery=k)
    plt.plot(features, np.array(jmi) * 100, marker='X', markevery=k)

    plt.xlabel('Number of Features', fontsize=14, color=font_color)
    plt.xticks(fontsize=14, color=font_color)
    plt.ylabel('Classification Accuracy (%)', fontsize=14, color=font_color)
    plt.yticks(fontsize=14, color=font_color)
    plt.legend(['MIFS', 'MRMR', 'CIFE', 'JMI'], fontsize=14, labelcolor=font_color)
    plt.title(f'{dataset_name} dataset performance with {model_name} algorithm', fontsize=16, color=font_color)

    if save:
        plt.savefig(f'./results/result_{dataset_name}_{model_name}.png', dpi=400)
    plt.show()
    plt.clf()


def plot_over_features_2(dataset_name, title, n_features, mrmr, mifs, jmi, cife, save=True):
    features = list(range(1, n_features+1))

    sns.set(font_scale=1.4, palette='bright')
    sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":"})
    k = int(np.log(n_features) / np.log(2))
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(k))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(14, 6)
    ax1.plot(features, np.array([i[0] for i in mifs[0]]) * 100, marker='s', markevery=k)
    ax1.plot(features, np.array([i[0] for i in mrmr[0]]) * 100, marker='o', markevery=k)
    ax1.plot(features, np.array([i[0] for i in cife[0]]) * 100, marker='D', markevery=k)
    ax1.plot(features, np.array([i[0] for i in jmi[0]]) * 100, marker='X', markevery=k)
    ax1.set_title('Simple entropy estimator')

    ax2.plot(features, np.array([i[0] for i in mifs[1]]) * 100, marker='s', markevery=k)
    ax2.plot(features, np.array([i[0] for i in mrmr[1]]) * 100, marker='o', markevery=k)
    ax2.plot(features, np.array([i[0] for i in cife[1]]) * 100, marker='D', markevery=k)
    ax2.plot(features, np.array([i[0] for i in jmi[1]]) * 100, marker='X', markevery=k)
    ax2.set_title('Complex entropy estimator')

    ax1.set_xlabel('Number of Features')
    ax2.set_xlabel('Number of Features')
    ax1.set_ylabel('Classification Accuracy (%)')
    ax2.set_ylabel('Classification Accuracy (%)')
    fig.legend(['MIFS', 'MRMR', 'CIFE', 'JMI'])
    # fig.suptitle(title)

    if save:
        fig.savefig(f'./results/result_entropy_two_{dataset_name}.png', dpi=400)
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

def plot_performance_8(dataset_name, n_features, mrmr, mifs, jmi, cife, mrmr_complex, mifs_complex, jmi_complex, cife_complex, save=True):
    features = list(range(1, n_features+1))

    font_color = '#000000'
    figure(figsize=(8, 6), dpi=100)
    sns.set(font_scale=2, palette='bright')
    sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":"})

    k = int(np.log(n_features) / np.log(2))
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(k))

    plt.plot(features, np.array(mifs_complex), marker='o', linestyle='-', markevery=k)
    plt.plot(features, np.array(mrmr_complex), marker='s', linestyle='--', markevery=k)
    plt.plot(features, np.array(cife_complex), marker='^', linestyle='-.', markevery=k)
    plt.plot(features, np.array(jmi_complex), marker='*', linestyle=':', markevery=k)
    plt.plot(features, np.array(mifs), linestyle='-')
    plt.plot(features, np.array(mrmr), linestyle='--')
    plt.plot(features,np.array(cife), linestyle='-.')
    plt.plot(features, np.array(jmi), linestyle=':')

    plt.xlabel('Number of Features', fontsize=14, color=font_color)
    plt.xticks(fontsize=14, color=font_color)
    plt.ylabel('Seconds', fontsize=14, color=font_color)
    plt.yticks(fontsize=14, color=font_color)
    plt.legend([ 'MIFS complex', 'MRMR complex', 'CIFE complex', 'JMI complex', 'MIFS', 'MRMR', 'CIFE', 'JMI'],
               fontsize=14, labelcolor=font_color)
    plt.title(f'{dataset_name} dataset runtime performance', fontsize=16, color=font_color)

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

def plot_performance_two(title, n_features, mrmr, mifs, jmi, cife, mrmr_complex, mifs_complex, jmi_complex, cife_complex, save=True):
    features = list(range(1, n_features + 1))

    sns.set(font_scale=1.4, palette='bright')
    sns.set_style('whitegrid', {"grid.color": ".6", "grid.linestyle": ":"})
    k = int(np.log(n_features) / np.log(2))
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(k))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(14, 8)
    ax1.plot(features, np.array(mifs), linestyle='-')
    ax1.plot(features, np.array(mrmr), linestyle='--')
    ax1.plot(features, np.array(cife), linestyle='-.')
    ax1.plot(features, np.array(jmi), linestyle=':')
    ax1.plot(features, np.array(mifs_complex), marker='o', linestyle='-', markevery=k)
    ax1.plot(features, np.array(mrmr_complex), marker='s', linestyle='--', markevery=k)
    ax1.plot(features, np.array(cife_complex), marker='^', linestyle='-.', markevery=k)
    ax1.plot(features, np.array(jmi_complex), marker='*', linestyle=':', markevery=k)


    ax2.plot(features, np.array(mifs), linestyle='-')
    ax2.plot(features, np.array(mrmr), linestyle='--')
    ax2.plot(features, np.array(cife), linestyle='-.')
    ax2.plot(features, np.array(jmi), linestyle=':')

    ax1.set_xlabel('Number of Features')
    ax2.set_xlabel('Number of Features')
    ax1.set_ylabel('Seconds')
    ax2.set_ylabel('Seconds')
    fig.legend(['MIFS', 'MRMR', 'CIFE', 'JMI', 'MIFS complex', 'MRMR complex', 'CIFE complex', 'JMI complex'],
               bbox_to_anchor=(0.8, 0), loc=4, ncol=4)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    # fig.suptitle(title)
    fig.show()

    if save:
        fig.savefig(f'./results/result_side_runtime.png', dpi=400)
