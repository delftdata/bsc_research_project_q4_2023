import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import figure
import seaborn as sns

def plot_over_features(dataset_name, model_name, n_features, mrmr, mifs, jmi, cife, save=True):
    features = list(range(1, n_features+1))

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
        plt.savefig(f'./results/result_{dataset_name}_{model_name}.png')
    plt.show()
    plt.clf()


def plot_performance(dataset_name, n_features, mrmr, mifs, jmi, cife, save=True):
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
        plt.savefig(f'./results/result_{dataset_name}_runtime.png')
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
        plt.savefig(f'./results/result_{dataset_name}_runtime.png')
    plt.show()
    plt.clf()