import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import figure


def plot_over_features(model_name, n_features, mrmr, mifs, jmi, cife, save=True):
    features = list(range(1, n_features+1))

    font_color = '#017188'
    figure(figsize=(8, 6), dpi=100)

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(4))


    plt.plot(features, np.array(mifs) * (-1), marker='s', markevery=4)
    plt.plot(features, np.array(mrmr) * (-1), marker='o', markevery=4)
    plt.plot(features, np.array(cife) * (-1), marker='D', markevery=4)
    plt.plot(features, np.array(jmi) * (-1), marker='X', markevery=4)

    plt.xlabel('Number of Features', fontsize=14, color=font_color)
    plt.xticks(fontsize=14, color=font_color)
    plt.ylabel('Root Mean Squared Error (RMSE) in 10^10', fontsize=14, color=font_color)
    plt.yticks(fontsize=14, color=font_color)
    plt.legend(['MIFS', 'MRMR', 'CIFE', 'JMI'], fontsize=14, labelcolor=font_color)
    plt.title(f'Housing prices dataset performance with {model_name} algorithm', fontsize=16, color=font_color)

    if save:
        plt.savefig(f'./results/result_{model_name}.png')
    plt.show()
    plt.clf()

def plot_performance(model_name, n_features, mrmr, mifs, jmi, cife, save=True):
    features = list(range(1, n_features+1))

    font_color = '#017188'
    figure(figsize=(8, 6), dpi=100)

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))


    plt.plot(features, np.array(mifs), marker='s', markevery=2)
    plt.plot(features, np.array(mrmr), marker='o', markevery=2)
    plt.plot(features, np.array(cife), marker='D', markevery=2)
    plt.plot(features, np.array(jmi), marker='X', markevery=2)

    plt.xlabel('Number of Features', fontsize=14, color=font_color)
    plt.xticks(fontsize=14, color=font_color)
    plt.ylabel('Seconds', fontsize=14, color=font_color)
    plt.yticks(fontsize=14, color=font_color)
    plt.legend(['MIFS', 'MRMR', 'CIFE', 'JMI'], fontsize=14, labelcolor=font_color)
    plt.title(f'Housing prices dataset run-time performance', fontsize=16, color=font_color)

    if save:
        plt.savefig(f'./results/result_{model_name}.png')
    plt.show()
    plt.clf()
