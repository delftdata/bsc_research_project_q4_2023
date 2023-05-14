import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import figure


def plot_over_features(model_name, n_features, mrmr, mifs, jmi, cife, save=True):
    features = list(range(1, n_features+1))

    font_color = '#017188'
    figure(figsize=(8, 6), dpi=100)

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
    plt.plot(features, np.array(mrmr) * 100, marker='o')
    plt.plot(features, np.array(mifs) * 100, marker='s')
    plt.plot(features, np.array(jmi) * 100, marker='X')
    plt.plot(features, np.array(cife) * 100, marker='D')

    plt.xlabel('Number of Features', fontsize=14, color=font_color)
    plt.xticks(fontsize=14, color=font_color)
    plt.ylabel('Classification Accuracy (%)', fontsize=14, color=font_color)
    plt.yticks(fontsize=14, color=font_color)
    plt.legend(['MRMR', 'MIFS', 'JMI', 'CIFE'], fontsize=14, labelcolor=font_color)
    plt.title(f'Steel plates\' faults dataset accuracy with {model_name}', fontsize=16, color=font_color)

    if save:
        plt.savefig(f'./results/result_{model_name}.png')
    plt.show()
    plt.clf()
