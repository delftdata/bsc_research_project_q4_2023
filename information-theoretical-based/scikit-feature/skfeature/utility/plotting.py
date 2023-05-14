import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import figure


def plot_over_features(model_name, n_features, mrmr, mifs, jmi, cife):
    features = list(range(1, n_features+1))

    figure(figsize=(8, 6), dpi=100)

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
    plt.plot(features, np.array(mrmr) * 100, marker='o')
    plt.plot(features, np.array(mifs) * 100, marker='s')
    plt.plot(features, np.array(jmi) * 100, marker='X')
    plt.plot(features, np.array(cife) * 100, marker='D')
    plt.xlabel('Number of Features', fontsize=14, color='#017188')
    plt.xticks(fontsize=14, color='#017188')
    plt.ylabel('Classification Accuracy (%)', fontsize=14, color='#017188')
    plt.yticks(fontsize=14, color='#017188')
    plt.legend(['MRMR', 'MIFS', 'JMI', 'CIFE'], fontsize=14, labelcolor='#017188')
    plt.title('Steel plates\' faults dataset accuracy with XGBoost', fontsize=16, color='#017188')

    plt.savefig(f'result_{model_name}.png')
    plt.show()
    plt.clf()