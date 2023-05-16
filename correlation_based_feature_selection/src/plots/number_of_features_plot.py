import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import figure


evaluation_metrics_options = {
    'accuracy': 'Accuracy',
}

def plot_over_number_of_features(algorithm, number_of_features, evaluation_metric,
                                 pearson_performance, spearman_performance,
                                 cramersv_performance, su_performance):
    evaluation_metric_name = evaluation_metrics_options.get(evaluation_metric)
    number_of_features_iteration = list(range(1, number_of_features))

    figure(figsize=(8, 6), dpi=100)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))

    plt.plot(number_of_features_iteration, np.array(pearson_performance) * 100, marker='o', color='red')
    plt.plot(number_of_features_iteration, np.array(spearman_performance) * 100, marker='o', color='green')
    plt.plot(number_of_features_iteration, np.array(cramersv_performance) * 100, marker='o', color='blue')
    plt.plot(number_of_features_iteration, np.array(su_performance) * 100, marker='o', color='orange')

    plt.xlabel('Number of Features')
    plt.ylabel(str(evaluation_metric_name) + ' (%)')
    plt.legend(['Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty'])
    plt.title(f'The change of the ' + str(evaluation_metric_name) + f' metric for {algorithm} '
              f'with the increase of the selected features')

    plt.savefig(f'./results/result_{algorithm} (1).png')
    plt.show()
    plt.clf()