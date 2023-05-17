import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import figure

# TODO: Add all metrics
evaluation_metrics_options = {
    'accuracy': 'Accuracy',
    'rmse': 'Root Mean Square Error',
}


def plot_over_number_of_features(algorithm, number_of_features, evaluation_metric,
                                 pearson_performance, spearman_performance,
                                 cramersv_performance, su_performance):
    evaluation_metric_name = evaluation_metrics_options.get(evaluation_metric)
    number_of_features_iteration = list(range(1, number_of_features))

    figure(figsize=(8, 6), dpi=100)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))

    # TODO: Consider using the percentage (e.g. for accuracy)
    plt.plot(number_of_features_iteration, np.array(pearson_performance),
             marker='o', color='#10A5D6')
    plt.plot(number_of_features_iteration, np.array(spearman_performance),
             marker='o', color='#C6209B')
    plt.plot(number_of_features_iteration, np.array(cramersv_performance),
             marker='o', color='#4361A3')
    plt.plot(number_of_features_iteration, np.array(su_performance),
             marker='o', color='#8A5C7E')

    plt.xlabel('Number of Features')
    plt.ylabel(str(evaluation_metric_name))
    plt.legend(['Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty'])
    plt.title('The change of the ' + str(evaluation_metric_name)
              + f' metric for {algorithm} with the increase of the selected features')

    # plt.savefig(f'./results/result_{algorithm} (1).png')
    plt.show()
    plt.clf()
