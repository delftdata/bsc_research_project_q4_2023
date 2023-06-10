import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import figure

# TODO: Add all metrics
evaluation_metrics_options = {
    'accuracy': 'accuracy',
    'rmse': 'root mean square error',
}


def plot_over_number_of_features(dataset_name, algorithm,
                                 number_of_features, dataset_type,
                                 evaluation_metric, pearson_performance,
                                 spearman_performance, cramersv_performance,
                                 su_performance, baseline_performance):
    evaluation_metric_name = evaluation_metrics_options.get(evaluation_metric)
    number_of_features_iteration = list(range(1, number_of_features))

    figure(figsize=(8, 6), dpi=100)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))

    # TODO: Consider using the percentage (e.g. for accuracy)
    plt.plot(number_of_features_iteration, np.array(pearson_performance),
             marker='D', color='#10A5D6')
    plt.plot(number_of_features_iteration, np.array(spearman_performance),
             marker='*', color='#C6209B')
    plt.plot(number_of_features_iteration, np.array(cramersv_performance),
             marker='>', color='#D3B813')
    plt.plot(number_of_features_iteration, np.array(su_performance),
             marker='s', color='#2EB835')
    plt.plot(number_of_features_iteration[-1], baseline_performance,
             marker='o', color='#D9D9D9')

    plt.xlabel('Number of Features')
    plt.ylabel(str(evaluation_metric_name))
    plt.legend(['Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty', 'Baseline'])
    plt.title('Change of ' + str(evaluation_metric_name)
              + f' for {algorithm} on {dataset_name} dataset ({dataset_type}) '
              + 'with the increase of selected features')

    # Create the directory if it doesn't exist
    directory = "./results"
    os.makedirs(directory, exist_ok=True)
    # Save the figure to folder
    plt.savefig(f'./results/result_{dataset_name}_{dataset_type}_{algorithm}.png')
    # plt.show()
    plt.clf()
