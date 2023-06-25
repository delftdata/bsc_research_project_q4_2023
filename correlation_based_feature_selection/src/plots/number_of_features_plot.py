import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import figure

evaluation_metrics_options = {
    'accuracy': 'Accuracy (%)',
    'rmse': 'Root mean square error',
}

plt.style.use('seaborn-darkgrid')
plt.rc_context({"context": "paper"})

def plot_over_number_of_features(dataset_name, algorithm,
                                 number_of_features, dataset_type,
                                 evaluation_metric, pearson_performance,
                                 spearman_performance, cramersv_performance,
                                 su_performance, baseline_performance):
    evaluation_metric_name = evaluation_metrics_options.get(evaluation_metric)
    number_of_features_iteration = list(range(1, number_of_features))

    figure(figsize=(8, 6), dpi=100)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))

    plt.plot(number_of_features_iteration, np.array(pearson_performance) * 100,
             marker='D', color='#10A5D6')
    plt.plot(number_of_features_iteration, np.array(spearman_performance) * 100,
             marker='*', color='#045A8D')
    plt.plot(number_of_features_iteration, np.array(cramersv_performance) * 100,
             marker='>', color='#BF9000')
    plt.plot(number_of_features_iteration, np.array(su_performance) * 100,
             marker='s', color='#CA0020')
    plt.plot(number_of_features_iteration[-1], baseline_performance * 100,
             marker='o', color='#000000')

    plt.xlabel('Number of features')
    plt.ylabel(str(evaluation_metric_name))
    plt.legend(['Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty', 'Baseline'])
    # plt.title('Change of ' + str(evaluation_metric_name)
    #           + f' for {algorithm} on {dataset_name} dataset ({dataset_type}) '
    #           + 'with the increase of selected features')

    # Set background
    ax = plt.gca()
    ax.set_facecolor('#F0F0F0')
    ax.grid(color='white')

    # Create the directory if it doesn't exist
    directory = "./results_performance"
    os.makedirs(directory, exist_ok=True)
    # Save the figure to folder
    plt.savefig(f'./results_performance/result_{dataset_name}_{dataset_type}_{algorithm}.png')
    # plt.show()
    plt.clf()
