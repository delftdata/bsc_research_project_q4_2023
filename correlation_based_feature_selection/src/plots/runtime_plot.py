import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import figure


plt.style.use('seaborn-darkgrid')
plt.rc_context({"context": "paper"})


def plot_over_runtime(dataset_name, algorithm, number_of_features, dataset_type,
                      pearson_duration, spearman_duration, cramersv_duration,
                      su_duration, baseline_duration):
    number_of_features_iteration = list(range(1, number_of_features))

    figure(figsize=(8, 6), dpi=100)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))

    plt.plot(number_of_features_iteration, np.array(pearson_duration),
             marker='D', color='#10A5D6')
    plt.plot(number_of_features_iteration, np.array(spearman_duration),
             marker='*', color='#C6209B')
    plt.plot(number_of_features_iteration, np.array(cramersv_duration),
             marker='>', color='#BF9000')
    plt.plot(number_of_features_iteration, np.array(su_duration),
             marker='s', color='#2EB835')
    plt.plot(number_of_features_iteration[-1], baseline_duration,
             marker='o', color='#000000')

    plt.xlabel('Number of features')
    plt.ylabel('Milliseconds')
    plt.legend(['Pearson', 'Spearman', 'Cramér\'s V', 'Symmetric Uncertainty', 'Baseline'])
    # plt.title('Change of ' + str(evaluation_metric_name)
    #           + f' for {algorithm} on {dataset_name} dataset ({dataset_type}) '
    #           + 'with the increase of selected features')

    # Set background
    ax = plt.gca()
    ax.set_facecolor('#F0F0F0')
    ax.grid(color='white')

    # Create the directory if it doesn't exist
    directory = "./results_runtime"
    os.makedirs(directory, exist_ok=True)
    # Save the figure to folder
    plt.savefig(f'./results_runtime/result_{dataset_name}_{dataset_type}_{algorithm}.png')
    # plt.show()
    plt.clf()
