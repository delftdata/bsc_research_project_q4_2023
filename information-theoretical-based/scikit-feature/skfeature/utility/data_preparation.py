import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset, TabularPredictor


def get_hyperparameters(train_data, y_label, algorithms, model_names):
    """
    Helper function to get hyperparameters of a model
    Args:
        train_data: training data
        y_label: name of y label
        algorithms: name of algorithms
        model_names: name of models

    Returns:
        Array of hyperparameters for each model.
    """
    hyperparameters = []
    for algorithm, model_name in zip(algorithms, model_names):
        # Train model on entire dataset
        initial_linear_predictor = TabularPredictor(label=y_label,
                                                    verbosity=0) \
            .fit(train_data=train_data, hyperparameters={algorithm: {}})
        initial_training_results = initial_linear_predictor.info()
        # Get tuned hyper-parameters
        hyperparameters.append(initial_training_results['model_info'][model_name]['hyperparameters'])

    return hyperparameters


def analysis(datasets):
    """
    Helper to plot and print the distribution of values for the tables.
    """
    for dataset in datasets:
        print(dataset['name'])
        train_data = pd.read_csv(dataset['path'])

        train_data = TabularDataset(train_data)
        train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                    enable_text_ngram_features=False).fit_transform(X=train_data)
        train_data.fillna(0, inplace=True)

        arr = train_data.describe().T

        print(arr[['count', 'mean', 'std', 'min', 'max']].to_string())

        ncols = int(np.sqrt(len(train_data.columns)))
        nrows = int(np.ceil(len(train_data.columns) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows * 4, ncols * 4))
        axes = axes.ravel()

        for index, col in enumerate(train_data.columns):
            if train_data[col].dtype == 'category':
                sns.histplot(train_data[col].cat.as_ordered(), ax=axes[index])
            else:
                ax = sns.histplot(train_data[col], ax=axes[index], kde=True)
                ax.lines[0].set_color('orange')
            axes[index].tick_params(axis="x", rotation=90)
            axes[index].set_title("Distribution of %s" % col)

        fig.tight_layout()
        fig.show()
        fig.savefig(f'results/kde_{dataset["name"]}.pdf', dpi=400)
