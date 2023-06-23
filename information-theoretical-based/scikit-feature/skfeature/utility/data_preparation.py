import pandas as pd

from sklearn.preprocessing import MinMaxScaler

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
