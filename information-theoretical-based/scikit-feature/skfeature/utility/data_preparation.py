import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from autogluon.tabular import TabularDataset, TabularPredictor


def prepare_data_for_ml(dataframe):
    df = dataframe.fillna(-1)
    df = df.apply(lambda x: pd.factorize(x)[0] if x.dtype == object else x)

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(df)
    normalized_X = pd.DataFrame(scaled_X, columns=df.columns)

    return normalized_X


def get_hyperparameters(train_data, y_label, algorithms, model_names):
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
