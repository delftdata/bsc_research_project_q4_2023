import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC

from warnings import filterwarnings
import logging
from ast import literal_eval

from skfeature.utility.data_preparation import get_hyperparameters
from skfeature.utility.experiments import select_jmi, select_cife, select_mrmr, select_mifs, select_mifs_beta
from plotting import *

filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)
filterwarnings("ignore", category=FutureWarning)


def perform_feature_selection_single(fs_algorithm, n_features, train, y_label):
    """
    Helper function to perform the feature selection for a single algorithm.

    Args:
        fs_algorithm: the algorithm to use to select featurs
        n_features: the number of features to select
        train: training set
        y_label: name of y label

    Returns:
        An array of tuples, where the selected features are first element
        and time it took to select is second tuple element.
    """
    # Perform feature selection
    train_data = TabularDataset(train)
    np.random.seed(42)
    idx, _, _, times = fs_algorithm(train_data.drop(y_label, axis=1).to_numpy(), train_data[y_label].to_numpy(), n_selected_features=n_features)
    result = [idx[0:i] for i in range(1, len(idx)+1)]
    result = list(zip(result, times))
    print(result)
    return result


def perform_feature_selection_single_mifs(fs_algorithm, n_features, train, y_label, beta):
    """
    Helper function to perform the feature selection for MIFS with hyperparameter beta.

    Args:
        fs_algorithm: the algorithm to use to select featurs
        n_features: the number of features to select
        train: training set
        y_label: name of y label
        beta: the hyperparameter for MIFS

    Returns:
        An array of tuples, where the selected features are first element
        and time it took to select is second tuple element.
    """
    # Perform feature selection
    train_data = TabularDataset(train)
    np.random.seed(42)
    idx, _, _, times = fs_algorithm(train_data.drop(y_label, axis=1).to_numpy(), train_data[y_label].to_numpy(), n_selected_features=n_features, beta=beta)
    result = [idx[0:i] for i in range(1, len(idx)+1)]
    result = list(zip(result, times))
    print(result)
    return result


def perform_feature_selection_mifs(n_features, train, y_label):
    """
    Helper function to perform the feature selection for five possible MIFS configurations.

    Args:
        n_features: the number of features
        train: the training set
        y_label: the name of the y label

    Returns:
        arrays with performed feature selection for each of the five possibilities
    """
    mifs_000 = perform_feature_selection_single_mifs(select_mifs_beta, n_features, train, y_label, 0.0)
    logging.error(mifs_000)
    mifs_025 = perform_feature_selection_single_mifs(select_mifs_beta, n_features, train, y_label, 0.25)
    logging.error(mifs_025)
    mifs_050 = perform_feature_selection_single_mifs(select_mifs_beta, n_features, train, y_label, 0.5)
    logging.error(mifs_050)
    mifs_075 = perform_feature_selection_single_mifs(select_mifs_beta, n_features, train, y_label, 0.75)
    logging.error(mifs_075)
    mifs_100 = perform_feature_selection_single_mifs(select_mifs_beta, n_features, train, y_label, 1.0)
    logging.error(mifs_100)
    return mifs_000, mifs_025, mifs_050, mifs_075, mifs_100


def perform_feature_selection_all(n_features, train, y_label):
    """
    Helper function to perform the feature selection for MIFS, MRMR, CIFE, and JMI.

    Args:
        n_features: number of features to select
        train: the training set
        y_label: the name of y label

    Returns:
        arrays with performed feature selection for each of the four possibilities
    """
    mrmr_result = perform_feature_selection_single(select_mrmr, n_features, train, y_label)
    logging.error(mrmr_result)
    mifs_result = perform_feature_selection_single(select_mifs, n_features, train, y_label)
    logging.error(mifs_result)
    jmi_result = perform_feature_selection_single(select_jmi, n_features, train, y_label)
    logging.error(jmi_result)
    cife_result = perform_feature_selection_single(select_cife, n_features, train, y_label)
    logging.error(cife_result)
    return cife_result, jmi_result, mifs_result, mrmr_result


def evaluate_SVM(train, test, fs_results, y_label):
    """
    Evaluates datasets under SVM model for a single feature selection algorithm.

    Args:
        train: the training set
        test: the test set
        fs_results: an array with the result from performing feature selection
        y_label: the name of the y label

    Returns:
        accuracy for each subset of selected features for SVM
    """
    if train[y_label].nunique() > 20:
        is_regression = True
    else:
        is_regression = False

    result = []
    for idx, duration in fs_results:
        # obtain the dataset on the selected features
        idx = list(set(idx))
        n_features = len(idx)
        picked_columns = [list(train.drop(y_label, axis=1).columns)[i] for i in idx[0:n_features]]

        # Train model on the smaller dataset with tuned hyperparameters
        if is_regression:
            linear_predictor = SVR()
        else:
            linear_predictor = SVC()

        linear_predictor.fit(train[picked_columns], train[y_label])
        # Get accuracy on test data
        test_data = TabularDataset(test[picked_columns])
        prediction = linear_predictor.predict(test_data)

        if is_regression:
            score = np.sqrt(mean_squared_error(test[y_label], prediction))
        else:
            score = accuracy_score(test[y_label], prediction)
        print(score)
        result.append((score, duration))

    return result


def evaluate_performance_SVM():
    """
    Evaluates the performance of several feature selection algorithm
    provided an input file with the results from selecting features.

    Returns:
        None, but the results are stored on disk
    """
    with open('results/logs/fs_complex.txt', "r") as file:
        data = file.readlines()

    data = [data[i:i + 5] for i in range(0, len(data), 5)]
    for i in range(len(data)):
        dataset = data[i]

        mrmr_result = literal_eval(dataset[1].replace('array', ''))
        mifs_result = literal_eval(dataset[2].replace('array', ''))
        jmi_result = literal_eval(dataset[3].replace('array', ''))
        cife_result = literal_eval(dataset[4].replace('array', ''))

        mat = pd.read_csv([x['path'] for x in datasets if x['path'] == '../.' + dataset[0].strip()][0])
        y_label = [x['y_label'] for x in datasets if x['path'] == '../.' + dataset[0].strip()][0]

        train_data = TabularDataset(mat)
        train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                    enable_text_ngram_features=False).fit_transform(X=train_data)
        train_data.fillna(0, inplace=True)
        columns = train_data.columns
        train_data = pd.DataFrame(MinMaxScaler((-1, 1)).fit_transform(train_data), columns=columns)
        logging.error(dataset[0])
        print(dataset[0])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns=[y_label]), train_data[y_label],
                                                            test_size=0.2, random_state=42)
        train = X_train.copy()
        train[y_label] = y_train
        test = X_test.copy()
        test[y_label] = y_test

        # Perform evaluation
        mrmr = evaluate_SVM(train, test, mrmr_result, y_label)
        print(mrmr)
        logging.error(mrmr)

        mifs = evaluate_SVM(train, test, mifs_result, y_label)
        print(mifs)
        logging.error(mifs)

        jmi = evaluate_SVM(train, test, jmi_result, y_label)
        print(jmi)
        logging.error(jmi)

        cife = evaluate_SVM(train, test, cife_result, y_label)
        print(cife)
        logging.error(cife)


def evaluate_model(train, test, fs_results, y_label, algorithms, hyperparameters):
    """
    Evaluates datasets under ML algorithms for a single feature selection algorithm.

    Args:
        train: the training set
        test: the test set
        fs_results: an array with the result from performing feature selection
        y_label: the name of the y label
        algorithms: ML algorithms
        hyperparameters: hyperparameters for the ML algorithms

    Returns:
        accuracy for each subset of selected features for each ML algorithm
    """
    results = []
    for algorithm, hyperparameter in zip(algorithms, hyperparameters):
        result = []
        for idx, duration in fs_results:
            # obtain the dataset on the selected features
            idx = list(set(idx))
            n_features = len(idx)
            picked_columns = [list(train.drop(y_label, axis=1).columns)[i] for i in idx[0:n_features]]
            picked_columns.append(y_label)
            features = train[picked_columns]

            # Train model on the smaller dataset with tuned hyperparameters
            linear_predictor = TabularPredictor(label=y_label,
                                                verbosity=0) \
                .fit(train_data=features, hyperparameters={algorithm: hyperparameter})

            # Get accuracy on test data
            test_data = TabularDataset(test[picked_columns])
            accuracy = linear_predictor.evaluate(test_data)
            if 'accuracy' in accuracy:
                accuracy = accuracy['accuracy']
            else:
                accuracy = -1 * accuracy['root_mean_squared_error']
            print(accuracy)
            result.append((accuracy, duration))

        results.append(result)
    return results


def run_pipeline(mat, y_label, algorithm, model_name, n_features):
    """
    Runs all steps from feature selection to model evaluation

    Args:
        mat: entire dataset
        y_label: name of y label
        algorithm: ML algorithm to train in
        model_name: name of ML algorithm in AutoGluon
        n_features: number of features to select

    Returns:
        Arrays with results from evaluation for each feature selection method.
    """
    # Encode features
    train_data = TabularDataset(mat)
    train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                enable_text_ngram_features=False).fit_transform(X=train_data)

    # Tune hyperparameters
    hyperparameters = get_hyperparameters(train_data, y_label, algorithm, model_name)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns=[y_label]), train_data[y_label],
                                                        test_size=0.2, random_state=42)
    train = X_train.copy()
    train[y_label] = y_train
    test = X_test.copy()
    test[y_label] = y_test

    # Perform feature selection
    cife_result, jmi_result, mifs_result, mrmr_result = perform_feature_selection_all(n_features, train, y_label)

    # Perform evaluation
    mrmr = evaluate_model(train, test, mrmr_result, y_label, algorithm, hyperparameters)
    mifs = evaluate_model(train, test, mifs_result, y_label, algorithm, hyperparameters)
    jmi = evaluate_model(train, test, jmi_result, y_label, algorithm, hyperparameters)
    cife = evaluate_model(train, test, cife_result, y_label, algorithm, hyperparameters)

    logging.error(mrmr)
    logging.error(mifs)
    logging.error(jmi)
    logging.error(cife)

    return mrmr, mifs, jmi, cife


def perform_feature_selection_for_multiple_datasets():
    """
    Performs the feature selection for all provided datasets for MIFS, MRMR, CIFE, and JMI.
    """
    for dataset in datasets:
        mat = pd.read_csv(dataset['path'])
        y_label = dataset['y_label']
        n_features = len(mat.columns) - 1

        # Encode features
        train_data = TabularDataset(mat)
        train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                    enable_text_ngram_features=False).fit_transform(X=train_data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns=[y_label]), train_data[y_label],
                                                            test_size=0.2, random_state=42)
        train = X_train.copy()
        train[y_label] = y_train

        # Perform feature selection
        print(dataset['path'])
        logging.error(dataset['path'])
        perform_feature_selection_all(n_features, train, y_label)


def perform_mifs_comparison():
    """
    Performs feature selection for all MIFS scenarios.
    """
    for dataset in datasets:
        mat = pd.read_csv(dataset['path'])
        y_label = dataset['y_label']
        n_features = len(mat.columns) - 1

        # Encode features
        train_data = TabularDataset(mat)
        train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                    enable_text_ngram_features=False).fit_transform(X=train_data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns=[y_label]), train_data[y_label],
                                                            test_size=0.2, random_state=42)
        train = X_train.copy()
        train[y_label] = y_train

        # Perform feature selection
        print(dataset['path'])
        logging.error(dataset['path'])
        perform_feature_selection_mifs(n_features, train, y_label)


def evaluate_performance():
    """
    Evaluates the performance of MIFS, MRMR, CIFE, and JMI.
    provided an input file with the results from selecting features.

    Returns:
        None, but the results are stored on disk
    """
    with open('results/logs/fs_complex.txt', "r") as file:
        data = file.readlines()

    data = [data[i:i + 6] for i in range(0, len(data), 6)]
    for i in range(len(data)):
        dataset = data[i]

        mrmr_result = literal_eval(dataset[1].replace('array', ''))
        mifs_result = literal_eval(dataset[2].replace('array', ''))
        jmi_result = literal_eval(dataset[3].replace('array', ''))
        cife_result = literal_eval(dataset[4].replace('array', ''))

        mat = pd.read_csv([x['path'] for x in datasets if x['path'] == '../.' + dataset[0].strip()][0])
        y_label = [x['y_label'] for x in datasets if x['path'] == '../.' + dataset[0].strip()][0]
        n_features = len(mat.columns) - 1

        algorithm = ['XGB', 'LR']
        model_name = ['XGBoost', 'LinearModel']
        train_data = TabularDataset(mat)
        train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                    enable_text_ngram_features=False).fit_transform(X=train_data)

        logging.error(f'{dataset[0]}, {algorithm}')
        hyperparameters = get_hyperparameters(train_data, y_label, algorithm, model_name)
        print(hyperparameters)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns=[y_label]), train_data[y_label],
                                                            test_size=0.2, random_state=42)
        train = X_train.copy()
        train[y_label] = y_train
        test = X_test.copy()
        test[y_label] = y_test

        # Perform evaluation
        mrmr = evaluate_model(train, test, mrmr_result, y_label, algorithm, hyperparameters)
        mifs = evaluate_model(train, test, mifs_result, y_label, algorithm, hyperparameters)
        jmi = evaluate_model(train, test, jmi_result, y_label, algorithm, hyperparameters)
        cife = evaluate_model(train, test, cife_result, y_label, algorithm, hyperparameters)

        print(mrmr)
        logging.error(mrmr)
        print(mifs)
        logging.error(mifs)
        print(jmi)
        logging.error(jmi)
        print(cife)
        logging.error(cife)


def evaluate_performance_mifs():
    """
    Evaluates the performance of MIFS under different hyperparameters.
    provided an input file with the results from selecting features.

    Returns:
        None, but the results are stored on disk
    """
    with open('results/logs/mifs.txt', "r") as file:
        data = file.readlines()

    data = [data[i:i + 6] for i in range(0, len(data), 6)]
    for i in range(len(data)):
        dataset = data[i]

        mifs_000 = literal_eval(dataset[1].replace('array', ''))
        mifs_025 = literal_eval(dataset[2].replace('array', ''))
        mifs_050 = literal_eval(dataset[3].replace('array', ''))
        mifs_075 = literal_eval(dataset[4].replace('array', ''))
        mifs_100 = literal_eval(dataset[5].replace('array', ''))

        mat = pd.read_csv([x['path'] for x in datasets if x['path'] == '../.' + dataset[0].strip()][0])
        y_label = [x['y_label'] for x in datasets if x['path'] == '../.' + dataset[0].strip()][0]
        n_features = len(mat.columns) - 1

        algorithm = ['XGB', 'LR']
        model_name = ['XGBoost', 'LinearModel']
        train_data = TabularDataset(mat)
        train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                    enable_text_ngram_features=False).fit_transform(X=train_data)

        logging.error(f'{dataset[0]}, {algorithm}')
        hyperparameters = get_hyperparameters(train_data, y_label, algorithm, model_name)
        print(hyperparameters)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns=[y_label]), train_data[y_label],
                                                            test_size=0.2, random_state=42)
        train = X_train.copy()
        train[y_label] = y_train
        test = X_test.copy()
        test[y_label] = y_test

        # Perform evaluation
        mifs_000 = evaluate_model(train, test, mifs_000, y_label, algorithm, hyperparameters)
        print(mifs_000)
        logging.error(mifs_000)
        mifs_025 = evaluate_model(train, test, mifs_025, y_label, algorithm, hyperparameters)
        print(mifs_025)
        logging.error(mifs_025)
        mifs_050 = evaluate_model(train, test, mifs_050, y_label, algorithm, hyperparameters)
        print(mifs_050)
        logging.error(mifs_050)
        mifs_075 = evaluate_model(train, test, mifs_075, y_label, algorithm, hyperparameters)
        print(mifs_075)
        logging.error(mifs_075)
        mifs_100 = evaluate_model(train, test, mifs_100, y_label, algorithm, hyperparameters)
        print(mifs_100)
        logging.error(mifs_100)


def evaluate_performance_mifs_SVM():
    """
    Evaluates the performance of MIFS under SVM.
    provided an input file with the results from selecting features.

    Returns:
        None, but the results are stored on disk
    """
    with open('results/logs/mifs.txt', "r") as file:
        data = file.readlines()

    data = [data[i:i + 6] for i in range(0, len(data), 6)]
    for i in range(len(data)):
        dataset = data[i]

        mifs_000 = literal_eval(dataset[1].replace('array', ''))
        mifs_025 = literal_eval(dataset[2].replace('array', ''))
        mifs_050 = literal_eval(dataset[3].replace('array', ''))
        mifs_075 = literal_eval(dataset[4].replace('array', ''))
        mifs_100 = literal_eval(dataset[5].replace('array', ''))

        mat = pd.read_csv([x['path'] for x in datasets if x['path'] == '../.' + dataset[0].strip()][0])
        y_label = [x['y_label'] for x in datasets if x['path'] == '../.' + dataset[0].strip()][0]

        train_data = TabularDataset(mat)
        train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                    enable_text_ngram_features=False).fit_transform(X=train_data)
        train_data.fillna(0, inplace=True)
        columns = train_data.columns
        train_data = pd.DataFrame(MinMaxScaler((-1, 1)).fit_transform(train_data), columns=columns)
        logging.error(dataset[0])
        print(dataset[0])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns=[y_label]), train_data[y_label],
                                                            test_size=0.2, random_state=42)
        train = X_train.copy()
        train[y_label] = y_train
        test = X_test.copy()
        test[y_label] = y_test

        # Perform evaluation
        mifs_000 = evaluate_SVM(train, test, mifs_000, y_label)
        print(mifs_000)
        logging.error(mifs_000)
        mifs_025 = evaluate_SVM(train, test, mifs_025, y_label)
        print(mifs_025)
        logging.error(mifs_025)
        mifs_050 = evaluate_SVM(train, test, mifs_050, y_label)
        print(mifs_050)
        logging.error(mifs_050)
        mifs_075 = evaluate_SVM(train, test, mifs_075, y_label)
        print(mifs_075)
        logging.error(mifs_075)
        mifs_100 = evaluate_SVM(train, test, mifs_100, y_label)
        print(mifs_100)
        logging.error(mifs_100)


if __name__ == '__main__':
    # logging.basicConfig(filename='app.log', filemode='a', level=logging.ERROR, format='%(message)s')

    datasets = []
    datasets.append({'name': 'Bike sharing', 'path': '../../datasets/bike-sharing/hour.csv', 'y_label': 'cnt', 'n_features': 15, 'is_classification': False})
    datasets.append({'name': 'Census Income', 'path': '../../datasets/CensusIncome/CensusIncome.csv', 'y_label': 'income_label', 'n_features': 15, 'is_classification': True})
    datasets.append({'name': 'Breast cancer', 'path': '../../datasets/breast-cancer/data.csv', 'y_label': 'diagnosis', 'n_features': 30, 'is_classification': True})
    datasets.append({'name': 'Housing prices', 'path': '../../datasets/housing-prices/train.csv', 'y_label': 'SalePrice', 'n_features': 80, 'is_classification': False})
    datasets.append({'name': 'Steel plates faults', 'path': '../../datasets/steel-plates-faults/steel_faults_train.csv', 'y_label': 'Class', 'n_features': 33, 'is_classification': True})
    datasets.append({'name': 'Gisette', 'path': '../../datasets/gisette/gisette_train.csv', 'y_label': 'Class', 'n_features': 250, 'is_classification': True})
    datasets.append({'name': 'Internet advertisements', 'path': '../../datasets/internet_advertisements/internet_advertisements.csv', 'y_label': 'class', 'n_features': 200, 'is_classification': True})

    # Uncomment the following to perform feature selection for the datasets above
    # for MIFS, MRMR, CIFE, and JMI
    # perform_feature_selection_for_multiple_datasets()
    # evaluate_performance()
    # evaluate_performance_SVM()

    # Uncomment the following to perform feature selection for the datasets above
    # for different beta values for MIFS approach
    # perform_mifs_comparison()
    # evaluate_performance_mifs()
    # evaluate_performance_mifs_SVM()

    # Uncomment the following to plot the results. Note that you need to have a look at
    # the folders that lead to the performance and feature selection logs
    # plot_feature_selection_three_models(datasets)
    # plot_feature_selection_three_models_mifs(datasets)
    # plot_feature_selection_two_side_by_side()
    # plot_feature_selection_runtime()
    # plot_feature_selection_performance()
