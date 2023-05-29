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

from skfeature.utility.data_preparation import prepare_data_for_ml, get_hyperparameters
from skfeature.utility.plotting import *
from skfeature.utility.experiments import select_jmi, select_cife, select_mrmr, select_mifs

filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)
filterwarnings("ignore", category=FutureWarning)


def perform_feature_selection_single(fs_algorithm, n_features, train, y_label):
    # Perform feature selection
    train_data = TabularDataset(train)
    np.random.seed(42)
    idx, _, _, times = fs_algorithm(train_data.drop(y_label, axis=1).to_numpy(), train_data[y_label].to_numpy(), n_selected_features=n_features)
    result = [idx[0:i] for i in range(1, len(idx)+1)]
    result = list(zip(result, times))
    print(result)
    return result


def perform_feature_selection_all(n_features, train, y_label):
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
    with open('results/me.txt', "r") as file:
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


def visualize_results(dataset_name, model_names, n, mrmrs, mifss, jmis, cifes):
    i = 0
    for mrmr, mifs, jmi, cife in zip(mrmrs, mifss, jmis, cifes):
        model_name = model_names[i]

        mrmr_one = [i[0] for i in mrmr]
        mifs_one = [i[0] for i in mifs]
        jmi_one = [i[0] for i in jmi]
        cife_one = [i[0] for i in cife]
        plot_over_features(dataset_name, model_name, n, mrmr_one, mifs_one, jmi_one, cife_one)

        mrmr_one = [i[1] for i in mrmr]
        mifs_one = [i[1] for i in mifs]
        jmi_one = [i[1] for i in jmi]
        cife_one = [i[1] for i in cife]
        plot_performance(dataset_name, n, mrmr_one, mifs_one, jmi_one, cife_one)
        i += 1

def plot_feature_selection_performance_svm():
    with open('results/logs/performance_complex_svm.txt', "r") as file:
        data = file.readlines()

    datasets = [data[i:i + 5] for i in range(0, len(data), 5)]
    for i in range(len(datasets)):
        dataset = datasets[i]
        dataset_name = dataset[0].split('datasets/')[1].split('/')[0]
        mrmr_result = literal_eval(dataset[1].replace('array', ''))
        mifs_result = literal_eval(dataset[2].replace('array', ''))
        jmi_result = literal_eval(dataset[3].replace('array', ''))
        cife_result = literal_eval(dataset[4].replace('array', ''))

        mrmr_one = [i[0] for i in mrmr_result]
        mifs_one = [i[0] for i in mifs_result]
        jmi_one = [i[0] for i in jmi_result]
        cife_one = [i[0] for i in cife_result]

        model_name = 'SVM'

        plot_over_features(dataset_name, model_name, len(mrmr_one), mrmr_one, mifs_one, jmi_one, cife_one)


def plot_feature_selection_performance():
    # with open('results/22-5-2023/performance_simple.txt', "r") as file:
    #     data = file.readlines()
    # with open('results/22-5-2023/performance_complex.txt', "r") as file:
    #     data_complex = file.readlines()

    with open('results/me.txt', "r") as file:
        data = file.readlines()

    data_complex = []
    datasets = [data[i:i + 5] for i in range(0, len(data), 5)]
    datasets_complex = [data_complex[i:i + 5] for i in range(0, len(data_complex), 5)]
    for i in range(len(datasets)):
        dataset = datasets[i]
        dataset_name = dataset[0].split('datasets/')[1].split('/')[0]
        mrmr_result = literal_eval(dataset[1].replace('array', ''))
        mifs_result = literal_eval(dataset[2].replace('array', ''))
        jmi_result = literal_eval(dataset[3].replace('array', ''))
        cife_result = literal_eval(dataset[4].replace('array', ''))

        model_names = ['XGBoost', 'LinearModel']
        j = 0
        for mrmr, mifs, jmi, cife in zip(mrmr_result, mifs_result, jmi_result, cife_result):
            mrmr_one = [i[0] for i in mrmr]
            mifs_one = [i[0] for i in mifs]
            jmi_one = [i[0] for i in jmi]
            cife_one = [i[0] for i in cife]

            model_name = model_names[j]

            plot_over_features(dataset_name, model_name, len(mrmr_one), mrmr_one, mifs_one, jmi_one, cife_one)
            j += 1

    for i in range(len(datasets_complex)):
        dataset_complex = datasets_complex[i]
        dataset_name = dataset_complex[0].split('datasets/')[1].split('/')[0]
        mrmr_result_complex = literal_eval(dataset_complex[1].replace('array', ''))
        mifs_result_complex = literal_eval(dataset_complex[2].replace('array', ''))
        jmi_result_complex = literal_eval(dataset_complex[3].replace('array', ''))
        cife_result_complex = literal_eval(dataset_complex[4].replace('array', ''))

        model_names = ['XGBoost', 'LinearModel']

        j = 0
        for mrmr, mifs, jmi, cife in zip(mrmr_result_complex, mifs_result_complex, jmi_result_complex, cife_result_complex):
            mrmr_one = [i[0] for i in mrmr]
            mifs_one = [i[0] for i in mifs]
            jmi_one = [i[0] for i in jmi]
            cife_one = [i[0] for i in cife]

            model_name = model_names[j]

            plot_over_features(dataset_name + '_complex_', model_name, len(mrmr_one), mrmr_one, mifs_one, jmi_one, cife_one)
            j += 1


def plot_feature_selection_two_side_by_side():
    with open('results/22-5-2023/performance_simple.txt', "r") as file:
        data = file.readlines()
    with open('results/22-5-2023/performance_complex.txt', "r") as file:
        data_complex = file.readlines()

    datasets = [data[i:i + 5] for i in range(0, len(data), 5)]
    datasets_complex = [data_complex[i:i + 5] for i in range(0, len(data_complex), 5)]

    temp = []
    for i in range(len(datasets)):
        dataset = datasets[i]
        if 'steel' in dataset[0]:
            temp.append(dataset)

    for i in range(len(datasets_complex)):
        dataset = datasets_complex[i]
        if 'steel' in dataset[0]:
            temp.append(dataset)

    mrmr = [literal_eval(temp[0][1].replace('array', ''))[1], literal_eval(temp[1][1].replace('array', ''))[1]]
    mifs = [literal_eval(temp[0][2].replace('array', ''))[1], literal_eval(temp[1][2].replace('array', ''))[1]]
    jmi = [literal_eval(temp[0][3].replace('array', ''))[1], literal_eval(temp[1][3].replace('array', ''))[1]]
    cife = [literal_eval(temp[0][4].replace('array', ''))[1], literal_eval(temp[1][4].replace('array', ''))[1]]

    plot_over_features_2('steel', 'Steel Plate Faults dataset performance on Linear Regression model',
                         len(mrmr[0]), mrmr, mifs, jmi, cife)


def plot_feature_selection_runtime():
    with open('results/22-5-2023/fs_simple.txt', "r") as file:
        data = file.readlines()
    with open('results/22-5-2023/fs_complex.txt', "r") as file:
        data_complex = file.readlines()

    datasets = [data[i:i + 5] for i in range(0, len(data), 5)]
    datasets_complex = [data_complex[i:i + 5] for i in range(0, len(data_complex), 5)]
    for i in range(len(datasets_complex)):
        dataset = datasets[i]
        dataset_name = dataset[0].split('datasets/')[1].split('/')[0]
        mrmr_result = literal_eval(dataset[1].replace('array', ''))
        mifs_result = literal_eval(dataset[2].replace('array', ''))
        jmi_result = literal_eval(dataset[3].replace('array', ''))
        cife_result = literal_eval(dataset[4].replace('array', ''))

        mrmr_one = [i[1] for i in mrmr_result]
        mifs_one = [i[1] for i in mifs_result]
        jmi_one = [i[1] for i in jmi_result]
        cife_one = [i[1] for i in cife_result]

        dataset_complex = datasets_complex[i]
        mrmr_result_complex = literal_eval(dataset_complex[1].replace('array', ''))
        mifs_result_complex = literal_eval(dataset_complex[2].replace('array', ''))
        jmi_result_complex = literal_eval(dataset_complex[3].replace('array', ''))
        cife_result_complex = literal_eval(dataset_complex[4].replace('array', ''))

        mrmr_complex_one = [i[1] for i in mrmr_result_complex]
        mifs_complex_one = [i[1] for i in mifs_result_complex]
        jmi_complex_one = [i[1] for i in jmi_result_complex]
        cife_complex_one = [i[1] for i in cife_result_complex]


        print(dataset_name)
        print(round(mifs_one[-1], 2))
        print(round(mrmr_one[-1], 2))
        print(round(cife_one[-1], 2))
        print(round(jmi_one[-1], 2))
        print(round(mifs_complex_one[-1], 2))
        print(round(mrmr_complex_one[-1], 2))
        print(round(cife_complex_one[-1], 2))
        print(round(jmi_complex_one[-1], 2))
        # plot_performance(dataset_name, len(mrmr_one), mrmr_one, mifs_one, jmi_one, cife_one, False)

        plot_performance_8(dataset_name, len(mrmr_one), mrmr_one, mifs_one, jmi_one, cife_one, mrmr_complex_one, mifs_complex_one, jmi_complex_one, cife_complex_one)
        # plot_performance_two('Breast cancer dataset runtime performance', len(mrmr_one), mrmr_one, mifs_one, jmi_one, cife_one, mrmr_complex_one, mifs_complex_one, jmi_complex_one, cife_complex_one)

def perform_feature_selection_for_multiple_datasets():

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


def evaluate_performance():
    with open('results/logs/fs_simple.txt', "r") as file:
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


if __name__ == '__main__':
    # logging.basicConfig(filename='app.log', filemode='a', level=logging.ERROR, format='%(message)s')

    datasets = []
    datasets.append({'path': '../../datasets/bike-sharing/hour.csv', 'y_label': 'cnt', 'n_features': 15})
    datasets.append({'path': '../../datasets/BankMarketing/bank.csv', 'y_label': 'y', 'n_features': 20})
    datasets.append({'path': '../../datasets/CensusIncome/CensusIncome.csv', 'y_label': 'income_label', 'n_features': 15})
    datasets.append({'path': '../../datasets/breast-cancer/data.csv', 'y_label': 'diagnosis', 'n_features': 30})
    datasets.append({'path': '../../datasets/housing-prices/train.csv', 'y_label': 'SalePrice', 'n_features': 80})
    datasets.append({'path': '../../datasets/steel-plates-faults/steel_faults_train.csv', 'y_label': 'Class', 'n_features': 33})
    datasets.append({'path': '../../datasets/gisette/gisette_train.csv', 'y_label': 'Class', 'n_features': 250})
    datasets.append({'path': '../../datasets/internet_advertisements/internet_advertisements.csv', 'y_label': 'class', 'n_features': 200})

    # perform_feature_selection_for_multiple_datasets()
    # evaluate_performance()
    # plot_feature_selection_runtime()
    # plot_feature_selection_performance()
    # plot_two_side_by_side()
    # evaluate_performance_SVM()
    plot_feature_selection_performance_svm()
