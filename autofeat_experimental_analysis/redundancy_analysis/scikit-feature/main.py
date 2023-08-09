import pandas as pd
import numpy as np

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import train_test_split

from warnings import filterwarnings
import logging
from skfeature.utility.experiments import select_jmi, select_cife, select_mrmr, select_mifs, select_cmim

from group_lasso import GroupLasso
import time
from itertools import compress

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
    idx, J_CMI, _, times = fs_algorithm(train_data.drop(y_label, axis=1).to_numpy(), train_data[y_label].to_numpy(), n_selected_features=n_features)
    result = [idx[0:i] for i in range(1, len(idx)+1)]
    result = list(zip(result, times))
    print(result)

    result = pd.DataFrame(result, columns=['features', 'time'])
    result['k'] = result.index + 1
    result['score'] = J_CMI
    return result


def perform_feature_selection_all(name, train, y_label):
    """
    Helper function to perform the feature selection for MIFS, MRMR, CIFE, JMI, and CMIM.

    Args:
        train: the training set
        y_label: the name of y label

    Returns:
        arrays with performed feature selection for each of the four possibilities
    """
    n_features = len(train.columns)
    mrmr_result = perform_feature_selection_single(select_mrmr, n_features, train, y_label)
    mrmr_result['method_name'] = 'MRMR'

    mifs_result = perform_feature_selection_single(select_mifs, n_features, train, y_label)
    mifs_result['method_name'] = 'MIFS'

    jmi_result = perform_feature_selection_single(select_jmi, n_features, train, y_label)
    jmi_result['method_name'] = 'JMI'

    cife_result = perform_feature_selection_single(select_cife, n_features, train, y_label)
    cife_result['method_name'] = 'CIFE'

    cmim_result = perform_feature_selection_single(select_cmim, n_features, train, y_label)
    cmim_result['method_name'] = 'CMIM'


    result = pd.concat([mifs_result, mrmr_result, cife_result, jmi_result, cmim_result])
    result['dataset'] = name
    try:
        prev_result = pd.read_csv('results/result.csv')
        result = pd.concat([prev_result, result])
        result.to_csv('results/result.csv', index=False)
    except:
        result.to_csv('results/result.csv', index=False)
    return result


def perform_feature_selection_for_multiple_datasets():
    """
    Performs the feature selection for all provided datasets for MIFS, MRMR, CIFE, JMI, and CMIM.
    """
    for dataset in datasets:
        mat = pd.read_csv(dataset['path'])
        y_label = dataset['y_label']

        # Encode features
        train_data = TabularDataset(mat)
        train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                    enable_text_ngram_features=False).fit_transform(X=train_data)
        train_data = train_data.apply(lambda x: x.fillna(x.value_counts().index[0]))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns=[y_label]), train_data[y_label],
                                                            test_size=0.2, random_state=42)
        train = X_train.copy()
        train[y_label] = y_train

        # Perform feature selection
        print(dataset['path'])
        logging.error(dataset['path'])
        perform_feature_selection_all(dataset['name'], train, y_label)


def evaluate_gbm():
    """
    Helper function to evaluate models consisting of a subset of features using LightGBM from AutoGluon.
    """
    data = pd.read_csv('results/result.csv')
    data.features = data.features.apply(lambda x: [int(y) for y in x[1:-1].replace('\t', ' ').split(' ') if y != ''])

    result = []
    for index, row in data.iterrows():
        idx = row['features']
        dataset = row['dataset']

        mat = pd.read_csv([x['path'] for x in datasets if x['name'] == dataset][0])
        y_label = [x['y_label'] for x in datasets if x['name'] == dataset][0]

        train_data = TabularDataset(mat)
        train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                    enable_text_ngram_features=False).fit_transform(X=train_data)

        train_data = train_data.apply(lambda x: x.fillna(x.value_counts().index[0]))

        algorithm = ['GBM']
        X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns=[y_label]), train_data[y_label],
                                                            test_size=0.2, random_state=42)
        train = X_train.copy()
        train[y_label] = y_train
        test = X_test.copy()
        test[y_label] = y_test

        print(index)

        # obtain the dataset on the selected features
        idx = list(set(idx))
        n_features = len(idx)
        a = list(train.drop(y_label, axis=1).columns)
        picked_columns = [a[i-1] for i in idx[0:n_features] if i < len(a)]
        picked_columns.append(y_label)
        features = train[picked_columns]

        # Train model on the smaller dataset with tuned hyperparameters
        linear_predictor = TabularPredictor(label=y_label,
                                            verbosity=0,
                                            path='./AutoGluon') \
            .fit(train_data=features, hyperparameters={algorithm[0]: {}})

        # Get accuracy on test data
        test_data = TabularDataset(test[picked_columns])
        accuracy = linear_predictor.evaluate(test_data)
        if 'accuracy' in accuracy:
            accuracy = accuracy['accuracy']
        else:
            accuracy = abs(accuracy['root_mean_squared_error'])
        print(accuracy)

        result.append(accuracy)

    data['accuracy'] = result
    data.to_csv('evaluated.csv', index=False)


def evaluate_group_lasso():
    """
    Helper function to select features with GroupLasso and evaluate resulting models.
    """

    df = pd.read_csv(datasets[0]['path'])
    df = TabularDataset(df)
    df = pd.DataFrame(AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                     enable_text_ngram_features=False).fit_transform(X=df))
    df = pd.DataFrame(df).apply(pd.to_numeric).fillna(0)

    time_before = time.time()
    for groups in [1, 10, 50, 100, 169]:
        N = len(df.columns) - 1
        K = groups
        array = []

        for i in range(K):
            array.extend([i + 1] * (N // K))
        array.extend([K + 1] * (N % K))
        # groups = np.concatenate(
        #     [size * [i] for i, size in enumerate(group_sizes)]
        # ).reshape(-1, 1)

        array = np.array(array).reshape(-1, 1)
        gl = GroupLasso(
            groups=array,
            group_reg=5,
            l1_reg=0,
            frobenius_lipschitz=True,
            scale_reg="inverse_group_size",
            subsampling_scheme=1,
            supress_warning=True,
            n_iter=1000,
            tol=1e-3,
            warm_start=True,  # Warm start to start each subsequent fit with previous weights
        )

        gl.fit(df.drop(datasets[0]['y_label'], axis=1).to_numpy(), df[[datasets[0]['y_label']]].to_numpy())

        pred_c = gl.predict(df.drop(datasets[0]['y_label'], axis=1))
        sparsity_mask = gl.sparsity_mask_

        time_after = time.time()

        # print(groups)

        a = df.drop(datasets[0]['y_label'], axis=1)
        a = a[a.columns[sparsity_mask]]
        a['y_label'] = df[[datasets[0]['y_label']]]
        linear_predictor = TabularPredictor(label='y_label',
                                            verbosity=0,
                                            path='./AutoGluon') \
            .fit(train_data=a, hyperparameters={'GBM': {}})
        test_data = TabularDataset(a)
        accuracy = linear_predictor.evaluate(test_data)
        # print(accuracy['accuracy'])
        # print(time_after - time_before)

        print(f'"{str(list(compress(range(0, len(df)), sparsity_mask)))}",{time_after - time_before},{groups},,GroupLasso,{datasets[0]["name"]},,{abs(accuracy["accuracy"])}')


if __name__ == '__main__':

    datasets = []
    datasets.append({'name': 'BreastCancer', 'path': '../../autofeat_datasets/breast_cancer/breast_cancer.csv', 'y_label': 'diagnosis', 'is_classification': True})
    datasets.append({'name': 'SpamEmail', 'path': '../../autofeat_datasets/SPAM/spam.csv', 'y_label': 'class', 'is_classification': True})
    datasets.append({'name': 'Musk', 'path': '../../autofeat_datasets/Musk/musk.csv', 'y_label': 'class', 'is_classification': True})
    datasets.append({'name': 'Arrhythmia', 'path': '../../autofeat_datasets/Arrhythmia/arrhythmia.csv', 'y_label': 'binaryClass', 'is_classification': True})
    datasets.append({'name': 'InternetAds', 'path': '../../autofeat_datasets/internet_advertisements/internet_advertisements.csv', 'y_label': 'class', 'is_classification': True})
    datasets.append({'name': 'Gisette', 'path': '../../autofeat_datasets/Gisette/gisette.csv', 'y_label': 'Class', 'is_classification': True})

    # Use this to start the feature selection process
    perform_feature_selection_for_multiple_datasets()
    # Use this to evaluate resulting subsets of datasets with LightGBM
    evaluate_gbm()
