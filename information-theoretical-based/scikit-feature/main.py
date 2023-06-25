import pandas as pd
import numpy as np

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

from sklearn.model_selection import train_test_split

from warnings import filterwarnings
import logging

from skfeature.utility.data_preparation import prepare_data_for_ml, get_hyperparameters
from skfeature.utility.plotting import plot_over_features, plot_performance
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


def evaluate_model(train, test, fs_results, y_label, eval_metric, algorithms, hyperparameters, n_features):
    results = []
    for algorithm, hyperparameter in zip(algorithms, hyperparameters):
        result = []
        for idx, duration in fs_results:
            # obtain the dataset on the selected features
            picked_columns = [list(train.drop(y_label, axis=1).columns)[i] for i in idx[0:n_features]]
            picked_columns.append(y_label)
            features = train[picked_columns]

            # Train model on the smaller dataset with tuned hyperparameters
            linear_predictor = TabularPredictor(label=y_label,
                                                eval_metric=eval_metric,
                                                verbosity=0) \
                .fit(train_data=features, hyperparameters={algorithm: hyperparameter})

            # Get accuracy on test data
            test_data = TabularDataset(test)
            accuracy = linear_predictor.evaluate(test_data)[eval_metric]
            print(accuracy)
            result.append((accuracy, duration))

        results.append(result)
    return results


def run_pipeline(mat, y_label, eval_metric, algorithm, model_name, n_features):
    # Encode features
    train_data = TabularDataset(mat)
    train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                enable_text_ngram_features=False).fit_transform(X=train_data)

    # Tune hyperparameters
    hyperparameters = get_hyperparameters(train_data, y_label, eval_metric, algorithm, model_name)

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
    mrmr = evaluate_model(train, test, mrmr_result, y_label, eval_metric, algorithm, hyperparameters, n_features)
    mifs = evaluate_model(train, test, mifs_result, y_label, eval_metric, algorithm, hyperparameters, n_features)
    jmi = evaluate_model(train, test, jmi_result, y_label, eval_metric, algorithm, hyperparameters, n_features)
    cife = evaluate_model(train, test, cife_result, y_label, eval_metric, algorithm, hyperparameters, n_features)

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
        plot_performance(dataset_name, model_name + '_time', n, mrmr_one, mifs_one, jmi_one, cife_one)
        i += 1


def main():
    datasets = []
    datasets.append({'path': 'skfeature/data/housing_train.csv', 'y_label': 'SalePrice', 'n_features': 40})
    datasets.append({'path': 'skfeature/data/Gisette/gisette_train.csv', 'y_label': 'Class', 'n_features': 250})

    for dataset in datasets:
        mat = pd.read_csv(dataset['path'])
        y_label = dataset['y_label']
        n_features = dataset['n_features']

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
        logging.error(dataset['path'])
        perform_feature_selection_all(n_features, train, y_label)


if __name__ == '__main__':
    logging.basicConfig(filename='app.log', filemode='a', level=logging.ERROR, format='%(message)s')
    main()
