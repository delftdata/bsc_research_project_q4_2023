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

def perform_feature_selection(fs_algorithm, n_features, train, y_label):

    # Perform feature selection
    train_data = TabularDataset(train)
    np.random.seed(42)
    idx, _, _, times = fs_algorithm(train_data.drop(y_label, axis=1).to_numpy(), train_data[y_label].to_numpy(), n_selected_features=n_features)
    result = [idx[0:i] for i in range(1, len(idx)+1)]
    result = list(zip(result, times))
    print(result)
    return result


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
    mrmr_result = perform_feature_selection(select_mrmr, n_features, train, y_label)
    logging.error(mrmr_result)
    mifs_result = perform_feature_selection(select_mifs, n_features, train, y_label)
    logging.error(mifs_result)
    jmi_result = perform_feature_selection(select_jmi, n_features, train, y_label)
    logging.error(jmi_result)
    cife_result = perform_feature_selection(select_cife, n_features, train, y_label)
    logging.error(cife_result)

    # Perform evaluation
    mrmr = evaluate_model(train, test, mrmr_result, y_label, eval_metric, algorithm, hyperparameters, n_features)
    mifs = evaluate_model(train, test, mifs_result, y_label, eval_metric, algorithm, hyperparameters, n_features)
    jmi = evaluate_model(train, test, jmi_result, y_label, eval_metric, algorithm, hyperparameters, n_features)
    cife = evaluate_model(train, test, cife_result, y_label, eval_metric, algorithm, hyperparameters, n_features)

    return mrmr, mifs, jmi, cife


def main():
    mat = pd.read_csv('skfeature/data/steel_faults_train.csv')
    # mat = prepare_data_for_ml(mat)
    y_label = 'Class'
    eval_metric = 'accuracy'
    algorithm = 'XGB'
    n = 20
    model_name = 'XGBoost'

    mrmr, mifs, jmi, cife = run_pipeline(mat, y_label, eval_metric, algorithm, model_name, n)
    mrmr = [i[0] for i in mrmr]
    mifs = [i[0] for i in mifs]
    jmi = [i[0] for i in jmi]
    cife = [i[0] for i in cife]
    plot_over_features(model_name, n, mrmr, mifs, jmi, cife)

    algorithm = 'LR'
    model_name = 'LinearModel'

    mrmr, mifs, jmi, cife = run_pipeline(mat, y_label, eval_metric, algorithm, model_name, n)

    mrmr = [i[0] for i in mrmr]
    mifs = [i[0] for i in mifs]
    jmi = [i[0] for i in jmi]
    cife = [i[0] for i in cife]
    plot_over_features(model_name, n, mrmr, mifs, jmi, cife)


if __name__ == '__main__':
    logging.basicConfig(filename='app.log', filemode='w', level=logging.ERROR, format='%(message)s')
    np.random.seed(42)
    main()
