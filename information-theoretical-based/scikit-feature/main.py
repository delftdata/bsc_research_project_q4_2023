import pandas as pd

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

from sklearn.model_selection import train_test_split

from multiprocessing import Pool, cpu_count
from itertools import repeat

from warnings import filterwarnings
import logging
import time

from skfeature.utility.data_preparation import prepare_data_for_ml, get_hyperparameters
from skfeature.utility.plotting import plot_over_features
from skfeature.utility.experiments import select_jmi, select_cife, select_mrmr, select_mifs

filterwarnings("ignore", category=UserWarning)


def auto_gluon(mat, y_label, eval_metric, algorithm, model_name, n_features, fs_algorithm):
    print(n_features)
    results = []

    # Encode features
    train_data = TabularDataset(mat)
    train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                enable_text_ngram_features=False).fit_transform(X=train_data)

    # Tune hyperparameters
    hyperparameters = get_hyperparameters(train_data, y_label, eval_metric, algorithm, model_name)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns=[y_label]), train_data[y_label],
                                                        test_size=0.2,
                                                        random_state=42)
    train = X_train
    train[y_label] = y_train

    test = X_test
    test[y_label] = y_test

    train_data = TabularDataset(train)

    time_now = time.time()
    idx, _, _ = fs_algorithm(train.drop(y_label, axis=1).to_numpy(), train[y_label].to_numpy(), n_selected_features=n_features)
    time_after = time.time()

    # obtain the dataset on the selected features
    picked_columns = [list(train.drop(y_label, axis=1).columns)[i] for i in idx[0:n_features]]
    picked_columns.append(y_label)
    features = train_data[picked_columns]

    # Train model on the smaller dataset with tuned hyper-parameters
    linear_predictor = TabularPredictor(label=y_label,
                                        eval_metric=eval_metric,
                                        verbosity=0) \
        .fit(train_data=features, hyperparameters={algorithm: hyperparameters})

    # Get accuracy on test data
    test_data = TabularDataset(test)
    accuracy = linear_predictor.evaluate(test_data)['accuracy']
    print(accuracy)
    results.append(accuracy)

    return accuracy, time_after - time_now


def run_all(mat, y_label, eval_metric, algorithm, model_name, n):

    with Pool(n) as p:
        mrmr = p.starmap(auto_gluon, zip(repeat(mat), repeat(y_label), repeat(eval_metric), repeat(algorithm), repeat(model_name), list(range(1, n + 1)), repeat(select_mrmr)))
    logging.error(mrmr)

    with Pool(n) as p:
        mifs = p.starmap(auto_gluon, zip(repeat(mat), repeat(y_label), repeat(eval_metric), repeat(algorithm), repeat(model_name), list(range(1, n + 1)), repeat(select_mifs)))
    logging.error(mifs)

    with Pool(n) as p:
        jmi = p.starmap(auto_gluon, zip(repeat(mat), repeat(y_label), repeat(eval_metric), repeat(algorithm), repeat(model_name), list(range(1, n + 1)), repeat(select_jmi)))
    logging.error(jmi)

    with Pool(n) as p:
        cife = p.starmap(auto_gluon, zip(repeat(mat), repeat(y_label), repeat(eval_metric), repeat(algorithm), repeat(model_name), list(range(1, n + 1)), repeat(select_cife)))
    logging.error(cife)

    return mrmr, mifs, jmi, cife


def main():
    mat = pd.read_csv('skfeature/data/steel_faults_train.csv')
    # mat = prepare_data_for_ml(mat)
    y_label = 'Class'
    eval_metric = 'accuracy'
    algorithm = 'XGB'
    n = 2
    model_name = 'XGBoost'

    mrmr, mifs, jmi, cife = run_all(mat, y_label, eval_metric, algorithm, model_name, n)
    mrmr = [i[0] for i in mrmr]
    mifs = [i[0] for i in mifs]
    jmi = [i[0] for i in jmi]
    cife = [i[0] for i in cife]
    plot_over_features(model_name, n, mrmr, mifs, jmi, cife)

    algorithm = 'LR'
    model_name = 'LinearModel'

    mrmr, mifs, jmi, cife = run_all(mat, y_label, eval_metric, algorithm, model_name, n)

    mrmr = [i[0] for i in mrmr]
    mifs = [i[0] for i in mifs]
    jmi = [i[0] for i in jmi]
    cife = [i[0] for i in cife]
    plot_over_features(model_name, n, mrmr, mifs, jmi, cife)


if __name__ == '__main__':
    logging.basicConfig(filename='app.log', filemode='w', level=logging.ERROR, format='%(message)s')
    main()
