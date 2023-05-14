import pandas as pd
import numpy as np

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


from multiprocessing import Pool, cpu_count
from warnings import filterwarnings
from itertools import repeat
import logging

from skfeature.information_theoretical_based import JMI, MIFS, CIFE, MRMR
from skfeature.utility.data_preparation import prepare_data_for_ml
from skfeature.utility.plotting import plot_over_features

filterwarnings("ignore", category=UserWarning)

def select_mifs(X, y, n_selected_features):
    return MIFS.mifs(X, y, n_selected_features=n_selected_features)


def select_mrmr(X, y, n_selected_features):
    return MRMR.mrmr(X, y, n_selected_features=n_selected_features)


def select_jmi(X, y, n_selected_features):
    return JMI.jmi(X, y, n_selected_features=n_selected_features)


def select_cife(X, y, n_selected_features):
    return CIFE.cife(X, y, n_selected_features=n_selected_features)


def get_results(n_features, X, y, model, fs_algorithm):
    # split data into 10 folds
    ss = KFold(n_splits=10, shuffle=True, random_state=42)

    print(n_features)
    correct = 0
    for train, test in ss.split(X):
        # obtain the index of each feature on the training set
        idx, _, _ = fs_algorithm(X, y, train, n_features)

        # obtain the dataset on the selected features
        features = X[:, idx[0:n_features]]

        # train a model with the selected features on the training dataset
        model.fit(features[train], y[train])

        # predict the class labels of test data
        y_predict = model.predict(features[test])

        # obtain the classification accuracy on the test data
        acc = accuracy_score(y[test], y_predict)
        correct = correct + acc

    # output the average accuracy over all 10 folds
    print(n_features)
    print('Accuracy:', float(correct) / 10)

    return float(correct) / 10


def run_parallel_results(n_features, X, y, model, fs_algorithm):
    # Parallelize over the number of features to compare against

    with Pool(min(n_features, cpu_count())) as p:
        results = p.starmap(get_results, zip(list(range(1, n_features + 1)), repeat(X), repeat(y), repeat(model), repeat(fs_algorithm)))

    return results


def main():
    mat = pd.read_csv('skfeature/data/steel_faults_train.csv')
    X = mat[mat.columns[:-2]].to_numpy()
    y = mat[mat.columns[-1]].to_numpy()
    n_features = 20
    model = LinearSVC()

    for model in [LinearSVC(random_state=42), LogisticRegression(random_state=42), DecisionTreeClassifier(random_state=42)]:
        mrmr = run_parallel_results(n_features, X, y, model, select_mrmr)
        mifs = run_parallel_results(n_features, X, y, model, select_mifs)
        jmi = run_parallel_results(n_features, X, y, model, select_jmi)
        cife = run_parallel_results(n_features, X, y, model, select_cife)
        print(mrmr)
        print(mifs)
        print(jmi)
        print(cife)

        plot_over_features(type(model).__name__, n_features, mrmr, mifs, jmi, cife)


def run_one():
    mat = pd.read_csv('skfeature/data/football_train.csv')
    y_label = 'win'
    X = mat.drop(y_label, axis=1)
    X = prepare_data_for_ml(X).to_numpy()
    y = mat[y_label].to_numpy()
    n_features = 10
    model = LogisticRegression()
    mrmr = run_parallel_results(n_features, X, y, model, select_mrmr)
    print(mrmr)


def auto_gluon(mat, y_label, eval_metric, algorithm, model_name, n, fs_algorithm):
    results = []

    # Encode features
    train_data = TabularDataset(mat)
    train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                enable_text_ngram_features=False).fit_transform(X=train_data)

    # Tune hyperparameters
    hyperparameters = get_hyperparameters(train_data, y_label, eval_metric, algorithm, model_name)

    # Run feature selection
    for n_features in range(1, n+1):
        idx, _, _ = fs_algorithm(train_data.drop(y_label, axis=1).to_numpy(), train_data[y_label].to_numpy(), n_selected_features=n_features)

        # obtain the dataset on the selected features
        picked_columns = [list(train_data.drop(y_label, axis=1).columns)[i] for i in idx[0:n_features]]
        picked_columns.append(y_label)
        features = train_data[picked_columns]

        # Train model on the smaller dataset with tuned hyper-parameters
        linear_predictor = TabularPredictor(label=y_label,
                                            eval_metric=eval_metric,
                                            verbosity=0) \
            .fit(train_data=features, hyperparameters={algorithm: hyperparameters})

        # Get accuracy
        training_results = linear_predictor.info()
        accuracy = training_results["model_info"][model_name]['val_score']
        print(accuracy)
        results.append(accuracy)

    return results

if __name__ == '__main__':
    # main()
    # run_one()
    mat = pd.read_csv('skfeature/data/steel_faults_train.csv')
    # mat = prepare_data_for_ml(mat)
    y_label = 'Class'
    eval_metric = 'accuracy'
    algorithm = 'XGB'
    n = 20
    model_name = 'XGBoost'

    logging.basicConfig(filename='app.log', filemode='w', level=logging.ERROR, format='%(message)s')
    mrmr = auto_gluon(mat, y_label, eval_metric, algorithm, model_name, n, select_mrmr)
    logging.error(mrmr)
    mifs = auto_gluon(mat, y_label, eval_metric, algorithm, model_name, n, select_mifs)
    logging.error(mifs)
    jmi = auto_gluon(mat, y_label, eval_metric, algorithm, model_name, n, select_jmi)
    logging.error(jmi)
    cife = auto_gluon(mat, y_label, eval_metric, algorithm, model_name, n, select_cife)
    logging.error(cife)

    plot_over_features(model_name, n, mrmr, mifs, jmi, cife)

    algorithm = 'LR'
    model_name = 'LinearModel'

    mrmr = auto_gluon(mat, y_label, eval_metric, algorithm, model_name, n, select_mrmr)
    logging.error(mrmr)
    mifs = auto_gluon(mat, y_label, eval_metric, algorithm, model_name, n, select_mifs)
    logging.error(mifs)
    jmi = auto_gluon(mat, y_label, eval_metric, algorithm, model_name, n, select_jmi)
    logging.error(jmi)
    cife = auto_gluon(mat, y_label, eval_metric, algorithm, model_name, n, select_cife)
    logging.error(cife)

    plot_over_features(model_name, n, mrmr, mifs, jmi, cife)

