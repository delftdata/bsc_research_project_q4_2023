import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from multiprocessing import Pool, cpu_count
from itertools import repeat

from skfeature.information_theoretical_based import JMI, MIFS, CIFE, MRMR, CMIM
from skfeature.utility.plotting_helpers import plot_over_features


def run_parallel_results(n_features, X, y, model, fs_algorithm):
    """
    Parallelize over the number of features to compare against

    Args:
        n_features: number of features
        X: training set without target column
        y: target column
        model: name of ML algorithm
        fs_algorithm: the feature selection algorithm

    Returns:
        Results of feature selection for each ML algorithm.
    """
    with Pool(min(n_features, cpu_count())) as p:
        results = p.starmap(run_kfold, zip(list(range(1, n_features + 1)), repeat(X), repeat(y), repeat(model), repeat(fs_algorithm)))

    return results

def run_one():
    """
    Runs all feature selection for a single dataset.
    """
    mat = pd.read_csv('skfeature/data/football_train.csv')
    y_label = 'win'
    X = mat.drop(y_label, axis=1)
    X = prepare_data_for_ml(X).to_numpy()
    y = mat[y_label].to_numpy()
    n_features = 10
    model = LogisticRegression()
    mrmr = run_parallel_results(n_features, X, y, model, select_mrmr)
    print(mrmr)


def run_kfold(n_features, X, y, model, fs_algorithm):
    """
    Performs KFold validation for SVM while performing feature selection.

    Args:
        n_features: number of features to select
        X: training set without target column
        y: target column
        model: name of ML algorithm
        fs_algorithm: feature selection algorithm

    Returns:
        The accuracy averaged over 10 runs.
    """
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

def select_mifs_beta(X, y, n_selected_features, beta):
    """
    Helper function to run MIFS with beta hyperparam.

    Args:
        X: training set without target column
        y: target column
        n_selected_features: number of features to select
        beta: beta hyperparameter

    Returns:
        Result from feature selection
    """
    return MIFS.mifs(X, y, n_selected_features=n_selected_features, beta=beta)

def select_mifs(X, y, n_selected_features):
    """
     Helper function to run MIFS feature selection.

    Args:
        X: training set without target column
        y: target column
        n_selected_features: number of features to select

    Returns:
        Result from feature selection
    """
    return MIFS.mifs(X, y, n_selected_features=n_selected_features)


def select_mrmr(X, y, n_selected_features):
    """
     Helper function to run MRMR feature selection.

    Args:
        X: training set without target column
        y: target column
        n_selected_features: number of features to select

    Returns:
        Result from feature selection
    """
    return MRMR.mrmr(X, y, n_selected_features=n_selected_features)


def select_jmi(X, y, n_selected_features):
    """
     Helper function to run JMI feature selection.

    Args:
        X: training set without target column
        y: target column
        n_selected_features: number of features to select

    Returns:
        Result from feature selection
    """
    return JMI.jmi(X, y, n_selected_features=n_selected_features)


def select_cife(X, y, n_selected_features):
    """
     Helper function to run CIFE feature selection.

    Args:
        X: training set without target column
        y: target column
        n_selected_features: number of features to select

    Returns:
        Result from feature selection
    """
    return CIFE.cife(X, y, n_selected_features=n_selected_features)


def select_cmim(X, y, n_selected_features):
    """
     Helper function to run CIFE feature selection.

    Args:
        X: training set without target column
        y: target column
        n_selected_features: number of features to select

    Returns:
        Result from feature selection
    """
    return CMIM.cmim(X, y, n_selected_features=n_selected_features)


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