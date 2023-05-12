import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

from multiprocessing import Pool, cpu_count
from warnings import filterwarnings
from itertools import repeat

from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.information_theoretical_based import CIFE

filterwarnings("ignore", category=UserWarning)


def prepare_data_for_ml(dataframe):
    df = dataframe.fillna(-1)
    df = df.apply(lambda x: pd.factorize(x)[0] if x.dtype == object else x)

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(df)
    normalized_X = pd.DataFrame(scaled_X, columns=df.columns)

    return normalized_X


def plot_over_features(model_name, n_features, mrmr, mifs, jmi, cife):
    features = list(range(1, n_features+1))
    plt.plot(features, mrmr)
    plt.plot(features, mifs)
    plt.plot(features, jmi)
    plt.plot(features, cife)
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.legend(['MRMR', 'MIFS', 'JMI', 'CIFE'])
    plt.title('Steel faults dataset accuracy with SVM')

    plt.savefig(f'result_{model_name}.png')
    plt.show()
    plt.clf()


def select_mifs(X, y, train, n_features):
    return MIFS.mifs(X[train], y[train], n_selected_features=n_features)


def select_mrmr(X, y, train, n_features):
    return MRMR.mrmr(X[train], y[train], n_selected_features=n_features)


def select_jmi(X, y, train, n_features):
    return JMI.jmi(X[train], y[train], n_selected_features=n_features)


def select_cife(X, y, train, n_features):
    return CIFE.cife(X[train], y[train], n_selected_features=n_features)


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

    columns = list(mat.columns)
    columns.remove('win')

    X = prepare_data_for_ml(mat[columns]).to_numpy()
    y = mat['win'].to_numpy()
    n_features = 2
    model = LinearSVC()
    mrmr = get_results(n_features, X, y, model, select_mrmr)
    print(mrmr)


if __name__ == '__main__':
    # main()
    run_one()
