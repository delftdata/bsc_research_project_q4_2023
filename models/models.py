from autogluon.tabular import TabularPredictor
from autogluon.features.generators import IdentityFeatureGenerator
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


class AutogluonModel():

    def __init__(self, problem_type: str, label: str, data_preprocessing: bool = False, test_size: float = 0.2, hyperparameters: dict[str, dict] = []):
        self.data_preprocessing = data_preprocessing
        self.test_size = test_size
        self.predictor = TabularPredictor(
            problem_type=problem_type, label=label)
        self.hyperparameters = hyperparameters
        self.df_train = []
        self.df_test = []

    def fit(self, df):
        self.df_train, self.df_test = train_test_split(
            df, test_size=self.test_size, random_state=1)

        self.predictor.fit(self.df_train, presets='best_quality', hyperparameters=self.hyperparameters,
                           feature_generator=IdentityFeatureGenerator())

    def evaluate(self):
        return self.predictor.evaluate(self.df_test)


class SVMModel():

    def __init__(self, problem_type: str, label: str, data_preprocessing: bool = False, test_size: float = 0.2):
        self.data_preprocessing = data_preprocessing
        self.test_size = test_size
        self.problem_type = problem_type
        self.predictor = SVR() if problem_type == 'regression' else SVC()
        self.label = label
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.y_pred = []

    def fit(self, df):
        X = df.drop([self.label], axis=1)
        y = df[self.label]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=1)

        self.predictor.fit(self.X_train, self.y_train)
        self.y_pred = self.predictor.predict(self.X_test)

    def grid_search(self, df, param_grid: list[dict[str, any]], file=""):
        X = df.drop([self.label], axis=1)
        y = df[self.label]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=1)
        model = SVR() if self.problem_type == 'regression' else SVC()

        grid = GridSearchCV(model, param_grid, refit=True, verbose=3, cv=2)
        grid.fit(self.X_train, self.y_train)

        if file != "":
            with open(file, 'a+') as f:
                f.write(str(grid.best_estimator_) + '\n')
        else:
            print(grid.best_params_)
            print(grid.best_estimator_)

    def evaluate(self):
        if self.problem_type == 'regression':
            return np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        return accuracy_score(self.y_test, self.y_pred)
