from autogluon.tabular import TabularPredictor
from autogluon.features.generators import IdentityFeatureGenerator
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


class AutogluonModel():

    def __init__(self, label: str, problem_type: str = None, data_preprocessing: bool = True, test_size: float = 0.2, hyperparameters: dict[str, dict] = []):
        """
        Autogluon model constructor. It creates a TabularPredictor instance that runs models supported by AutoGluon.
        Supported algorithms are available in AutoGluon's documentation: https://auto.gluon.ai/0.1.0/api/autogluon.task.html
        Args:
            label (str): Label of the target column.
            problem_type (str, optional): Type of problem: 'binary', 'multiclass' or 'regression'. If None, Autogluon will infer it automatically. 
            data_preprocessing (bool, optional): Whether to do automatic data preprocessing or not. Defaults to True.
            test_size (float, optional): Percentage (between 0 and 1) of the dataset size to allocate for testing. Defaults to 0.2.
            hyperparameters (dict[str, dict], optional): Defines which algorithms AutoGluon should run. Defaults to all the supported algorithms.
        """
        self.data_preprocessing = data_preprocessing
        self.test_size = test_size
        self.predictor = TabularPredictor(
            problem_type=problem_type, label=label)
        self.hyperparameters = hyperparameters
        self.df_train = []
        self.df_test = []

    def fit(self, df):
        """
        Fits machine learning models.

        Args:
            df (Pandas Dataframe): Dataframe to train and test the models on.
        """
        self.df_train, self.df_test = train_test_split(
            df, test_size=self.test_size, random_state=1)

        if not self.data_preprocessing:
            self.predictor.fit(self.df_train, presets='best_quality', hyperparameters=self.hyperparameters,
                               feature_generator=IdentityFeatureGenerator())
        else:
            self.predictor.fit(self.df_train, presets='best_quality',
                               hyperparameters=self.hyperparameters)

    def evaluate(self):
        """
        Evaluates machine learning models and returns the results of the best performing one.
        """
        return self.predictor.evaluate(self.df_test)


class SVMModel():

    def __init__(self,  label: str, problem_type: str, data_preprocessing: bool = False, test_size: float = 0.2):
        """_summary_

        Args:
            label (str):  Label of the target column.
            problem_type (str): Type of problem: 'classification' (uses SVC) or 'regression'(uses SVR).
            data_preprocessing (bool, optional): Whether to do automatic data preprocessing or not. Defaults to False. Feature is not implemented.
            test_size (float, optional): _description_. Percentage (between 0 and 1) of the dataset size to allocate for testing. Defaults to 0.2.
        """
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
        """
        Fits machine learning models.

        Args:
            df (Pandas Dataframe): Dataframe to train and test the models on.
        """
        X = df.drop([self.label], axis=1)
        y = df[self.label]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=1)

        self.predictor.fit(self.X_train, self.y_train)
        self.y_pred = self.predictor.predict(self.X_test)

    def grid_search(self, df, param_grid: list[dict[str, any]], file=""):
        """
        Performs grid search to find best hyperparameters for SVMs.

        Args:
            df (Pandas Dataframe): Dataframe to be used.
            param_grid (list[dict[str, any]]): Grid of hyperparameters to choose from. Format is available in GridSearchCV's documentation:
                https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            file (str, optional): File to save the results to. In case no file is specified, results will be printed in the console.
        """
        X = df.drop([self.label], axis=1)
        y = df[self.label]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=1)
        model = SVR() if self.problem_type == 'regression' else SVC()

        grid = GridSearchCV(model, param_grid, refit=True, verbose=3, cv=5)
        grid.fit(self.X_train, self.y_train)

        if file != "":
            with open(file, 'a+') as f:
                f.write(str(grid.best_estimator_) + '\n')
        else:
            print(grid.best_params_)
            print(grid.best_estimator_)

    def evaluate(self):
        """
        Evaluates machine learning models and returns the MSE (for regression) and accuracy (for classification).
        """
        if self.problem_type == 'regression':
            return np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        return accuracy_score(self.y_test, self.y_pred)
