import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score


class SVMModel:
    def __init__(self,  label: str, problem_type: str,
                 data_preprocessing: bool = False, test_size: float = 0.2):
        """_summary_

        Args:
            label (str):  Label of the target column.
            problem_type (str): Type of problem: 'classification' (uses SVC) or 'regression'(uses SVR).
            data_preprocessing (bool, optional): Whether to do automatic data preprocessing or not.
                                                 Defaults to False. Feature is not implemented.
            test_size (float, optional): _description_. Percentage (between 0 and 1) of the
                                         dataset size to allocate for testing. Defaults to 0.2.
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

    def fit(self, dataframe):
        """
        Fits machine learning models.

        Args:
            dataframe (Pandas Dataframe): Dataframe to train and test the models on.
        """
        X = dataframe.drop([self.label], axis=1)
        y = dataframe[self.label]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=1)

        self.predictor.fit(self.X_train, self.y_train)
        self.y_pred = self.predictor.predict(self.X_test)

    def grid_search(self, dataframe, param_grid: list[dict[str, any]], file=""):
        """
        Performs grid search to find the best hyperparameters for SVMs.

        Args:
            dataframe (Pandas Dataframe): Dataframe to be used.
            param_grid (list[dict[str, any]]): Grid of hyperparameters to choose from.
                                               Format is available in GridSearchCV's documentation:
                                               https://scikit-learn.org/stable/modules/generated/
                                               sklearn.model_selection.GridSearchCV.html
            file (str, optional): File to save the results to.
                                  In case no file is specified, results will be printed in the console.
        """
        X = dataframe.drop([self.label], axis=1)
        y = dataframe[self.label]
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
        Evaluates machine learning models and returns the MSE (for regression) and
        accuracy (for classification).
        """
        if self.problem_type == 'regression':
            return np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        return accuracy_score(self.y_test, self.y_pred)
