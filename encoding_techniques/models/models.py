from autogluon.tabular import TabularPredictor
from autogluon.features.generators import IdentityFeatureGenerator
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, r2_score
from sklearn.preprocessing import LabelBinarizer


def writeToFile(file="", content=""):
    print(file)
    if file != "":
        with open(file, 'a+') as f:
            f.write(str(content) + '\n')
    else:
        print(content)


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
        self.problem_type = problem_type

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
                               feature_generator=IdentityFeatureGenerator(), ag_args_fit={'num_gpus': 1})
        else:
            self.predictor.fit(self.df_train, presets='best_quality',
                               hyperparameters=self.hyperparameters, ag_args_fit={'num_gpus': 1})

    def evaluate(self, datasetName, modelName, encoderName, duration, encodeDuration, test_type = 'normal', additional_content = ""):
        """
        Evaluates machine learning models and returns the results of the best performing one.
        """

        runtimeInfo = f'Encode Duration: {encodeDuration}. Total Duration: {duration}\n'
        metrics = self.predictor.evaluate(self.df_test)

        
        
        
        if(self.problem_type == 'binary'):
            score =  round(metrics['accuracy']*100,2)
            auc = round(metrics['roc_auc'],2)
            f1 = round(metrics['f1'],2)
            precision = round(metrics['precision'],2)
            recall = round(metrics['recall'],2)    
            content = f'& {modelName} & {score} & {auc} & {f1} & {precision}/{recall} & {int(duration)} \n'
        elif(self.problem_type == 'multiclass'):
            score =  round(metrics['accuracy']*100,2)
            mcc = round(metrics['mcc'],2)
            f1 = round(metrics['f1'],2)
            precision = round(metrics['precision'],2)
            recall = round(metrics['recall'],2)
            content = f'& {modelName} & {score} & {mcc} & {f1} & {precision}  & {recall} & {int(duration)} \n'
        else:
            rmse = int(abs((metrics['root_mean_squared_error'])))
            mse =  int(abs((metrics['mean_squared_error'])))
            mae = round(metrics['mean_absolute_error'],2)
            r2 = round(metrics['r2'],2)           
            content = f'& {modelName} & {rmse} & {mse}  & {mae} & {r2} & {int(duration)} \n'
            # mse = metrics['mean_squared_error']

        folder = 'results/' if test_type == 'normal' else 'results_combined/'
        
        
        if test_type == 'normal':
            writeToFile(folder  + datasetName + '/' + encoderName + '.txt' , content)
            # writeToFile(folder + datasetName + '/' + encoderName + '/hyperparameters/' + modelName + '-AutoGluon.json',
            #         self.predictor.info())
        elif test_type == 'combinedFinal':
            writeToFile(folder  + datasetName + '/' + 'combined-results.txt' , content)
        else:
            return score

        return self.predictor.evaluate(self.df_test)


class SVMModel():

    def __init__(self,  label: str, problem_type: str, data_preprocessing: bool = False, test_size: float = 0.2, hyperparameters: dict = dict()):
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
        if len(hyperparameters) > 0:
            # self.predictor = SVR() if problem_type == 'regression' else SVC(decision_function_shape='ovo')
            self.predictor = SVR(C=hyperparameters['C'], gamma=hyperparameters['gamma'], kernel=hyperparameters['kernel'], degree=hyperparameters['degree']) if problem_type == 'regression' else SVC(C=hyperparameters['C'], gamma=hyperparameters['gamma'], kernel=hyperparameters['kernel'], degree=hyperparameters['degree'])
        else: 
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

    def grid_search(self, df, datasetName, encoderName, param_grid: list[dict[str, any]]):
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
        
        writeToFile('results/' + datasetName + '/' + encoderName + '/hyperparameters/SVM.json',
        grid.best_estimator_)
        
    def evaluate(self, datasetName, encoderName, duration, encodeDuration, test_type = 'normal'):
        """
        Evaluates machine learning models and returns the MSE (for regression) and accuracy (for classification).
        """
        content = ""
        if self.problem_type == 'regression':
            mse = mean_squared_error(self.y_test, self.y_pred)
            rmse =  np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, self.y_pred)
            r2 = r2_score(self.y_test, self.y_pred)
            content = f'& {encoderName} & SVM &{int(abs(rmse))} & {int(abs(mse))} & {round(mae, 2)}& {round(r2, 2)} & {int(duration)}\n'

        elif self.problem_type == 'binary':
            score =  round(accuracy_score(self.y_test, self.y_pred)*100,2)
            auc = round(roc_auc_score(self.y_test, self.y_pred),2)
            f1 = round(f1_score(self.y_test, self.y_pred),2)
            precision = round(precision_score(self.y_test, self.y_pred),2)
            recall = round(recall_score(self.y_test, self.y_pred),2)
            content = f'& SVM & {encoderName} & {score} & {auc} & {f1} & {precision}/{recall} & {int(duration)} \n'
                
        else:
            score =  round(accuracy_score(self.y_test, self.y_pred)*100,2)
            mcc = round(matthews_corrcoef(self.y_test, self.y_pred),2)
            content = f'& SVM & {encoderName} & {score} & {mcc} &  &  & {int(duration)} \n'

        folder = 'results/' if test_type == 'normal' else 'results_combined/'

        if test_type == 'normal':
            writeToFile(folder  + datasetName + '/' + encoderName + '.txt', content)
            # writeToFile(folder + datasetName + '/' + encoderName + '/hyperparameters/' + 'svm' + '-AutoGluon.json',
            #         self.predictor.info())
        elif test_type == 'combinedFinal':
            writeToFile(folder  + datasetName + '/' + 'combined-results.txt' , content)
        else:
            return score
        

def multiclass_roc_auc_score(truth, pred, average="macro"):

    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return roc_auc_score(truth, pred, average=average)