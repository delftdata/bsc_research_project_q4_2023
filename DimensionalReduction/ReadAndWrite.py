import csv
import enum
import os

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLassoCV
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn
from sklearn.preprocessing import OneHotEncoder
import matlab.engine
from sklearn import preprocessing
from sklearn import datasets, cluster



class DataToUse(enum.Enum):
    ALL = 1
    TRAIN = 2
    TEST = 3

class DataSet(enum.Enum):
    CIFAR_10 = 1
    BCWD = 2
    FONTS = 3

class DRMethod(enum.Enum):
    PCA = 1
    LDA = 2
    GDA = 3
    LASSO = 4
    NONE = 5
    FA = 6

class ClassificationMethod(enum.Enum):
    LOGISTIC_REGRESSION = 1
    LINEAR_REGRESSION = 2
    SVM = 3
    DECISION_TREE = 4




class Handle:
    def __init__(self):
        self.variance_to_keep = 0.95

    def readAll(self, data_set):
        file_to_read = self.fileToRead(data_set)

        if data_set == DataSet.BCWD:
            df = pd.read_csv(file_to_read)
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_label = train_df['diagnosis']
            train_matrix = train_df.drop(['id', 'diagnosis'], axis=1)
            test_label = test_df['diagnosis']
            test_matrix = test_df.drop(['id', 'diagnosis'], axis=1)

        elif data_set == DataSet.CIFAR_10:
            df = pd.read_csv(file_to_read, dtype=np.uint8)
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_df = train_df.head(100)
            test_df = test_df.head(10)
            train_label = train_df['label']
            train_matrix = train_df.drop('label', axis=1)
            test_label = test_df['label']
            test_matrix = test_df.drop('label', axis=1)

        header = df.columns


        return header, train_matrix, train_label, test_matrix, test_label


    def split(self, data_set):
        if data_set == DataSet.CIFAR_10:
            file_to_read = "data/CIFAR-10/data_train_original.csv"
            df = pd.read_csv(file_to_read, dtype=np.uint8)
            # df = df.iloc[:1000]
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_df.to_csv('data/CIFAR-10/data_train.csv', index=False)
            test_df.to_csv('data/CIFAR-10/data.csv', index=False)

        elif data_set == DataSet.FONTS:
            map_to_read = "data/Fonts/"
            df = []
            for file in os.listdir(map_to_read):
                if file.endswith(".csv") and not file.startswith("data"):
                    file_to_read = map_to_read + file
                    df.append(pd.read_csv(file_to_read))
            df = pd.concat(df)
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_df.to_csv('data/Fonts/data_train.csv', index=False)
            test_df.to_csv('data/Fonts/data.csv', index=False)

        elif data_set == DataSet.BCWD:
            map_to_read = "data/BCWD/"
            df = []
            for file in os.listdir(map_to_read):
                if file.startswith("data_train_or") or file.startswith("data_test_or"):
                    file_to_read = map_to_read + file
                    df.append(pd.read_csv(file_to_read))
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_df.to_csv('data/BCWD/data_train.csv', index=False)
            test_df.to_csv('data/BCWD/data.csv', index=False)



    def fileToRead (self, data_set):
        file_to_read = "../datasets"

        if data_set == DataSet.BCWD:
            file_to_read += "/BreastCancer"
        elif data_set == DataSet.CIFAR_10:
            file_to_read += "/CIFAR-10"
        elif data_set == DataSet.FONTS:
            file_to_read += "/Fonts"

        file_to_read += "/data.csv"
        return file_to_read


    def dimensionalReduce (self, training_data, training_label, test_data, drMethod, scale_lasso=False):
        training_data_result = []
        test_data_result =[]

        if drMethod == DRMethod.NONE:
            return training_data, test_data

        elif drMethod == DRMethod.PCA:
            pca = PCA(n_components= self.variance_to_keep)
            pca.fit(training_data)
            training_data_result = pca.transform(training_data)
            test_data_result = pca.transform(test_data)

        elif drMethod == DRMethod.LDA:
            lda = LinearDiscriminantAnalysis()
            lda.fit(training_data, training_label)
            explained_variances = lda.explained_variance_ratio_

            n_components = 0
            new_total_variance = 0.0
            while new_total_variance < self.variance_to_keep:
                new_total_variance += explained_variances[n_components]
                n_components += 1
            lda.n_components = (n_components -1)

            training_data_result = lda.transform(training_data)
            test_data_result = lda.transform(test_data)

        elif drMethod == DRMethod.LASSO:

            # Initialize the Lasso model and one hot encode classes
            lasso = Lasso(alpha=0.1)
            encoder = OneHotEncoder()
            encode_labels = encoder.fit_transform(training_label.to_numpy().reshape(-1,1)).toarray()
            lasso.fit(training_data, encode_labels)

            list = [None] * ( len(lasso.coef_[0]))
            for i in range (len(lasso.coef_[0])):
                put = 0
                for j in range(len(lasso.coef_)):
                    put += abs(lasso.coef_[j][i])
                list[i] = (put/ len(lasso.coef_))

            training_data_result = pd.DataFrame()
            test_data_result = pd.DataFrame()
            for i, feature_name in enumerate(training_data):
                if list[i] == 0:
                    continue
                else:
                    if scale_lasso == True:
                        new_column_training = training_data[feature_name] * list[i]
                        new_column_test = test_data[feature_name] * list[i]
                    else:
                        new_column_training = training_data[feature_name]
                        new_column_test = test_data[feature_name]
                    training_data_result[feature_name] = new_column_training

                    test_data_result[feature_name] = new_column_test
        elif drMethod == DRMethod.FA:
            agglo = cluster.FeatureAgglomeration(n_clusters=10)
            agglo.fit(training_data, training_label)
            training_data_result = agglo.transform(training_data)
            test_data_result = agglo.transform(test_data)

        elif drMethod == DRMethod.GDA:
            eng = matlab.engine.start_matlab()

            new_test_data = np.transpose(test_data.to_numpy())
            new_training_data = np.transpose(training_data.to_numpy())
            new_training_label =training_label.to_numpy()
            le = preprocessing.LabelEncoder()
            le.fit(new_training_label)
            new_training_label = le.transform(new_training_label)
            new_training_label += 1

            mappedData2 =eng.gda(new_training_data, new_training_data,
                                 new_training_label)
            training_data_result = np.transpose(np.array(mappedData2))
            mappedData2 = eng.gda(new_test_data, new_training_data,
                                 new_training_label)
            test_data_result = np.transpose(np.array(mappedData2))

            eng.quit()

        return training_data_result, test_data_result


    def predict (self, training_data, training_label, test_data, test_label, classificationMethod):
        predicted = []
        if classificationMethod == ClassificationMethod.SVM:
            classifier = sklearn.svm.SVC()
            classifier.fit(training_data, training_label)
            predicted = classifier.predict(test_data)
        if classificationMethod == ClassificationMethod.LOGISTIC_REGRESSION:
            classifier = sklearn.linear_model.LogisticRegression()
            classifier.fit(training_data, training_label)
            predicted = classifier.predict(test_data)
        if classificationMethod == ClassificationMethod.DECISION_TREE:
            classifier = tree.DecisionTreeClassifier()
            classifier.fit(training_data, training_label)
            predicted = classifier.predict(test_data)
        if classificationMethod == ClassificationMethod.LINEAR_REGRESSION:
            classifier = sklearn.linear_model.LinearRegression()
            classifier.fit(training_data, training_label)
            predicted = classifier.predict(test_data)

        right = 0
        wrong = 0

        for prediction, real in zip(predicted, test_label):
            if prediction == real:
                right += 1
            else:
                wrong += 1
        print("right: ", right, "  wrong: ", wrong)

        accuracy = right / (right + wrong)

        return accuracy, predicted

