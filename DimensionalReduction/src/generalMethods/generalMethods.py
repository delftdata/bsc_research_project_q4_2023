import enum
import os

from sklearn.linear_model import Lasso

import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn
from sklearn.preprocessing import OneHotEncoder
import matlab.engine
# import oct2py
from sklearn import preprocessing
from sklearn import datasets, cluster
import time
from sklearn.metrics.pairwise import pairwise_kernels
from DimensionalReduction.src.generalMethods.gda2 import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis




class DataToUse(enum.Enum):
    ALL = 1
    TRAIN = 2
    TEST = 3

class DataSet(enum.Enum):
    CIFAR_10 = 1
    BCWD = 2
    FONTS = 3
    CROPS = 4
    MNIST = 5

class DRMethod(enum.Enum):
    PCA = 1
    LDA = 2
    GDA = 3
    LASSO = 4
    NONE = 5
    FA = 6
    QDA =7

class ClassificationMethod(enum.Enum):
    LOGISTIC_REGRESSION = 1
    LINEAR_REGRESSION = 2
    SVM = 3
    DECISION_TREE = 4




class Handle:
    def __init__(self):
        self.variance_to_keep = 0.95
        self.alpha = 0.01
        self.numberOfClasses = 10

    def readSplit(self, data_set):
        file_to_read = self.fileToRead(data_set)
        df = None

        if data_set == DataSet.BCWD:
            df = pd.read_csv(file_to_read)
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_label = train_df['diagnosis']
            train_matrix = train_df.drop(['id', 'diagnosis'], axis=1)
            test_label = test_df['diagnosis']
            test_matrix = test_df.drop(['id', 'diagnosis'], axis=1)

        elif data_set == DataSet.CIFAR_10:
            df = pd.read_csv(file_to_read, dtype=np.uint8)
            df = df.sample(n=10000, random_state=42)
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_label = train_df['label']
            train_matrix = train_df.drop('label', axis=1)
            test_label = test_df['label']
            test_matrix = test_df.drop('label', axis=1)

        elif data_set == DataSet.MNIST:
            df = pd.read_csv(file_to_read)
            df = df.sample(n=10000, random_state=42)
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_label = train_df['label']
            train_matrix = train_df.drop(['label'], axis=1)
            test_label = test_df['label']
            test_matrix = test_df.drop(['label'], axis=1)


        elif data_set == DataSet.CROPS:
            df = pd.read_csv(file_to_read)
            df = df.sample(n=10000, random_state=42)
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_label = train_df['label']
            train_matrix = train_df.drop(['label'], axis=1)
            test_label = test_df['label']
            test_matrix = test_df.drop(['label'], axis=1)


        elif data_set == DataSet.FONTS:
            map_to_read = "../../datasets/CharacterFontImages/"
            df = []
            print("number of fonts: ", self.numberOfClasses)
            for i, file in enumerate(os.listdir(map_to_read)):
                if file.endswith(".csv") and i < self.numberOfClasses:
                    print(file)
                    file_to_read = map_to_read + file
                    df.append(pd.read_csv(file_to_read))
            df = pd.concat(df)
            # nnn = self.numberOfClasses *1000
            nnn = 1900
            print(nnn)
            df = df.sample(n=nnn, random_state=42)
            # print(df)
            # print(df.shape)
            train_df, test_df = train_test_split(df, test_size=0.526, random_state=42)
            train_label = train_df['font']
            train_matrix = train_df.drop(['font', 'fontVariant'], axis=1)
            test_label = test_df['font']
            test_matrix = test_df.drop(['font', 'fontVariant'], axis=1)
            self.numberOfClasses += 10




        header = df.columns
        print("done reading data with shape: ", df.shape)


        return header, train_matrix, train_label, test_matrix, test_label

    def readAll(self, data_set):
        file_to_read = self.fileToRead(data_set)

        if data_set == DataSet.BCWD:
            df = pd.read_csv(file_to_read)
            df.rename({"diagnosis": "label"})
            # print(df)
            label = df['label']
            matrix = df.drop(['id', 'label'], axis=1)

        elif data_set == DataSet.CIFAR_10:
            df = pd.read_csv(file_to_read, dtype=np.uint8)
            label = df['label']
            matrix = df.drop('label', axis=1)

        elif data_set == DataSet.MNIST:
            df = pd.read_csv(file_to_read, dtype=np.uint8)
            label = df['label']
            matrix = df.drop('label', axis=1)

        header = df.columns

        return header, matrix, label


    def split(self, data_set):
        if data_set == DataSet.CIFAR_10:
            file_to_read = "data/CIFAR-10/data_train_original.csv"
            df = pd.read_csv(file_to_read, dtype=np.uint8)
            # df = df.iloc[:1000]
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_df.to_csv('data/CIFAR-10/data_train.csv', index=False)
            test_df.to_csv('data/CIFAR-10/breast_cancer.csv', index=False)

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
            test_df.to_csv('data/Fonts/breast_cancer.csv', index=False)

        elif data_set == DataSet.BCWD:
            map_to_read = "data/BCWD/"
            df = []
            for file in os.listdir(map_to_read):
                if file.startswith("data_train_or") or file.startswith("data_test_or"):
                    file_to_read = map_to_read + file
                    df.append(pd.read_csv(file_to_read))
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_df.to_csv('data/BCWD/data_train.csv', index=False)
            test_df.to_csv('data/BCWD/breast_cancer.csv', index=False)


    def print_files_in_path(self, path):
        files = os.listdir(path)
        for file in files:
            print(file)

    def fileToRead (self, data_set):
        file_to_read = "../../datasets"

        # directory_path = '/datasets'
        # self.print_files_in_path(directory_path)
        # print("_________________________________________")

        if data_set == DataSet.BCWD:
            file_to_read += "/breast-cancer"
        elif data_set == DataSet.CIFAR_10:
            file_to_read += "/CIFAR-10"
        elif data_set == DataSet.FONTS:
            file_to_read += "/Fonts"
        elif data_set == DataSet.CROPS:
            file_to_read += "/crops"
        elif data_set == DataSet.MNIST:
            file_to_read += "/MNIST"

        file_to_read += "/breast_cancer.csv"
        return file_to_read



    def dimensionalReduce (self, training_data, training_label, test_data, drMethod, scale_lasso=False, var: float = 0.95, alpha = 0.01):
        training_data_result = []
        test_data_result =[]
        self.variance_to_keep = var
        self.alpha = alpha

        if drMethod == DRMethod.NONE:
            start_time = 0
            end_time = 0
            training_data_result = training_data
            test_data_result = test_data

        elif drMethod == DRMethod.PCA:
            pca = PCA(n_components= self.variance_to_keep)
            start_time = time.time()
            pca.fit(training_data)
            training_data_result = pca.transform(training_data)
            end_time = time.time()
            test_data_result = pca.transform(test_data)


        elif drMethod == DRMethod.LDA:
            lda = LinearDiscriminantAnalysis(n_components=var)
            start_time = time.time()
            try:
                lda.fit(training_data, training_label)
            except:
                ("number of components to big")
            # explained_variances = lda.explained_variance_ratio_
            #
            # n_components = 0
            # new_total_variance = 0.0
            # while new_total_variance < self.variance_to_keep:
            #     new_total_variance += explained_variances[n_components]
            #     n_components += 1
            # lda.n_components = (n_components -1)

            training_data_result = lda.transform(training_data)
            end_time = time.time()
            test_data_result = lda.transform(test_data)




        elif drMethod == DRMethod.LASSO:
            # Initialize the Lasso model and one hot encode classes
            lasso = Lasso(alpha=self.alpha)
            encoder = OneHotEncoder()
            encode_labels = encoder.fit_transform(training_label.to_numpy().reshape(-1,1)).toarray()
            start_time = time.time()
            lasso.fit(training_data, encode_labels)

            list = [None] * ( len(lasso.coef_[0]))
            for i in range (len(lasso.coef_[0])):
                put = 0
                for j in range(len(lasso.coef_)):
                    put += abs(lasso.coef_[j][i])
                list[i] = (put/ len(lasso.coef_))
            training_data_result = pd.DataFrame()
            test_data_result = pd.DataFrame()
            new_columns_training = []
            new_columns_test = []

            for i, feature_name in enumerate(training_data):
                if list[i] == 0:
                    continue
                else:
                    if scale_lasso == True:
                        new_columns_training.append(training_data[feature_name] * list[i])
                        new_columns_test.append(test_data[feature_name] * list[i])
                    else:
                        new_columns_training.append(training_data[feature_name])
                        new_columns_test.append(test_data[feature_name])
            try:
                training_data_result = pd.concat(new_columns_training, axis=1)
                test_data_result = pd.concat(new_columns_test, axis=1)
            except:
                print("alpha to big")

            end_time = time.time()

        elif drMethod == DRMethod.FA:
            agglo = cluster.FeatureAgglomeration(n_clusters=10)
            start_time = time.time()
            agglo.fit(training_data, training_label)
            training_data_result = agglo.transform(training_data)
            test_data_result = agglo.transform(test_data)
            end_time = time.time()

        elif drMethod == DRMethod.GDA:
            eng = matlab.engine.start_matlab()
            # folder_path = '.\src'
            # eng.cd(folder_path, nargout=0)

            new_test_data = np.transpose(test_data.to_numpy())
            new_training_data = np.transpose(training_data.to_numpy())
            new_training_label = training_label.to_numpy()
            le = preprocessing.LabelEncoder()
            le.fit(new_training_label)
            new_training_label = le.transform(new_training_label)
            new_training_label += 1
            # print("new train: ", new_training_data)
            # print(type(new_training_data))
            # print("new label: ", new_training_label)
            # print(type(new_training_label))
            # print("new test: ", new_test_data)
            # print(type(new_test_data))

            #
            start_time = time.time()
            gda = GDA(n_components=1, kernel="poly", kernel_params={"degree": 3})
            training_data = training_data.to_numpy()  # Convert DataFrame to numpy array
            training_label = training_label.to_numpy().ravel()  # Convert Series to numpy array

            # mappedData_train = gda.fit_transform(training_data, training_label)
            # mappedData_test = gda.transform(test_data)
            # mappedData_train = self.gda(new_training_data, new_training_data, new_training_label, var)
            # mappedData_test = self.gda(new_test_data, new_training_data, new_training_label, var)
            try:
                mappedData_train = eng.gda(new_training_data, new_training_data,
                                     new_training_label, var)
                mappedData_test = eng.gda(new_test_data, new_training_data,
                                     new_training_label, var)
            except:
                print("numebr of components to big")
            # mappedData_train = oc.gda(new_training_data, new_training_label, new_training_label)
            # mappedData_test = oc.gda( new_test_data, new_training_data, new_training_label)
            end_time = time.time()
            #
            training_data_result = np.transpose(np.array(mappedData_train))
            test_data_result = np.transpose(np.array(mappedData_test))
            # oc.exit()
            eng.quit()

        elif drMethod == DRMethod.QDA:
            start_time = time.time()
            gda = QuadraticDiscriminantAnalysis(kernel='poly')
            gda.fit(training_data, training_label)
            training_data_result = gda.transform(training_data)
            end_time = time.time()
            test_data_result = gda.transform(test_data)



        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_to_transform = ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        return training_data_result, test_data_result, time_to_transform


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
        # print("right: ", right, "  wrong: ", wrong)

        accuracy = right / (right + wrong)

        return accuracy, predicted

    def enumToName(self, enum):
        if enum == DRMethod.PCA:
            return "PCA"
        elif enum == DRMethod.LDA:
            return "LDA"
        elif enum == DRMethod.GDA:
            return "GDA"
        elif enum == DRMethod.LASSO:
            return "Lasso"
        elif enum == DRMethod.NONE:
            return "Default"
        else:
            return "unknown enum name"


    # def gda(self, data, trainData, trainLabel, nDim=None, options=None):
    #     # Check dimensions
    #     print(data.shape)
    #     print(trainData.shape)
    #     trainLabel = trainLabel.reshape(1, -1)
    #     print(trainLabel.shape)
    #     if data.shape[0] != trainData.shape[0]:
    #         raise ValueError('DATA and TRAINDATA must be in the same space with equal dimensions.')
    #     if trainData.shape[1] != trainLabel.shape[1]:
    #         raise ValueError('The length of the TRAINLABEL must be equal to the number of columns in TRAINDATA.')
    #
    #     # Set default options if not provided
    #     if options is None:
    #         options = {'KernelType': 'poly'}
    #
    #     # Separate samples of each class in a list
    #     c = np.max(trainLabel)
    #     dataCell = [trainData[:, np.where(trainLabel[0] == i)[0]] for i in range(1, c+1)]
    #
    #     # Create class-specific kernel for the training data
    #     kTrainCell = []
    #     for p in range(c):
    #         print("p = ",p)
    #         for q in range(c):
    #             classP = dataCell[p]
    #             classQ = dataCell[q]
    #             Kpq = pairwise_kernels(classP.T, classQ.T, metric='poly')
    #             print(Kpq.shape)
    #             kTrainCell.append(Kpq)
    #     kTrain = np.concatenate(kTrainCell, axis=1)
    #
    #     # Make data have zero mean
    #     n = trainData.shape[1]
    #     One = np.ones((n, n)) / n
    #     zeroMeanKtrain = kTrain - One @ kTrain - kTrain @ One + One @ kTrain @ One
    #
    #     # Create the block-diagonal W matrix
    #     wTrainCell = []
    #     for p in range(c):
    #         for q in range(c):
    #             if p == q:
    #                 wTrainCell.append(np.ones((dataCell[p].shape[1], dataCell[q].shape[1])) / dataCell[p].shape[1])
    #             else:
    #                 wTrainCell.append(np.zeros((dataCell[p].shape[1], dataCell[q].shape[1])))
    #     wTrain = np.concatenate(wTrainCell, axis=1)
    #
    #     # Decompose zeroMeanKtrain using eigen-decomposition
    #     gamma, P = np.linalg.eig(zeroMeanKtrain)
    #     index = np.argsort(gamma)[::-1]
    #     gamma = gamma[index]
    #     P = P[:, index]
    #
    #     # Remove eigenvalues with relatively small value
    #     maxEigVal = np.max(np.abs(gamma))
    #     zeroEigIndex = np.where(np.abs(gamma) / maxEigVal < 1e-6)[0]
    #     gamma = np.delete(gamma, zeroEigIndex)
    #     P = np.delete(P, zeroEigIndex, axis=1)
    #
    #     # Normalize eigenvectors
    #     nEig = len(gamma)
    #     for i in range(nEig):
    #         P[:, i] /= np.linalg.norm(P[:, i])
    #
    #     # Compute eigenvectors (beta) and eigenvalues (lambda)
    #     BB = P.T @ wTrain @ P
    #     lambda_, beta = np.linalg.eig(BB)
    #     index = np.argsort(lambda_)[::-1]
    #     lambda_ = lambda_[index]
    #     beta = beta[:, index]

        # Remove eigenvalues with relatively small value
        maxEigVal = np.max(np.abs(lambda_))
        zeroEigIndex = np.where(np.abs(lambda_) / maxEigVal < 1e-6)[0]
        lambda_ = np.delete(lambda_, zeroEigIndex)
        beta = np.delete(beta, zeroEigIndex, axis=1)

        # Compute eigenvectors (alpha) and normalize them
        gamma = np.diag(gamma)
        alpha = P @ np.linalg.inv(gamma) @ beta
        nEig = len(lambda_)
        for i in range(nEig):
            scalar = np.sqrt((alpha[:, i].T @ zeroMeanKtrain @ alpha[:, i]))
            alpha[:, i] /= scalar

        # Dimensionality reduction (if nDim is not given, nEig dimensions are retained)
        if nDim is None:
            nDim = nEig
        elif nDim > nEig:
            print(f'Target dimensionality reduced to {nEig}.')
        w = alpha[:, :nDim]  # Projection matrix

        # Create class-specific kernel for all data points
        kDataCell = [pairwise_kernels(data.T, classP.T, metric="poly") for classP in dataCell]
        kData = np.concatenate(kDataCell, axis=1)

        # Make data zero mean
        nPrime = data.shape[1]
        Oneprime = np.ones((n, nPrime)) / n
        zeroMeanKdata = kData - kTrain @ Oneprime - One @ kData + One @ kTrain @ Oneprime

        # Project all data points non-linearly onto a new lower-dimensional subspace (w)
        mappedData = w.T @ zeroMeanKdata

        return mappedData
