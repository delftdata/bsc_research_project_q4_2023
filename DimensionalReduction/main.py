import csv
from ReadAndWrite import *
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset
import matlab.engine

from numba import jit, cuda

if __name__ == '__main__':
    handle = Handle()
    # handle.split(DataSet.CIFAR_10)
    header_train, matrix_train, label_train = handle.readAll(DataSet.BCWD, DataToUse.TRAIN)
    header_test, matrix_test, label_test = handle.readAll(DataSet.BCWD, DataToUse.TEST)

    matrix_test = (matrix_test-matrix_test.mean())/matrix_test.std()
    matrix_train = (matrix_train-matrix_train.mean())/matrix_train.std()



    list = [DRMethod.GDA]

    for x in list:
        new_matrix_train, new_matrix_test = handle.dimensionalReduce(matrix_train, label_train, matrix_test, x)
        print("matrix train : ", new_matrix_train.shape)

        print("_________________________________________________________________________________________")
        print("deminsional reduction: ", x)
        accuracy_logistic, predicted_logistic = handle.predict(new_matrix_train, label_train, new_matrix_test, label_test,
                                                               ClassificationMethod.LOGISTIC_REGRESSION)
        print("accuracy logistic regression: ", accuracy_logistic)
        accuracy_tree, predicted_tree = handle.predict(new_matrix_train, label_train, new_matrix_test, label_test,
                                                       ClassificationMethod.DECISION_TREE)
        print("accuracy decision tree: ", accuracy_tree)
        accuracy_svm, predicted_svm = handle.predict(new_matrix_train, label_train, new_matrix_test, label_test,
                                                     ClassificationMethod.SVM)
        print("accuracy SVM: ", accuracy_svm)












    # header_train, matrix_train, label_train = handle.readAll(DataSet.CIFAR_10, DataToUse.TRAIN)
    # print(matrix_train)
    # print(label_train)
    # header_test, matrix_test, label_test = handle.readAll(DataSet.CIFAR_10, DataToUse.TEST)

    # print(header)
    # print("______________________________________________")
    # print(id_diagnosis)
    # print("______________________________________________")
    # print(matrix)
    # training_pca, test_pca = handle.dimensionalReduce(matrix_train, matrix_test, DRMethod.PCA)

    # accuracy_base_svm, predicted_base_svm = handle.predict(matrix_train, label_train, matrix_test, label_test, ClassificationMethod.SVM)
    # print("accuracy svm: " , accuracy_base_svm)
    # accuracy_base_lr, predicted_base_lr = handle.predict(matrix_train, label_train, matrix_test, label_test,
    #                                                        ClassificationMethod.LOGISTIC_REGRESSION)
    # print("accuracy logistic regresion: " , accuracy_base_lr)
    # accuracy_base_dt, predicted_base_dt = handle.predict(matrix_train, label_train, matrix_test, label_test,
    #                                                        ClassificationMethod.DECISION_TREE)
    # print("accuracy desision tree: ", + accuracy_base_dt)
    # accuracy_pca, predicted_pca = handle.predict(training_pca, label_train, test_pca, label_test,
    #                                                ClassificationMethod.SVM)
    # print("accuracy base: ", accuracy_base)
    # print("accuracy pca: " , accuracy_pca)
