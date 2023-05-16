from ReadAndWrite import *


if __name__ == '__main__':
    handle = Handle()
    # handle.split(DataSet.CIFAR_10)
    header, matrix_train, label_train, matrix_test, label_test = handle.readAll(DataSet.BCWD)

    matrix_test = (matrix_test-matrix_test.mean())/matrix_test.std()
    matrix_train = (matrix_train-matrix_train.mean())/matrix_train.std()

    drMethods = [DRMethod.PCA, DRMethod.LDA, DRMethod.GDA, DRMethod.LASSO, DRMethod.FA, DRMethod.NONE]

    for x in drMethods:
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


