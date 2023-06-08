
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


from .generalMethods.generalMethods import *
from .models.models import *

def compare_models(drMethods, algorithmsToTest, matrix_train, label_train, matrix_test, label_test):

    numberOfFeatures = {}
    accuracyOfMethods = {}
    handle = Handle()

    for i, x in enumerate(drMethods):
        new_matrix_train, new_matrix_test, time_to_transform = handle.dimensionalReduce(matrix_train, label_train, matrix_test, x)

        # Refactor data for autogluon
        new_label_train = np.reshape(np.array(label_train), (-1,1))
        new_label_test = np.reshape(np.array(label_test), (-1, 1))
        df_train = pd.DataFrame(new_matrix_train)
        df_test = pd.DataFrame(new_matrix_test)
        df_train["label"] = new_label_train
        df_test["label"] = new_label_test

        print("_________________________________________________________________________________________")
        print("deminsional reduction: ", x)
        print("time to transform: ", time_to_transform)
        print("number of features left: ", df_train.shape[1] -1)
        numberOfFeatures[handle.enumToName(x)] = df_train.shape[1] -1


        for hyper in algorithmsToTest:
            predictor = TabularPredictor(label='label', problem_type='multiclass', verbosity=0)
            predictor.fit(train_data=df_train, hyperparameters=hyper)
            performance = predictor.evaluate(df_test)
            print("results: ", hyper, ": ", performance)
            accuracyOfMethods[handle.enumToName(x)] = performance.get("accuracy")

    plt.bar(accuracyOfMethods.keys(), accuracyOfMethods.values(), width=0.4 )
    plt.show()

def plotModelParameter (drMethod, algorithmToTest, matrix_train, label_train, matrix_test, label_test):
    handle = Handle()
    varValues = []
    varResults = []
    numberOfFeatures = []

    if drMethod == DRMethod.PCA:
        var = 0.9

        while var <= 1:
            new_matrix_train, new_matrix_test, time_to_transform = handle.dimensionalReduce(matrix_train, label_train,
                                                                                            matrix_test, drMethod, var=var)

            # Refactor data for autogluon
            new_label_train = np.reshape(np.array(label_train), (-1, 1))
            new_label_test = np.reshape(np.array(label_test), (-1, 1))
            df_train = pd.DataFrame(new_matrix_train)
            df_test = pd.DataFrame(new_matrix_test)
            df_train['label'] = new_label_train
            df_test['label'] = new_label_test

            predictor = TabularPredictor(label='label', problem_type='multiclass', verbosity=0)
            predictor.fit(train_data=df_train, hyperparameters=algorithmToTest)
            performance = predictor.evaluate(df_test)
            print("results: ", var, ": ", performance)
            print(performance.get("accuracy"))
            varValues.append(var)
            varResults.append(performance.get("accuracy"))
            numberOfFeatures.append(df_train.shape[1] - 1)

            if var == 0.95:
                var = 0.99
            else:
                var += 0.05


        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle(('accuracy vs #features ', handle.enumToName(drMethod)))
        # ax1.plot(varValues, varResults)
        # ax2.plot(varValues, numberOfFeatures)
        # plt.show()

    if drMethod == DRMethod.LDA:
        var = 1

        while True:
            try:
                new_matrix_train, new_matrix_test, time_to_transform = handle.dimensionalReduce(matrix_train, label_train,
                                                                                                matrix_test, drMethod,
                                                                                                var=var)

                # Refactor data for autogluon
                new_label_train = np.reshape(np.array(label_train), (-1, 1))
                new_label_test = np.reshape(np.array(label_test), (-1, 1))
                df_train = pd.DataFrame(new_matrix_train)
                df_test = pd.DataFrame(new_matrix_test)
                df_train['label'] = new_label_train
                df_test['label'] = new_label_test

                predictor = TabularPredictor(label='label', problem_type='multiclass', verbosity=0)
                predictor.fit(train_data=df_train, hyperparameters=algorithmToTest)
                performance = predictor.evaluate(df_test)
                # print("results: ", var, ": ", performance)
                # print(performance.get("accuracy"))
                varValues.append(var)
                varResults.append(performance.get("accuracy"))
                numberOfFeatures.append(df_train.shape[1] - 1)

                var += 1
            except:
                break


    elif drMethod == DRMethod.LASSO:
        alpha = 0.01
        varValues = []
        varResults = []
        numberOfFeatures = []
        while alpha <= 0.5:
            new_matrix_train, new_matrix_test, time_to_transform = handle.dimensionalReduce(matrix_train, label_train,
                                                                                            matrix_test, drMethod,
                                                                                            alpha=alpha)

            # Refactor data for autogluon
            try:
                new_label_train = np.reshape(np.array(label_train), (-1, 1))
                new_label_test = np.reshape(np.array(label_test), (-1, 1))
                df_train = pd.DataFrame(new_matrix_train)
                df_test = pd.DataFrame(new_matrix_test)
                df_train["label"] = new_label_train
                df_test["label"] = new_label_test

                predictor = TabularPredictor(label='label', problem_type='binary', verbosity=0)
                predictor.fit(train_data=df_train, hyperparameters=algorithmToTest)
                performance = predictor.evaluate(df_test)
                # print("results: ", alpha, ": ", performance)
                # print(performance.get("accuracy"))
                varValues.append(alpha)
                varResults.append(performance.get("accuracy"))
                numberOfFeatures.append(df_train.shape[1] - 1)

                alpha += 0.01
            except:
                break

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle(('accuracy vs #features', handle.enumToName(drMethod)))
        # ax1.plot(alphaValues, alphaResults)
        # ax2.plot(alphaValues, numberOfFeatures)
        # plt.show()

    print("dr method: ", drMethod, " |classification methods: ", algorithmToTest)
    for i in range(0, len(varValues)):
        print("var value: ", varValues[i], " |accuracy: ", varResults[i], " |number of features: ", numberOfFeatures[i])


if __name__ == '__main__':
    handle = Handle()
    header, matrix_train, label_train, matrix_test, label_test = handle.readSplit(DataSet.FONTS)

    scaler = preprocessing.StandardScaler().fit(matrix_train)
    matrix_train = pd.DataFrame(scaler.transform(matrix_train))
    matrix_test = pd.DataFrame(scaler.transform(matrix_test))

    le = preprocessing.LabelEncoder()
    le.fit(label_train)
    label_train = pd.DataFrame(le.transform(label_train))
    label_train += 1
    label_test = pd.DataFrame(le.transform(label_test))
    label_test += 1

    # matrix_test = (matrix_test-matrix_test.mean())/matrix_test.std()
    # matrix_train = (matrix_train-matrix_train.mean())/matrix_train.std()

    # drMethods = [DRMethod.PCA, DRMethod.GDA, DRMethod.LDA, DRMethod.LASSO, DRMethod.NONE]
    # hyperparameters = [{'LR': {}}]
    # compare_models(drMethods, hyperparameters, matrix_train, label_train, matrix_test, label_test)
    plotModelParameter(DRMethod.PCA, {'LR': {}}, matrix_train, label_train, matrix_test, label_test)
    plotModelParameter(DRMethod.PCA, {'RF': {}}, matrix_train, label_train, matrix_test, label_test)
    plotModelParameter(DRMethod.LDA, {'LR': {}}, matrix_train, label_train, matrix_test, label_test)
    plotModelParameter(DRMethod.LDA, {'RF': {}}, matrix_train, label_train, matrix_test, label_test)
    plotModelParameter(DRMethod.LASSO, {'LR': {}}, matrix_train, label_train, matrix_test, label_test)
    plotModelParameter(DRMethod.LASSO, {'RF': {}}, matrix_train, label_train, matrix_test, label_test)



# for hyperparameter in hyperparameters:
#     # au = AutogluonModel(label="label", problem_type="binary", hyperparameters=hyperparameter)
#     au = AutogluonModel(label="label", problem_type="binary", hyperparameters=hyperparameter)
#     au.fit_split_data(df_train, df_test)
#     results = au.evaluate()
#     print("results: ", hyperparameter, ": ", results)


# accuracy_logistic, predicted_logistic = handle.predict(new_matrix_train, label_train, new_matrix_test, label_test,
#                                                        ClassificationMethod.LOGISTIC_REGRESSION)
# print("accuracy logistic regression: ", accuracy_logistic)
# accuracy_tree, predicted_tree = handle.predict(new_matrix_train, label_train, new_matrix_test, label_test,
#                                                ClassificationMethod.DECISION_TREE)
# print("accuracy decision tree: ", accuracy_tree)
# accuracy_svm, predicted_svm = handle.predict(new_matrix_train, label_train, new_matrix_test, label_test,
#                                              ClassificationMethod.SVM)
# print("accuracy SVM: ", accuracy_svm)
