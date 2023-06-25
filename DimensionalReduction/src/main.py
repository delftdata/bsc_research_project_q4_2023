
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics


from generalMethods.generalMethods import *
from models.models import *

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
        var = 0.05

        while var <= 1:
            new_matrix_train, new_matrix_test, time_to_transform = handle.dimensionalReduce(matrix_train, label_train,
                                                                                            matrix_test, drMethod, var=var)
            print("time to  transform: ", time_to_transform)
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
                print("time to  transform: ", time_to_transform)
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

                print("var: ", var, " |performance: ", performance)

                var += 1
            except:
                break

    if drMethod == DRMethod.GDA:
        var = 1

        while True:
            try:
                new_matrix_train, new_matrix_test, time_to_transform = handle.dimensionalReduce(matrix_train, label_train,
                                                                                                matrix_test, drMethod,
                                                                                                var=var)
                print("time to  transform: ", time_to_transform)
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

                print("var: ", var, " |performance: ", performance)

                var += 1
            except:
                break


    elif drMethod == DRMethod.LASSO:
        alpha = 0.01
        varValues = []
        varResults = []
        numberOfFeatures = []
        while alpha <= 0.4:
            new_matrix_train, new_matrix_test, time_to_transform = handle.dimensionalReduce(matrix_train, label_train,
                                                                                            matrix_test, drMethod,
                                                                                            alpha=alpha)
            print("time to  transform: ", time_to_transform)
            # Refactor data for autogluon
            try:
            # if True:

                new_label_train = np.reshape(np.array(label_train), (-1, 1))
                new_label_test = np.reshape(np.array(label_test), (-1, 1))
                df_train = pd.DataFrame(new_matrix_train)
                df_test = pd.DataFrame(new_matrix_test)
                df_train["label"] = new_label_train
                df_test["label"] = new_label_test

                predictor = TabularPredictor(label='label', problem_type='multiclass', verbosity=0)
                predictor.fit(train_data=df_train, hyperparameters=algorithmToTest)
                performance = predictor.evaluate(df_test)
                # print("results: ", alpha, ": ", performance)
                # print(performance.get("accuracy"))
                varValues.append(alpha)
                varResults.append(performance.get("accuracy"))
                numberOfFeatures.append(df_train.shape[1] - 1)

                print("var: ", alpha, " |performance: ", performance)

                alpha += 0.01
                print("alpha NOT to big")
            except:
                print("alpha is to big or other error")
                break

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle(('accuracy vs #features', handle.enumToName(drMethod)))
        # ax1.plot(alphaValues, alphaResults)
        # ax2.plot(alphaValues, numberOfFeatures)
        # plt.show()

    print("dr method: ", drMethod, " |classification methods: ", algorithmToTest)
    for i in range(0, len(varValues)):
        print("var value: ", varValues[i], " |accuracy: ", varResults[i], " |number of features: ", numberOfFeatures[i])


def plotModelParameterSVM (drMethod, algorithmToTest, matrix_train, label_train, matrix_test, label_test):
    handle = Handle()
    varValues = []
    varResults = []
    numberOfFeatures = []

    if drMethod == DRMethod.PCA:
        var = 0.05

        while var < 1:
            new_matrix_train, new_matrix_test, time_to_transform = handle.dimensionalReduce(matrix_train, label_train,
                                                                                            matrix_test, drMethod, var=var)

            print("step 1")
            print(new_matrix_train.shape)
            print(new_matrix_test.shape)
            # Refactor data for autogluon
            # new_label_train = np.reshape(np.array(label_train), (-1, 1))
            # new_label_test = np.reshape(np.array(label_test), (-1, 1))
            # df_train = pd.DataFrame(new_matrix_train)
            # df_test = pd.DataFrame(new_matrix_test)
            # df_train['label'] = new_label_train
            # df_test['label'] = new_label_test

            # predictor = svm.SVC(kernel='linear')
            # predictor.fit(new_matrix_train, label_train)
            # predicted = predictor.predict(new_matrix_test)
            # accuracy = metrics.accuracy_score(label_test, predicted)
            # print("results: ", alpha, ": ", accuracy)
            # print(accuracy)
            # varValues.append(alpha)
            # varResults.append(accuracy)
            # numberOfFeatures.append(new_matrix_test.shape[1])

            predictor = svm.SVC(kernel='linear', max_iter=10000)
            predictor.fit(new_matrix_train, label_train)
            print("step 1.5")
            predicted = predictor.predict(new_matrix_test)
            accuracy = metrics.accuracy_score(label_test, predicted)
            print("step 2")
            print("results: ", var, ": ", accuracy)
            print(accuracy)
            varValues.append(var)
            varResults.append(accuracy)
            numberOfFeatures.append(new_matrix_test.shape[1] )

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

                predictor = svm.SVC(kernel='linear',max_iter=10000)
                predictor.fit(new_matrix_train, label_train)
                predicted = predictor.predict(new_matrix_test)
                accuracy = metrics.accuracy_score(label_test, predicted)
                print("results: ", var, ": ", accuracy)
                print(accuracy)
                varValues.append(var)
                varResults.append(accuracy)
                numberOfFeatures.append(new_matrix_test.shape[1])

                var += 1
            except:
                break

    if drMethod == DRMethod.GDA:
        var = 1

        while True:
            try:
                new_matrix_train, new_matrix_test, time_to_transform = handle.dimensionalReduce(matrix_train, label_train,
                                                                                                matrix_test, drMethod,
                                                                                                var=var)

                predictor = svm.SVC(kernel='linear',max_iter=10000)
                predictor.fit(new_matrix_train, label_train)
                predicted = predictor.predict(new_matrix_test)
                accuracy = metrics.accuracy_score(label_test, predicted)
                print("results: ", var, ": ", accuracy)
                print(accuracy)
                varValues.append(var)
                varResults.append(accuracy)
                numberOfFeatures.append(new_matrix_test.shape[1])

                var += 1
            except:
                break


    elif drMethod == DRMethod.LASSO:
        alpha = 0.01
        varValues = []
        varResults = []
        numberOfFeatures = []
        while alpha <= 0.4:
            new_matrix_train, new_matrix_test, time_to_transform = handle.dimensionalReduce(matrix_train, label_train,
                                                                                            matrix_test, drMethod,
                                                                                            alpha=alpha)
            print(new_matrix_train.shape)
            try:

                predictor = svm.SVC(kernel='linear', max_iter=10000)
                predictor.fit(new_matrix_train, label_train)
                predicted = predictor.predict(new_matrix_test)
                accuracy = metrics.accuracy_score(label_test, predicted)
                print("results: ", alpha, ": ", accuracy)
                print(accuracy)
                varValues.append(alpha)
                varResults.append(accuracy)
                numberOfFeatures.append(new_matrix_test.shape[1])

                alpha += 0.01
                print("alpha NOT to big")
            except:
                print("alpha is to big or other error")
                break

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle(('accuracy vs #features', handle.enumToName(drMethod)))
        # ax1.plot(alphaValues, alphaResults)
        # ax2.plot(alphaValues, numberOfFeatures)
        # plt.show()

    print("dr method: ", drMethod, " |classification methods: ", "SVM")
    for i in range(0, len(varValues)):
        print("var value: ", varValues[i], " |accuracy: ", varResults[i], " |number of features: ", numberOfFeatures[i])

def plotModelParameterGDA (drMethod, algorithmToTest, matrix_train, label_train, matrix_test, label_test):
    handle = Handle()
    alpha = 0.01

    varValues1 = []
    varValues2 = []
    varValues3 = []
    varResults1 = []
    varResults2 = []
    varResults3 = []
    numberOfFeatures1 = []
    numberOfFeatures2 = []
    numberOfFeatures3 = []

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
            predictor.fit(train_data=df_train, hyperparameters={'LR': {}})
            performance = predictor.evaluate(df_test)
            # print("results: ", var, ": ", performance)
            # print(performance.get("accuracy"))
            varValues1.append(var)
            varResults1.append(performance.get("accuracy"))
            numberOfFeatures1.append(df_train.shape[1] - 1)

            print("var: ", var, " |performance: ", performance)

            predictor = TabularPredictor(label='label', problem_type='multiclass', verbosity=0)
            predictor.fit(train_data=df_train, hyperparameters={'RF': {}})
            performance = predictor.evaluate(df_test)
            # print("results: ", var, ": ", performance)
            # print(performance.get("accuracy"))
            varValues2.append(var)
            varResults2.append(performance.get("accuracy"))
            numberOfFeatures2.append(df_train.shape[1] - 1)

            print("var: ", var, " |performance: ", performance)

            predictor = svm.SVC(kernel='linear', max_iter=10000)
            predictor.fit(new_matrix_train, label_train)
            predicted = predictor.predict(new_matrix_test)
            accuracy = metrics.accuracy_score(label_test, predicted)
            print("results: ", var, ": ", accuracy)
            print(accuracy)
            varValues3.append(var)
            varResults3.append(accuracy)
            numberOfFeatures3.append(new_matrix_test.shape[1])

            var += 1
        except:
            break


        print("dr method: ", drMethod, " |classification methods: ", "LR")
        for i in range(0, len(varValues1)):
            print("var value: ", varValues1[i], " |accuracy: ", varResults1[i], " |number of features: ",
                  numberOfFeatures1[i])

        print("dr method: ", drMethod, " |classification methods: ", "RF")
        for i in range(0, len(varValues2)):
            print("var value: ", varValues2[i], " |accuracy: ", varResults2[i], " |number of features: ",
                  numberOfFeatures2[i])

        print("dr method: ", drMethod, " |classification methods: ", "SVM")
        for i in range(0, len(varValues3)):
            print("var value: ", varValues3[i], " |accuracy: ", varResults3[i], " |number of features: ", numberOfFeatures3[i])


def plotGraph():

    pcaVar = np.arange(0.05, 1, 0.05).tolist()
    pcaFeatures = [1,1,1,1,1,2,2,3,3,4,4,5,6,7,9,12,15,21,31]
    lrPCAAcc = [0.543, 0.543, 0.543, 0.543, 0.543, 0.844, 0.844, 0.866, 0.866, 0.893, 0.893, 0.939, .944, 0.954, 0.956, 0.957, 0.964, 0.969, 0.979]
    rfPCAAcc = [0.475 , 0.475, 0.475, 0.475, 0.475 ,0.856  , 0.856, 0.904  , 0.904  , 0.922  ,0.922, 0.959, 0.962, 0.967, 0.971, 0.975, 0.98  , 0.976 , 0.982                   ]
    svmPCAAcc = [0.549  , 0.549  , 0.549  ,0.549  ,0.549 ,0.853  ,0.853 ,0.873  ,0.873  ,0.905  ,0.905  ,0.949  , 0.945  ,0.95  , 0.954, 0.959,0.967,0.975 ,0.979             ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('accuracy vs #features PCA')


    ax1.plot(pcaVar, lrPCAAcc, color='red', label='Logistic Regression')  # x=y
    ax1.plot(pcaVar, rfPCAAcc, color="blue", label="Random Forest")
    ax1.plot(pcaVar, svmPCAAcc, color="yellow", label='SVM')
    ax1.legend()
    ax2.plot(pcaVar, pcaFeatures, color='red', label='#Features')
    ax2.legend()
    plt.show()


def read_csv_files(directory, method):
    # Initialize empty dataframes
    dfLR = pd.DataFrame()
    dfRF = pd.DataFrame()
    dfSVM = pd.DataFrame()

    # Get a list of all CSV files starting with "PCA_" in the directory
    file_list = [file for file in os.listdir(directory) if file.startswith(str(method+"_")) and file.endswith(".csv")]

    # Read each CSV file into a separate dataframe
    for file in file_list:
        file_path = os.path.join(directory, file)
        if file.endswith("LR.csv"):
            dfLR = pd.read_csv(file_path)
        elif file.endswith("RF.csv"):
            dfRF = pd.read_csv(file_path)
        elif file.endswith("SVM.csv"):
            dfSVM = pd.read_csv(file_path)

    return dfLR, dfRF, dfSVM

def plot(dataset, method):
    # Plotting accuracy vs. number of features for all dataframes
    dfLR, dfRF, dfSVM = read_csv_files(("results_"+dataset), method)
    try:
        plt.plot(dfLR['number of features'], dfLR['accuracy'], label='LR', color="blue")
    except:
        print("no LR for ", dataset, " ", method)
    try:
        plt.plot(dfRF['number of features'], dfRF['accuracy'], label='RF', color="red")
    except:
        print("no RF for ", dataset, " ", method)
    try:
        plt.plot(dfSVM['number of features'], dfSVM['accuracy'], label='SVM', color="yellow")
    except:
        print("no VSM for ", dataset, " ", method)
    # Set labels and title for the plot
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Features for ' + method + "(" + dataset + " dataset)")

    # Display legend
    plt.legend()
    plt.show()



if __name__ == '__main__':
    handle = Handle()
    datasetsToRun = ["fonts10_small"]
    methodsToRun = ["PCA", "LDA", "GDA", "LASSO"]
    for datas in datasetsToRun:
        for meth in methodsToRun:
            plot(datas, meth)

    i =0
    # plotGraph()
    while i < 1:

        header, matrix_train, label_train, matrix_test, label_test = handle.readSplit(DataSet.MNIST)

        scaler = preprocessing.StandardScaler().fit(matrix_train)
        matrix_train = pd.DataFrame(scaler.transform(matrix_train))
        matrix_test = pd.DataFrame(scaler.transform(matrix_test))

        le = preprocessing.LabelEncoder()
        le.fit(label_train)
        label_train = pd.DataFrame(le.transform(label_train))
        label_train += 1
        label_test = pd.DataFrame(le.transform(label_test))
        label_test += 1

        # new_matrix_train, new_matrix_test, time_to_transformPCA = handle.dimensionalReduce(matrix_train, label_train,
        #                                                                                 matrix_test, DRMethod.PCA, var=0.95)
        # new_matrix_train, new_matrix_test, time_to_transformLDA = handle.dimensionalReduce(matrix_train, label_train,
        #                                                                                 matrix_test, DRMethod.LDA, var=1)
        # new_matrix_train, new_matrix_test, time_to_transformLASSO = handle.dimensionalReduce(matrix_train, label_train,
        #                                                                                 matrix_test, DRMethod.LASSO, var=0.01)
        # new_matrix_train, new_matrix_test, time_to_transformGDA = handle.dimensionalReduce(matrix_train, label_train,
        #                                                                                  matrix_test, DRMethod.GDA, var=1)

        # print("PCA: ", time_to_transformPCA)
        # print("LDA: ", time_to_transformLDA)
        # print("LASSO: ", time_to_transformLASSO)
        # matrix_test = (matrix_test-matrix_test.mean())/matrix_test.std()
        # matrix_train = (matrix_train-matrix_train.mean())/matrix_train.std()

        # drMethods = [DRMethod.PCA, DRMethod.GDA, DRMethod.LDA, DRMethod.LASSO, DRMethod.NONE]

        # compare_models(drMethods, hyperparameters, matrix_train, label_train, matrix_test, label_test)
        # plotModelParameterGDA(DRMethod.GDA, {'RF': {}}, matrix_train, label_train, matrix_test, label_test)
        # plotModelParameter(DRMethod.PCA, {'LR': {}}, matrix_train, label_train, matrix_test, label_test)
        # # plotModelParameter(DRMethod.PCA, {'RF': {}}, matrix_train, label_train, matrix_test, label_test)
        # plotModelParameter(DRMethod.LDA, {'LR': {}}, matrix_train, label_train, matrix_test, label_test)
        # plotModelParameter(DRMethod.LDA, {'RF': {}}, matrix_train, label_train, matrix_test, label_test)
        plotModelParameter(DRMethod.LASSO, {'LR': {}}, matrix_train, label_train, matrix_test, label_test)
        # plotModelParameter(DRMethod.LASSO, {'RF': {}}, matrix_train, label_train, matrix_test, label_test)
        # plotModelParameterSVM(DRMethod.LDA, {'SVM': {}}, matrix_train, label_train, matrix_test, label_test)
        # plotModelParameterSVM(DRMethod.LASSO, {'SVM': {}}, matrix_train, label_train, matrix_test, label_test)
        # plotModelParameterSVM(DRMethod.PCA, {'SVM': {}}, matrix_train, label_train, matrix_test, label_test)
        plotModelParameter(DRMethod.GDA, {'LR': {}}, matrix_train, label_train, matrix_test, label_test)
        # plotModelParameter(DRMethod.GDA, {'RF': {}}, matrix_train, label_train, matrix_test, label_test)
        # plotModelParameterSVM(DRMethod.GDA, {'SVM': {}}, matrix_train, label_train, matrix_test, label_test)
    #
        i+=1

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
