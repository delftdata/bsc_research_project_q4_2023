from ast import literal_eval

from skfeature.utility.plotting_helpers import *


def visualize_results(dataset_name, model_names, n, mrmrs, mifss, jmis, cifes):
    """
    Plots the efficiency and effectiveness of each FS method

    Args:
        dataset_name: name of dataset
        model_names: name of ML algorithms
        n: number of features
        mrmrs: results from MRMR
        mifss: results from MIFS
        jmis: results from JMI
        cifes: results from CIFE
    """
    i = 0
    for mrmr, mifs, jmi, cife in zip(mrmrs, mifss, jmis, cifes):
        model_name = model_names[i]

        mrmr_one = [i[0] for i in mrmr]
        mifs_one = [i[0] for i in mifs]
        jmi_one = [i[0] for i in jmi]
        cife_one = [i[0] for i in cife]
        plot_over_features(dataset_name, model_name, n, mrmr_one, mifs_one, jmi_one, cife_one)

        mrmr_one = [i[1] for i in mrmr]
        mifs_one = [i[1] for i in mifs]
        jmi_one = [i[1] for i in jmi]
        cife_one = [i[1] for i in cife]
        plot_performance(dataset_name, n, mrmr_one, mifs_one, jmi_one, cife_one)
        i += 1


def plot_feature_selection_performance():
    """
    Plots effectiveness for simple and complex entropy estimator for XGB and LR.
    """
    with open('results/logs/performance_simple.txt', "r") as file:
        data = file.readlines()
    with open('results/logs/performance_complex.txt', "r") as file:
        data_complex = file.readlines()

    datasets = [data[i:i + 5] for i in range(0, len(data), 5)]
    datasets_complex = [data_complex[i:i + 5] for i in range(0, len(data_complex), 5)]
    for i in range(len(datasets)):
        dataset = datasets[i]
        dataset_name = dataset[0].split('datasets/')[1].split('/')[0]
        mrmr_result = literal_eval(dataset[1].replace('array', ''))
        mifs_result = literal_eval(dataset[2].replace('array', ''))
        jmi_result = literal_eval(dataset[3].replace('array', ''))
        cife_result = literal_eval(dataset[4].replace('array', ''))

        model_names = ['XGBoost', 'LinearModel']
        j = 0
        for mrmr, mifs, jmi, cife in zip(mrmr_result, mifs_result, jmi_result, cife_result):
            mrmr_one = [i[0] for i in mrmr]
            mifs_one = [i[0] for i in mifs]
            jmi_one = [i[0] for i in jmi]
            cife_one = [i[0] for i in cife]

            model_name = model_names[j]

            plot_over_features(dataset_name, model_name, len(mrmr_one), mrmr_one, mifs_one, jmi_one, cife_one)
            j += 1

    for i in range(len(datasets_complex)):
        dataset_complex = datasets_complex[i]
        dataset_name = dataset_complex[0].split('datasets/')[1].split('/')[0]
        mrmr_result_complex = literal_eval(dataset_complex[1].replace('array', ''))
        mifs_result_complex = literal_eval(dataset_complex[2].replace('array', ''))
        jmi_result_complex = literal_eval(dataset_complex[3].replace('array', ''))
        cife_result_complex = literal_eval(dataset_complex[4].replace('array', ''))

        model_names = ['XGBoost', 'LinearModel']

        j = 0
        for mrmr, mifs, jmi, cife in zip(mrmr_result_complex, mifs_result_complex, jmi_result_complex, cife_result_complex):
            mrmr_one = [i[0] for i in mrmr]
            mifs_one = [i[0] for i in mifs]
            jmi_one = [i[0] for i in jmi]
            cife_one = [i[0] for i in cife]

            model_name = model_names[j]

            plot_over_features(dataset_name + '_complex_', model_name, len(mrmr_one), mrmr_one, mifs_one, jmi_one, cife_one)
            j += 1


def plot_feature_selection_three_models(datasets):
    """
    Plots the effectiveness of the feature selection algorithms over the
    three ML algorithms.

    Note that here you can select whether you want an individual plot or three plots.
    """
    with open('results/logs/performance_complex.txt', "r") as file:
        data = file.readlines()

    data = [data[i:i + 6] for i in range(0, len(data), 6)]

    for i in range(len(data)):
        dataset = data[i]
        dataset_name = dataset[0].split('datasets/')[1].split('/')[0]
        mrmr_result = [[entry[0] for entry in model] for model in literal_eval(dataset[1].replace('array', ''))]
        mifs_result = [[entry[0] for entry in model] for model in literal_eval(dataset[2].replace('array', ''))]
        jmi_result = [[entry[0] for entry in model] for model in literal_eval(dataset[3].replace('array', ''))]
        cife_result = [[entry[0] for entry in model] for model in literal_eval(dataset[4].replace('array', ''))]
        base_results = [[entry[0] for entry in model] for model in literal_eval(dataset[5].replace('array', ''))]

        is_classification = [x for x in datasets if x['path'] == '../.' + dataset[0].strip()][0]['is_classification']
        name = [x for x in datasets if x['path'] == '../.' + dataset[0].strip()][0]['name']
        title = f'Performance on {name} dataset'
        # plot_over_features(dataset_name, name, len(mrmr_result[0]),
        #                    mrmr_result, mifs_result, jmi_result, cife_result, base_results,
        #                    is_classification)
        # plot_over_features_mifs(dataset_name, name, len(mrmr_result[0]),
        #                    mrmr_result, mifs_result, jmi_result, cife_result, base_results,
        #                    is_classification)
        plot_over_features_3(dataset_name, title, len(mrmr_result[0]),
                             mrmr_result, mifs_result, jmi_result, cife_result, base_results,
                             is_classification)


def plot_feature_selection_three_models_mifs(datasets):
    """
    Plots the effectiveness of MIFS over the three ML algorithms.
    """
    with open('results/logs/performance_mifs.txt', "r") as file:
        data = file.readlines()

    data = [data[i:i + 6] for i in range(0, len(data), 6)]

    for i in range(len(data)):
        dataset = data[i]
        dataset_name = dataset[0].split('datasets/')[1].split('/')[0]
        mifs_000 = [[entry[0] for entry in model] for model in literal_eval(dataset[1].replace('array', ''))]
        mifs_025 = [[entry[0] for entry in model] for model in literal_eval(dataset[2].replace('array', ''))]
        mifs_050 = [[entry[0] for entry in model] for model in literal_eval(dataset[3].replace('array', ''))]
        mifs_075 = [[entry[0] for entry in model] for model in literal_eval(dataset[4].replace('array', ''))]
        mifs_100 = [[entry[0] for entry in model] for model in literal_eval(dataset[5].replace('array', ''))]

        is_classification = [x for x in datasets if x['path'] == '../.' + dataset[0].strip()][0]['is_classification']
        name = [x for x in datasets if x['path'] == '../.' + dataset[0].strip()][0]['name']
        title = f'Performance on {name} dataset'
        plot_mifs_3(dataset_name, title, len(mifs_100[0]), mifs_000, mifs_025, mifs_050, mifs_075, mifs_100, is_classification)


def plot_feature_selection_two_side_by_side():
    """
    Plots the difference in effectiveness of simple and complex entropy estimator.

    Note that the function will only plot for steel, but that can be modified.
    """
    with open('results/logs/performance_simple.txt', "r") as file:
        data = file.readlines()
    with open('results/logs/performance_complex.txt', "r") as file:
        data_complex = file.readlines()

    datasets = [data[i:i + 6] for i in range(0, len(data), 6)]
    datasets_complex = [data_complex[i:i + 6] for i in range(0, len(data_complex), 6)]

    temp = []
    for i in range(len(datasets)):
        dataset = datasets[i]
        if 'steel' in dataset[0]:
            temp.append(dataset)

    for i in range(len(datasets_complex)):
        dataset = datasets_complex[i]
        if 'steel' in dataset[0]:
            temp.append(dataset)

    mrmr = [literal_eval(temp[0][1].replace('array', ''))[1], literal_eval(temp[1][1].replace('array', ''))[1]]
    mifs = [literal_eval(temp[0][2].replace('array', ''))[1], literal_eval(temp[1][2].replace('array', ''))[1]]
    jmi = [literal_eval(temp[0][3].replace('array', ''))[1], literal_eval(temp[1][3].replace('array', ''))[1]]
    cife = [literal_eval(temp[0][4].replace('array', ''))[1], literal_eval(temp[1][4].replace('array', ''))[1]]
    base = [literal_eval(temp[0][5].replace('array', ''))[1], literal_eval(temp[1][5].replace('array', ''))[1]]

    plot_over_features_2('steel', 'Steel Plate Faults dataset performance on Linear Regression model',
                         len(mrmr[0]), mrmr, mifs, jmi, cife, base)


def plot_feature_selection_runtime():
    """
    Plots runtime difference between simple and complex entropy estimators.
    """
    with open('results/logs/fs_simple.txt', "r") as file:
        data = file.readlines()
    with open('results/logs/fs_complex.txt', "r") as file:
        data_complex = file.readlines()

    datasets = [data[i:i + 6] for i in range(0, len(data), 6)]
    datasets_complex = [data_complex[i:i + 6] for i in range(0, len(data_complex), 6)]
    for i in range(len(datasets_complex)):
        dataset = datasets[i]
        if 'breast' not in dataset[0]:
            continue
        dataset_name = dataset[0].split('datasets/')[1].split('/')[0]
        mrmr_result = literal_eval(dataset[1].replace('array', ''))
        mifs_result = literal_eval(dataset[2].replace('array', ''))
        jmi_result = literal_eval(dataset[3].replace('array', ''))
        cife_result = literal_eval(dataset[4].replace('array', ''))
        # base_result = literal_eval(dataset[5].replace('array', ''))

        mrmr_one = [i[1] for i in mrmr_result]
        mifs_one = [i[1] for i in mifs_result]
        jmi_one = [i[1] for i in jmi_result]
        cife_one = [i[1] for i in cife_result]
        base_one = [0.012706518173217773 for i in cife_result]

        dataset_complex = datasets_complex[i]
        mrmr_result_complex = literal_eval(dataset_complex[1].replace('array', ''))
        mifs_result_complex = literal_eval(dataset_complex[2].replace('array', ''))
        jmi_result_complex = literal_eval(dataset_complex[3].replace('array', ''))
        cife_result_complex = literal_eval(dataset_complex[4].replace('array', ''))
        base_result_complex = literal_eval(dataset_complex[5].replace('array', ''))

        mrmr_complex_one = [i[1] for i in mrmr_result_complex]
        mifs_complex_one = [i[1] for i in mifs_result_complex]
        jmi_complex_one = [i[1] for i in jmi_result_complex]
        cife_complex_one = [i[1] for i in cife_result_complex]
        base_complex_one = [i[1] for i in base_result_complex]

        print(dataset_name)
        print(round(mifs_one[-1], 2))
        print(round(mrmr_one[-1], 2))
        print(round(cife_one[-1], 2))
        print(round(jmi_one[-1], 2))
        print(round(base_one[-1], 2))
        print(round(mifs_complex_one[-1], 2))
        print(round(mrmr_complex_one[-1], 2))
        print(round(cife_complex_one[-1], 2))
        print(round(jmi_complex_one[-1], 2))
        print(round(base_complex_one[-1], 2))
        # plot_performance(dataset_name, len(mrmr_one), mrmr_one, mifs_one, jmi_one, cife_one, False)

        # plot_performance_8(dataset_name, len(mrmr_one), mrmr_one, mifs_one, jmi_one, cife_one, mrmr_complex_one, mifs_complex_one, jmi_complex_one, cife_complex_one)
        plot_performance_two('Breast cancer dataset runtime performance', len(mrmr_one), mrmr_one, mifs_one, jmi_one, cife_one, base_one, mrmr_complex_one, mifs_complex_one, jmi_complex_one, cife_complex_one, base_complex_one)


def print_list_of_features():
    """
    Prints the order of selecting features for MIFS, MRMR, CIFE and JMI.
    """
    with open('results/logs/fs_complex.txt', "r") as file:
        data_complex = file.readlines()

    datasets_complex = [data_complex[i:i + 6] for i in range(0, len(data_complex), 6)]
    for i in range(len(datasets_complex)):
        mat = pd.read_csv('../.' + datasets_complex[i][0].strip())
        columns_mrmr = literal_eval(datasets_complex[i][1].replace('array', ''))[-1][0]
        columns_mifs = literal_eval(datasets_complex[i][2].replace('array', ''))[-1][0]
        columns_jmi = literal_eval(datasets_complex[i][3].replace('array', ''))[-1][0]
        columns_cife = literal_eval(datasets_complex[i][4].replace('array', ''))[-1][0]
        columns_base = literal_eval(datasets_complex[i][5].replace('array', ''))[-1][0]
        db = [x for x in datasets if x['path'] == '../.' + datasets_complex[i][0].strip()][0]
        y_label = db['y_label']

        print(db['path'])
        train_data = TabularDataset(mat)
        train_data = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                    enable_text_ngram_features=False).fit_transform(X=train_data)

        print('mifs')
        mifs = [list(train_data.drop(y_label, axis=1).columns)[i] for i in columns_mifs[0:len(columns_base)]]
        print(mifs)
        print('mrmr')
        mrmr =[list(train_data.drop(y_label, axis=1).columns)[i] for i in columns_mrmr[0:len(columns_base)]]
        print(mrmr)
        print('cife')
        cife = [list(train_data.drop(y_label, axis=1).columns)[i] for i in columns_cife[0:len(columns_base)]]
        print(cife)
        print('jmi')
        jmi = [list(train_data.drop(y_label, axis=1).columns)[i] for i in columns_jmi[0:len(columns_base)]]
        print(jmi)
        print('ig')
        ig = [list(train_data.drop(y_label, axis=1).columns)[i] for i in columns_base[0:len(columns_base)]]
        print(ig)
        print()

        if len(list(train_data.drop(y_label, axis=1).columns)) != len(columns_mifs):
            continue

        columns_mifs = [dict((v, i) for i, v in enumerate(columns_mifs)).get(v, -1) for v in range(0, len(columns_base))]
        columns_mrmr = [dict((v, i) for i, v in enumerate(columns_mrmr)).get(v, -1) for v in range(0, len(columns_base))]
        columns_cife = [dict((v, i) for i, v in enumerate(columns_cife)).get(v, -1) for v in range(0, len(columns_base))]
        columns_jmi = [dict((v, i) for i, v in enumerate(columns_jmi)).get(v, -1) for v in range(0, len(columns_base))]
        columns_base = [dict((v, i) for i, v in enumerate(columns_base)).get(v, -1) for v in range(0, len(columns_base))]


        df = pd.DataFrame({'columns': list(train_data.drop(y_label, axis=1).columns),
                           'mifs': columns_mifs,
                           'mrmr': columns_mrmr,
                           'cife': columns_cife,
                           'jmi': columns_jmi,
                           'ig': columns_base})

        print(df.to_string())
