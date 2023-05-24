import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, minmax_scale

def scale(array):
    # return minmax_scale(array, (0.7,1))
    return array
autogluon = np.array([
    [84.55, 87.55, 87.62, 86.11, 79.88], 
    [88.99, 90.27, 90.12, 90.24, 88.39],
    [78.01, 98.96, 98.88, 97.49, 89.0],
    [65.97, 86.55, 86.43, 82.9, 83.05]         
    ])

onehot = np.array([
    [84.81, 85.78, 85.91, 84.24, 79.88],
    [89.77, 90.77, 90.51, 90.41, 88.39],
    [92.4, 99.96, 100, 99.61, 100],
    [77.51, 86.89, 87.06, 84.51, 88.85]
])

ordinal = np.array([
    [80.78, 85.28, 87.53 , 83.98, 79.88],
    [88.99, 90.31, 90.12, 89.97, 88.39],
    [78.01, 98.88, 98.88, 97.3, 89.0],
    [65.98, 86.55, 86.67, 82.9,  83.85]
])

target = np.array([
    [83.07, 85.27, 85.39, 84.76, 79.88],
    [89.33, 90.36, 90.38, 90.3, 88.39],
    [85.69, 99.5, 99.58, 99.07, 86.5],
    [74.58, 84.78, 84.72, 78.72, 80.78]
])


catboost = np.array([
    [83.04, 85.35, 85.44, 85.08, 79.88],
    [89.34, 90.41, 90.35, 90.31, 88.39],
    [85.69, 99.61, 99.65, 99.07, 86.5],
    [74.5, 84.65, 84.57, 78.95, 80.78]
])

count = np.array([
    [79.92, 85.32, 85.24, 84.36, 82.50 ],
    [89.52, 90.66, 90.88, 90.56, 88.39],
    # [0 ,0 , 0 , 0 , 0],
    [65.92, 82.53, 82.82, 74.4, 66.12]
])

autogluon = scale(autogluon)
onehot = scale(onehot)
ordinal = scale(ordinal)
catboost = scale(catboost)
target = scale(target)
count = scale(count)

print(autogluon)
autogluon_time = np.array([
    [30, 30, 35, 3, 10], 
    [34, 70, 110, 123, 19],
    [53, 162, 309, 314, 1],
    [188, 429, 1145, 1196, 155]         
    ])

onehot_time = np.array([
    [57, 32, 39, 5, 33], 
    [53, 74, 111, 122, 28],
    [73, 157, 272, 275, 6],
    [233, 494, 1132, 1189, 176]  
])

ordinal_time = np.array([
    [23, 30, 30, 4, 10], 
    [23, 54, 89, 100, 19],
    [51, 159, 307, 313, 1],
    [174, 406, 1083, 1132, 153]  
])

target_time = np.array([
    [22, 28, 34, 5, 11], 
    [24, 55, 90, 101, 20],
    [48, 137, 258, 163, 1],
    [72, 250, 831, 905, 121]  
])


catboost_time = np.array([
    [30, 35, 30, 4, 10], 
    [23, 53, 88, 99, 20],
    [47, 138, 261, 266, 1],
    [68, 228, 635, 703, 133]  
])

count_time = np.array([
    [22, 29, 35, 6, 111], 
    [23, 54, 90, 103, 20],
    # [],
    [66, 256, 799, 877, 242]  
])

algorithm_names = ['Linear model', 'XGBoost', 'LightGBM', 'Random Forest', 'SVM']



def baseline_algorithm(index):
    """
    Args:
        index:
        0 - linear
        1 - xgb
        2 - gbm
        3 - rf
        4 - svm
    """
    
    plt.axhline(y=0, color='r', linestyle='dotted')
    autogluon_mean = np.mean(autogluon, axis = 0)[index]
    onehot_mean = np.mean(onehot, axis = 0)[index]
    ordinal_mean = np.mean(ordinal, axis = 0)[index]
    target_mean = np.mean(target, axis = 0)[index]
    catboost_mean = np.mean(catboost, axis = 0)[index]
    count_mean = np.mean(count, axis = 0)[index]

    plt.scatter(
    ['onehot', 'ordinal', 'target', 'catboost', 'count'], 
    [onehot_mean-autogluon_mean, ordinal_mean-autogluon_mean, target_mean-autogluon_mean, catboost_mean-autogluon_mean, count_mean-autogluon_mean])
    plt.xlabel('encoder')
    # plt.yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3], [-5, -4, -3, -2, -1, 'AutoGluon\nEncoding', 1, 2, 3])
    plt.ylabel('deviation in accuracy from baseline')

    plt.title(f"Accuracy Mean on {algorithm_names[index]}")
    # plt.show()
    plt.savefig(f'dark/accuracy/{algorithm_names[index]}_baseline_mean_deviation.png')
    plt.clf()

def baseline_algorithm_mean(index):
    """
    Args:
        index:
        0 - linear
        1 - xgb
        2 - gbm
        3 - rf
        4 - svm
    """
    
    autogluon_mean = np.mean(autogluon, axis = 0)[index]
    plt.axhline(y=autogluon_mean, color='r', linestyle='dotted')
    onehot_mean = np.mean(onehot, axis = 0)[index]
    ordinal_mean = np.mean(ordinal, axis = 0)[index]
    target_mean = np.mean(target, axis = 0)[index]
    catboost_mean = np.mean(catboost, axis = 0)[index]
    count_mean = np.mean(count, axis = 0)[index]

    onehot_std = np.std(onehot, axis = 0)[index]
    ordinal_std = np.std(ordinal, axis = 0)[index]
    target_std = np.std(target, axis = 0)[index]
    catboost_std = np.std(catboost, axis = 0)[index]
    count_std = np.std(count, axis = 0)[index]

    plt.errorbar(
    ['onehot', 'ordinal', 'target', 'catboost', 'count'], 
    [onehot_mean, ordinal_mean, target_mean, catboost_mean, count_mean], 
    [onehot_std, ordinal_std, target_std, catboost_std, count_std],
    linestyle='None', marker='^', capsize=3)
    plt.xlabel('encoder')
    # plt.yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3], [-5, -4, -3, -2, -1, 'AutoGluon\nEncoding', 1, 2, 3])
    plt.ylabel('mean accuracy and std')
    # plt.ylim(0.2, 1)

    plt.title(f"Accuracy Mean/Std on {algorithm_names[index]}")
    # plt.show()
    plt.savefig(f'dark/accuracy/std-{algorithm_names[index]}_baseline_mean_deviation.png')
    plt.clf()

def baseline_all_encoders_mean():
    
    autogluon_mean = np.mean(autogluon)
    plt.axhline(y=autogluon_mean, color='r', linestyle='dotted')
    onehot_mean = np.mean(onehot)
    ordinal_mean = np.mean(ordinal)
    target_mean = np.mean(target)
    catboost_mean = np.mean(catboost)
    count_mean = np.mean(count)

    onehot_std = np.std(onehot)
    ordinal_std = np.std(ordinal)
    target_std = np.std(target)
    catboost_std = np.std(catboost)
    count_std = np.std(count)

    print(onehot_mean)
    plt.errorbar(
    ['onehot', 'ordinal', 'target', 'catboost', 'count'], 
    [onehot_mean, ordinal_mean, target_mean, catboost_mean, count_mean], 
    [onehot_std, ordinal_std, target_std, catboost_std, count_std],
    linestyle='None', marker='^', capsize=2)
    plt.xlabel('encoder')
    plt.ylabel('mean accuracy and std')
    # plt.ylim(0.2, 1)

    plt.title(f"Accuracy Mean/Std on all ml algorithms")
    # plt.show()
    plt.savefig(f'dark/accuracy/std-all-algs_baseline_mean_deviation.png')
    plt.clf()



def baseline_all_encoders():
    plt.axhline(y=0, color='r', linestyle='dotted')

    autogluon_mean = np.mean(autogluon)
    onehot_mean = np.mean(onehot)
    ordinal_mean = np.mean(ordinal)
    target_mean = np.mean(target)
    catboost_mean = np.mean(catboost)
    count_mean = np.mean(count)
    
    plt.scatter(
        ['onehot', 'ordinal', 'target', 'catboost', 'count'], 
        [onehot_mean-autogluon_mean, ordinal_mean-autogluon_mean, target_mean-autogluon_mean, catboost_mean-autogluon_mean, count_mean-autogluon_mean])
    plt.xlabel('encoder')
    plt.yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3], [-5, -4, -3, -2, -1, f'AutoGluon\n({round(autogluon_mean,2)})', 1, 2, 3])
    plt.ylabel('deviation in accuracy from baseline')

    plt.title("Fig1. Accuracy mean on all ml algorithms")
    # plt.show()
    plt.savefig(f'dark/accuracy/canva_all_algs_baseline_mean_deviation.png')
    plt.clf()


def time_baseline_algorithm(index):
    """
    Args:
        index:
        0 - linear
        1 - xgb
        2 - gbm
        3 - rf
        4 - svm
    """
    
    plt.axhline(y=0, color='r', linestyle='dotted')
    autogluon_mean = np.mean(autogluon_time, axis = 0)[index]
    onehot_mean = np.mean(onehot_time, axis = 0)[index]
    ordinal_mean = np.mean(ordinal_time, axis = 0)[index]
    target_mean = np.mean(target_time, axis = 0)[index]
    catboost_mean = np.mean(catboost_time, axis = 0)[index]
    count_mean = np.mean(count_time, axis = 0)[index]

    plt.scatter(
    ['onehot', 'ordinal', 'target', 'catboost', 'count'], 
    [onehot_mean-autogluon_mean, ordinal_mean-autogluon_mean, target_mean-autogluon_mean, catboost_mean-autogluon_mean, count_mean-autogluon_mean])
    plt.xlabel('encoder')
    # plt.yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3], [-5, -4, -3, -2, -1, 'AutoGluon\nEncoding', 1, 2, 3])
    plt.ylabel('deviation in accuracy from baseline')

    plt.title(f"Runtime Mean on {algorithm_names[index]}")
    # plt.show()
    plt.savefig(f'dark/time/{algorithm_names[index]}_baseline_mean_deviation.png')
    plt.clf()


def time_baseline_all_encoders():
    plt.axhline(y=0, color='r', linestyle='dotted')

    autogluon_mean = np.mean(autogluon_time)
    onehot_mean = np.mean(onehot_time)
    ordinal_mean = np.mean(ordinal_time)
    target_mean = np.mean(target_time)
    catboost_mean = np.mean(catboost_time)
    count_mean = np.mean(count_time)
    
    plt.scatter(
        ['onehot', 'ordinal', 'target', 'catboost', 'count'], 
        [onehot_mean-autogluon_mean, ordinal_mean-autogluon_mean, target_mean-autogluon_mean, catboost_mean-autogluon_mean, count_mean-autogluon_mean])
    plt.xlabel('encoder')
    plt.yticks([-70, -60, -50, -40, -30, -20, -10, 0, 10], [-70, -60, -50, -40, -30, -20, -10, f'AutoGluon\n({round(autogluon_mean,2)})', 10])
    plt.ylabel('deviation in accuracy from baseline')

    plt.title("Fig2. Runtime mean on all ml algorithms")
    # plt.show()
    plt.savefig(f'dark/time/canva_all_algs_baseline_mean_deviation.png')
    plt.clf()


plt.style.use('ggplot')

# baseline_all_encoders()
# for i, algorithm in enumerate(algorithm_names):
#     baseline_algorithm(i)


# time_baseline_all_encoders()
# for i, algorithm in enumerate(algorithm_names):
#     time_baseline_algorithm(i)




baseline_all_encoders_mean()
for i, algorithm in enumerate(algorithm_names):
    baseline_algorithm_mean(i)
