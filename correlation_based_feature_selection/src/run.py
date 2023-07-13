from .autofeat_pipeline import MLPipeline
import pandas as pd
from scipy.io import arff
from sklearn.datasets import fetch_openml


# def evaluate_census_income_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/CensusIncome/CensusIncome.csv', dataset_name='CensusIncome',
#         target_label='income_label', evaluation_metric='accuracy', features_to_select=14)
#
#     dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='classification')
#     dataset_evaluator.evaluate_all_models_select_above_c()
#     dataset_evaluator.evaluate_support_vector_machine_model(problem_type='classification')
#     dataset_evaluator.evaluate_all_models()


# def evaluate_feature_selection_census_income_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/CensusIncome/CensusIncome.csv', dataset_name='CensusIncome',
#         target_label='income_label', evaluation_metric='accuracy')
#
#     dataset_evaluator.evaluate_feature_selection_step()


# def evaluate_feature_selection_breast_cancer_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/BreastCancer/data.csv', dataset_name='BreastCancer',
#         target_label='diagnosis', evaluation_metric='accuracy')
#
#     dataset_evaluator.evaluate_feature_selection_step()


# def evaluate_steel_plates_fault_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/SteelPlatesFaults/steel_faults_train.csv', dataset_name='SteelPlatesFaults',
#         target_label='Class', evaluation_metric='accuracy', features_to_select=33)
#
#     # dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='classification')
#     # dataset_evaluator.evaluate_all_models_select_above_c()
#     # dataset_evaluator.evaluate_all_models()
#     dataset_evaluator.evaluate_support_vector_machine_model(problem_type='classification')
#     dataset_evaluator.evaluate_all_models()


# def evaluate_feature_selection_steel_plates_faults_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/SteelPlatesFaults/steel_faults_train.csv', dataset_name='SteelPlatesFaults',
#         target_label='Class', evaluation_metric='accuracy')
#
#     dataset_evaluator.evaluate_feature_selection_step()


# def evaluate_connect4_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/Connect4/connect4.csv', dataset_name='Connect4',
#         target_label='label', evaluation_metric='accuracy', features_to_select=42)
#
#     dataset_evaluator.evaluate_support_vector_machine_model(problem_type='classification')
#     dataset_evaluator.evaluate_all_models()


# def evaluate_feature_selection_connect4_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/Connect4/connect4.csv', dataset_name='Connect4',
#         target_label='label', evaluation_metric='accuracy')
#
#     dataset_evaluator.evaluate_feature_selection_step()


# def evaluate_feature_selection_housing_prices_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/HousingPrices/housing_prices.csv', dataset_name='Housing Prices',
#         target_label='SalePrice', evaluation_metric='root_mean_squared_error')
#
#     dataset_evaluator.evaluate_feature_selection_step()


# def evaluate_feature_selection_gisette_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/Gisette/gisette_train.csv', dataset_name='Gisette',
#         target_label='Class', evaluation_metric='accuracy')
#
#     dataset_evaluator.evaluate_support_vector_machine_model()


# def evaluate_bike_sharing_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/BikeSharing/hour.csv', dataset_name='BikeSharing',
#         target_label='cnt', evaluation_metric='root_mean_squared_error', features_to_select=16)
#
#     # dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='regression')
#     # dataset_evaluator.evaluate_all_models_select_above_c()
#     dataset_evaluator.evaluate_support_vector_machine_model(problem_type='classification')
#     dataset_evaluator.evaluate_all_models()


# def evaluate_feature_selection_bike_sharing_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/BikeSharing/hour.csv', dataset_name='BikeSharing',
#         target_label='cnt', evaluation_metric='root_mean_squared_error')
#
#     dataset_evaluator.evaluate_feature_selection_step()


# def evaluate_arrhythmia_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/Arrhythmia/arrhythmia.csv', dataset_name='Arrhythmia',
#         target_label='Class', evaluation_metric='accuracy', features_to_select=200)
#
#     # dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='classification')
#     # dataset_evaluator.evaluate_all_models_select_above_c()
#     dataset_evaluator.evaluate_support_vector_machine_model(problem_type='classification')
#     dataset_evaluator.evaluate_all_models()


# def evaluate_feature_selection_arrhythmia_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/Arrhythmia/arrhythmia.csv', dataset_name='Arrhythmia',
#         target_label='Class', evaluation_metric='accuracy')
#
#     dataset_evaluator.evaluate_feature_selection_step()


# def evaluate_feature_selection_internet_advertisements_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/InternetAdvertisements/internet_advertisements.csv',
#         dataset_name='InternetAds',
#         target_label='class',
#         evaluation_metric='accuracy')
#
#     dataset_evaluator.evaluate_feature_selection_step()


# def evaluate_nursery_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/Nursery/nursery.csv', dataset_name='Nursery',
#         target_label='label', evaluation_metric='accuracy', features_to_select=8)
#
#     # dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='classification')
#     # dataset_evaluator.evaluate_all_models_select_above_c()
#     # dataset_evaluator.evaluate_all_models()
#     dataset_evaluator.evaluate_support_vector_machine_model(problem_type='classification')
#     dataset_evaluator.evaluate_all_models()


# def evaluate_feature_selection_nursery_dataset():
#     dataset_evaluator = MLPipeline(
#         dataset_file='../datasets/Nursery/nursery.csv', dataset_name='Nursery',
#         target_label='label', evaluation_metric='accuracy')
#
#     dataset_evaluator.evaluate_feature_selection_step()


def evaluate_dataframe(dataframe):
    print("Number of rows:", dataframe.shape[0])
    print("Number of columns:", dataframe.shape[1])
    print(dataframe.dtypes)
    print(dataframe.head())


def evaluate_breast_cancer_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/BreastCancer/data.csv', dataset_name='BreastCancer',
        target_label='diagnosis', evaluation_metric='accuracy', features_to_select='small')

    dataset_evaluator.evaluate_all_models()


def evaluate_spam_email_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/SPAM/spam.csv', dataset_name='SpamEmail',
        target_label='class', evaluation_metric='accuracy', features_to_select='small')

    dataset_evaluator.evaluate_all_models()


def evaluate_musk_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/Musk/musk.csv', dataset_name='Musk',
        target_label='class', evaluation_metric='accuracy', features_to_select='medium')

    dataset_evaluator.evaluate_all_models()


def evaluate_arrhythmia_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/Arrhythmia/arrhythmia.csv', dataset_name='Arrhythmia',
        target_label='binaryClass', evaluation_metric='accuracy', features_to_select='medium')

    dataset_evaluator.evaluate_all_models()


def evaluate_internet_advertisements_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/InternetAdvertisements/internet_advertisements.csv',
        dataset_name='InternetAds',
        target_label='class',
        evaluation_metric='accuracy',
        features_to_select='large')

    dataset_evaluator.evaluate_all_models()


def evaluate_gisette_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/Gisette/gisette_train.csv',
        dataset_name='Gisette',
        target_label='Class',
        evaluation_metric='accuracy',
        features_to_select='large')

    dataset_evaluator.evaluate_all_models()


def evaluate_housing_prices_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/HousingPrices/housing_prices.csv', dataset_name='HousingPrices',
        target_label='SalePrice', evaluation_metric='root_mean_squared_error', features_to_select='small')

    dataset_evaluator.evaluate_all_models()


def evaluate_topo_2_1_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/TOPO-2-1/topo_2_1.csv', dataset_name='TOPO',
        target_label='oz267', evaluation_metric='root_mean_squared_error', features_to_select='medium')

    dataset_evaluator.evaluate_all_models()


def evaluate_qsar_tid_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/QSAR-TID-11109/qsar.csv', dataset_name='QSAR',
        target_label='MEDIAN_PXC50', evaluation_metric='root_mean_squared_error', features_to_select='large')

    dataset_evaluator.evaluate_all_models()


# def arff_to_csv(file_path, csv_file_path):
#     data = arff.loadarff(file_path)
#     dataframe = pd.DataFrame(data[0])
#
#     cat_columns = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
#     dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.str.decode('utf8'))
#     dataframe.to_csv(csv_file_path, index=False)


if __name__ == '__main__':
    # Binary classification
    evaluate_breast_cancer_dataset()
    # evaluate_internet_advertisements_dataset()
    # evaluate_gisette_dataset()
    # evaluate_spam_email_dataset()
    # evaluate_musk_dataset()
    evaluate_arrhythmia_dataset()

    # Regression
    # evaluate_housing_prices_dataset()
    # evaluate_topo_2_1_dataset()
    # evaluate_qsar_tid_dataset()
