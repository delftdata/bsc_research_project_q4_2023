from .pipeline import MLPipeline
from .plots.number_of_features_plot2 import parse_data, plot_over_number_of_features, \
    plot_over_number_of_features_custom
from .plots.runtime_plot import plot_over_number_of_features_runtime, plot_over_number_of_features_runtime_custom
from .plots.runtime_plot2 import plot_over_runtime
from .plots.avg_number_of_features_plot import parse_data_all, plot_average_over_number_of_features, parse_data_all2
from .plots.thresold_value_plot import parse_data_all_threshold, plot_average_over_number_of_features_threshold
from .plots.avg_algorithm import plot_average_over_number_of_features_alg


def evaluate_census_income_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/CensusIncome/CensusIncome.csv', dataset_name='CensusIncome',
        target_label='income_label', evaluation_metric='accuracy', features_to_select=14)

    dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='classification')
    dataset_evaluator.evaluate_all_models_select_above_c()


def evaluate_feature_selection_census_income_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/CensusIncome/CensusIncome.csv', dataset_name='CensusIncome',
        target_label='income_label', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_breast_cancer_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/BreastCancer/data.csv', dataset_name='BreastCancer',
        target_label='diagnosis', evaluation_metric='accuracy', features_to_select=31)

    dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='classification')
    dataset_evaluator.evaluate_all_models_select_above_c()


def evaluate_feature_selection_breast_cancer_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/BreastCancer/data.csv', dataset_name='BreastCancer',
        target_label='diagnosis', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_steel_plates_fault_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/SteelPlatesFaults/steel_faults_train.csv', dataset_name='SteelPlatesFaults',
        target_label='Class', evaluation_metric='accuracy', features_to_select=33)

    # dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='classification')
    # dataset_evaluator.evaluate_all_models_select_above_c()
    dataset_evaluator.evaluate_all_models()


def evaluate_feature_selection_steel_plates_faults_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/SteelPlatesFaults/steel_faults_train.csv', dataset_name='SteelPlatesFaults',
        target_label='Class', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_connect4_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Connect4/connect4.csv', dataset_name='Connect4',
        target_label='label', evaluation_metric='accuracy', features_to_select=42)

    dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='classification')
    dataset_evaluator.evaluate_all_models_select_above_c()


def evaluate_feature_selection_connect4_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Connect4/connect4.csv', dataset_name='Connect4',
        target_label='label', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_housing_prices_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/HousingPrices/train.csv', dataset_name='HousingPrices',
        target_label='SalePrice', evaluation_metric='root_mean_squared_error', features_to_select=80)

    dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='regression')
    dataset_evaluator.evaluate_all_models_select_above_c()


def evaluate_feature_selection_housing_prices_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/HousingPrices/train.csv', dataset_name='Housing Prices',
        target_label='SalePrice', evaluation_metric='root_mean_squared_error')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_gisette_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Gisette/gisette_train.csv',
        dataset_name='Gisette',
        target_label='Class',
        evaluation_metric='accuracy',
        features_to_select=200)

    dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='classification')
    dataset_evaluator.evaluate_all_models_select_above_c()


def evaluate_feature_selection_gisette_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Gisette/gisette_train.csv', dataset_name='Gisette',
        target_label='Class', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_support_vector_machine_model()


def evaluate_bank_marketing_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/BankMarketing/bank.csv', dataset_name='BankMarketing',
        target_label='y', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


def evaluate_feature_selection_bank_marketing_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/BankMarketing/bank.csv', dataset_name='BankMarketing',
        target_label='y', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_feature_selection_nasa_numeric_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/NasaNumeric/nasa_numeric.csv', dataset_name='NasaNumeric',
        target_label='act_effort', evaluation_metric='root_mean_squared_error')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_bike_sharing_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/BikeSharing/hour.csv', dataset_name='BikeSharing',
        target_label='cnt', evaluation_metric='root_mean_squared_error', features_to_select=16)

    dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='regression')
    dataset_evaluator.evaluate_all_models_select_above_c()


def evaluate_feature_selection_bike_sharing_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/BikeSharing/hour.csv', dataset_name='BikeSharing',
        target_label='cnt', evaluation_metric='root_mean_squared_error')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_arrhythmia_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Arrhythmia/arrhythmia.csv', dataset_name='Arrhythmia',
        target_label='Class', evaluation_metric='accuracy', features_to_select=200)

    dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='classification')
    dataset_evaluator.evaluate_all_models_select_above_c()


def evaluate_feature_selection_arrhythmia_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Arrhythmia/arrhythmia.csv', dataset_name='Arrhythmia',
        target_label='Class', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_feature_selection_crop_mapping_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Crop/arrhythmia.csv', dataset_name='Arrhythmia',
        target_label='Class', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_internet_advertisements_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/InternetAdvertisements/internet_advertisements.csv',
        dataset_name='InternetAds',
        target_label='class',
        evaluation_metric='accuracy',
        features_to_select=200)

    dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='classification')
    dataset_evaluator.evaluate_all_models_select_above_c()


def evaluate_feature_selection_internet_advertisements_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/InternetAdvertisements/internet_advertisements.csv',
        dataset_name='InternetAds',
        target_label='class',
        evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_nursery_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Nursery/nursery.csv', dataset_name='Nursery',
        target_label='label', evaluation_metric='accuracy', features_to_select=8)

    # dataset_evaluator.evaluate_support_vector_machine_model_select_above_c(problem_type='classification')
    # dataset_evaluator.evaluate_all_models_select_above_c()
    dataset_evaluator.evaluate_all_models()


def evaluate_feature_selection_nursery_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Nursery/nursery.csv', dataset_name='Nursery',
        target_label='label', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_dataframe(dataframe):
    print("Number of rows:", dataframe.shape[0])
    print("Number of columns:", dataframe.shape[1])
    print(dataframe.dtypes)
    print(dataframe.head())


if __name__ == '__main__':
    # plot_average_over_number_of_features_alg()
    # parse_data_all2()
    # plot_average_over_number_of_features_threshold()
    # parse_data_all_threshold()
    # plot_average_over_number_of_features()
    # plot_over_runtime()
    # parse_data()
    # plot_over_number_of_features()
    # plot_over_number_of_features_custom()
    # plot_over_number_of_features_runtime()
    # plot_over_number_of_features_runtime_custom()

    # binary classification
    # evaluate_census_income_dataset()
    # evaluate_breast_cancer_dataset()
    # evaluate_steel_plates_fault_dataset()
    # evaluate_arrhythmia_dataset()
    # evaluate_internet_advertisements_dataset()
    # evaluate_gisette_dataset()
    # multi-class classification
    evaluate_nursery_dataset()
    # evaluate_connect4_dataset()

    # regression
    # evaluate_housing_prices_dataset()
    # evaluate_bike_sharing_dataset()

    # databases that have SMALL number of instances
    # evaluate_feature_selection_breast_cancer_dataset()
    # evaluate_feature_selection_steel_plates_faults_dataset()
    # evaluate_feature_selection_arrhythmia_dataset()
    # evaluate_feature_selection_housing_prices_dataset()
    # evaluate_feature_selection_internet_advertisements_dataset()

    # databases that have LARGE number of instances
    # evaluate_feature_selection_gisette_dataset()
    # evaluate_feature_selection_bike_sharing_dataset()
    # evaluate_feature_selection_nursery_dataset()
    # evaluate_feature_selection_connect4_dataset()
    # evaluate_feature_selection_census_income_dataset()
