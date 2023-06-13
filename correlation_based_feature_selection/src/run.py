from .pipeline import MLPipeline
from .plots.runtime_plot2 import plot_over_runtime, parse_results, plot_over_runtime_large_datasets


def evaluate_census_income_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/CensusIncome/CensusIncome.csv', dataset_name='CensusIncome',
        target_label='income_label', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


def evaluate_feature_selection_census_income_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/CensusIncome/CensusIncome.csv', dataset_name='CensusIncome',
        target_label='income_label', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_breast_cancer_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/BreastCancer/data.csv', dataset_name='BreastCancer',
        target_label='diagnosis', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


def evaluate_feature_selection_breast_cancer_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/BreastCancer/data.csv', dataset_name='BreastCancer',
        target_label='diagnosis', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_steel_plates_fault_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/SteelPlatesFaults/steel_faults_train.csv', dataset_name='SteelPlatesFaults',
        target_label='Class', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


def evaluate_feature_selection_steel_plates_faults_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/SteelPlatesFaults/steel_faults_train.csv', dataset_name='SteelPlatesFaults',
        target_label='Class', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()

def evaluate_connect4_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Connect4/connect4.csv', dataset_name='Connect4',
        target_label='label', evaluation_metric='accuracy')


    dataset_evaluator.evaluate_all_models()


def evaluate_feature_selection_connect4_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Connect4/connect4.csv', dataset_name='Connect4',
        target_label='label', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_housing_prices_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/HousingPrices/train.csv', dataset_name='HousingPrices',
        target_label='SalePrice', evaluation_metric='root_mean_squared_error')

    dataset_evaluator.evaluate_all_models()


def evaluate_feature_selection_housing_prices_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/HousingPrices/train.csv', dataset_name='Housing Prices',
        target_label='SalePrice', evaluation_metric='root_mean_squared_error')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_gisette_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Gisette/gisette_train.csv', dataset_name='Gisette',
        target_label='Class', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


def evaluate_feature_selection_gisette_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Gisette/gisette_train.csv', dataset_name='Gisette',
        target_label='Class', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


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


def evaluate_feature_selection_bike_sharing_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/BikeSharing/hour.csv', dataset_name='BikeSharing',
        target_label='cnt', evaluation_metric='root_mean_squared_error')

    print("Number of rows:", dataset_evaluator.dataframe.shape[0])
    print("Number of columns:", dataset_evaluator.dataframe.shape[1])
    print(dataset_evaluator.dataframe.dtypes)
    print(dataset_evaluator.dataframe.head())
    #dataset_evaluator.evaluate_feature_selection_step()


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


def evaluate_feature_selection_internet_advertisements_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/InternetAdvertisements/internet_advertisements.csv',
        dataset_name='InternetAds',
        target_label='class',
        evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


def evaluate_feature_selection_nursery_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Nursery/nursery.csv', dataset_name='Nursery',
        target_label='label', evaluation_metric='accuracy')

    dataset_evaluator.evaluate_feature_selection_step()


if __name__ == '__main__':
    # evaluate_census_income_dataset()
    # evaluate_breast_cancer_dataset()
    # evaluate_steel_plates_fault_dataset()
    # evaluate_connect4_dataset()
    # evaluate_housing_prices_dataset()
    # evaluate_gisette_dataset()
    # evaluate_bank_marketing_dataset()

    # Databases that have SMALL number of instances
    # evaluate_feature_selection_breast_cancer_dataset()
    # evaluate_feature_selection_bank_marketing_dataset()
    # evaluate_feature_selection_steel_plates_faults_dataset()
    # evaluate_feature_selection_arrhythmia_dataset()
    # evaluate_feature_selection_housing_prices_dataset()
    # evaluate_feature_selection_nasa_numeric_dataset()
    # evaluate_feature_selection_internet_advertisements_dataset()

    # Databases that have LARGE number of instances
    evaluate_feature_selection_gisette_dataset()
    evaluate_feature_selection_bike_sharing_dataset()
    evaluate_feature_selection_nursery_dataset()
    evaluate_feature_selection_connect4_dataset()
    evaluate_feature_selection_census_income_dataset()

    # plot_over_runtime()
    # plot_over_runtime_large_datasets()
