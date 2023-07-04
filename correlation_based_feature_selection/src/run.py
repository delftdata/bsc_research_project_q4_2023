from .autofeat_pipeline import MLPipeline


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
#         dataset_file='../datasets/HousingPrices/train.csv', dataset_name='Housing Prices',
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
        dataset_file='../datasets/BreastCancer/data.csv', dataset_name='BreastCancer',
        target_label='diagnosis', evaluation_metric='accuracy', features_to_select='small')

    dataset_evaluator.evaluate_all_models()


def evaluate_internet_advertisements_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/InternetAdvertisements/internet_advertisements.csv',
        dataset_name='InternetAds',
        target_label='class',
        evaluation_metric='accuracy',
        features_to_select='large')

    dataset_evaluator.evaluate_all_models()


def evaluate_gisette_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Gisette/gisette_train.csv',
        dataset_name='Gisette',
        target_label='Class',
        evaluation_metric='accuracy',
        features_to_select='large')

    dataset_evaluator.evaluate_all_models()


def evaluate_housing_prices_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/HousingPrices/train.csv', dataset_name='HousingPrices',
        target_label='SalePrice', evaluation_metric='root_mean_squared_error', features_to_select='small')

    dataset_evaluator.evaluate_all_models()



if __name__ == '__main__':
    # Binary classification
    evaluate_breast_cancer_dataset()
    evaluate_internet_advertisements_dataset()

    # Regression
    # evaluate_housing_prices_dataset()

    # binary classification
    # evaluate_census_income_dataset()
    # evaluate_breast_cancer_dataset()
    # evaluate_steel_plates_fault_dataset()
    # evaluate_arrhythmia_dataset()
    # evaluate_internet_advertisements_dataset()
    # evaluate_gisette_dataset()
    # multi-class classification
    # evaluate_nursery_dataset()
    # evaluate_connect4_dataset()

    # regression
    # evaluate_housing_prices_dataset()
    # evaluate_bike_sharing_dataset()
