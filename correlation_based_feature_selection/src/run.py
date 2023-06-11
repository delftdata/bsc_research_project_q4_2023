from .pipeline import MLPipeline


def evaluate_census_income_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/CensusIncome/CensusIncome.csv',
        dataset_name='CensusIncome',
        target_label='income_label',
        evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


def evaluate_breast_cancer_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/BreastCancer/data.csv',
        dataset_name='BreastCancer',
        target_label='diagnosis',
        evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


def evaluate_steel_plates_fault_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/SteelPlatesFaults/steel_faults_train.csv',
        dataset_name='SteelPlatesFaults',
        target_label='Class',
        evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


def evaluate_connect4_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Connect-4/Connect-4.csv',
        dataset_name='Connect4',
        target_label='label',
        evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


def evaluate_housing_prices_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/HousingPrices/train.csv',
        dataset_name='Housing Prices',
        target_label='SalePrice',
        evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


def evaluate_gisette_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../datasets/Gisette/gisette_train.csv',
        dataset_name='Gisette',
        target_label='Class',
        evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


if __name__ == '__main__':
    # evaluate_census_income_dataset()
    # evaluate_breast_cancer_dataset()
    # evaluate_steel_plates_fault_dataset()
    evaluate_connect4_dataset()
    # evaluate_housing_prices_dataset()
    # evaluate_gisette_dataset()
