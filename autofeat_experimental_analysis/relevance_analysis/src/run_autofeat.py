from .autofeat_pipeline import MLPipeline


def evaluate_breast_cancer_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/breast_cancer/breast_cancer.csv', dataset_name='BreastCancer',
        target_label='diagnosis', evaluation_metric='accuracy', features_to_select='small')

    dataset_evaluator.evaluate_all_models()


def evaluate_spam_email_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/spam/spam.csv', dataset_name='SpamEmail',
        target_label='class', evaluation_metric='accuracy', features_to_select='small')

    dataset_evaluator.evaluate_all_models()


def evaluate_musk_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/musk/musk.csv', dataset_name='Musk',
        target_label='class', evaluation_metric='accuracy', features_to_select='medium')

    dataset_evaluator.evaluate_all_models()


def evaluate_arrhythmia_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/arrhythmia/arrhythmia.csv', dataset_name='Arrhythmia',
        target_label='binaryClass', evaluation_metric='accuracy', features_to_select='medium')

    dataset_evaluator.evaluate_all_models()


def evaluate_internet_advertisements_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/internet_advertisements/internet_advertisements.csv',
        dataset_name='InternetAds',
        target_label='class',
        evaluation_metric='accuracy',
        features_to_select='large')

    dataset_evaluator.evaluate_all_models()


def evaluate_gisette_dataset():
    dataset_evaluator = MLPipeline(
        dataset_file='../autofeat_datasets/gisette/gisette.csv',
        dataset_name='Gisette',
        target_label='Class',
        evaluation_metric='accuracy',
        features_to_select='large')

    dataset_evaluator.evaluate_all_models()


if __name__ == '__main__':
    # AutoFeat - Binary classification datasets

    # Small datasets
    evaluate_breast_cancer_dataset()
    evaluate_spam_email_dataset()
    # Medium datasets
    evaluate_musk_dataset()
    evaluate_arrhythmia_dataset()
    # Large datasets
    evaluate_internet_advertisements_dataset()
    evaluate_gisette_dataset()