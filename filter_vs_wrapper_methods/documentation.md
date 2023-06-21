# Automatic Feature Discovery: A comparative study between filter and wrapper feature selection techniques

## Project Documentation

The document provides a comprehensive technical and theoretical overview of the project,
focusing on the structure and potential manipulations that can be performed on the files.

## Table of Contents

-   [Datasets](#datasets)
-   [Experiments](#experiments)
-   [Editing the arguments_main.json and arguments_plot.json](#editing-the-arguments-file)
-   [Project Structure](#project-structure)

## Datasets

The datasets included in this GitHub repository were collected collaboratively with the peers, who also contributed
to the creation of the other folders in the repository.

The collected datasets exhibit diversity in terms of the number of samples, features, and types of features.

For detailed information about the datasets, please refer to the `datasets/datasets_summary.md` file.

However, please note that the information provided for some datasets may not be entirely accurate.
To provide a more reliable reference, the following table presents the interpretation of each dataset based
on the number of samples, categorical features, discrete features, and numerical features
as recorded after performing experiment 4.

| Dataset                 | #Samples | #Categorical | #Discrete | #Continuous |
| ----------------------- | -------- | ------------ | --------- | ----------- |
| Arrhythmia              | 452      | 0            | 146       | 116         |
| Bank Marketing          | 45211    | 9            | 7         | 0           |
| Bike Sharing            | 17379    | 1            | 11        | 4           |
| Breast Cancer           | 569      | 0            | 1         | 30          |
| Census Income           | 48842    | 8            | 6         | 0           |
| Character Font Images   | 745000   | 0            | 408       | 1           |
| Housing Prices          | 1460     | 43           | 34        | 3           |
| Internet Advertisements | 3279     | 4            | 1554      | 0           |
| Nasa Numeric            | 93       | 19           | 0         | 3           |
| Steel Plates Faults     | 1941     | 0            | 20        | 13          |

## Experiments

As outlined in the research paper, four experiments were carried out to evaluate the performance of filter
and wrapper feature selection techniques.

In the codebase, these experiments are labeled as `experiment1`, `experiment2`, `experiment3`, `experiment4`,
and `experiment5`.

For detailed information and a comprehensive analysis of the experiments, please refer to the research paper,
where each experiment is extensively discussed.

## Editing the arguments_main.json and arguments_plot.json

1. arguments_main.json

    The `main` function in `main.py` expects arguments to follow the specified format:

    ```json
    {
        "experiment_name": "experiment2",
        "imputation_strategy": "mean",
        "normalization": true,
        "evaluate_on_svm": false,
        "dataset": "steel_plates_faults"
    }
    ```

    Here is the meaning of each key:

    - `"experiment_name"`: Indicates the experiment to be executed. Valid options are: `"experiment1"`,
      `"experiment2"`, `"experiment3"`, `"experiment4"`, `"experiment5"`.
    - `"imputation_strategy"`: Specifies the imputation strategy to be applied to continuous data.
      Acceptable values are `"mean"`, `"median"`.
    - `"normalization"`: Determines whether normalization should be performed for ANOVA, Forward Selection,
      and Backward Elimination. Possible values are: `true`, `false`.
    - `"evaluate_on_svm"`: Specifies whether the experiments should be conducted using support vector machines or
      Autogluon models (Light GBM, Random Forest, XGBoost, Linear/Logistic Regression).
      Accepted values are: `true`, `false`.
    - `"dataset"`: Indicates the dataset to be used for the experiments. Supported datasets include: `"bank_marketing"`,
      `"breast_cancer"`, `"steel_plates_faults"`,`"housing_prices"`, `"bike_sharing"`, `"census_income"`,
      `"arrhythmia"`, `"crop"`, `"character_font_images"`, `"internet_advertisements"`, `"nasa_numeric"`.

2. arguments_plot.json

    The `main` function in `plot.py` expects arguments to follow the specified format:

    ```json
    {
        "experiment_name": "experiment1",
        "dataset": "steel_plates_faults",
        "plot_type": "average_runtime",
        "leaderboard": "all"
    }
    ```

    Here is the meaning of each key:

    - `"experiment_name"`: Specifies the experimental results to be plotted. Valid options are:
      `"experiment1"`, `"experiment2"`, `"experiment3"`, `"experiment4"`, `"experiment5"`.
    - `"dataset"`: Specifies the dataset for which the results should be considered by `main`. Supported datasets
      include:`"bank_marketing"`, `"breast_cancer"`, `"steel_plates_faults"`,
      `"housing_prices"`, `"bike_sharing"`, `"census_income"`, `"arrhythmia"`, `"crop"`,
      `"character_font_images"`, `"internet_advertisements"`, `"nasa_numeric"`.
    - `"plot_type"`: Specifies the type of plot to generate. Here is an overview of the possible values for
      `"plot_type"` and their meanings:
        - `"results"`: Plots the results for the specified `"dataset"`.
        - `"all_results"`: Plots the results for all available datasets corresponding to the given `"experiment_name"`.
          Ignores the value of `"dataset"`.
        - `"average_runtime"`: Plots the average runtime across all datasets for the specified `"experiment_name"`.
          Ignores the value of `"dataset"`.
        - `"baseline"`: Plots the baseline-based graphs for the specified `"experiment_name"`.
        - `"all_baselines"`: Plots the baseline-based graphs for all datasets for the specified `"experiment_name"`.
        - `"average_baseline"`: Plots the average baseline-based graphs across all datasets for the specified
          `"experiment_name"`.
    - `"leaderboard"`: Specifies which method to display on the baseline-based graphs. Valid options are:
      `"best"`, `"second"`, `"third"`, `"worst"`, and `"all"` to use all first four options during a single run.

## Project Structure

The project structure of the ``filter_vs_wrapper_methods` project is as follows:

-   `documentation.md`: Markdown file containing the project documentation.
-   `src`: Source code directory
-   `README.md`: Markdown file providing instructions for installation and usage of the project.

The `src` directory contains the implementation logic for conducting the comparative study between filter and
wrapper feature selection techniques. The presence of the `__init__.py` file in all subdirectories of `src`
is required to make Python treat those directories as packages<sup>[1]</sup>.

### `feature_selection_methods` directory

-   `__init__.py`
-   `filter.py`: Contains the `rank_features_descending_filter` function, which ranks feature importance
    using Chi-Squared or ANOVA tests.
-   `wrapper.py`: Contains the `rank_features_descending_wrapper` function, which ranks feature importance
    using Forward Selection or Backward Elimination.

### `processing` directory

-   `__init__.py`
-   `filter_preprocessing.py`: Contains functions for preprocessing data for filter methods.
-   `imputer.py`: Contains functions for imputing missing values in data.
-   `postprocessing.py`: Contains functions for postprocessing the results of experiments.
-   `preprocessing.py`: Contains functions for preprocessing data, such as scaling, discretizing, and normalizing.
-   `splitter.py`: Contains functions for splitting the data into subsets based on various criteria.
-   `wrapper_preprocessing.py`: Contains functions for preprocessing data for wrapper methods.

### `reader` directory

-   `data` directory: Contains the datasets used in the experiments.
-   `__init__.py`
-   `dataset_info.py`: Defines the DatasetInfo dataclass for storing dataset information.
-   `reader.py`: Defines the Reader class for reading and storing data from files into pandas DataFrames.

### `writer` directory

-   `__init__.py`
-   `writer.py`: Defines the Writer class for writing data to files.

### `plotter` directory

-   `__init__.py`
-   `plotter.py`: Contains functions for plotting the results of experiments.

### `results` directory

-   `experiment1` directory
-   `experiment2` directory
-   `experiment3` directory
-   `experiment4` directory
-   `experiment5` directory
-   `runtime` directory

### `evaluator` directory

It contains the following items:

-   `__init__.py`
-   `evaluator.py`: Defines the Evaluator class for performing experiments on Autogluon models and SVM variants.

In addition to the directories, there are several files in the `src` directory:

-   `.pylintrc`: Configuration file for the Python linter.
-   `arguments_main.json`: JSON file containing arguments for the main.py function.
-   `arguments_plot.json`: JSON file containing arguments for the plot.py function.
-   `Dockerfile`: File for creating a Docker container for the project.
-   `main.py`: Function for executing the experiments.
-   `plot.py`: Function for creating plots based on the experiment results.

[1]: https://docs.python.org/3/tutorial/modules.html
