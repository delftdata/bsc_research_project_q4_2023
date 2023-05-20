| Method / Feature type | Discrete | Continuous | Ordinal | Nominal |
| --------------------- | -------- | ---------- | ------- | ------- |
| Chi2                  | x        |            | x       | x       |
| ANOVA                 | x        | x          |         |         |
| Forward Selection     | x        | x          | x       | x       |
| Backward Elimination  | x        | x          | x       | x       |

For each of the following experiments imputation strategies for missing values may be applied.
Mean/Median/Mode/Constant imputation replaces missing values with with the mean, median, mode of the corresponding
feature or a constant, respectively.

-   Mean imputation - continuous
-   Median imputation - continuous
-   Mode imputation - numerical, categorical
-   Constant imputation - numerical, categorical

Another option is to drop the rows containing at least one missing value.

Experiment 1: Use the data as it is. Do not perform any preprocessing.

-   Will probably result in errors.

Experiment 2: Preprocess the data by converting it to the appropriate type for each method.

-   Discrete -> Continuous: MixMax Scaling
-   Continuous -> Nominal: KBinsDiscretizer
-   Nominal -> Discrete: OrdinalEncoding
-   Ordinal -> Discrete: OrdinalEncoding

Experiment 3: Drop the features from the data if they do not match the desired type. For this experiment you can
compare only the remaining features performance before and after applying the appropriate feature selection
algorithm. No comparison between feature selection techniques can be made, since some of them require different types
of data, thus different subsets of features.
