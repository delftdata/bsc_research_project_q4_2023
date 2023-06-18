from dataclasses import dataclass
from typing import Literal


@dataclass
class DatasetInfo:
    """
    A dataclass used to represent the dataset information needed to perform the experiments.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset.
    dataset_path : str
        Path of the dataset.
    target_label : str
        Name of the target label.
    results_path : str
        Path to the file where the results obtained from performing the experiments on the dataset should be stored.
    eval_metric : Literal["accuracy", "neg_root_mean_squared_error"], optional
        Evaluation metric of the dataset (default: "accuracy").
        For classification tasks, use "accuracy".
        For regression tasks, use "neg_root_mean_squared_error".
    file_names : str, optional
        Only used by the character font images dataset, since it is split into multiple files (default: "").
    """
    dataset_name: str
    dataset_path: str
    target_label: str
    results_path: str
    eval_metric: Literal["accuracy", "neg_root_mean_squared_error"] = "accuracy"
    file_names: str = ""
