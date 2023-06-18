from __future__ import annotations

import json
import os


class Writer:

    @staticmethod
    def write_selected_features(results_path: str, selected_features: list[str], method: str):
        """Writes the selected features to files for the given method.

        Parameters
        ----------
        results_path : str
            The path to the results directory.
        selected_features : list[str]
            The list of selected features.
        method : str
            The name of the method used for feature selection.

        """
        for selected_feature in selected_features:
            Writer.write_to_file(f"{results_path}/selected_features", f"{method}.txt", selected_feature)

    @staticmethod
    def write_runtime(results_path: str, runtime: float, method: str):
        """Writes the runtime of a specific method to a file.

        Parameters
        ----------
        results_path : str
            The path to the results directory.
        runtime : float
            The runtime of the method.
        method : str
            The name of the method.

        Raises
        ------
        OSError
            If there is an error while writing the runtime to the file.
        """
        Writer.write_to_file(f"{results_path}/runtime", f"{method}.txt", str(runtime))

    @staticmethod
    def write_performance(results_path: str, performance: dict[str, list[float]], method):
        """Writes the performance of different algorithms for a specific method to separate files.

        Parameters
        ----------
        results_path : str
            The path to the results directory.
        performance : dict[str, list[float]]
            A dictionary where the keys are the algorithm names and the values are lists of performance scores.
        method : str
            The name of the method.

        Raises
        ------
        OSError
            If there is an error while writing the performance to the files.
        """
        for (algorithm, performance_algorithm) in performance.items():
            content = ",".join([str(x) for x in performance_algorithm])
            Writer.write_to_file(f"{results_path}/{method}", f"{algorithm}.txt", content)

    @staticmethod
    def write_to_file(path: str, results_file_name: str, content: str, mode="a+"):
        """Writes the provided content to a file specified by the path and filename.

        Parameters
        ----------
        path : str
            The path to the directory where the file should be stored.
        results_file_name : str
            The name of the file to write the content to.
        content : str
            The content to write to the file.
        mode : str, optional
            The mode in which the file should be opened (default: "a+").
            Supported modes: "r", "w", "a", "r+", "w+", "a+", etc.
        """
        path_to_file = f"{path}/{results_file_name}"
        try:
            if not os.path.isdir(path):
                os.makedirs(path)
            with open(path_to_file, mode=mode, encoding="utf-8") as file:
                file.write(content + "\n")
        except OSError as error:
            print(f"An error occurred: {error}")

    @staticmethod
    def write_json_to_file(path: str, results_file_name: str, json_content: dict | list, mode="w"):
        """Writes the provided content to a file specified by the path and filename.

        Parameters
        ----------
        path : str
            The path to the directory where the file should be stored.
        results_file_name : str
            The name of the file to write the content to.
        json_content
            The json content to write to the file.
        mode : str, optional
            The mode in which the file should be opened (default: "a+").
            Supported modes: "r", "w", "a", "r+", "w+", "a+", etc.
        """
        path_to_file = f"{path}/{results_file_name}"
        try:
            if not os.path.isdir(path):
                os.makedirs(path)
            with open(path_to_file, mode=mode, encoding="utf-8") as file:
                json.dump(json_content, file)
        except OSError as error:
            print(f"An error occurred: {error}")
