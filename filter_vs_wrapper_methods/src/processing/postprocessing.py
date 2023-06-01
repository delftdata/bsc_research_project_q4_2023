import numpy as np


def postprocess_results(raw_metrics: list[str]) -> np.ndarray:
    parsed_metrics = [[float(y) for y in x.split(",")] for x in raw_metrics]

    average_rows = [sum(metric) / len(metric) for metric in parsed_metrics]
    average = sum(average_rows) / len(average_rows)
    max_length = max([len(metric) for metric in parsed_metrics])

    processed_metrics = [([average] * (max_length - len(metric))) + metric for metric in parsed_metrics]
    mean_metrics = np.mean(np.array(processed_metrics), axis=0)

    return mean_metrics
