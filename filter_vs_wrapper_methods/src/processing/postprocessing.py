import numpy as np


def postprocess_results(raw_metrics: list[str]) -> np.ndarray:
    metrics = np.array([np.array([float(y) for y in x.split(",")], dtype=np.float64)
                        for x in raw_metrics])
    metrics = np.mean(metrics, axis=0)
    return metrics
