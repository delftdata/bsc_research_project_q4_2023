from typing import Literal

import pandas as pd
import seaborn as sns


class Plotter:

    @staticmethod
    def plot_metric_sea_born(performance: dict[str, list[float]],
                             scoring: Literal["Accuracy", "Mean Squared Error"],
                             x_axis="Percentage of selected features",
                             legend="Method"):

        data_frame_dictionary: dict[str, list] = dict()

        for (key_percentage_or_method, values_percentage_or_metric) in performance.items():
            data_frame_dictionary[key_percentage_or_method] = values_percentage_or_metric

        df = pd.DataFrame(data_frame_dictionary)
        df_melt = pd.melt(df, [x_axis])
        df_melt.rename(columns={"value": scoring, "variable": legend}, inplace=True)

        return sns.lineplot(x=x_axis, y=scoring, hue=legend, data=df_melt)
