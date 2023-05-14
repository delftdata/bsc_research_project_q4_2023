from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def prepare_data_for_ml(dataframe):
    df = dataframe.fillna(-1)
    df = df.apply(lambda x: pd.factorize(x)[0] if x.dtype == object else x)

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(df)
    normalized_X = pd.DataFrame(scaled_X, columns=df.columns)

    return normalized_X