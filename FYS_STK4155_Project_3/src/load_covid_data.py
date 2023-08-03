import pandas as pd
import numpy as np

def load_covid_data():
    """
        Loading data from data/covid_data.csv and parsing content. Changing dates to unix eopch time to
        represent dates in numerical values.

        Returns
        -------
        Array
            Array with column name strings
        np.ndarray
            Two dimensional numpy array with all parsed data
    """
    data = pd.read_csv("../data/covid_data.csv", index_col=False, low_memory = False)

    headers = data.columns.tolist()

    headers = list(map(lambda x: x.replace("DATE_DIED", "DIED"), headers))

    data["DATE_DIED"] = pd.to_datetime(data["DATE_DIED"], dayfirst=True)
    data["DATE_DIED"] = pd.to_numeric(data["DATE_DIED"])

    data = data.to_numpy()
    data[:,4] = np.where(data[:, 4] == -2208988800000000000, 0, 1)

    Y = data[:,4]
    X = np.delete(data, 4, 1)

    headers = np.delete(headers, 4) #delete died from headers

    return headers, X, Y
