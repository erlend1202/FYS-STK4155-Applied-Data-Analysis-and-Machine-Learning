import pandas as pd
import numpy as np

def load_data():
    """
        Loading data from data/data.csv and parsing content. Changing dates to unix eopch time to
        represent dates in numerical values.
        Returns
        -------
        Array
            Array with column name strings
        np.ndarray
            Two dimensional numpy array with all parsed data
    """
    data = pd.read_csv("../data/data.csv", index_col=False, low_memory = False)

    headers = data.columns.tolist()


    data = data.to_numpy()

    headers = np.delete(headers, 7) #Removing country-year as we already have it seperatly
    headers = np.delete(headers, 10) #Removing generation

    data = np.delete(data, 7, 1) #Removing country-year as we already have it seperatly
    data = np.delete(data, 10, 1) #Removing generation




    ##This is to change from nan to 0 for HDI
    data[:,7] = np.where(data[:,7] != data[:,7], 0, data[:,7])

    return headers, data 
