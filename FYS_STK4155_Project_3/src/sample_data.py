from enum import Enum
from imageio import imread
import numpy as np
from FrankeFunction import FrankeFunctionNoised

class DataSamplesType(Enum): 
    REAL = 0
    TEST = 1

def create_data_samples_with_real_data(path, n):
    """
    Getting x, y, z terrain data from image file.
    Paramters
    ---------
    n: int
        Amount of datapoints
    Returns
    -------
    x: np.array
        Array with the x coordinates as meshgrid
    y: np.array
        Array with the y coordinates as meshgrid
    z: np.array
        Array with the z coordinates as meshgrid
    """
    terrain = imread(path)
    terrain = terrain[:n, :n]

    x = np.linspace(0, 1, np.shape(terrain)[0])
    y = np.linspace(0, 1, np.shape(terrain)[1])
    x, y = np.meshgrid(x, y)
    z = terrain

    return x, y, z

def create_data_samples_with_franke(max_noise = 0.01): 
    """
    Creating synthetic test x, y, z terrain data using the franke function.
    Paramters
    ---------
    max_noise: float
        Noise used in franke function
    Returns
    -------
    x: np.array
        Array with the x coordinates as meshgrid
    y: np.array
        Array with the y coordinates as meshgrid
    z: np.array
        Array with the z coordinates as meshgrid
    """

    x = np.arange(0, 1, 0.075)
    y = np.arange(0, 1, 0.075)
    x, y = np.meshgrid(x,y)
    z = FrankeFunctionNoised(x,y,max_noise)

    return x, y, z

def create_data_samples(data_samples_type: DataSamplesType, real_data_path="data/SRTM_data_Norway_1.tif", real_data_n = 1000, test_data_noise = 0.01): 
    """
    Creating x, y, z data samples which is either synthetic test data or real terrain data.
    Paramters
    ---------
    data_samples_type: DataSamplesType
        Chooses between using test data or real data. Value can be either set to TEST or REAL
    real_data_path: str
        Setting the path to the real terrain data image. Defaults to data/SRTM_data_Norway_1.tif
    
    real_data_n: int
        Amount of datapoints to use when using real data. Defaults to 1000.
    test_data_noise: float
        The noise to use for the franke function when using test data. Defaults to 0.01
    Returns
    -------
    x: np.array
        Array with the x coordinates as meshgrid
    y: np.array
        Array with the y coordinates as meshgrid
    z: np.array
        Array with the z coordinates as meshgrid
    """
    if data_samples_type == DataSamplesType.REAL:
        return create_data_samples_with_real_data(real_data_path, real_data_n)
    elif data_samples_type == DataSamplesType.TEST:
        return create_data_samples_with_franke(test_data_noise)

    raise Exception("Invalid DataSamplesType")