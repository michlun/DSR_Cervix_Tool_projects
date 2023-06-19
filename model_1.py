"""
Import libraries
"""
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = 'cervical_cancer_dataset.csv'
COLUMNS_TO_DROP = ['STDs: Time since last diagnosis', 'STDs: Time since first diagnosis']


def preprocess_cervical_data(dataset_path: str, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Preprocesses the cervical cancer dataset by performing the following steps:

    1. Imports the dataset from the specified CSV file.
    2. Replaces '?' values with NaN.
    3. Drops the specified columns from the dataset.
    4. Converts object type columns to numeric type.
    5. Replaces null values with column means.

    Parameters:
        - dataset_path (str): The file path of the cervical cancer dataset in CSV format.
        - columns_to_drop (List[str]): A list of column names to be dropped from the dataset.
    Returns:
        pd.DataFrame: The preprocessed cervical cancer dataset.
    """

    cervical_data = pd.read_csv(dataset_path)
    cervical_data = cervical_data.replace('?', np.nan)
    cervical_data = cervical_data.drop(columns_to_drop, axis=1)
    cervical_data = cervical_data.apply(pd.to_numeric)
    cervical_data = cervical_data.fillna(cervical_data.mean())

    return cervical_data


preprocessed_data = preprocess_cervical_data(DATASET_PATH, COLUMNS_TO_DROP)

