"""
This module provides utility functions for the preprocessing, training, and evaluation
of a cervical cancer prediction model using XGBoost.

It includes the following functions:

- wrangling_cervical_data(dataset_path: str, columns_to_drop: List[str]) -> pd.DataFrame:
    Preprocesses the cervical cancer dataset by replacing missing values, dropping columns,
    converting data types, and filling null values with column means.

- preprocess_target_column(data: pd.DataFrame, target_column: str, test_size: float = 0.3,
                           random_state: int = 4) -> Tuple:
    Preprocesses the dataset by splitting it into feature and target variables, converting
    them to the appropriate data types, applying feature scaling, and splitting the data
    into training, validation, and testing sets.

- calculate_metrics(xgb_model, x__train, y__train, x__test, y__test) -> dict:
    Calculates the evaluation metrics for an XGBoost model, including accuracy scores,
    predicted values, and a classification report.

- predict_cervical_cancer_risk(model, data) -> np.ndarray:
    Predicts the risk of cervical cancer for a single case using a trained model.

- train_xgboost_model(x__train: np.ndarray, y__train: np.ndarray) -> xgb.XGBClassifier:
    Trains an XGBoost classifier model using the provided training data.

The module imports necessary libraries and packages such as pandas, numpy, scikit-learn,
and xgboost. It provides type hints for the function parameters and returns, ensuring
clear understanding of the expected inputs and outputs.

Author: [Francesco Cocciro]
"""

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import xgboost as xgb


def wrangling_cervical_data(dataset_path: str, columns_to_drop: List[str]) -> pd.DataFrame:
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


def preprocess_target_column(data: pd.DataFrame,
                             target_column: str,
                             test_size: float = 0.3,
                             random_state: int = 4) -> Tuple:
    """
    Preprocesses the dataset by splitting it into feature and target variables,
    converting them to the appropriate data types, applying feature scaling,
    and splitting the data into training, validation, and testing sets.

    Args:
        data (pd.DataFrame): The input dataset.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
            Default is 0.2.
        random_state (int): The seed used by the random number generator. Default is 42.

    Returns:
        tuple: A tuple containing the preprocessed feature and target splits of the dataset.
               The tuple elements are: (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    features = data.drop(target_column, axis=1)
    target = data[target_column]

    features = np.array(features).astype('float32')
    target = np.array(target).astype('float32')

    scalar = MinMaxScaler()
    features = scalar.fit_transform(features)

    x__train, x__test, y__train, y__test = train_test_split(
        features, target, test_size=test_size, random_state=random_state)

    return x__train, x__test, y__train,  y__test


def calculate_metrics(xgb_model, x__train, y__train, x__test, y__test) -> dict:
    """
    Calculates the evaluation metrics for an XGBoost model.

    Parameters:
        - xgb: The trained XGBoost model instance.
        - X_train: The training set features.
        - y_train: The training set target variable.
        - X_test: The test set features.
        - y_test: The test set target variable.

    Returns:
        A dictionary containing the following calculated metrics:
        - accuracy_train: Accuracy on the training set.
        - accuracy_test: Accuracy on the test set.
        - y_hat: Predicted values on the test set.
        - classification_report: Classification report on the test set.
    """
    accuracy_train = xgb_model.score(x__train, y__train)
    accuracy_test = xgb_model.score(x__test, y__test)
    y__hat = xgb.predict(x__test)
    report = classification_report(y__test, y__hat)

    metrics_dict = {
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test,
        'y_hat': y__hat,
        'classification_report': report
    }

    for key, value in metrics_dict.items():
        print('\n', key, '\n', value)


def predict_cervical_cancer_risk(model, data) -> np.ndarray:
    """
    Predicts the risk of cervical cancer for a single case using a trained model.

    Parameters:
        - model: The trained cervical cancer risk prediction model.
        - data: The data of a single case for prediction.

    Returns:
        - prediction: The predicted risk of cervical cancer for the given case.
    """
    data = np.array(data).astype('float32').reshape(1, -1)
    prediction = model.predict(data)

    if prediction[0] == 0:
        print('The predicted risk of cervical cancer is low.')
    else:
        print('The predicted risk of cervical cancer is high.')

    return prediction[0]


def train_xgboost_model(x__train: np.ndarray, y__train: np.ndarray) -> xgb.XGBClassifier:
    """
    Trains an XGBoost classifier model using the provided training data.

    Parameters:
        - X_train (np.ndarray): The feature array of the training data.
        - y_train (np.ndarray): The target array of the training data.

    Returns:
        xgb.XGBClassifier: The trained XGBoost classifier model.
    """
    xgb_model = xgb.XGBClassifier(learning_rate=0.1, max_depth=20, n_estimators=100)
    xgb_model.fit(x__train, y__train)
    return xgb_model
