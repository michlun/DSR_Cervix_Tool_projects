"""
This module provides utility functions for the preprocessing, training, and evaluation
of a cervical cancer prediction model.

It includes the following functions:

- wrangling_cervical_data(dataset_path: str, columns_to_drop: List[str]) -> pd.DataFrame:
    Preprocesses the cervical cancer dataset by replacing missing values, setting columns,
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

- train_model(x__train: np.ndarray, y__train: np.ndarray):
    Trains a model using the provided training data.

The module imports necessary libraries and packages.
It provides type hints for the function parameters and returns, ensuring
clear understanding of the expected inputs and outputs.

Author: [Francesco Cocciro]
"""

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import sys
import numpy as np
import pandas as pd
import joblib as jl
import ai_apy


def wrangling_cervical_data(dataset_path: str, selected_features: List[str]) -> pd.DataFrame:
    """
    Preprocesses the cervical cancer dataset by performing the following steps:

    1. Imports the dataset from the specified CSV file.
    2. Replaces '?' values with NaN.
    3. Set the selected columns from the dataset.
    4. Converts object type columns to numeric type.
    5. Replaces null values with column means.

    Parameters:
        - dataset_path (str): The file path of the cervical cancer dataset in CSV format.
        - selected_features (List[str]): The selected columns from the dataset.
    Returns:
        pd.DataFrame: The preprocessed cervical cancer dataset.
    """

    cervical_data_original = pd.read_csv(dataset_path)
    cervical_data_original = cervical_data_original.replace('?', np.nan)
    cervical_data = cervical_data_original[selected_features]
    cervical_data = cervical_data.apply(pd.to_numeric)
    cervical_data = cervical_data.fillna(cervical_data.mean())

    return cervical_data


def preprocess_target_column(data: pd.DataFrame,
                             test_size: float = 0.2,
                             random_state: int = 42) -> Tuple:
    """
    Preprocesses the dataset by splitting it into feature and target variables,
    converting them to the appropriate data types, applying feature scaling,
    and splitting the data into training, validation, and testing sets.

    Args:
        data (pd.DataFrame): The input dataset.
        test_size (float): The proportion of the dataset to include in the test split.
            Default is 0.2.
        random_state (int): The seed used by the random number generator. Default is 42.

    Returns:
        tuple: A tuple containing the preprocessed feature and target splits of the dataset.
               The tuple elements are: (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    data['Target'] = np.logical_or.reduce([
        data['Citology'], data['Hinselmann'], data['Schiller'], data['Biopsy']]).astype(int)

    features = data.drop(['Target', 'Citology' ,'Hinselmann', 'Schiller', 'Biopsy'], axis=1)
    target = data['Target']

    features = np.array(features).astype('float32')
    target = np.array(target).astype('float32')

    scalar = MinMaxScaler()
    features = scalar.fit_transform(features)

    x__train, x__test, y__train, y__test = train_test_split(
        features, target, test_size=test_size, random_state=random_state)

    return x__train, x__test, y__train,  y__test


def calculate_metrics(model, x__train, y__train, x__test, y__test) -> dict:
    """
    Calculates the evaluation metrics for an XGBoost model.

    Parameters:
        - model: The trained model instance.
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
    accuracy_train = model.score(x__train, y__train)
    accuracy_test = model.score(x__test, y__test)
    y__hat = model.predict(x__test)
    report = classification_report(y__test, y__hat)

    metrics_dict = {
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test,
        'y_hat': y__hat,
        'classification_report': report
    }

    for key, value in metrics_dict.items():
        print('\n', key, '\n', value)


def predict_cervical_cancer_risk(data) -> Tuple:
    """
    Predicts the risk of cervical cancer for a single case using trained model.

    Parameters:
        - data: The data of a single case for prediction.

    Returns:
        - prediction: The predicted risk of cervical cancer for the given case.
    """
    try:
        # load the model
        model = jl.load('model_1.pk1')
        data = np.array(data).astype('float32').reshape(1, -1)
        prediction = model.predict(data)

        if prediction[0] == 0:
            prediction_str = 'Please generate only a text of 3 rows about the following prediction: ' \
                             'The predicted risk of cervical cancer is low.'
            text_prediction = '0'  # ai_apy.generate_response(prediction_str)
            return text_prediction
        else:
            prediction_str = 'Please generate only a text of 3 rows about the following prediction: ' \
                             'The predicted risk of cervical cancer is high.'
            text_prediction = '1'  # ai_apy.generate_response(prediction_str)
            return text_prediction

    except Exception as e:
        print(sys.exc_info())


def train_model(x__train: np.ndarray, y__train: np.ndarray):
    """
    Trains a model using the provided training data.

    Parameters:
        - X_train (np.ndarray): The feature array of the training data.
        - y_train (np.ndarray): The target array of the training data.

    Returns:
        The trained model.
    """
    # Add class_weight='balanced' to the model to have more weight on the minority class
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(x__train, y__train)

    jl.dump(lr, 'model_1.pk1')


def get_input_values(input_dict):
    """
       Extracts and returns the input values from the given input dictionary based on the selected features.

       Args:
           input_dict (dict): A dictionary containing the input values, where the keys represent the features.

       Returns:
           list: A list of input values extracted from the input dictionary, in the order of the selected features.
       """
    selected_features = ['Age',
                         'First sexual intercourse',
                         'Num of pregnancies',
                         'Smokes (years)',
                         'Smokes (packs/year)',
                         'Hormonal Contraceptives (years)',
                         'IUD (years)',
                         'STDs',
                         'STDs (number)',
                         'STDs:condylomatosis',
                         'STDs:cervical condylomatosis',
                         'STDs:vaginal condylomatosis',
                         'STDs:vulvo-perineal condylomatosis',
                         'STDs:syphilis',
                         'STDs:pelvic inflammatory disease',
                         'STDs:genital herpes',
                         'STDs:molluscum contagiosum',
                         'STDs:AIDS',
                         'STDs:HIV',
                         'STDs:Hepatitis B',
                         'STDs:HPV',
                         'STDs: Number of diagnosis',
                         'Dx:Cancer',
                         'Dx:CIN',
                         'Dx:HPV']
    sorted_dict = {key: input_dict[key] for key in selected_features if key in input_dict}
    input_value = [value for value in sorted_dict.values()]
    return input_value
