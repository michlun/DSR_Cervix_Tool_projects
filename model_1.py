"""
Import libraries
"""
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


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
                             test_size: float = 0.2,
                             random_state: int = 42) -> Tuple:
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
    x = data.drop(target_column, axis=1)
    y = data[target_column]

    x = np.array(x).astype('float32')
    y = np.array(y).astype('float32')

    scalar = StandardScaler()
    x = scalar.fit_transform(x)

    x__train, x__test, y__train, y__test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

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

    return metrics_dict


DATASET_PATH = 'cervical_cancer_dataset.csv'
COLUMNS_TO_DROP = ['STDs: Time since last diagnosis', 'STDs: Time since first diagnosis']
# Wrangle the dataset
cleaned_data = wrangling_cervical_data(DATASET_PATH, COLUMNS_TO_DROP)
# Preprocess the target column and split the data
X_train, X_test, y_train, y_test = preprocess_target_column(cleaned_data, 'Biopsy')

# Train the model
xgb = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=20)
xgb.fit(X_train, y_train)

metrics = calculate_metrics(xgb, X_train, y_train, X_test, y_test)

# print element from dictionary
for k, v in metrics.items():
    print('\n', k, '\n', v)
