"""
Cervical Cancer Prediction

This module utilizes the functions from `module_utils` to preprocess the cervical cancer dataset,
train an XGBoost classifier, and predict the risk of cervical cancer based on input data.
"""

import model_1_utils as m1u

DATASET_PATH = 'cervical_cancer_dataset.csv'
COLUMNS_TO_DROP = ['STDs: Time since last diagnosis', 'STDs: Time since first diagnosis',
                   'STDs:cervical condylomatosis', 'STDs:AIDS', 'Smokes', 'IUD',
                   'Hormonal Contraceptives', 'Smokes (packs/year)']

# Wrangle the dataset
cleaned_data = m1u.wrangling_cervical_data(DATASET_PATH, COLUMNS_TO_DROP)

# Preprocess the target column and split the data
X_train, X_test, y_train, y_test = m1u.preprocess_target_column(cleaned_data, 'Biopsy')

# Train the model
xgb = m1u.train_xgboost_model(X_train, y_train)

# Example usage: Predicting the risk of cervical cancer for a single case
single_case_data = [54.0, 3.0, 27.0, 6.0, 34.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

predicted_risk = m1u.predict_cervical_cancer_risk(xgb, single_case_data)
