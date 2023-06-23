"""
This module utilizes the functions from `module_utils` to preprocess the cervical cancer dataset,
train an XGBoost classifier, and save the model.
"""


import model_1_utils as m1u

DATASET_PATH = 'cervical_cancer_dataset.csv'
COLUMNS_TO_DROP = ['STDs: Time since last diagnosis', 'STDs: Time since first diagnosis',
                   'STDs:cervical condylomatosis', 'STDs:AIDS', 'Smokes', 'IUD',
                   'Hormonal Contraceptives', 'Smokes (packs/year)']

# Wrangle the dataset
cleaned_data = m1u.wrangling_cervical_data(DATASET_PATH, COLUMNS_TO_DROP)
target_column = ['Biopsy']

# Preprocess the target column and split the data
X_train, X_test, y_train, y_test = m1u.preprocess_target_column(cleaned_data, target_column)

# Train the model and save the model
m1u.train_xgboost_model(X_train, y_train)
