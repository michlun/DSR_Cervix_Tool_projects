"""
This module utilizes the functions from `module_utils` to preprocess the cervical cancer dataset,
train and save the model.
"""


import model_1_utils as m1u

DATASET_PATH = 'cervical_cancer_dataset.csv'
# STDs (number), First sexual intercourse,Num of pregnancies
SELECTED_FEATURES = [
    'Age',
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
    'Dx:HPV',
    'Citology', 'Hinselmann', 'Schiller', 'Biopsy'
]

# Wrangle the dataset
cleaned_data = m1u.wrangling_cervical_data(DATASET_PATH, SELECTED_FEATURES)

# Preprocess the target column and split the data
X_train, X_test, y_train, y_test = m1u.preprocess_target_column(cleaned_data)

# Train the model and save the model
m1u.train_model(X_train, y_train)
