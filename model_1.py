"""
Cervical Cancer Prediction

This module is used to predict the risk of cervical cancer based on input data.
"""

import model_1_utils as m1u
import joblib as jl

# Load the data
load_data = jl.load('model_1.pk1')

# Example usage: Predicting the risk of cervical cancer for a single case
single_case_data = [54.0, 3.0, 27.0, 6.0, 34.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

predicted_risk = m1u.predict_cervical_cancer_risk(load_data, single_case_data)
