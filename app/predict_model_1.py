"""
Cervical Cancer Prediction

This module is used to predict the risk of cervical cancer based on input data.
"""
import sys
import pandas as pd
import model_1_utils as m1u
import joblib as jl


class Predict:
    def __init__(self):
        pass

    @staticmethod
    def predict(feature):
        try:
            load_model = jl.load('model_1.pk1')
            single_case_data = feature
            predicted_risk = m1u.predict_cervical_cancer_risk(load_model, single_case_data)
            return predicted_risk
        except Exception as e:
            print(sys.exc_info())


class CustomData:
    def __init__(self, input_dict):
        self.selected_features = ['Age',
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
        self.sorted_dict = {key: input_dict[key] for key in self.selected_features if key in input_dict}
        self.input_value = [value for value in self.sorted_dict.values()]




#
# # Load the data
# load_model = jl.load('model_1.pk1')
#
# # Example usage: Predicting the risk of cervical cancer for a single case
# single_case_data = [54.0, 3.0, 27.0, 6.0, 34.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
#
# predicted_risk = m1u.predict_cervical_cancer_risk(load_model, single_case_data)
