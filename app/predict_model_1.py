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

    def predict(self, feature):
        load_model = jl.load('model_1.pk1')


class CustomData:
    def __init__(self,
                 age,
                 first_sexual_intercourse,
                 num_of_pregnancies,
                 smokes_years,
                 smokes_packs_year,
                 hormonal_contraceptives_years,
                 iud_years,
                 stds,
                 stds_number,
                 stds_condylomatosis,
                 stds_cervical_condylomatosis,
                 stds_vaginal_condylomatosis,
                 stds_vulvo_perineal_condylomatosis,
                 stds_syphilis,
                 stds_pelvic_inflammatory_disease,
                 stds_genital_herpes,
                 stds_molluscum_contagiosum,
                 stds_aids,
                 stds_hiv,
                 stds_hepatitis_b,
                 stds_hpv,
                 stds_number_of_diagnosis,
                 dx_cancer,
                 dx_cin,
                 dx_hpv):
        self.age = age
        self.first_sexual_intercourse = first_sexual_intercourse
        self.num_of_pregnancies = num_of_pregnancies
        self.smokes_years = smokes_years
        self.smokes_packs_year = smokes_packs_year
        self.hormonal_contraceptives_years = hormonal_contraceptives_years
        self.iud_years = iud_years
        self.stds = stds
        self.stds_number = stds_number
        self.stds_condylomatosis = stds_condylomatosis
        self.stds_cervical_condylomatosis = stds_cervical_condylomatosis
        self.stds_vaginal_condylomatosis = stds_vaginal_condylomatosis
        self.stds_vulvo_perineal_condylomatosis = stds_vulvo_perineal_condylomatosis
        self.stds_syphilis = stds_syphilis
        self.stds_pelvic_inflammatory_disease = stds_pelvic_inflammatory_disease
        self.stds_genital_herpes = stds_genital_herpes
        self.stds_molluscum_contagiosum = stds_molluscum_contagiosum
        self.stds_aids = stds_aids
        self.stds_hiv = stds_hiv
        self.stds_hepatitis_b = stds_hepatitis_b
        self.stds_hpv = stds_hpv
        self.stds_number_of_diagnosis = stds_number_of_diagnosis
        self.dx_cancer = dx_cancer
        self.dx_cin = dx_cin
        self.dx_hpv = dx_hpv



#
# # Load the data
# load_model = jl.load('model_1.pk1')
#
# # Example usage: Predicting the risk of cervical cancer for a single case
# single_case_data = [54.0, 3.0, 27.0, 6.0, 34.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
#
# predicted_risk = m1u.predict_cervical_cancer_risk(load_model, single_case_data)
