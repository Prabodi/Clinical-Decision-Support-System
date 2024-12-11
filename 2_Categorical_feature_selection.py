# 1. Import Libraries

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import statistics
from collections import Counter  # to count class labels distribution
import math
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # Gaussian Naive Bayes
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, matthews_corrcoef, \
    f1_score, auc, roc_curve, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer  # from sklearn.metrics import fbeta_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.impute import KNNImputer
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE  # Recursive feature elimination
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)  # None

# ------------------------------------------------------------------------------------------------------------------------------------

# 2. Import files

df_train_all = pd.read_csv(sys.argv[1])  # after prepocessing - training set only
df_test_all = pd.read_csv(sys.argv[2])  # after prepocessing - test set only

# from data set, drop 'hadm_id'

df_train_all = df_train_all.drop('hadm_id', axis=1)
df_test_all = df_test_all.drop('hadm_id', axis=1)

# Rounding-off values of selected columns (which have a lot of decimal points) into 2 decimal points

df_train_all = df_train_all.round(
    {'heart_rate_mean': 2, 'sbp_mean': 2, 'dbp_mean': 2, 'mbp_mean': 2, 'resp_rate_mean': 2, 'temperature_mean': 2,
     'spo2_mean': 2, 'glucose_mean': 2})
df_train_all = df_train_all.round(
    {'heart_rate_mean': 2, 'sbp_mean': 2, 'dbp_mean': 2, 'mbp_mean': 2, 'resp_rate_mean': 2, 'temperature_mean': 2,
     'spo2_mean': 2, 'glucose_mean': 2})

df_test_all = df_test_all.round(
    {'heart_rate_mean': 2, 'sbp_mean': 2, 'dbp_mean': 2, 'mbp_mean': 2, 'resp_rate_mean': 2, 'temperature_mean': 2,
     'spo2_mean': 2, 'glucose_mean': 2})
df_test_all = df_test_all.round(
    {'heart_rate_mean': 2, 'sbp_mean': 2, 'dbp_mean': 2, 'mbp_mean': 2, 'resp_rate_mean': 2, 'temperature_mean': 2,
     'spo2_mean': 2, 'glucose_mean': 2})

print(df_train_all['heart_rate_mean'].head())
# col_names = ['encoder__first_careunit_Cardiac Vascular Intensive Care Unit (CVICU)', 'encoder__first_careunit_Coronary Care Unit (CCU)', 'encoder__first_careunit_Medical Intensive Care Unit (MICU)', 'encoder__first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)', 'encoder__first_careunit_Neuro Intermediate', 'encoder__first_careunit_Neuro Stepdown', 'encoder__first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)', 'encoder__first_careunit_Surgical Intensive Care Unit (SICU)', 'encoder__first_careunit_Trauma SICU (TSICU)', 'encoder__admission_type_DIRECT EMER.', 'encoder__admission_type_DIRECT OBSERVATION', 'encoder__admission_type_ELECTIVE', 'encoder__admission_type_EU OBSERVATION', 'encoder__admission_type_EW EMER.', 'encoder__admission_type_OBSERVATION ADMIT', 'encoder__admission_type_SURGICAL SAME DAY ADMISSION', 'encoder__admission_type_URGENT', 'encoder__admission_location_AMBULATORY SURGERY TRANSFER', 'encoder__admission_location_CLINIC REFERRAL', 'encoder__admission_location_EMERGENCY ROOM', 'encoder__admission_location_INFORMATION NOT AVAILABLE', 'encoder__admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'encoder__admission_location_PACU', 'encoder__admission_location_PHYSICIAN REFERRAL', 'encoder__admission_location_PROCEDURE SITE', 'encoder__admission_location_TRANSFER FROM HOSPITAL', 'encoder__admission_location_TRANSFER FROM SKILLED NURSING FACILITY', 'encoder__admission_location_WALK-IN/SELF REFERRAL', 'encoder__hep_types_LMWH', 'encoder__hep_types_UFH', 'encoder__hep_types_both', 'encoder__treatment_types_P', 'encoder__treatment_types_T', 'encoder__treatment_types_both', 'encoder__lactate_min_status_elevated', 'encoder__lactate_min_status_low', 'encoder__lactate_min_status_normal', 'encoder__lactate_min_status_not ordered', 'encoder__lactate_max_status_elevated', 'encoder__lactate_max_status_low', 'encoder__lactate_max_status_normal', 'encoder__lactate_max_status_not ordered', 'encoder__ph_min_status_elevated', 'encoder__ph_min_status_low', 'encoder__ph_min_status_normal', 'encoder__ph_min_status_not ordered', 'encoder__ph_max_status_elevated', 'encoder__ph_max_status_low', 'encoder__ph_max_status_normal', 'encoder__ph_max_status_not ordered', 'encoder__totalco2_bg_min_status_elevated', 'encoder__totalco2_bg_min_status_low', 'encoder__totalco2_bg_min_status_normal', 'encoder__totalco2_bg_min_status_not ordered', 'encoder__totalco2_bg_max_status_elevated', 'encoder__totalco2_bg_max_status_low', 'encoder__totalco2_bg_max_status_normal', 'encoder__totalco2_bg_max_status_not ordered', 'encoder__pco2_bg_min_status_elevated', 'encoder__pco2_bg_min_status_low', 'encoder__pco2_bg_min_status_normal', 'encoder__pco2_bg_min_status_not ordered', 'encoder__pco2_bg_max_status_elevated', 'encoder__pco2_bg_max_status_low', 'encoder__pco2_bg_max_status_normal', 'encoder__pco2_bg_max_status_not ordered', 'encoder__ast_min_status_elevated', 'encoder__ast_min_status_normal', 'encoder__ast_min_status_not ordered', 'encoder__ast_max_status_elevated', 'encoder__ast_max_status_normal', 'encoder__ast_max_status_not ordered', 'encoder__alp_min_status_elevated', 'encoder__alp_min_status_low', 'encoder__alp_min_status_normal', 'encoder__alp_min_status_not ordered', 'encoder__alp_max_status_elevated', 'encoder__alp_max_status_low', 'encoder__alp_max_status_normal', 'encoder__alp_max_status_not ordered', 'encoder__alt_min_status_elevated', 'encoder__alt_min_status_normal', 'encoder__alt_min_status_not ordered', 'encoder__alt_max_status_elevated', 'encoder__alt_max_status_normal', 'encoder__alt_max_status_not ordered', 'encoder__bilirubin_total_min_status_elevated', 'encoder__bilirubin_total_min_status_normal', 'encoder__bilirubin_total_min_status_not ordered', 'encoder__bilirubin_total_max_status_elevated', 'encoder__bilirubin_total_max_status_normal', 'encoder__bilirubin_total_max_status_not ordered', 'encoder__albumin_min_status_elevated', 'encoder__albumin_min_status_low', 'encoder__albumin_min_status_normal', 'encoder__albumin_min_status_not ordered', 'encoder__albumin_max_status_elevated', 'encoder__albumin_max_status_low', 'encoder__albumin_max_status_normal', 'encoder__albumin_max_status_not ordered', 'encoder__pco2_bg_art_min_status_elevated', 'encoder__pco2_bg_art_min_status_low', 'encoder__pco2_bg_art_min_status_normal', 'encoder__pco2_bg_art_min_status_not ordered', 'encoder__pco2_bg_art_max_status_elevated', 'encoder__pco2_bg_art_max_status_low', 'encoder__pco2_bg_art_max_status_normal', 'encoder__pco2_bg_art_max_status_not ordered', 'encoder__po2_bg_art_min_status_elevated', 'encoder__po2_bg_art_min_status_low', 'encoder__po2_bg_art_min_status_normal', 'encoder__po2_bg_art_min_status_not ordered', 'encoder__po2_bg_art_max_status_elevated', 'encoder__po2_bg_art_max_status_low', 'encoder__po2_bg_art_max_status_normal', 'encoder__po2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_min_status_elevated', 'encoder__totalco2_bg_art_min_status_low', 'encoder__totalco2_bg_art_min_status_normal', 'encoder__totalco2_bg_art_min_status_not ordered', 'encoder__totalco2_bg_art_max_status_elevated', 'encoder__totalco2_bg_art_max_status_low', 'encoder__totalco2_bg_art_max_status_normal', 'encoder__totalco2_bg_art_max_status_not ordered', 'encoder__ld_ldh_min_status_elevated', 'encoder__ld_ldh_min_status_low', 'encoder__ld_ldh_min_status_normal', 'encoder__ld_ldh_min_status_not ordered', 'encoder__ld_ldh_max_status_elevated', 'encoder__ld_ldh_max_status_low', 'encoder__ld_ldh_max_status_normal', 'encoder__ld_ldh_max_status_not ordered', 'encoder__ck_cpk_min_status_elevated', 'encoder__ck_cpk_min_status_low', 'encoder__ck_cpk_min_status_normal', 'encoder__ck_cpk_min_status_not ordered', 'encoder__ck_cpk_max_status_elevated', 'encoder__ck_cpk_max_status_low', 'encoder__ck_cpk_max_status_normal', 'encoder__ck_cpk_max_status_not ordered', 'encoder__ck_mb_min_status_elevated', 'encoder__ck_mb_min_status_normal', 'encoder__ck_mb_min_status_not ordered', 'encoder__ck_mb_max_status_elevated', 'encoder__ck_mb_max_status_normal', 'encoder__ck_mb_max_status_not ordered', 'encoder__fio2_bg_art_min_status_no ref range', 'encoder__fio2_bg_art_min_status_not ordered', 'encoder__fio2_bg_art_max_status_no ref range', 'encoder__fio2_bg_art_max_status_not ordered', 'encoder__so2_bg_art_min_status_no ref range', 'encoder__so2_bg_art_min_status_not ordered', 'encoder__so2_bg_art_max_status_no ref range', 'encoder__so2_bg_art_max_status_not ordered', 'encoder__fibrinogen_min_status_elevated', 'encoder__fibrinogen_min_status_low', 'encoder__fibrinogen_min_status_normal', 'encoder__fibrinogen_min_status_not ordered', 'encoder__fibrinogen_max_status_elevated', 'encoder__fibrinogen_max_status_low', 'encoder__fibrinogen_max_status_normal', 'encoder__fibrinogen_max_status_not ordered', 'encoder__thrombin_min_status_elevated', 'encoder__thrombin_min_status_normal', 'encoder__thrombin_min_status_not ordered', 'encoder__thrombin_max_status_elevated', 'encoder__thrombin_max_status_normal', 'encoder__thrombin_max_status_not ordered', 'encoder__d_dimer_min_status_elevated', 'encoder__d_dimer_min_status_normal', 'encoder__d_dimer_min_status_not ordered', 'encoder__d_dimer_max_status_elevated', 'encoder__d_dimer_max_status_normal', 'encoder__d_dimer_max_status_not ordered', 'encoder__methemoglobin_min_status_elevated', 'encoder__methemoglobin_min_status_normal', 'encoder__methemoglobin_min_status_not ordered', 'encoder__methemoglobin_max_status_elevated', 'encoder__methemoglobin_max_status_normal', 'encoder__methemoglobin_max_status_not ordered', 'encoder__ggt_min_status_elevated', 'encoder__ggt_min_status_low', 'encoder__ggt_min_status_normal', 'encoder__ggt_min_status_not ordered', 'encoder__ggt_max_status_elevated', 'encoder__ggt_max_status_low', 'encoder__ggt_max_status_normal', 'encoder__ggt_max_status_not ordered', 'encoder__globulin_min_status_elevated', 'encoder__globulin_min_status_low', 'encoder__globulin_min_status_normal', 'encoder__globulin_min_status_not ordered', 'encoder__globulin_max_status_elevated', 'encoder__globulin_max_status_low', 'encoder__globulin_max_status_normal', 'encoder__globulin_max_status_not ordered', 'encoder__atyps_min_status_elevated', 'encoder__atyps_min_status_not ordered', 'encoder__atyps_max_status_elevated', 'encoder__atyps_max_status_not ordered', 'encoder__total_protein_min_status_elevated', 'encoder__total_protein_min_status_low', 'encoder__total_protein_min_status_normal', 'encoder__total_protein_min_status_not ordered', 'encoder__total_protein_max_status_elevated', 'encoder__total_protein_max_status_low', 'encoder__total_protein_max_status_normal', 'encoder__total_protein_max_status_not ordered', 'encoder__carboxyhemoglobin_min_status_elevated', 'encoder__carboxyhemoglobin_min_status_normal', 'encoder__carboxyhemoglobin_min_status_not ordered', 'encoder__carboxyhemoglobin_max_status_elevated', 'encoder__carboxyhemoglobin_max_status_normal', 'encoder__carboxyhemoglobin_max_status_not ordered', 'encoder__amylase_min_status_elevated', 'encoder__amylase_min_status_normal', 'encoder__amylase_min_status_not ordered', 'encoder__amylase_max_status_elevated', 'encoder__amylase_max_status_normal', 'encoder__amylase_max_status_not ordered', 'encoder__aado2_bg_art_min_status_no ref range', 'encoder__aado2_bg_art_min_status_not ordered', 'encoder__aado2_bg_art_max_status_no ref range', 'encoder__aado2_bg_art_max_status_not ordered', 'encoder__bilirubin_direct_min_status_elevated', 'encoder__bilirubin_direct_min_status_normal', 'encoder__bilirubin_direct_min_status_not ordered', 'encoder__bilirubin_direct_max_status_elevated', 'encoder__bilirubin_direct_max_status_normal', 'encoder__bilirubin_direct_max_status_not ordered', 'encoder__nrbc_min_status_elevated', 'encoder__nrbc_min_status_not ordered', 'encoder__nrbc_max_status_elevated', 'encoder__nrbc_max_status_not ordered', 'encoder__bands_min_status_elevated', 'encoder__bands_min_status_normal', 'encoder__bands_min_status_not ordered', 'encoder__bands_max_status_elevated', 'encoder__bands_max_status_normal', 'encoder__bands_max_status_not ordered', 'remainder__gender', 'remainder__anchor_age', 'remainder__base_platelets', 'remainder__heart_rate_min', 'remainder__heart_rate_max', 'remainder__heart_rate_mean', 'remainder__sbp_min', 'remainder__sbp_max', 'remainder__sbp_mean', 'remainder__dbp_min', 'remainder__dbp_max', 'remainder__dbp_mean', 'remainder__mbp_min', 'remainder__mbp_max', 'remainder__mbp_mean', 'remainder__resp_rate_min', 'remainder__resp_rate_max', 'remainder__resp_rate_mean', 'remainder__temperature_min', 'remainder__temperature_max', 'remainder__temperature_mean', 'remainder__spo2_min', 'remainder__spo2_max', 'remainder__spo2_mean', 'remainder__glucose_min', 'remainder__glucose_max', 'remainder__glucose_mean', 'remainder__hematocrit_min', 'remainder__hematocrit_max', 'remainder__hemoglobin_min', 'remainder__hemoglobin_max', 'remainder__bicarbonate_min', 'remainder__bicarbonate_max', 'remainder__calcium_min', 'remainder__calcium_max', 'remainder__chloride_min', 'remainder__chloride_max', 'remainder__sodium_min', 'remainder__sodium_max', 'remainder__potassium_min', 'remainder__potassium_max', 'remainder__platelets_min', 'remainder__platelets_max', 'remainder__wbc_min', 'remainder__wbc_max', 'remainder__aniongap_min', 'remainder__aniongap_max', 'remainder__bun_min', 'remainder__bun_max', 'remainder__creatinine_min', 'remainder__creatinine_max', 'remainder__inr_min', 'remainder__inr_max', 'remainder__pt_min', 'remainder__pt_max', 'remainder__ptt_min', 'remainder__ptt_max', 'remainder__gcs_min', 'label']

# --------------------------------------------------------------------------------------------------------------

# categorical feature selection

df_train_cat = df_train_all[['first_careunit', 'admission_type', 'admission_location', 'gender', 'hep_types',
                             'treatment_types', 'thrombin_min_status', 'thrombin_max_status', 'd_dimer_max_status',
                             'd_dimer_min_status', 'methemoglobin_min_status', 'methemoglobin_max_status',
                             'ggt_min_status', 'ggt_max_status', 'globulin_min_status', 'globulin_max_status',
                             'total_protein_min_status', 'total_protein_max_status', 'atyps_max_status',
                             'atyps_min_status', 'carboxyhemoglobin_min_status', 'carboxyhemoglobin_max_status',
                             'amylase_max_status', 'amylase_min_status', 'aado2_bg_art_max_status',
                             'aado2_bg_art_min_status', 'bilirubin_direct_min_status', 'bilirubin_direct_max_status',
                             'bicarbonate_bg_min_status', 'bicarbonate_bg_max_status', 'fio2_bg_art_min_status',
                             'fio2_bg_art_max_status', 'nrbc_max_status', 'nrbc_min_status', 'bands_min_status',
                             'bands_max_status', 'so2_bg_art_min_status', 'so2_bg_art_max_status',
                             'fibrinogen_max_status', 'fibrinogen_min_status', 'hematocrit_bg_min_status',
                             'hematocrit_bg_max_status', 'hemoglobin_bg_min_status', 'hemoglobin_bg_max_status',
                             'temperature_bg_max_status', 'temperature_bg_min_status', 'chloride_bg_min_status',
                             'chloride_bg_max_status', 'sodium_bg_max_status', 'sodium_bg_min_status',
                             'glucose_bg_max_status', 'glucose_bg_min_status', 'ck_cpk_max_status', 'ck_cpk_min_status',
                             'ck_mb_max_status', 'ck_mb_min_status', 'ld_ldh_max_status', 'ld_ldh_min_status',
                             'calcium_bg_max_status', 'calcium_bg_min_status', 'pco2_bg_art_min_status',
                             'po2_bg_art_max_status', 'totalco2_bg_art_max_status', 'totalco2_bg_art_min_status',
                             'pco2_bg_art_max_status', 'po2_bg_art_min_status', 'potassium_bg_min_status',
                             'potassium_bg_max_status', 'albumin_max_status', 'albumin_min_status',
                             'bilirubin_total_min_status', 'bilirubin_total_max_status', 'alt_max_status',
                             'alt_min_status', 'alp_max_status', 'alp_min_status', 'ast_min_status', 'ast_max_status',
                             'pco2_bg_max_status', 'pco2_bg_min_status', 'totalco2_bg_min_status',
                             'totalco2_bg_max_status', 'ph_min_status', 'ph_max_status', 'lactate_min_status',
                             'lactate_max_status', "label"]]

df_test_cat = df_test_all[['first_careunit', 'admission_type', 'admission_location', 'gender', 'hep_types',
                           'treatment_types', 'thrombin_min_status', 'thrombin_max_status', 'd_dimer_max_status',
                           'd_dimer_min_status', 'methemoglobin_min_status', 'methemoglobin_max_status',
                           'ggt_min_status', 'ggt_max_status', 'globulin_min_status', 'globulin_max_status',
                           'total_protein_min_status', 'total_protein_max_status', 'atyps_max_status',
                           'atyps_min_status', 'carboxyhemoglobin_min_status', 'carboxyhemoglobin_max_status',
                           'amylase_max_status', 'amylase_min_status', 'aado2_bg_art_max_status',
                           'aado2_bg_art_min_status', 'bilirubin_direct_min_status', 'bilirubin_direct_max_status',
                           'bicarbonate_bg_min_status', 'bicarbonate_bg_max_status', 'fio2_bg_art_min_status',
                           'fio2_bg_art_max_status', 'nrbc_max_status', 'nrbc_min_status', 'bands_min_status',
                           'bands_max_status', 'so2_bg_art_min_status', 'so2_bg_art_max_status',
                           'fibrinogen_max_status', 'fibrinogen_min_status', 'hematocrit_bg_min_status',
                           'hematocrit_bg_max_status', 'hemoglobin_bg_min_status', 'hemoglobin_bg_max_status',
                           'temperature_bg_max_status', 'temperature_bg_min_status', 'chloride_bg_min_status',
                           'chloride_bg_max_status', 'sodium_bg_max_status', 'sodium_bg_min_status',
                           'glucose_bg_max_status', 'glucose_bg_min_status', 'ck_cpk_max_status', 'ck_cpk_min_status',
                           'ck_mb_max_status', 'ck_mb_min_status', 'ld_ldh_max_status', 'ld_ldh_min_status',
                           'calcium_bg_max_status', 'calcium_bg_min_status', 'pco2_bg_art_min_status',
                           'po2_bg_art_max_status', 'totalco2_bg_art_max_status', 'totalco2_bg_art_min_status',
                           'pco2_bg_art_max_status', 'po2_bg_art_min_status', 'potassium_bg_min_status',
                           'potassium_bg_max_status', 'albumin_max_status', 'albumin_min_status',
                           'bilirubin_total_min_status', 'bilirubin_total_max_status', 'alt_max_status',
                           'alt_min_status', 'alp_max_status', 'alp_min_status', 'ast_min_status', 'ast_max_status',
                           'pco2_bg_max_status', 'pco2_bg_min_status', 'totalco2_bg_min_status',
                           'totalco2_bg_max_status', 'ph_min_status', 'ph_max_status', 'lactate_min_status',
                           'lactate_max_status', "label"]]

x_train_cat = df_train_cat.drop(['label'], axis=1)
y_train_cat = df_train_cat['label']
x_test_cat = df_test_cat.drop(['label'], axis=1)
y_test_cat = df_test_cat['label']

x_train_cat = x_train_cat.astype(str)

print(x_train_cat.head())
print(y_train_cat.head())

def prepare_inputs(x_train, x_test):
    oe = OrdinalEncoder(handle_unknown='use_encoded_value',
                        unknown_value=-1)  # handle_unknown{‘error’, ‘use_encoded_value’}, default=’error’ # OneHotEncoder()
    oe.fit(x_train)
    x_train_enc = oe.transform(x_train)
    x_test_enc = oe.transform(x_test)
    return x_train_enc, x_test_enc


# prepare input data
x_train_enc, x_test_enc = prepare_inputs(x_train_cat, x_test_cat)


# prepare output data
# y_train_enc, y_test_enc = prepare_targets(y_train_cat, y_test_cat)


# Pearson’s chi-squared statistical hypothesis test is an example of a test for independence between categorical variables.
# The results of this test can be used for feature selection, where those features that are independent of the target variable can be removed from the dataset.
# Can do this two ways.

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Method 1 - selecting the top k most relevant features

def select_features(x_train, y_train, x_test, k_value='all'):
    fs = SelectKBest(score_func=chi2, k=k_value)
    fs.fit(x_train, y_train)
    x_train_fs = fs.transform(x_train)
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs, fs


# feature selection
x_train_fs, x_test_fs, fs = select_features(x_train_enc, y_train_cat, x_test_enc)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))

# Feature 0: 86.457760
# Feature 1: 2.915373
# Feature 2: 28.669582
# Feature 3: 21.181538
# Feature 4: 0.000019
# Feature 5: 71.339573
# Feature 6: 0.000425
# Feature 7: 0.000425
# Feature 8: 0.021521
# Feature 9: 0.021521
# Feature 10: 0.000338
# Feature 11: 0.000582
# Feature 12: 0.017568
# Feature 13: 0.017568
# Feature 14: 0.000156
# Feature 15: 0.000204
# Feature 16: 0.007888
# Feature 17: 0.005683
# Feature 18: 6.879290
# Feature 19: 6.126057
# Feature 20: 0.020257
# Feature 21: 0.021082
# Feature 22: 0.686899
# Feature 23: 0.664450
# Feature 24: 0.364467
# Feature 25: 0.364467
# Feature 26: 9.982556
# Feature 27: 10.052835
# Feature 28: 0.001083
# Feature 29: 0.000005
# Feature 30: 1.468859
# Feature 31: 1.468859
# Feature 32: 17.572476
# Feature 33: 17.429204
# Feature 34: 11.949734
# Feature 35: 12.967815
# Feature 36: 4.516367
# Feature 37: 4.516367
# Feature 38: 22.791096
# Feature 39: 23.846168
# Feature 40: 4.732299
# Feature 41: 4.732299
# Feature 42: 6.288319
# Feature 43: 5.900317
# Feature 44: 13.577677
# Feature 45: 13.577677
# Feature 46: 1.435979
# Feature 47: 2.007689
# Feature 48: 4.481955
# Feature 49: 4.390474
# Feature 50: 19.068578
# Feature 51: 11.916697
# Feature 52: 6.667160
# Feature 53: 6.011116
# Feature 54: 10.302071
# Feature 55: 10.048803
# Feature 56: 91.743950
# Feature 57: 87.005947
# Feature 58: 16.095536
# Feature 59: 20.413389
# Feature 60: 21.935338
# Feature 61: 46.914160
# Feature 62: 17.275614
# Feature 63: 24.225288
# Feature 64: 26.284067
# Feature 65: 30.666408
# Feature 66: 6.711999
# Feature 67: 13.267939
# Feature 68: 57.774710
# Feature 69: 61.314364
# Feature 70: 153.280099
# Feature 71: 171.340156
# Feature 72: 128.374068
# Feature 73: 114.644039
# Feature 74: 145.095368
# Feature 75: 127.247784
# Feature 76: 187.282518
# Feature 77: 208.788834
# Feature 78: 96.500530
# Feature 79: 74.655204
# Feature 80: 65.701192
# Feature 81: 50.221835
# Feature 82: 69.791444
# Feature 83: 54.235337
# Feature 84: 116.562797
# Feature 85: 187.856659

names = []
values = []
for i in range(len(fs.scores_)):
    names.append(df_train_cat.columns[i])
    values.append(fs.scores_[i])
chi_list = zip(names, values)
print(values)

# [86.45776043890886, 2.915372600555852, 28.66958234457939, 21.181537775384783, 1.8517269981360844e-05, 71.3395734780463, 0.0004248793048501003, 0.0004248793048501003, 0.021520626496648362, 0.021520626496648362, 0.0003381099947259906, 0.000581698499512631, 0.017568106234415994, 0.017568106234415994, 0.00015638422250629976, 0.00020405433469241046, 0.007887559228814981, 0.005682525731117499, 6.87928992550838, 6.126057423829615, 0.020256892360501, 0.02108200979030579, 0.6868986171624648, 0.6644502557438754, 0.364466989249121, 0.364466989249121, 9.98255616491565, 10.052834639187987, 0.0010831297781481956, 4.8158128894195415e-06, 1.468859150882997, 1.468859150882997, 17.572475799206842, 17.42920446516616, 11.949733944443715, 12.967814709549668, 4.516366543693526, 4.516366543693526, 22.791096239665396, 23.846168033591105, 4.732299331696845, 4.732299331696845, 6.28831909489972, 5.900317055254234, 13.577677030116146, 13.577677030116146, 1.4359790553997416, 2.0076887503909155, 4.481954928592113, 4.390474451220125, 19.068578229671022, 11.916696611732412, 6.667160160640506, 6.011116160163928, 10.30207113108315, 10.048802647138467, 91.74394963797462, 87.0059470411619, 16.095535541824994, 20.413389424727036, 21.935337649348117, 46.91415977646422, 17.275613972853705, 24.22528787041047, 26.284066603450416, 30.66640764717752, 6.711999093824896, 13.267939209085052, 57.77471044453384, 61.314364392907535, 153.28009890449013, 171.34015637835978, 128.37406771249385, 114.64403945113634, 145.09536835123126, 127.24778416747725, 187.2825182941218, 208.78883394500798, 96.50053035154572, 74.65520375679009, 65.70119208989051, 50.22183465186658, 69.79144370335234, 54.235337286165496, 116.56279691933948, 187.85665887155062]

# plot the scores
plt.figure(figsize=(10, 4))
sns.barplot(x=names, y=values)
plt.xticks(fontsize="4",
           rotation=90)
plt.title('Chi-square scores of categorical features')
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Method 2 - Using function 'chi2'. This is the approach we have acually used here, and this straightforward compared to previous method. The results from bith methods are same.

# calculate P-values
# refer - https://medium.com/analytics-vidhya/categorical-feature-selection-using-chi-squared-test-e4c0d0af6b7e
# refer - research paper - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4966396/

# The chi-squared test is a statistical test used to determine if there is a significant association between two categorical variables. It compares the observed frequencies of occurrences in each category with the expected frequencies if there were no association between the variables.

f_score_train = chi2(x_train_enc, y_train_cat)
print(f_score_train)

# (array([8.64577604e+01, 2.91537260e+00, 2.86695823e+01, 2.11815378e+01,
#        1.85172700e-05, 7.13395735e+01, 4.24879305e-04, 4.24879305e-04,
#        2.15206265e-02, 2.15206265e-02, 3.38109995e-04, 5.81698500e-04,
#        1.75681062e-02, 1.75681062e-02, 1.56384223e-04, 2.04054335e-04,
#        7.88755923e-03, 5.68252573e-03, 6.87928993e+00, 6.12605742e+00,
#        2.02568924e-02, 2.10820098e-02, 6.86898617e-01, 6.64450256e-01,
#        3.64466989e-01, 3.64466989e-01, 9.98255616e+00, 1.00528346e+01,
#        1.08312978e-03, 4.81581289e-06, 1.46885915e+00, 1.46885915e+00,
#        1.75724758e+01, 1.74292045e+01, 1.19497339e+01, 1.29678147e+01,
#        4.51636654e+00, 4.51636654e+00, 2.27910962e+01, 2.38461680e+01,
#        4.73229933e+00, 4.73229933e+00, 6.28831909e+00, 5.90031706e+00,
#        1.35776770e+01, 1.35776770e+01, 1.43597906e+00, 2.00768875e+00,
#        4.48195493e+00, 4.39047445e+00, 1.90685782e+01, 1.19166966e+01,
#        6.66716016e+00, 6.01111616e+00, 1.03020711e+01, 1.00488026e+01,
#        9.17439496e+01, 8.70059470e+01, 1.60955355e+01, 2.04133894e+01,
#        2.19353376e+01, 4.69141598e+01, 1.72756140e+01, 2.42252879e+01,
#        2.62840666e+01, 3.06664076e+01, 6.71199909e+00, 1.32679392e+01,
#        5.77747104e+01, 6.13143644e+01, 1.53280099e+02, 1.71340156e+02,
#        1.28374068e+02, 1.14644039e+02, 1.45095368e+02, 1.27247784e+02,
#        1.87282518e+02, 2.08788834e+02, 9.65005304e+01, 7.46552038e+01,
#        6.57011921e+01, 5.02218347e+01, 6.97914437e+01, 5.42353373e+01,
#        1.16562797e+02, 1.87856659e+02]), array([1.42754010e-20, 8.77391454e-02, 8.58415320e-08, 4.17769540e-06,
#        9.96566578e-01, 3.00742942e-17, 9.83554689e-01, 9.83554689e-01,
#        8.83369528e-01, 8.83369528e-01, 9.85329511e-01, 9.80758145e-01,
#        8.94553443e-01, 8.94553443e-01, 9.90022420e-01, 9.88602799e-01,
#        9.29231375e-01, 9.39910359e-01, 8.72002001e-03, 1.33203590e-02,
#        8.86821950e-01, 8.84555767e-01, 4.07221283e-01, 4.14993218e-01,
#        5.46035790e-01, 5.46035790e-01, 1.58030152e-03, 1.52113704e-03,
#        9.73745611e-01, 9.98249047e-01, 2.25525780e-01, 2.25525780e-01,
#        2.76562450e-05, 2.98208824e-05, 5.46551847e-04, 3.16891684e-04,
#        3.35720572e-02, 3.35720572e-02, 1.80600479e-06, 1.04349398e-06,
#        2.96012874e-02, 2.96012874e-02, 1.21536260e-02, 1.51381595e-02,
#        2.28891320e-04, 2.28891320e-04, 2.30791127e-01, 1.56503588e-01,
#        3.42545177e-02, 3.61402548e-02, 1.26104127e-05, 5.56330463e-04,
#        9.82055477e-03, 1.42160325e-02, 1.32880986e-03, 1.52446994e-03,
#        9.86494322e-22, 1.08195350e-20, 6.02258472e-05, 6.23917398e-06,
#        2.81993149e-06, 7.41649308e-12, 3.23309579e-05, 8.56996314e-07,
#        2.94710068e-07, 3.06424404e-08, 9.57663073e-03, 2.69983829e-04,
#        2.93924403e-14, 4.86517805e-15, 3.32704026e-35, 3.77111285e-39,
#        9.29623813e-30, 9.41712906e-27, 2.04696826e-33, 1.63968833e-29,
#        1.24600722e-42, 2.52412853e-47, 8.92199332e-23, 5.60536888e-18,
#        5.24741179e-16, 1.37311438e-12, 6.59182032e-17, 1.77859383e-13,
#        3.57866638e-27, 9.33662130e-43]))


print(names)

# ['first_careunit', 'admission_type', 'admission_location', 'gender', 'hep_types', 'treatment_types', 'thrombin_min_status', 'thrombin_max_status', 'd_dimer_max_status', 'd_dimer_min_status', 'methemoglobin_min_status', 'methemoglobin_max_status', 'ggt_min_status', 'ggt_max_status', 'globulin_min_status', 'globulin_max_status', 'total_protein_min_status', 'total_protein_max_status', 'atyps_max_status', 'atyps_min_status', 'carboxyhemoglobin_min_status', 'carboxyhemoglobin_max_status', 'amylase_max_status', 'amylase_min_status', 'aado2_bg_art_max_status', 'aado2_bg_art_min_status', 'bilirubin_direct_min_status', 'bilirubin_direct_max_status', 'bicarbonate_bg_min_status', 'bicarbonate_bg_max_status', 'fio2_bg_art_min_status', 'fio2_bg_art_max_status', 'nrbc_max_status', 'nrbc_min_status', 'bands_min_status', 'bands_max_status', 'so2_bg_art_min_status', 'so2_bg_art_max_status', 'fibrinogen_max_status', 'fibrinogen_min_status', 'hematocrit_bg_min_status', 'hematocrit_bg_max_status', 'hemoglobin_bg_min_status', 'hemoglobin_bg_max_status', 'temperature_bg_max_status', 'temperature_bg_min_status', 'chloride_bg_min_status', 'chloride_bg_max_status', 'sodium_bg_max_status', 'sodium_bg_min_status', 'glucose_bg_max_status', 'glucose_bg_min_status', 'ck_cpk_max_status', 'ck_cpk_min_status', 'ck_mb_max_status', 'ck_mb_min_status', 'ld_ldh_max_status', 'ld_ldh_min_status', 'calcium_bg_max_status', 'calcium_bg_min_status', 'pco2_bg_art_min_status', 'po2_bg_art_max_status', 'totalco2_bg_art_max_status', 'totalco2_bg_art_min_status', 'pco2_bg_art_max_status', 'po2_bg_art_min_status', 'potassium_bg_min_status', 'potassium_bg_max_status', 'albumin_max_status', 'albumin_min_status', 'bilirubin_total_min_status', 'bilirubin_total_max_status', 'alt_max_status', 'alt_min_status', 'alp_max_status', 'alp_min_status', 'ast_min_status', 'ast_max_status', 'pco2_bg_max_status', 'pco2_bg_min_status', 'totalco2_bg_min_status', 'totalco2_bg_max_status', 'ph_min_status', 'ph_max_status', 'lactate_min_status', 'lactate_max_status']
p_value = pd.Series(f_score_train[1], index=x_train_cat.columns)
# p_value.sort_values(ascending=True, inplace=True)
print(p_value)

# first_careunit                  1.427540e-20
# admission_type                  8.773915e-02
# admission_location              8.584153e-08
# gender                          4.177695e-06
# hep_types                       9.965666e-01
# treatment_types                 3.007429e-17
# thrombin_min_status             9.835547e-01
# thrombin_max_status             9.835547e-01
# d_dimer_max_status              8.833695e-01
# d_dimer_min_status              8.833695e-01
# methemoglobin_min_status        9.853295e-01
# methemoglobin_max_status        9.807581e-01
# ggt_min_status                  8.945534e-01
# ggt_max_status                  8.945534e-01
# globulin_min_status             9.900224e-01
# globulin_max_status             9.886028e-01
# total_protein_min_status        9.292314e-01
# total_protein_max_status        9.399104e-01
# atyps_max_status                8.720020e-03
# atyps_min_status                1.332036e-02
# carboxyhemoglobin_min_status    8.868219e-01
# carboxyhemoglobin_max_status    8.845558e-01
# amylase_max_status              4.072213e-01
# amylase_min_status              4.149932e-01
# aado2_bg_art_max_status         5.460358e-01
# aado2_bg_art_min_status         5.460358e-01
# bilirubin_direct_min_status     1.580302e-03
# bilirubin_direct_max_status     1.521137e-03
# bicarbonate_bg_min_status       9.737456e-01
# bicarbonate_bg_max_status       9.982490e-01
# fio2_bg_art_min_status          2.255258e-01
# fio2_bg_art_max_status          2.255258e-01
# nrbc_max_status                 2.765624e-05
# nrbc_min_status                 2.982088e-05
# bands_min_status                5.465518e-04
# bands_max_status                3.168917e-04
# so2_bg_art_min_status           3.357206e-02
# so2_bg_art_max_status           3.357206e-02
# fibrinogen_max_status           1.806005e-06
# fibrinogen_min_status           1.043494e-06
# hematocrit_bg_min_status        2.960129e-02
# hematocrit_bg_max_status        2.960129e-02
# hemoglobin_bg_min_status        1.215363e-02
# hemoglobin_bg_max_status        1.513816e-02
# temperature_bg_max_status       2.288913e-04
# temperature_bg_min_status       2.288913e-04
# chloride_bg_min_status          2.307911e-01
# chloride_bg_max_status          1.565036e-01
# sodium_bg_max_status            3.425452e-02
# sodium_bg_min_status            3.614025e-02
# glucose_bg_max_status           1.261041e-05
# glucose_bg_min_status           5.563305e-04
# ck_cpk_max_status               9.820555e-03
# ck_cpk_min_status               1.421603e-02
# ck_mb_max_status                1.328810e-03
# ck_mb_min_status                1.524470e-03
# ld_ldh_max_status               9.864943e-22
# ld_ldh_min_status               1.081953e-20
# calcium_bg_max_status           6.022585e-05
# calcium_bg_min_status           6.239174e-06
# pco2_bg_art_min_status          2.819931e-06
# po2_bg_art_max_status           7.416493e-12
# totalco2_bg_art_max_status      3.233096e-05
# totalco2_bg_art_min_status      8.569963e-07
# pco2_bg_art_max_status          2.947101e-07
# po2_bg_art_min_status           3.064244e-08
# potassium_bg_min_status         9.576631e-03
# potassium_bg_max_status         2.699838e-04
# albumin_max_status              2.939244e-14
# albumin_min_status              4.865178e-15
# bilirubin_total_min_status      3.327040e-35
# bilirubin_total_max_status      3.771113e-39
# alt_max_status                  9.296238e-30
# alt_min_status                  9.417129e-27
# alp_max_status                  2.046968e-33
# alp_min_status                  1.639688e-29
# ast_min_status                  1.246007e-42
# ast_max_status                  2.524129e-47
# pco2_bg_max_status              8.921993e-23
# pco2_bg_min_status              5.605369e-18
# totalco2_bg_min_status          5.247412e-16
# totalco2_bg_max_status          1.373114e-12
# ph_min_status                   6.591820e-17
# ph_max_status                   1.778594e-13
# lactate_min_status              3.578666e-27
# lactate_max_status              9.336621e-43

selected_cat_features = (p_value[p_value < 0.05])  # to select statistical significant features
print("selected_cat_features")

print(selected_cat_features)

# first_careunit                 1.427540e-20
# admission_location             8.584153e-08
# gender                         4.177695e-06
# treatment_types                3.007429e-17
# atyps_max_status               8.720020e-03
# atyps_min_status               1.332036e-02
# bilirubin_direct_min_status    1.580302e-03
# bilirubin_direct_max_status    1.521137e-03
# nrbc_max_status                2.765624e-05
# nrbc_min_status                2.982088e-05
# bands_min_status               5.465518e-04
# bands_max_status               3.168917e-04
# so2_bg_art_min_status          3.357206e-02
# so2_bg_art_max_status          3.357206e-02
# fibrinogen_max_status          1.806005e-06
# fibrinogen_min_status          1.043494e-06
# hematocrit_bg_min_status       2.960129e-02
# hematocrit_bg_max_status       2.960129e-02
# hemoglobin_bg_min_status       1.215363e-02
# hemoglobin_bg_max_status       1.513816e-02
# temperature_bg_max_status      2.288913e-04
# temperature_bg_min_status      2.288913e-04
# sodium_bg_max_status           3.425452e-02
# sodium_bg_min_status           3.614025e-02
# glucose_bg_max_status          1.261041e-05
# glucose_bg_min_status          5.563305e-04
# ck_cpk_max_status              9.820555e-03
# ck_cpk_min_status              1.421603e-02
# ck_mb_max_status               1.328810e-03
# ck_mb_min_status               1.524470e-03
# ld_ldh_max_status              9.864943e-22
# ld_ldh_min_status              1.081953e-20
# calcium_bg_max_status          6.022585e-05
# calcium_bg_min_status          6.239174e-06
# pco2_bg_art_min_status         2.819931e-06
# po2_bg_art_max_status          7.416493e-12
# totalco2_bg_art_max_status     3.233096e-05
# totalco2_bg_art_min_status     8.569963e-07
# pco2_bg_art_max_status         2.947101e-07
# po2_bg_art_min_status          3.064244e-08
# potassium_bg_min_status        9.576631e-03
# potassium_bg_max_status        2.699838e-04
# albumin_max_status             2.939244e-14
# albumin_min_status             4.865178e-15
# bilirubin_total_min_status     3.327040e-35
# bilirubin_total_max_status     3.771113e-39
# alt_max_status                 9.296238e-30
# alt_min_status                 9.417129e-27
# alp_max_status                 2.046968e-33
# alp_min_status                 1.639688e-29
# ast_min_status                 1.246007e-42
# ast_max_status                 2.524129e-47
# pco2_bg_max_status             8.921993e-23
# pco2_bg_min_status             5.605369e-18
# totalco2_bg_min_status         5.247412e-16
# totalco2_bg_max_status         1.373114e-12
# ph_min_status                  6.591820e-17
# ph_max_status                  1.778594e-13
# lactate_min_status             3.578666e-27
# lactate_max_status             9.336621e-43

print(p_value.shape)  # (86,)
print(selected_cat_features.shape)  # (60,)

# plot the p-values

plt.figure(figsize=(10, 4))
sns.barplot(x=names, y=p_value)
plt.xticks(fontsize="4",
           rotation=90)
plt.xlabel('Categorical Feature')
plt.title('P-value')
plt.title('P-values of Categorical Features after Conducting Chi-Squared Test')
plt.show()

# python 2_Jan_3_combine_feature_selection_and_prediction.py /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/train_data_before_preprocessing.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/test_data_before_preprocessing.csv

