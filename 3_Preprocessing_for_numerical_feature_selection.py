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
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------------------------------------------------------------------------------

# 2. Import files

df_train_all = pd.read_csv(sys.argv[1])  # after prepocessing - training set only
df_test_all = pd.read_csv(sys.argv[2])  # after prepocessing - test set only

# from data set, drop 'hadm_id'

df_train_all = df_train_all.drop('hadm_id', axis=1)
df_test_all = df_test_all.drop('hadm_id', axis=1)

# col_names = ['encoder__first_careunit_Cardiac Vascular Intensive Care Unit (CVICU)', 'encoder__first_careunit_Coronary Care Unit (CCU)', 'encoder__first_careunit_Medical Intensive Care Unit (MICU)', 'encoder__first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)', 'encoder__first_careunit_Neuro Intermediate', 'encoder__first_careunit_Neuro Stepdown', 'encoder__first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)', 'encoder__first_careunit_Surgical Intensive Care Unit (SICU)', 'encoder__first_careunit_Trauma SICU (TSICU)', 'encoder__admission_type_DIRECT EMER.', 'encoder__admission_type_DIRECT OBSERVATION', 'encoder__admission_type_ELECTIVE', 'encoder__admission_type_EU OBSERVATION', 'encoder__admission_type_EW EMER.', 'encoder__admission_type_OBSERVATION ADMIT', 'encoder__admission_type_SURGICAL SAME DAY ADMISSION', 'encoder__admission_type_URGENT', 'encoder__admission_location_AMBULATORY SURGERY TRANSFER', 'encoder__admission_location_CLINIC REFERRAL', 'encoder__admission_location_EMERGENCY ROOM', 'encoder__admission_location_INFORMATION NOT AVAILABLE', 'encoder__admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'encoder__admission_location_PACU', 'encoder__admission_location_PHYSICIAN REFERRAL', 'encoder__admission_location_PROCEDURE SITE', 'encoder__admission_location_TRANSFER FROM HOSPITAL', 'encoder__admission_location_TRANSFER FROM SKILLED NURSING FACILITY', 'encoder__admission_location_WALK-IN/SELF REFERRAL', 'encoder__hep_types_LMWH', 'encoder__hep_types_UFH', 'encoder__hep_types_both', 'encoder__treatment_types_P', 'encoder__treatment_types_T', 'encoder__treatment_types_both', 'encoder__lactate_min_status_elevated', 'encoder__lactate_min_status_low', 'encoder__lactate_min_status_normal', 'encoder__lactate_min_status_not ordered', 'encoder__lactate_max_status_elevated', 'encoder__lactate_max_status_low', 'encoder__lactate_max_status_normal', 'encoder__lactate_max_status_not ordered', 'encoder__ph_min_status_elevated', 'encoder__ph_min_status_low', 'encoder__ph_min_status_normal', 'encoder__ph_min_status_not ordered', 'encoder__ph_max_status_elevated', 'encoder__ph_max_status_low', 'encoder__ph_max_status_normal', 'encoder__ph_max_status_not ordered', 'encoder__totalco2_bg_min_status_elevated', 'encoder__totalco2_bg_min_status_low', 'encoder__totalco2_bg_min_status_normal', 'encoder__totalco2_bg_min_status_not ordered', 'encoder__totalco2_bg_max_status_elevated', 'encoder__totalco2_bg_max_status_low', 'encoder__totalco2_bg_max_status_normal', 'encoder__totalco2_bg_max_status_not ordered', 'encoder__pco2_bg_min_status_elevated', 'encoder__pco2_bg_min_status_low', 'encoder__pco2_bg_min_status_normal', 'encoder__pco2_bg_min_status_not ordered', 'encoder__pco2_bg_max_status_elevated', 'encoder__pco2_bg_max_status_low', 'encoder__pco2_bg_max_status_normal', 'encoder__pco2_bg_max_status_not ordered', 'encoder__ast_min_status_elevated', 'encoder__ast_min_status_normal', 'encoder__ast_min_status_not ordered', 'encoder__ast_max_status_elevated', 'encoder__ast_max_status_normal', 'encoder__ast_max_status_not ordered', 'encoder__alp_min_status_elevated', 'encoder__alp_min_status_low', 'encoder__alp_min_status_normal', 'encoder__alp_min_status_not ordered', 'encoder__alp_max_status_elevated', 'encoder__alp_max_status_low', 'encoder__alp_max_status_normal', 'encoder__alp_max_status_not ordered', 'encoder__alt_min_status_elevated', 'encoder__alt_min_status_normal', 'encoder__alt_min_status_not ordered', 'encoder__alt_max_status_elevated', 'encoder__alt_max_status_normal', 'encoder__alt_max_status_not ordered', 'encoder__bilirubin_total_min_status_elevated', 'encoder__bilirubin_total_min_status_normal', 'encoder__bilirubin_total_min_status_not ordered', 'encoder__bilirubin_total_max_status_elevated', 'encoder__bilirubin_total_max_status_normal', 'encoder__bilirubin_total_max_status_not ordered', 'encoder__albumin_min_status_elevated', 'encoder__albumin_min_status_low', 'encoder__albumin_min_status_normal', 'encoder__albumin_min_status_not ordered', 'encoder__albumin_max_status_elevated', 'encoder__albumin_max_status_low', 'encoder__albumin_max_status_normal', 'encoder__albumin_max_status_not ordered', 'encoder__pco2_bg_art_min_status_elevated', 'encoder__pco2_bg_art_min_status_low', 'encoder__pco2_bg_art_min_status_normal', 'encoder__pco2_bg_art_min_status_not ordered', 'encoder__pco2_bg_art_max_status_elevated', 'encoder__pco2_bg_art_max_status_low', 'encoder__pco2_bg_art_max_status_normal', 'encoder__pco2_bg_art_max_status_not ordered', 'encoder__po2_bg_art_min_status_elevated', 'encoder__po2_bg_art_min_status_low', 'encoder__po2_bg_art_min_status_normal', 'encoder__po2_bg_art_min_status_not ordered', 'encoder__po2_bg_art_max_status_elevated', 'encoder__po2_bg_art_max_status_low', 'encoder__po2_bg_art_max_status_normal', 'encoder__po2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_min_status_elevated', 'encoder__totalco2_bg_art_min_status_low', 'encoder__totalco2_bg_art_min_status_normal', 'encoder__totalco2_bg_art_min_status_not ordered', 'encoder__totalco2_bg_art_max_status_elevated', 'encoder__totalco2_bg_art_max_status_low', 'encoder__totalco2_bg_art_max_status_normal', 'encoder__totalco2_bg_art_max_status_not ordered', 'encoder__ld_ldh_min_status_elevated', 'encoder__ld_ldh_min_status_low', 'encoder__ld_ldh_min_status_normal', 'encoder__ld_ldh_min_status_not ordered', 'encoder__ld_ldh_max_status_elevated', 'encoder__ld_ldh_max_status_low', 'encoder__ld_ldh_max_status_normal', 'encoder__ld_ldh_max_status_not ordered', 'encoder__ck_cpk_min_status_elevated', 'encoder__ck_cpk_min_status_low', 'encoder__ck_cpk_min_status_normal', 'encoder__ck_cpk_min_status_not ordered', 'encoder__ck_cpk_max_status_elevated', 'encoder__ck_cpk_max_status_low', 'encoder__ck_cpk_max_status_normal', 'encoder__ck_cpk_max_status_not ordered', 'encoder__ck_mb_min_status_elevated', 'encoder__ck_mb_min_status_normal', 'encoder__ck_mb_min_status_not ordered', 'encoder__ck_mb_max_status_elevated', 'encoder__ck_mb_max_status_normal', 'encoder__ck_mb_max_status_not ordered', 'encoder__fio2_bg_art_min_status_no ref range', 'encoder__fio2_bg_art_min_status_not ordered', 'encoder__fio2_bg_art_max_status_no ref range', 'encoder__fio2_bg_art_max_status_not ordered', 'encoder__so2_bg_art_min_status_no ref range', 'encoder__so2_bg_art_min_status_not ordered', 'encoder__so2_bg_art_max_status_no ref range', 'encoder__so2_bg_art_max_status_not ordered', 'encoder__fibrinogen_min_status_elevated', 'encoder__fibrinogen_min_status_low', 'encoder__fibrinogen_min_status_normal', 'encoder__fibrinogen_min_status_not ordered', 'encoder__fibrinogen_max_status_elevated', 'encoder__fibrinogen_max_status_low', 'encoder__fibrinogen_max_status_normal', 'encoder__fibrinogen_max_status_not ordered', 'encoder__thrombin_min_status_elevated', 'encoder__thrombin_min_status_normal', 'encoder__thrombin_min_status_not ordered', 'encoder__thrombin_max_status_elevated', 'encoder__thrombin_max_status_normal', 'encoder__thrombin_max_status_not ordered', 'encoder__d_dimer_min_status_elevated', 'encoder__d_dimer_min_status_normal', 'encoder__d_dimer_min_status_not ordered', 'encoder__d_dimer_max_status_elevated', 'encoder__d_dimer_max_status_normal', 'encoder__d_dimer_max_status_not ordered', 'encoder__methemoglobin_min_status_elevated', 'encoder__methemoglobin_min_status_normal', 'encoder__methemoglobin_min_status_not ordered', 'encoder__methemoglobin_max_status_elevated', 'encoder__methemoglobin_max_status_normal', 'encoder__methemoglobin_max_status_not ordered', 'encoder__ggt_min_status_elevated', 'encoder__ggt_min_status_low', 'encoder__ggt_min_status_normal', 'encoder__ggt_min_status_not ordered', 'encoder__ggt_max_status_elevated', 'encoder__ggt_max_status_low', 'encoder__ggt_max_status_normal', 'encoder__ggt_max_status_not ordered', 'encoder__globulin_min_status_elevated', 'encoder__globulin_min_status_low', 'encoder__globulin_min_status_normal', 'encoder__globulin_min_status_not ordered', 'encoder__globulin_max_status_elevated', 'encoder__globulin_max_status_low', 'encoder__globulin_max_status_normal', 'encoder__globulin_max_status_not ordered', 'encoder__atyps_min_status_elevated', 'encoder__atyps_min_status_not ordered', 'encoder__atyps_max_status_elevated', 'encoder__atyps_max_status_not ordered', 'encoder__total_protein_min_status_elevated', 'encoder__total_protein_min_status_low', 'encoder__total_protein_min_status_normal', 'encoder__total_protein_min_status_not ordered', 'encoder__total_protein_max_status_elevated', 'encoder__total_protein_max_status_low', 'encoder__total_protein_max_status_normal', 'encoder__total_protein_max_status_not ordered', 'encoder__carboxyhemoglobin_min_status_elevated', 'encoder__carboxyhemoglobin_min_status_normal', 'encoder__carboxyhemoglobin_min_status_not ordered', 'encoder__carboxyhemoglobin_max_status_elevated', 'encoder__carboxyhemoglobin_max_status_normal', 'encoder__carboxyhemoglobin_max_status_not ordered', 'encoder__amylase_min_status_elevated', 'encoder__amylase_min_status_normal', 'encoder__amylase_min_status_not ordered', 'encoder__amylase_max_status_elevated', 'encoder__amylase_max_status_normal', 'encoder__amylase_max_status_not ordered', 'encoder__aado2_bg_art_min_status_no ref range', 'encoder__aado2_bg_art_min_status_not ordered', 'encoder__aado2_bg_art_max_status_no ref range', 'encoder__aado2_bg_art_max_status_not ordered', 'encoder__bilirubin_direct_min_status_elevated', 'encoder__bilirubin_direct_min_status_normal', 'encoder__bilirubin_direct_min_status_not ordered', 'encoder__bilirubin_direct_max_status_elevated', 'encoder__bilirubin_direct_max_status_normal', 'encoder__bilirubin_direct_max_status_not ordered', 'encoder__nrbc_min_status_elevated', 'encoder__nrbc_min_status_not ordered', 'encoder__nrbc_max_status_elevated', 'encoder__nrbc_max_status_not ordered', 'encoder__bands_min_status_elevated', 'encoder__bands_min_status_normal', 'encoder__bands_min_status_not ordered', 'encoder__bands_max_status_elevated', 'encoder__bands_max_status_normal', 'encoder__bands_max_status_not ordered', 'remainder__gender', 'remainder__anchor_age', 'remainder__base_platelets', 'remainder__heart_rate_min', 'remainder__heart_rate_max', 'remainder__heart_rate_mean', 'remainder__sbp_min', 'remainder__sbp_max', 'remainder__sbp_mean', 'remainder__dbp_min', 'remainder__dbp_max', 'remainder__dbp_mean', 'remainder__mbp_min', 'remainder__mbp_max', 'remainder__mbp_mean', 'remainder__resp_rate_min', 'remainder__resp_rate_max', 'remainder__resp_rate_mean', 'remainder__temperature_min', 'remainder__temperature_max', 'remainder__temperature_mean', 'remainder__spo2_min', 'remainder__spo2_max', 'remainder__spo2_mean', 'remainder__glucose_min', 'remainder__glucose_max', 'remainder__glucose_mean', 'remainder__hematocrit_min', 'remainder__hematocrit_max', 'remainder__hemoglobin_min', 'remainder__hemoglobin_max', 'remainder__bicarbonate_min', 'remainder__bicarbonate_max', 'remainder__calcium_min', 'remainder__calcium_max', 'remainder__chloride_min', 'remainder__chloride_max', 'remainder__sodium_min', 'remainder__sodium_max', 'remainder__potassium_min', 'remainder__potassium_max', 'remainder__platelets_min', 'remainder__platelets_max', 'remainder__wbc_min', 'remainder__wbc_max', 'remainder__aniongap_min', 'remainder__aniongap_max', 'remainder__bun_min', 'remainder__bun_max', 'remainder__creatinine_min', 'remainder__creatinine_max', 'remainder__inr_min', 'remainder__inr_max', 'remainder__pt_min', 'remainder__pt_max', 'remainder__ptt_min', 'remainder__ptt_max', 'remainder__gcs_min', 'label']

# --------------------------------------------------------------------------------------------------------------

# categorical feature selection

# training dataset

df_train_categorical_selected = df_train_all[
    ['first_careunit',	'admission_location',	'gender',	'treatment_types',	'atyps_max_status',	'atyps_min_status',	'bilirubin_direct_min_status',	'bilirubin_direct_max_status',	'nrbc_max_status',	'nrbc_min_status',	'bands_min_status',	'bands_max_status',	'so2_bg_art_min_status',	'so2_bg_art_max_status',	'fibrinogen_max_status',	'fibrinogen_min_status',	'hematocrit_bg_min_status',	'hematocrit_bg_max_status',	'hemoglobin_bg_min_status',	'hemoglobin_bg_max_status',	'temperature_bg_max_status',	'temperature_bg_min_status',	'sodium_bg_max_status',	'sodium_bg_min_status',	'glucose_bg_max_status',	'glucose_bg_min_status',	'ck_cpk_max_status',	'ck_cpk_min_status',	'ck_mb_max_status',	'ck_mb_min_status',	'ld_ldh_max_status',	'ld_ldh_min_status',	'calcium_bg_max_status',	'calcium_bg_min_status',	'pco2_bg_art_min_status',	'po2_bg_art_max_status',	'totalco2_bg_art_max_status',	'totalco2_bg_art_min_status',	'pco2_bg_art_max_status',	'po2_bg_art_min_status',	'potassium_bg_min_status',	'potassium_bg_max_status',	'albumin_max_status',	'albumin_min_status',	'bilirubin_total_min_status',	'bilirubin_total_max_status',	'alt_max_status',	'alt_min_status',	'alp_max_status',	'alp_min_status',	'ast_min_status',	'ast_max_status',	'pco2_bg_max_status',	'pco2_bg_min_status',	'totalco2_bg_min_status',	'totalco2_bg_max_status',	'ph_min_status',	'ph_max_status',	'lactate_min_status',	'lactate_max_status']]
print(df_train_categorical_selected.shape)  # (10732, 60) - cat features

df_train_numerical_all = df_train_all[
    ['anchor_age',	'base_platelets',	'heart_rate_min',	'heart_rate_max',	'heart_rate_mean',	'sbp_min',	'sbp_max',	'sbp_mean',	'dbp_min',	'dbp_max',	'dbp_mean',	'mbp_min',	'mbp_max',	'mbp_mean',	'resp_rate_min',	'resp_rate_max',	'resp_rate_mean',	'spo2_min',	'spo2_max',	'spo2_mean',	'temperature_vital_min',	'temperature_vital_max',	'temperature_vital_mean',	'glucose_vital_min',	'glucose_vital_max',	'glucose_vital_mean',	'hematocrit_lab_min',	'hematocrit_lab_max',	'hemoglobin_lab_min',	'hemoglobin_lab_max',	'bicarbonate_lab_min',	'bicarbonate_lab_max',	'calcium_lab_min',	'calcium_lab_max',	'chloride_lab_min',	'chloride_lab_max',	'sodium_lab_min',	'sodium_lab_max',	'potassium_lab_min',	'potassium_lab_max',	'glucose_lab_min',	'glucose_lab_max',	'platelets_min',	'platelets_max',	'wbc_min',	'wbc_max',	'aniongap_min',	'aniongap_max',	'bun_min',	'bun_max',	'creatinine_min',	'creatinine_max',	'inr_min',	'inr_max',	'pt_min',	'pt_max',	'ptt_min',	'ptt_max',	'gcs_min']]


print(df_train_numerical_all.shape)  # (10732, 59)

df_train_cat_selected_num_all = pd.concat(
    [df_train_categorical_selected, df_train_numerical_all], axis=1)
print(df_train_cat_selected_num_all.shape)  # (10732, 119)

pd.set_option('display.max_columns', None)
print(df_train_cat_selected_num_all.head(10))

# ------------------------------------------------------------------------------------------------------------------------------------

# testing data set

df_test_categorical_selected = df_test_all[
    ['first_careunit',	'admission_location',	'gender',	'treatment_types',	'atyps_max_status',	'atyps_min_status',	'bilirubin_direct_min_status',	'bilirubin_direct_max_status',	'nrbc_max_status',	'nrbc_min_status',	'bands_min_status',	'bands_max_status',	'so2_bg_art_min_status',	'so2_bg_art_max_status',	'fibrinogen_max_status',	'fibrinogen_min_status',	'hematocrit_bg_min_status',	'hematocrit_bg_max_status',	'hemoglobin_bg_min_status',	'hemoglobin_bg_max_status',	'temperature_bg_max_status',	'temperature_bg_min_status',	'sodium_bg_max_status',	'sodium_bg_min_status',	'glucose_bg_max_status',	'glucose_bg_min_status',	'ck_cpk_max_status',	'ck_cpk_min_status',	'ck_mb_max_status',	'ck_mb_min_status',	'ld_ldh_max_status',	'ld_ldh_min_status',	'calcium_bg_max_status',	'calcium_bg_min_status',	'pco2_bg_art_min_status',	'po2_bg_art_max_status',	'totalco2_bg_art_max_status',	'totalco2_bg_art_min_status',	'pco2_bg_art_max_status',	'po2_bg_art_min_status',	'potassium_bg_min_status',	'potassium_bg_max_status',	'albumin_max_status',	'albumin_min_status',	'bilirubin_total_min_status',	'bilirubin_total_max_status',	'alt_max_status',	'alt_min_status',	'alp_max_status',	'alp_min_status',	'ast_min_status',	'ast_max_status',	'pco2_bg_max_status',	'pco2_bg_min_status',	'totalco2_bg_min_status',	'totalco2_bg_max_status',	'ph_min_status',	'ph_max_status',	'lactate_min_status',	'lactate_max_status']]

print(df_test_categorical_selected.shape)  # (2683, 60)

df_test_numerical_all = df_test_all[
    ['anchor_age',	'base_platelets',	'heart_rate_min',	'heart_rate_max',	'heart_rate_mean',	'sbp_min',	'sbp_max',	'sbp_mean',	'dbp_min',	'dbp_max',	'dbp_mean',	'mbp_min',	'mbp_max',	'mbp_mean',	'resp_rate_min',	'resp_rate_max',	'resp_rate_mean',	'spo2_min',	'spo2_max',	'spo2_mean',	'temperature_vital_min',	'temperature_vital_max',	'temperature_vital_mean',	'glucose_vital_min',	'glucose_vital_max',	'glucose_vital_mean',	'hematocrit_lab_min',	'hematocrit_lab_max',	'hemoglobin_lab_min',	'hemoglobin_lab_max',	'bicarbonate_lab_min',	'bicarbonate_lab_max',	'calcium_lab_min',	'calcium_lab_max',	'chloride_lab_min',	'chloride_lab_max',	'sodium_lab_min',	'sodium_lab_max',	'potassium_lab_min',	'potassium_lab_max',	'glucose_lab_min',	'glucose_lab_max',	'platelets_min',	'platelets_max',	'wbc_min',	'wbc_max',	'aniongap_min',	'aniongap_max',	'bun_min',	'bun_max',	'creatinine_min',	'creatinine_max',	'inr_min',	'inr_max',	'pt_min',	'pt_max',	'ptt_min',	'ptt_max',	'gcs_min']]


print(df_test_numerical_all.shape)  # (2683, 59)

df_test_cat_selected_num_all = pd.concat(
    [df_test_categorical_selected, df_test_numerical_all], axis=1)
print(df_test_cat_selected_num_all.shape)  # (2683, 119)

pd.set_option('display.max_columns', None)
print(df_test_cat_selected_num_all.head(10))

# ------------------------------------------------------------------------------------------------------------------------------------
# Convert Categorical features to Numerical - Dummy variable encoding # (18440, 209) (18440, 169)

# train set

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'),
                   df_train_categorical_selected.columns.tolist())],
    remainder='passthrough')

df_train_cat_selected_num_all = np.array(ct.fit_transform(
    df_train_cat_selected_num_all))  # here 'np' (NumPy) was added because, fit_transform itself doesn't return output in np array, so in order to train future machine learning models, np is added.

print(df_train_cat_selected_num_all.shape)  # (10732, 220)

# test set

df_test_cat_selected_num_all = np.array(ct.transform(df_test_cat_selected_num_all))  # handle_unknown = 'ignore'

x_axis_original = ct.get_feature_names_out().tolist()
print(x_axis_original)
# ['encoder__first_careunit_Coronary Care Unit (CCU)', 'encoder__first_careunit_Medical Intensive Care Unit (MICU)', 'encoder__first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)', 'encoder__first_careunit_Neuro Intermediate', 'encoder__first_careunit_Neuro Stepdown', 'encoder__first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)', 'encoder__first_careunit_Surgical Intensive Care Unit (SICU)', 'encoder__first_careunit_Trauma SICU (TSICU)', 'encoder__admission_type_DIRECT OBSERVATION', 'encoder__admission_type_ELECTIVE', 'encoder__admission_type_EU OBSERVATION', 'encoder__admission_type_EW EMER.', 'encoder__admission_type_OBSERVATION ADMIT', 'encoder__admission_type_SURGICAL SAME DAY ADMISSION', 'encoder__admission_type_URGENT', 'encoder__admission_location_CLINIC REFERRAL', 'encoder__admission_location_EMERGENCY ROOM', 'encoder__admission_location_INFORMATION NOT AVAILABLE', 'encoder__admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'encoder__admission_location_PACU', 'encoder__admission_location_PHYSICIAN REFERRAL', 'encoder__admission_location_PROCEDURE SITE', 'encoder__admission_location_TRANSFER FROM HOSPITAL', 'encoder__admission_location_TRANSFER FROM SKILLED NURSING FACILITY', 'encoder__admission_location_WALK-IN/SELF REFERRAL', 'encoder__gender_M', 'encoder__treatment_types_T', 'encoder__atyps_max_status_normal', 'encoder__atyps_max_status_not ordered', 'encoder__atyps_min_status_normal', 'encoder__atyps_min_status_not ordered', 'encoder__bilirubin_direct_min_status_normal', 'encoder__bilirubin_direct_min_status_not ordered', 'encoder__bilirubin_direct_max_status_normal', 'encoder__bilirubin_direct_max_status_not ordered', 'encoder__nrbc_max_status_normal', 'encoder__nrbc_max_status_not ordered', 'encoder__nrbc_min_status_normal', 'encoder__nrbc_min_status_not ordered', 'encoder__bands_min_status_normal', 'encoder__bands_min_status_not ordered', 'encoder__bands_max_status_normal', 'encoder__bands_max_status_not ordered', 'encoder__so2_bg_art_min_status_not ordered', 'encoder__so2_bg_art_max_status_not ordered', 'encoder__fibrinogen_max_status_low', 'encoder__fibrinogen_max_status_normal', 'encoder__fibrinogen_max_status_not ordered', 'encoder__fibrinogen_min_status_low', 'encoder__fibrinogen_min_status_normal', 'encoder__fibrinogen_min_status_not ordered', 'encoder__hematocrit_bg_min_status_not ordered', 'encoder__hematocrit_bg_max_status_not ordered', 'encoder__hemoglobin_bg_min_status_low', 'encoder__hemoglobin_bg_min_status_normal', 'encoder__hemoglobin_bg_min_status_not ordered', 'encoder__hemoglobin_bg_max_status_low', 'encoder__hemoglobin_bg_max_status_normal', 'encoder__hemoglobin_bg_max_status_not ordered', 'encoder__temperature_bg_max_status_not ordered', 'encoder__temperature_bg_min_status_not ordered', 'encoder__glucose_bg_max_status_low', 'encoder__glucose_bg_max_status_normal', 'encoder__glucose_bg_max_status_not ordered', 'encoder__glucose_bg_min_status_low', 'encoder__glucose_bg_min_status_normal', 'encoder__glucose_bg_min_status_not ordered', 'encoder__ck_cpk_max_status_low', 'encoder__ck_cpk_max_status_normal', 'encoder__ck_cpk_max_status_not ordered', 'encoder__ck_cpk_min_status_low', 'encoder__ck_cpk_min_status_normal', 'encoder__ck_cpk_min_status_not ordered', 'encoder__ck_mb_max_status_normal', 'encoder__ck_mb_max_status_not ordered', 'encoder__ck_mb_min_status_normal', 'encoder__ck_mb_min_status_not ordered', 'encoder__ld_ldh_max_status_low', 'encoder__ld_ldh_max_status_normal', 'encoder__ld_ldh_max_status_not ordered', 'encoder__ld_ldh_min_status_low', 'encoder__ld_ldh_min_status_normal', 'encoder__ld_ldh_min_status_not ordered', 'encoder__calcium_bg_max_status_low', 'encoder__calcium_bg_max_status_normal', 'encoder__calcium_bg_max_status_not ordered', 'encoder__calcium_bg_min_status_low', 'encoder__calcium_bg_min_status_normal', 'encoder__calcium_bg_min_status_not ordered', 'encoder__pco2_bg_art_min_status_low', 'encoder__pco2_bg_art_min_status_normal', 'encoder__pco2_bg_art_min_status_not ordered', 'encoder__po2_bg_art_max_status_low', 'encoder__po2_bg_art_max_status_normal', 'encoder__po2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_max_status_low', 'encoder__totalco2_bg_art_max_status_normal', 'encoder__totalco2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_min_status_low', 'encoder__totalco2_bg_art_min_status_normal', 'encoder__totalco2_bg_art_min_status_not ordered', 'encoder__pco2_bg_art_max_status_low', 'encoder__pco2_bg_art_max_status_normal', 'encoder__pco2_bg_art_max_status_not ordered', 'encoder__po2_bg_art_min_status_low', 'encoder__po2_bg_art_min_status_normal', 'encoder__po2_bg_art_min_status_not ordered', 'encoder__potassium_bg_min_status_low', 'encoder__potassium_bg_min_status_normal', 'encoder__potassium_bg_min_status_not ordered', 'encoder__potassium_bg_max_status_low', 'encoder__potassium_bg_max_status_normal', 'encoder__potassium_bg_max_status_not ordered', 'encoder__albumin_max_status_low', 'encoder__albumin_max_status_normal', 'encoder__albumin_max_status_not ordered', 'encoder__albumin_min_status_low', 'encoder__albumin_min_status_normal', 'encoder__albumin_min_status_not ordered', 'encoder__bilirubin_total_min_status_normal', 'encoder__bilirubin_total_min_status_not ordered', 'encoder__bilirubin_total_max_status_normal', 'encoder__bilirubin_total_max_status_not ordered', 'encoder__alt_max_status_normal', 'encoder__alt_max_status_not ordered', 'encoder__alt_min_status_normal', 'encoder__alt_min_status_not ordered', 'encoder__alp_max_status_low', 'encoder__alp_max_status_normal', 'encoder__alp_max_status_not ordered', 'encoder__alp_min_status_low', 'encoder__alp_min_status_normal', 'encoder__alp_min_status_not ordered', 'encoder__ast_min_status_normal', 'encoder__ast_min_status_not ordered', 'encoder__ast_max_status_normal', 'encoder__ast_max_status_not ordered', 'encoder__pco2_bg_max_status_low', 'encoder__pco2_bg_max_status_normal', 'encoder__pco2_bg_max_status_not ordered', 'encoder__pco2_bg_min_status_low', 'encoder__pco2_bg_min_status_normal', 'encoder__pco2_bg_min_status_not ordered', 'encoder__totalco2_bg_min_status_low', 'encoder__totalco2_bg_min_status_normal', 'encoder__totalco2_bg_min_status_not ordered', 'encoder__totalco2_bg_max_status_low', 'encoder__totalco2_bg_max_status_normal', 'encoder__totalco2_bg_max_status_not ordered', 'encoder__ph_min_status_low', 'encoder__ph_min_status_normal', 'encoder__ph_min_status_not ordered', 'encoder__ph_max_status_low', 'encoder__ph_max_status_normal', 'encoder__ph_max_status_not ordered', 'encoder__lactate_min_status_low', 'encoder__lactate_min_status_normal', 'encoder__lactate_min_status_not ordered', 'encoder__lactate_max_status_low', 'encoder__lactate_max_status_normal', 'encoder__lactate_max_status_not ordered', 'remainder__anchor_age', 'remainder__base_platelets', 'remainder__heart_rate_min', 'remainder__heart_rate_max', 'remainder__heart_rate_mean', 'remainder__sbp_min', 'remainder__sbp_max', 'remainder__sbp_mean', 'remainder__dbp_min', 'remainder__dbp_max', 'remainder__dbp_mean', 'remainder__mbp_min', 'remainder__mbp_max', 'remainder__mbp_mean', 'remainder__resp_rate_min', 'remainder__resp_rate_max', 'remainder__resp_rate_mean', 'remainder__spo2_min', 'remainder__spo2_max', 'remainder__spo2_mean', 'remainder__temperature_vital_min', 'remainder__temperature_vital_max', 'remainder__temperature_vital_mean', 'remainder__glucose_vital_min', 'remainder__glucose_vital_max', 'remainder__glucose_vital_mean', 'remainder__hematocrit_lab_min', 'remainder__hematocrit_lab_max', 'remainder__hemoglobin_lab_min', 'remainder__hemoglobin_lab_max', 'remainder__bicarbonate_lab_min', 'remainder__bicarbonate_lab_max', 'remainder__calcium_lab_min', 'remainder__calcium_lab_max', 'remainder__chloride_lab_min', 'remainder__chloride_lab_max', 'remainder__sodium_lab_min', 'remainder__sodium_lab_max', 'remainder__potassium_lab_min', 'remainder__potassium_lab_max', 'remainder__glucose_lab_min', 'remainder__glucose_lab_max', 'remainder__platelets_min', 'remainder__platelets_max', 'remainder__wbc_min', 'remainder__wbc_max', 'remainder__aniongap_min', 'remainder__aniongap_max', 'remainder__bun_min', 'remainder__bun_max', 'remainder__creatinine_min', 'remainder__creatinine_max', 'remainder__inr_min', 'remainder__inr_max', 'remainder__pt_min', 'remainder__pt_max', 'remainder__ptt_min', 'remainder__ptt_max', 'remainder__gcs_min']

print(len(x_axis_original))  # 220
print(df_train_cat_selected_num_all)

print(x_axis_original[159]) # encoder__lactate_max_status_not ordered - Therefore, upto (including) index = 159 column, are from categorical features encoding.
print(x_axis_original[160]) # index 160 =  - remainder__anchor_age

# ------------------------------------------------------------------------------------------------------------------------------------
# #------------------------------------------------------------------------------------------------------------------------------------
# Missing value imputation

ImputerKNN = KNNImputer(n_neighbors=2)
df_train_cat_selected_num_all = ImputerKNN.fit_transform(df_train_cat_selected_num_all)
df_test_cat_selected_num_all = ImputerKNN.transform(df_test_cat_selected_num_all)
# # ------------------------------------------------------------------------------------------------------------------------------------

# feature scaling - only for numerical(continuous features)


# In train_x_y , train_x_y:
#   first 117 column - cat features
#   from there to end - Numerical features

# we do feature scaling only on numerical features (coz, cat features are aleady encoded into 0 or 1)

sc = MinMaxScaler()

df_train_cat_selected_num_all[:, 160:] = sc.fit_transform(df_train_cat_selected_num_all[:, 160:])  # Here feature scaling not applied to dummy columns(first 3 columns), i.e. for France = 100,Spain=010 and Germany=001, because those column values are alread in between -3 and 3, and also, if feature scaling do to these columns, abnormal values may return
# Here 'fit method' calculate ,mean and the standard devation of each feature. 'Transform method' apply equation, { Xstand=[x-mean(x)]/standard devation(x) , where x -feature, here have to categoroed for x, which is salary and ange. which called 'Standarization'}, for each feature.

df_test_cat_selected_num_all[:, 160:] = sc.transform(df_test_cat_selected_num_all[:, 160:])  # Here, when do feature scaling in test set, test set should be scaled by using the same parameters used in training set.
# Also, x_test is the input for the prediction function got from training set. That's why here only transform method is using instead fit_transform.
# Means, here when apply standarization to each of two features (age and salary), the mean and the standard deviation used is the values got from training data. >> Xstand_test=[x_test-mean(x_train)]/standard devation(x_train)
# # ------------------------------------------------------------------------------------------------------------------------------------

# download files to feed into greedy

# training

train_cols = x_axis_original + ['label']

y1_1 = np.array([df_train_all['label']])  # (1, 10732)
y1_2 = np.transpose(y1_1)  # (10732, 1)
concat_xtrain = np.concatenate((df_train_cat_selected_num_all, y1_2), axis=1)

train_set_to_greedy = pd.DataFrame(concat_xtrain, columns=train_cols)

output_result_dir = '/Users/psenevirathn/Desktop/PhD/Coding/Python/output_csv_files'

save_train_loc = os.path.join(output_result_dir,
                              'training_set_to_greedy.csv')  # This Returns a path. os.path.join - https://www.geeksforgeeks.org/python-os-path-join-method/

#train_set_to_greedy.to_csv(save_train_loc)

# test

test_cols = x_axis_original + ['label']

y1_1 = np.array([df_test_all['label']])  # (1, 2683)
y1_2 = np.transpose(y1_1)  # (2683, 1)
concat_xtest = np.concatenate((df_test_cat_selected_num_all, y1_2), axis=1)

test_set_to_greedy = pd.DataFrame(concat_xtest, columns=test_cols)

output_result_dir = '/Users/psenevirathn/Desktop/PhD/Coding/Python/output_csv_files'

save_test_loc = os.path.join(output_result_dir,
                             'test_set_to_greedy.csv')  # This Returns a path. os.path.join - https://www.geeksforgeeks.org/python-os-path-join-method/

#test_set_to_greedy.to_csv(save_test_loc)

# The plot of p-values of categorical fetures for thesis coded in jupiter notbook - /Users/psenevirathn/PycharmProjects/scientificProject/Thesis_Figures_Chapter4_HIT_classifier.ipynb

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# python 3_Jan_3_Numerical_feature_selection_preprocessing.py /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/train_data_before_preprocessing.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/test_data_before_preprocessing.csv
