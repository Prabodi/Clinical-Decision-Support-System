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
from sklearn.ensemble import GradientBoostingClassifier
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
from sklearn.model_selection import cross_val_predict
# ------------------------------------------------------------------------------------------------------------------------------------

# 2. Import files

df_train_all = pd.read_csv(sys.argv[1])  # after prepocessing - training set only , Counter({0: 9141, 1: 1591})
df_test_all = pd.read_csv(sys.argv[2])  # after prepocessing - test set only , Counter({0: 2285, 1: 398})

df_updated_labels = pd.read_csv(sys.argv[3])  # updated HIT labels (new ground thuth) - for both train and test data

# with 'orginal label'

label_count_train_orginal_label = Counter(df_train_all['label']) # Counter({0: 9141, 1: 1591})
label_count_test_orginal_label = Counter(df_test_all['label']) # Counter({0: 2285, 1: 398})

print(label_count_train_orginal_label)
print(label_count_test_orginal_label)

# from data set, drop 'hadm_id'

df_train_all = df_train_all.drop(['label'], axis=1)
df_test_all = df_test_all.drop(['label'], axis=1)

df_train_all['label'] = pd.merge(df_train_all, df_updated_labels, on='hadm_id', how='left')['updated_HIT_label'] # column 'label' (with 'original label') is overwritten by 'updtaed label'
df_test_all['label'] = pd.merge(df_test_all, df_updated_labels, on='hadm_id', how='left')['updated_HIT_label']

# count class ditribution - HIT / No HIT

label_count_train = Counter(df_train_all['label']) # Counter({0: 9209, 1: 1523}) # For 68 previously OD patients in train set, ground truth changed from '1' to '0'.
label_count_test = Counter(df_test_all['label']) # Counter({0: 2293, 1: 390}) # For 8 previously OD patients in test set, ground truth changed from '1' to '0'.
label_count_full_data_set = Counter(pd.concat([df_train_all['label'], df_test_all['label']], axis = 0)) # Counter({0: 11502, 1: 1913})
print(label_count_train)
print(label_count_test)
print(label_count_full_data_set)

# from data set, drop 'hadm_id'

hadm_id_train = df_train_all['hadm_id']
hadm_id_test = df_test_all['hadm_id']

df_train_all = df_train_all.drop(['hadm_id'], axis=1)
df_test_all = df_test_all.drop(['hadm_id'], axis=1)

#print(df_train_all.columns.tolist()) # ['first_careunit', 'admission_type', 'admission_location', 'gender', 'anchor_age', 'base_platelets', 'hep_types', 'treatment_types', 'heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean', 'dbp_min', 'dbp_max', 'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean', 'resp_rate_min', 'resp_rate_max', 'resp_rate_mean', 'spo2_min', 'spo2_max', 'spo2_mean', 'temperature_vital_min', 'temperature_vital_max', 'temperature_vital_mean', 'glucose_vital_min', 'glucose_vital_max', 'glucose_vital_mean', 'hematocrit_lab_min', 'hematocrit_lab_max', 'hemoglobin_lab_min', 'hemoglobin_lab_max', 'bicarbonate_lab_min', 'bicarbonate_lab_max', 'calcium_lab_min', 'calcium_lab_max', 'chloride_lab_min', 'chloride_lab_max', 'sodium_lab_min', 'sodium_lab_max', 'potassium_lab_min', 'potassium_lab_max', 'glucose_lab_min', 'glucose_lab_max', 'platelets_min', 'platelets_max', 'wbc_min', 'wbc_max', 'aniongap_min', 'aniongap_max', 'bun_min', 'bun_max', 'creatinine_min', 'creatinine_max', 'inr_min', 'inr_max', 'pt_min', 'pt_max', 'ptt_min', 'ptt_max', 'gcs_min', 'thrombin_min_status', 'thrombin_max_status', 'd_dimer_max_status', 'd_dimer_min_status', 'methemoglobin_min_status', 'methemoglobin_max_status', 'ggt_min_status', 'ggt_max_status', 'globulin_min_status', 'globulin_max_status', 'total_protein_min_status', 'total_protein_max_status', 'atyps_max_status', 'atyps_min_status', 'carboxyhemoglobin_min_status', 'carboxyhemoglobin_max_status', 'amylase_max_status', 'amylase_min_status', 'aado2_bg_art_max_status', 'aado2_bg_art_min_status', 'bilirubin_direct_min_status', 'bilirubin_direct_max_status', 'bicarbonate_bg_min_status', 'bicarbonate_bg_max_status', 'fio2_bg_art_min_status', 'fio2_bg_art_max_status', 'nrbc_max_status', 'nrbc_min_status', 'bands_min_status', 'bands_max_status', 'so2_bg_art_min_status', 'so2_bg_art_max_status', 'fibrinogen_max_status', 'fibrinogen_min_status', 'hematocrit_bg_min_status', 'hematocrit_bg_max_status', 'hemoglobin_bg_min_status', 'hemoglobin_bg_max_status', 'temperature_bg_max_status', 'temperature_bg_min_status', 'chloride_bg_min_status', 'chloride_bg_max_status', 'sodium_bg_max_status', 'sodium_bg_min_status', 'glucose_bg_max_status', 'glucose_bg_min_status', 'ck_cpk_max_status', 'ck_cpk_min_status', 'ck_mb_max_status', 'ck_mb_min_status', 'ld_ldh_max_status', 'ld_ldh_min_status', 'calcium_bg_max_status', 'calcium_bg_min_status', 'pco2_bg_art_min_status', 'po2_bg_art_max_status', 'totalco2_bg_art_max_status', 'totalco2_bg_art_min_status', 'pco2_bg_art_max_status', 'po2_bg_art_min_status', 'potassium_bg_min_status', 'potassium_bg_max_status', 'albumin_max_status', 'albumin_min_status', 'bilirubin_total_min_status', 'bilirubin_total_max_status', 'alt_max_status', 'alt_min_status', 'alp_max_status', 'alp_min_status', 'ast_min_status', 'ast_max_status', 'pco2_bg_max_status', 'pco2_bg_min_status', 'totalco2_bg_min_status', 'totalco2_bg_max_status', 'ph_min_status', 'ph_max_status', 'lactate_min_status', 'lactate_max_status', 'label']
#print(len(df_train_all.columns.tolist())) # 146

# Rounding-off values of selected columns (which have a lot of decimal points) into 2 decimal points

df_train_all = df_train_all.round(
    {'heart_rate_mean': 2, 'sbp_mean': 2, 'dbp_mean': 2, 'mbp_mean': 2, 'resp_rate_mean': 2, 'temperature_mean': 2,
     'spo2_mean': 2, 'glucose_mean': 2})
df_test_all = df_test_all.round(
    {'heart_rate_mean': 2, 'sbp_mean': 2, 'dbp_mean': 2, 'mbp_mean': 2, 'resp_rate_mean': 2, 'temperature_mean': 2,
     'spo2_mean': 2, 'glucose_mean': 2})

# col_names = ['encoder__first_careunit_Cardiac Vascular Intensive Care Unit (CVICU)', 'encoder__first_careunit_Coronary Care Unit (CCU)', 'encoder__first_careunit_Medical Intensive Care Unit (MICU)', 'encoder__first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)', 'encoder__first_careunit_Neuro Intermediate', 'encoder__first_careunit_Neuro Stepdown', 'encoder__first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)', 'encoder__first_careunit_Surgical Intensive Care Unit (SICU)', 'encoder__first_careunit_Trauma SICU (TSICU)', 'encoder__admission_type_DIRECT EMER.', 'encoder__admission_type_DIRECT OBSERVATION', 'encoder__admission_type_ELECTIVE', 'encoder__admission_type_EU OBSERVATION', 'encoder__admission_type_EW EMER.', 'encoder__admission_type_OBSERVATION ADMIT', 'encoder__admission_type_SURGICAL SAME DAY ADMISSION', 'encoder__admission_type_URGENT', 'encoder__admission_location_AMBULATORY SURGERY TRANSFER', 'encoder__admission_location_CLINIC REFERRAL', 'encoder__admission_location_EMERGENCY ROOM', 'encoder__admission_location_INFORMATION NOT AVAILABLE', 'encoder__admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'encoder__admission_location_PACU', 'encoder__admission_location_PHYSICIAN REFERRAL', 'encoder__admission_location_PROCEDURE SITE', 'encoder__admission_location_TRANSFER FROM HOSPITAL', 'encoder__admission_location_TRANSFER FROM SKILLED NURSING FACILITY', 'encoder__admission_location_WALK-IN/SELF REFERRAL', 'encoder__hep_types_LMWH', 'encoder__hep_types_UFH', 'encoder__hep_types_both', 'encoder__treatment_types_P', 'encoder__treatment_types_T', 'encoder__treatment_types_both', 'encoder__lactate_min_status_elevated', 'encoder__lactate_min_status_low', 'encoder__lactate_min_status_normal', 'encoder__lactate_min_status_not ordered', 'encoder__lactate_max_status_elevated', 'encoder__lactate_max_status_low', 'encoder__lactate_max_status_normal', 'encoder__lactate_max_status_not ordered', 'encoder__ph_min_status_elevated', 'encoder__ph_min_status_low', 'encoder__ph_min_status_normal', 'encoder__ph_min_status_not ordered', 'encoder__ph_max_status_elevated', 'encoder__ph_max_status_low', 'encoder__ph_max_status_normal', 'encoder__ph_max_status_not ordered', 'encoder__totalco2_bg_min_status_elevated', 'encoder__totalco2_bg_min_status_low', 'encoder__totalco2_bg_min_status_normal', 'encoder__totalco2_bg_min_status_not ordered', 'encoder__totalco2_bg_max_status_elevated', 'encoder__totalco2_bg_max_status_low', 'encoder__totalco2_bg_max_status_normal', 'encoder__totalco2_bg_max_status_not ordered', 'encoder__pco2_bg_min_status_elevated', 'encoder__pco2_bg_min_status_low', 'encoder__pco2_bg_min_status_normal', 'encoder__pco2_bg_min_status_not ordered', 'encoder__pco2_bg_max_status_elevated', 'encoder__pco2_bg_max_status_low', 'encoder__pco2_bg_max_status_normal', 'encoder__pco2_bg_max_status_not ordered', 'encoder__ast_min_status_elevated', 'encoder__ast_min_status_normal', 'encoder__ast_min_status_not ordered', 'encoder__ast_max_status_elevated', 'encoder__ast_max_status_normal', 'encoder__ast_max_status_not ordered', 'encoder__alp_min_status_elevated', 'encoder__alp_min_status_low', 'encoder__alp_min_status_normal', 'encoder__alp_min_status_not ordered', 'encoder__alp_max_status_elevated', 'encoder__alp_max_status_low', 'encoder__alp_max_status_normal', 'encoder__alp_max_status_not ordered', 'encoder__alt_min_status_elevated', 'encoder__alt_min_status_normal', 'encoder__alt_min_status_not ordered', 'encoder__alt_max_status_elevated', 'encoder__alt_max_status_normal', 'encoder__alt_max_status_not ordered', 'encoder__bilirubin_total_min_status_elevated', 'encoder__bilirubin_total_min_status_normal', 'encoder__bilirubin_total_min_status_not ordered', 'encoder__bilirubin_total_max_status_elevated', 'encoder__bilirubin_total_max_status_normal', 'encoder__bilirubin_total_max_status_not ordered', 'encoder__albumin_min_status_elevated', 'encoder__albumin_min_status_low', 'encoder__albumin_min_status_normal', 'encoder__albumin_min_status_not ordered', 'encoder__albumin_max_status_elevated', 'encoder__albumin_max_status_low', 'encoder__albumin_max_status_normal', 'encoder__albumin_max_status_not ordered', 'encoder__pco2_bg_art_min_status_elevated', 'encoder__pco2_bg_art_min_status_low', 'encoder__pco2_bg_art_min_status_normal', 'encoder__pco2_bg_art_min_status_not ordered', 'encoder__pco2_bg_art_max_status_elevated', 'encoder__pco2_bg_art_max_status_low', 'encoder__pco2_bg_art_max_status_normal', 'encoder__pco2_bg_art_max_status_not ordered', 'encoder__po2_bg_art_min_status_elevated', 'encoder__po2_bg_art_min_status_low', 'encoder__po2_bg_art_min_status_normal', 'encoder__po2_bg_art_min_status_not ordered', 'encoder__po2_bg_art_max_status_elevated', 'encoder__po2_bg_art_max_status_low', 'encoder__po2_bg_art_max_status_normal', 'encoder__po2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_min_status_elevated', 'encoder__totalco2_bg_art_min_status_low', 'encoder__totalco2_bg_art_min_status_normal', 'encoder__totalco2_bg_art_min_status_not ordered', 'encoder__totalco2_bg_art_max_status_elevated', 'encoder__totalco2_bg_art_max_status_low', 'encoder__totalco2_bg_art_max_status_normal', 'encoder__totalco2_bg_art_max_status_not ordered', 'encoder__ld_ldh_min_status_elevated', 'encoder__ld_ldh_min_status_low', 'encoder__ld_ldh_min_status_normal', 'encoder__ld_ldh_min_status_not ordered', 'encoder__ld_ldh_max_status_elevated', 'encoder__ld_ldh_max_status_low', 'encoder__ld_ldh_max_status_normal', 'encoder__ld_ldh_max_status_not ordered', 'encoder__ck_cpk_min_status_elevated', 'encoder__ck_cpk_min_status_low', 'encoder__ck_cpk_min_status_normal', 'encoder__ck_cpk_min_status_not ordered', 'encoder__ck_cpk_max_status_elevated', 'encoder__ck_cpk_max_status_low', 'encoder__ck_cpk_max_status_normal', 'encoder__ck_cpk_max_status_not ordered', 'encoder__ck_mb_min_status_elevated', 'encoder__ck_mb_min_status_normal', 'encoder__ck_mb_min_status_not ordered', 'encoder__ck_mb_max_status_elevated', 'encoder__ck_mb_max_status_normal', 'encoder__ck_mb_max_status_not ordered', 'encoder__fio2_bg_art_min_status_no ref range', 'encoder__fio2_bg_art_min_status_not ordered', 'encoder__fio2_bg_art_max_status_no ref range', 'encoder__fio2_bg_art_max_status_not ordered', 'encoder__so2_bg_art_min_status_no ref range', 'encoder__so2_bg_art_min_status_not ordered', 'encoder__so2_bg_art_max_status_no ref range', 'encoder__so2_bg_art_max_status_not ordered', 'encoder__fibrinogen_min_status_elevated', 'encoder__fibrinogen_min_status_low', 'encoder__fibrinogen_min_status_normal', 'encoder__fibrinogen_min_status_not ordered', 'encoder__fibrinogen_max_status_elevated', 'encoder__fibrinogen_max_status_low', 'encoder__fibrinogen_max_status_normal', 'encoder__fibrinogen_max_status_not ordered', 'encoder__thrombin_min_status_elevated', 'encoder__thrombin_min_status_normal', 'encoder__thrombin_min_status_not ordered', 'encoder__thrombin_max_status_elevated', 'encoder__thrombin_max_status_normal', 'encoder__thrombin_max_status_not ordered', 'encoder__d_dimer_min_status_elevated', 'encoder__d_dimer_min_status_normal', 'encoder__d_dimer_min_status_not ordered', 'encoder__d_dimer_max_status_elevated', 'encoder__d_dimer_max_status_normal', 'encoder__d_dimer_max_status_not ordered', 'encoder__methemoglobin_min_status_elevated', 'encoder__methemoglobin_min_status_normal', 'encoder__methemoglobin_min_status_not ordered', 'encoder__methemoglobin_max_status_elevated', 'encoder__methemoglobin_max_status_normal', 'encoder__methemoglobin_max_status_not ordered', 'encoder__ggt_min_status_elevated', 'encoder__ggt_min_status_low', 'encoder__ggt_min_status_normal', 'encoder__ggt_min_status_not ordered', 'encoder__ggt_max_status_elevated', 'encoder__ggt_max_status_low', 'encoder__ggt_max_status_normal', 'encoder__ggt_max_status_not ordered', 'encoder__globulin_min_status_elevated', 'encoder__globulin_min_status_low', 'encoder__globulin_min_status_normal', 'encoder__globulin_min_status_not ordered', 'encoder__globulin_max_status_elevated', 'encoder__globulin_max_status_low', 'encoder__globulin_max_status_normal', 'encoder__globulin_max_status_not ordered', 'encoder__atyps_min_status_elevated', 'encoder__atyps_min_status_not ordered', 'encoder__atyps_max_status_elevated', 'encoder__atyps_max_status_not ordered', 'encoder__total_protein_min_status_elevated', 'encoder__total_protein_min_status_low', 'encoder__total_protein_min_status_normal', 'encoder__total_protein_min_status_not ordered', 'encoder__total_protein_max_status_elevated', 'encoder__total_protein_max_status_low', 'encoder__total_protein_max_status_normal', 'encoder__total_protein_max_status_not ordered', 'encoder__carboxyhemoglobin_min_status_elevated', 'encoder__carboxyhemoglobin_min_status_normal', 'encoder__carboxyhemoglobin_min_status_not ordered', 'encoder__carboxyhemoglobin_max_status_elevated', 'encoder__carboxyhemoglobin_max_status_normal', 'encoder__carboxyhemoglobin_max_status_not ordered', 'encoder__amylase_min_status_elevated', 'encoder__amylase_min_status_normal', 'encoder__amylase_min_status_not ordered', 'encoder__amylase_max_status_elevated', 'encoder__amylase_max_status_normal', 'encoder__amylase_max_status_not ordered', 'encoder__aado2_bg_art_min_status_no ref range', 'encoder__aado2_bg_art_min_status_not ordered', 'encoder__aado2_bg_art_max_status_no ref range', 'encoder__aado2_bg_art_max_status_not ordered', 'encoder__bilirubin_direct_min_status_elevated', 'encoder__bilirubin_direct_min_status_normal', 'encoder__bilirubin_direct_min_status_not ordered', 'encoder__bilirubin_direct_max_status_elevated', 'encoder__bilirubin_direct_max_status_normal', 'encoder__bilirubin_direct_max_status_not ordered', 'encoder__nrbc_min_status_elevated', 'encoder__nrbc_min_status_not ordered', 'encoder__nrbc_max_status_elevated', 'encoder__nrbc_max_status_not ordered', 'encoder__bands_min_status_elevated', 'encoder__bands_min_status_normal', 'encoder__bands_min_status_not ordered', 'encoder__bands_max_status_elevated', 'encoder__bands_max_status_normal', 'encoder__bands_max_status_not ordered', 'remainder__gender', 'remainder__anchor_age', 'remainder__base_platelets', 'remainder__heart_rate_min', 'remainder__heart_rate_max', 'remainder__heart_rate_mean', 'remainder__sbp_min', 'remainder__sbp_max', 'remainder__sbp_mean', 'remainder__dbp_min', 'remainder__dbp_max', 'remainder__dbp_mean', 'remainder__mbp_min', 'remainder__mbp_max', 'remainder__mbp_mean', 'remainder__resp_rate_min', 'remainder__resp_rate_max', 'remainder__resp_rate_mean', 'remainder__temperature_min', 'remainder__temperature_max', 'remainder__temperature_mean', 'remainder__spo2_min', 'remainder__spo2_max', 'remainder__spo2_mean', 'remainder__glucose_min', 'remainder__glucose_max', 'remainder__glucose_mean', 'remainder__hematocrit_min', 'remainder__hematocrit_max', 'remainder__hemoglobin_min', 'remainder__hemoglobin_max', 'remainder__bicarbonate_min', 'remainder__bicarbonate_max', 'remainder__calcium_min', 'remainder__calcium_max', 'remainder__chloride_min', 'remainder__chloride_max', 'remainder__sodium_min', 'remainder__sodium_max', 'remainder__potassium_min', 'remainder__potassium_max', 'remainder__platelets_min', 'remainder__platelets_max', 'remainder__wbc_min', 'remainder__wbc_max', 'remainder__aniongap_min', 'remainder__aniongap_max', 'remainder__bun_min', 'remainder__bun_max', 'remainder__creatinine_min', 'remainder__creatinine_max', 'remainder__inr_min', 'remainder__inr_max', 'remainder__pt_min', 'remainder__pt_max', 'remainder__ptt_min', 'remainder__ptt_max', 'remainder__gcs_min', 'label']

# --------------------------------------------------------------------------------------------------------------

# categorical feature selection

# training dataset

df_train_categorical_selected = df_train_all[  # (10090, 60)
    ['first_careunit', 'admission_location', 'gender', 'treatment_types', 'atyps_max_status', 'atyps_min_status',
     'bilirubin_direct_min_status', 'bilirubin_direct_max_status', 'nrbc_max_status', 'nrbc_min_status',
     'bands_min_status', 'bands_max_status', 'so2_bg_art_min_status', 'so2_bg_art_max_status', 'fibrinogen_max_status',
     'fibrinogen_min_status', 'hematocrit_bg_min_status', 'hematocrit_bg_max_status', 'hemoglobin_bg_min_status',
     'hemoglobin_bg_max_status', 'temperature_bg_max_status', 'temperature_bg_min_status', 'sodium_bg_max_status',
     'sodium_bg_min_status', 'glucose_bg_max_status', 'glucose_bg_min_status', 'ck_cpk_max_status', 'ck_cpk_min_status',
     'ck_mb_max_status', 'ck_mb_min_status', 'ld_ldh_max_status', 'ld_ldh_min_status', 'calcium_bg_max_status',
     'calcium_bg_min_status', 'pco2_bg_art_min_status', 'po2_bg_art_max_status', 'totalco2_bg_art_max_status',
     'totalco2_bg_art_min_status', 'pco2_bg_art_max_status', 'po2_bg_art_min_status', 'potassium_bg_min_status',
     'potassium_bg_max_status', 'albumin_max_status', 'albumin_min_status', 'bilirubin_total_min_status',
     'bilirubin_total_max_status', 'alt_max_status', 'alt_min_status', 'alp_max_status', 'alp_min_status',
     'ast_min_status', 'ast_max_status', 'pco2_bg_max_status', 'pco2_bg_min_status', 'totalco2_bg_min_status',
     'totalco2_bg_max_status', 'ph_min_status', 'ph_max_status', 'lactate_min_status', 'lactate_max_status']]

df_train_numerical_selected = df_train_all[  # (10090, 14)
    ['platelets_min', 'pt_max', 'creatinine_max', 'temperature_vital_min', 'bun_max', 'inr_max', 'inr_min',
     'anchor_age', 'resp_rate_min', 'bicarbonate_lab_max', 'bun_min', 'aniongap_max', 'wbc_max', 'hemoglobin_lab_min']]

df_train_cat_selected_numerical_selected = pd.concat(   # (10090, 74)
    [df_train_categorical_selected, df_train_numerical_selected], axis=1)

print(df_train_categorical_selected.shape)  # (10090, 60)
print(df_train_numerical_selected.shape)  # (10090, 14)
print(df_train_cat_selected_numerical_selected.shape)  # (10090, 74)

# ------------------------------------------------------------------------------------------------------------------------------------

# testing data set

df_test_categorical_selected = df_test_all[  # (2523, 60)
    ['first_careunit', 'admission_location', 'gender', 'treatment_types', 'atyps_max_status', 'atyps_min_status',
     'bilirubin_direct_min_status', 'bilirubin_direct_max_status', 'nrbc_max_status', 'nrbc_min_status',
     'bands_min_status', 'bands_max_status', 'so2_bg_art_min_status', 'so2_bg_art_max_status', 'fibrinogen_max_status',
     'fibrinogen_min_status', 'hematocrit_bg_min_status', 'hematocrit_bg_max_status', 'hemoglobin_bg_min_status',
     'hemoglobin_bg_max_status', 'temperature_bg_max_status', 'temperature_bg_min_status', 'sodium_bg_max_status',
     'sodium_bg_min_status', 'glucose_bg_max_status', 'glucose_bg_min_status', 'ck_cpk_max_status', 'ck_cpk_min_status',
     'ck_mb_max_status', 'ck_mb_min_status', 'ld_ldh_max_status', 'ld_ldh_min_status', 'calcium_bg_max_status',
     'calcium_bg_min_status', 'pco2_bg_art_min_status', 'po2_bg_art_max_status', 'totalco2_bg_art_max_status',
     'totalco2_bg_art_min_status', 'pco2_bg_art_max_status', 'po2_bg_art_min_status', 'potassium_bg_min_status',
     'potassium_bg_max_status', 'albumin_max_status', 'albumin_min_status', 'bilirubin_total_min_status',
     'bilirubin_total_max_status', 'alt_max_status', 'alt_min_status', 'alp_max_status', 'alp_min_status',
     'ast_min_status', 'ast_max_status', 'pco2_bg_max_status', 'pco2_bg_min_status', 'totalco2_bg_min_status',
     'totalco2_bg_max_status', 'ph_min_status', 'ph_max_status', 'lactate_min_status', 'lactate_max_status']]

df_test_numerical_selected = df_test_all[  # (2523, 14)
    ['platelets_min', 'pt_max', 'creatinine_max', 'temperature_vital_min', 'bun_max', 'inr_max', 'inr_min',
     'anchor_age', 'resp_rate_min', 'bicarbonate_lab_max', 'bun_min', 'aniongap_max', 'wbc_max', 'hemoglobin_lab_min']]

df_test_cat_selected_numerical_selected = pd.concat(  # (2523, 74)
    [df_test_categorical_selected, df_test_numerical_selected], axis=1)

print(df_test_categorical_selected.shape)  # (2523, 60)
print(df_test_numerical_selected.shape)  # (2523, 14)
print(df_test_cat_selected_numerical_selected.shape)  # (2523, 74)


# ------------------------------------------------------------------------------------------------------------------------------------

# train set

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'),
                   df_train_categorical_selected.columns.tolist())],
    remainder='passthrough')

df_train_cat_selected_numerical_selected = np.array(ct.fit_transform(
    df_train_cat_selected_numerical_selected))  # here 'np' (NumPy) was added because, fit_transform itself doesn't return output in np array, so in order to train future machine learning models, np is added.

print(df_train_cat_selected_numerical_selected.shape)  # (10732, 174)

# test set

df_test_cat_selected_numerical_selected = np.array(
    ct.transform(df_test_cat_selected_numerical_selected))  # handle_unknown = 'ignore'

x_axis_original = ct.get_feature_names_out().tolist()
print(x_axis_original)
# ['encoder__first_careunit_Coronary Care Unit (CCU)', 'encoder__first_careunit_Medical Intensive Care Unit (MICU)', 'encoder__first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)', 'encoder__first_careunit_Neuro Intermediate', 'encoder__first_careunit_Neuro Stepdown', 'encoder__first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)', 'encoder__first_careunit_Surgical Intensive Care Unit (SICU)', 'encoder__first_careunit_Trauma SICU (TSICU)', 'encoder__admission_location_CLINIC REFERRAL', 'encoder__admission_location_EMERGENCY ROOM', 'encoder__admission_location_INFORMATION NOT AVAILABLE', 'encoder__admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'encoder__admission_location_PACU', 'encoder__admission_location_PHYSICIAN REFERRAL', 'encoder__admission_location_PROCEDURE SITE', 'encoder__admission_location_TRANSFER FROM HOSPITAL', 'encoder__admission_location_TRANSFER FROM SKILLED NURSING FACILITY', 'encoder__admission_location_WALK-IN/SELF REFERRAL', 'encoder__gender_M', 'encoder__treatment_types_T', 'encoder__atyps_max_status_normal', 'encoder__atyps_max_status_not ordered', 'encoder__atyps_min_status_normal', 'encoder__atyps_min_status_not ordered', 'encoder__bilirubin_direct_min_status_normal', 'encoder__bilirubin_direct_min_status_not ordered', 'encoder__bilirubin_direct_max_status_normal', 'encoder__bilirubin_direct_max_status_not ordered', 'encoder__nrbc_max_status_normal', 'encoder__nrbc_max_status_not ordered', 'encoder__nrbc_min_status_normal', 'encoder__nrbc_min_status_not ordered', 'encoder__bands_min_status_normal', 'encoder__bands_min_status_not ordered', 'encoder__bands_max_status_normal', 'encoder__bands_max_status_not ordered', 'encoder__so2_bg_art_min_status_not ordered', 'encoder__so2_bg_art_max_status_not ordered', 'encoder__fibrinogen_max_status_low', 'encoder__fibrinogen_max_status_normal', 'encoder__fibrinogen_max_status_not ordered', 'encoder__fibrinogen_min_status_low', 'encoder__fibrinogen_min_status_normal', 'encoder__fibrinogen_min_status_not ordered', 'encoder__hematocrit_bg_min_status_not ordered', 'encoder__hematocrit_bg_max_status_not ordered', 'encoder__hemoglobin_bg_min_status_low', 'encoder__hemoglobin_bg_min_status_normal', 'encoder__hemoglobin_bg_min_status_not ordered', 'encoder__hemoglobin_bg_max_status_low', 'encoder__hemoglobin_bg_max_status_normal', 'encoder__hemoglobin_bg_max_status_not ordered', 'encoder__temperature_bg_max_status_not ordered', 'encoder__temperature_bg_min_status_not ordered', 'encoder__sodium_bg_max_status_low', 'encoder__sodium_bg_max_status_normal', 'encoder__sodium_bg_max_status_not ordered', 'encoder__sodium_bg_min_status_low', 'encoder__sodium_bg_min_status_normal', 'encoder__sodium_bg_min_status_not ordered', 'encoder__glucose_bg_max_status_low', 'encoder__glucose_bg_max_status_normal', 'encoder__glucose_bg_max_status_not ordered', 'encoder__glucose_bg_min_status_low', 'encoder__glucose_bg_min_status_normal', 'encoder__glucose_bg_min_status_not ordered', 'encoder__ck_cpk_max_status_low', 'encoder__ck_cpk_max_status_normal', 'encoder__ck_cpk_max_status_not ordered', 'encoder__ck_cpk_min_status_low', 'encoder__ck_cpk_min_status_normal', 'encoder__ck_cpk_min_status_not ordered', 'encoder__ck_mb_max_status_normal', 'encoder__ck_mb_max_status_not ordered', 'encoder__ck_mb_min_status_normal', 'encoder__ck_mb_min_status_not ordered', 'encoder__ld_ldh_max_status_low', 'encoder__ld_ldh_max_status_normal', 'encoder__ld_ldh_max_status_not ordered', 'encoder__ld_ldh_min_status_low', 'encoder__ld_ldh_min_status_normal', 'encoder__ld_ldh_min_status_not ordered', 'encoder__calcium_bg_max_status_low', 'encoder__calcium_bg_max_status_normal', 'encoder__calcium_bg_max_status_not ordered', 'encoder__calcium_bg_min_status_low', 'encoder__calcium_bg_min_status_normal', 'encoder__calcium_bg_min_status_not ordered', 'encoder__pco2_bg_art_min_status_low', 'encoder__pco2_bg_art_min_status_normal', 'encoder__pco2_bg_art_min_status_not ordered', 'encoder__po2_bg_art_max_status_low', 'encoder__po2_bg_art_max_status_normal', 'encoder__po2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_max_status_low', 'encoder__totalco2_bg_art_max_status_normal', 'encoder__totalco2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_min_status_low', 'encoder__totalco2_bg_art_min_status_normal', 'encoder__totalco2_bg_art_min_status_not ordered', 'encoder__pco2_bg_art_max_status_low', 'encoder__pco2_bg_art_max_status_normal', 'encoder__pco2_bg_art_max_status_not ordered', 'encoder__po2_bg_art_min_status_low', 'encoder__po2_bg_art_min_status_normal', 'encoder__po2_bg_art_min_status_not ordered', 'encoder__potassium_bg_min_status_low', 'encoder__potassium_bg_min_status_normal', 'encoder__potassium_bg_min_status_not ordered', 'encoder__potassium_bg_max_status_low', 'encoder__potassium_bg_max_status_normal', 'encoder__potassium_bg_max_status_not ordered', 'encoder__albumin_max_status_low', 'encoder__albumin_max_status_normal', 'encoder__albumin_max_status_not ordered', 'encoder__albumin_min_status_low', 'encoder__albumin_min_status_normal', 'encoder__albumin_min_status_not ordered', 'encoder__bilirubin_total_min_status_normal', 'encoder__bilirubin_total_min_status_not ordered', 'encoder__bilirubin_total_max_status_normal', 'encoder__bilirubin_total_max_status_not ordered', 'encoder__alt_max_status_normal', 'encoder__alt_max_status_not ordered', 'encoder__alt_min_status_normal', 'encoder__alt_min_status_not ordered', 'encoder__alp_max_status_low', 'encoder__alp_max_status_normal', 'encoder__alp_max_status_not ordered', 'encoder__alp_min_status_low', 'encoder__alp_min_status_normal', 'encoder__alp_min_status_not ordered', 'encoder__ast_min_status_normal', 'encoder__ast_min_status_not ordered', 'encoder__ast_max_status_normal', 'encoder__ast_max_status_not ordered', 'encoder__pco2_bg_max_status_low', 'encoder__pco2_bg_max_status_normal', 'encoder__pco2_bg_max_status_not ordered', 'encoder__pco2_bg_min_status_low', 'encoder__pco2_bg_min_status_normal', 'encoder__pco2_bg_min_status_not ordered', 'encoder__totalco2_bg_min_status_low', 'encoder__totalco2_bg_min_status_normal', 'encoder__totalco2_bg_min_status_not ordered', 'encoder__totalco2_bg_max_status_low', 'encoder__totalco2_bg_max_status_normal', 'encoder__totalco2_bg_max_status_not ordered', 'encoder__ph_min_status_low', 'encoder__ph_min_status_normal', 'encoder__ph_min_status_not ordered', 'encoder__ph_max_status_low', 'encoder__ph_max_status_normal', 'encoder__ph_max_status_not ordered', 'encoder__lactate_min_status_low', 'encoder__lactate_min_status_normal', 'encoder__lactate_min_status_not ordered', 'encoder__lactate_max_status_low', 'encoder__lactate_max_status_normal', 'encoder__lactate_max_status_not ordered', 'remainder__platelets_min', 'remainder__pt_max', 'remainder__creatinine_max', 'remainder__temperature_vital_min', 'remainder__bun_max', 'remainder__inr_max', 'remainder__inr_min', 'remainder__anchor_age', 'remainder__resp_rate_min', 'remainder__bicarbonate_lab_max', 'remainder__bun_min', 'remainder__aniongap_max', 'remainder__wbc_max', 'remainder__hemoglobin_lab_min']

#print(len(x_axis_original))  # 174
#print(x_axis_original[0]) # encoder__first_careunit_Coronary Care Unit (CCU)
#print(x_axis_original[159:162]) # ['encoder__lactate_max_status_not ordered', 'remainder__platelets_min', 'remainder__pt_max']
# ------------------------------------------------------------------------------------------------------------------------------------
# Missing value imputation

ImputerKNN = KNNImputer(n_neighbors=2)
df_train_cat_selected_numerical_selected = ImputerKNN.fit_transform(df_train_cat_selected_numerical_selected)
df_test_cat_selected_numerical_selected = ImputerKNN.transform(df_test_cat_selected_numerical_selected)

# # ------------------------------------------------------------------------------------------------------------------------------------
# feature scaling - only for numerical(continuous features)

# In train_x_y , train_x_y:
#   first 117 columns - cat features
#   from there to end - Numerical features

# we do feature scaling only on numerical features (coz, cat features are aleady encoded into 0 or 1)

sc = MinMaxScaler()

# numerical features starts from index 160

df_train_cat_selected_numerical_selected[:, 160:] = sc.fit_transform(df_train_cat_selected_numerical_selected[:,
                                                                     160:])  # Here feature scaling not applied to dummy columns(first 3 columns), i.e. for France = 100,Spain=010 and Germany=001, because those column values are alread in between -3 and 3, and also, if feature scaling do to these columns, abnormal values may return
# Here 'fit method' calculate ,mean and the standard devation of each feature. 'Transform method' apply equation, { Xstand=[x-mean(x)]/standard devation(x) , where x -feature, here have to categoroed for x, which is salary and ange. which called 'Standarization'}, for each feature.

df_test_cat_selected_numerical_selected[:, 160:] = sc.transform(df_test_cat_selected_numerical_selected[:,
                                                                160:])  # Here, when do feature scaling in test set, test set should be scaled by using the same parameters used in training set.
# Also, x_test is the input for the prediction function got from training set. That's why here only transform method is using instead fit_transform.
# Means, here when apply standarization to each of two features (age and salary), the mean and the standard deviation used is the values got from training data. >> Xstand_test=[x_test-mean(x_train)]/standard devation(x_train)

print(df_train_cat_selected_numerical_selected.shape)  # (10732, 174)
print(df_test_cat_selected_numerical_selected.shape)  # (2683, 174)
# # ------------------------------------------------------------------------------------------------------------------------------------

# # 15. ML Predictors

x_train = df_train_cat_selected_numerical_selected
x_test = df_test_cat_selected_numerical_selected

y_train = df_train_all['label']
y_test = df_test_all['label']

# # # # ------------------------------------------------------------------------------------------------------------------------------------
# # #
# # ## 15.1. ML model 1 - Naive-Bayes
#
# print('Naive-Bayes')
# # Training the Naive-Bayes:
# classifier_NB = GaussianNB()
# classifier_NB.fit(x_train, y_train)
#
# # Predict the classifier response for the Test dataset:
# y_pred_NB = classifier_NB.predict(x_test)
#
# # len(y_test) - 22206
# print(len(y_pred_NB[y_pred_NB == 0]))
# print(len(y_test[y_test == 0]))
#
# ## Evaluate the Performance of blind test
# blind_cm_NB = confusion_matrix(y_test, y_pred_NB)  # cm for confusion matrix , len(y_test) - 22206
# print(blind_cm_NB)
#
# blind_acc_NB = float(
#     round(balanced_accuracy_score(y_test, y_pred_NB), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
# print(blind_acc_NB)
#
# blind_recall_NB = float(round(recall_score(y_test, y_pred_NB), 3))  # tp / (tp + fn)
# print(blind_recall_NB)
#
# blind_precision_NB = float(round(precision_score(y_test, y_pred_NB), 3))  # tp / (tp + fp)
# print(blind_precision_NB)
#
# blind_f1_NB = float(round(f1_score(y_test, y_pred_NB),
#                           3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# # It is primarily used to compare the performance of two classifiers.
# # Suppose that classifier A has a higher recall, and classifier B has higher precision.
# # In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
# print(blind_f1_NB)
#
# blind__mcc_NB = float(round(matthews_corrcoef(y_test, y_pred_NB),
#                             3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
# print(blind__mcc_NB)
#
# blind_AUC_NB = float(round(roc_auc_score(y_test, (classifier_NB.predict_proba(x_test)[:, 1])), 3))
# print(blind_AUC_NB)
# # area under the ROC curve, which is the curve having False Positive Rate on the x-axis and True Positive Rate on the y-axis at all classification thresholds.
#
# blind_test_NB = [blind_acc_NB, blind_recall_NB, blind_precision_NB, blind_f1_NB, blind__mcc_NB, blind_AUC_NB]
# print(blind_test_NB)
#
# # roc
#
# y_pred_proba_NB = classifier_NB.predict_proba(x_test)[::,
#                   1]  # Start at the beginning, end when it ends, walk in steps of 1 , # first col is prob of y=0, while 2nd col is prob of y=1 . https://dev.to/rajat_naegi/simply-explained-predictproba-263i
#
# # roc_auc_score(y, clf.predict_proba(X)[:, 1])
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba_NB)
# auc = roc_auc_score(y_test, y_pred_proba_NB)
#
# # plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
# # plt.legend(loc=4)
# # plt.show()
#
# # Number of Folds to split the data:
# # folds = 10 # not stratified
#
# folds = StratifiedKFold(n_splits=10, shuffle=True,
#                         random_state=0)  # why 'shuffle' parameter - https://stackoverflow.com/questions/63236831/shuffle-parameter-in-sklearn-model-selection-stratifiedkfold
#
# # Call the function of cross-validation passing the parameters:
# cross_accuracy_all_NB = cross_val_score(estimator=classifier_NB, X=x_train, y=y_train, cv=folds,
#                                         scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.
#
# cross_precision_all_NB = cross_val_score(estimator=classifier_NB, X=x_train, y=y_train, cv=folds, scoring='precision')
#
# cross_recall_all_NB = cross_val_score(estimator=classifier_NB, X=x_train, y=y_train, cv=folds, scoring='recall')
#
# cross_f1_all_NB = cross_val_score(estimator=classifier_NB, X=x_train, y=y_train, cv=folds, scoring='f1')
#
# # no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
# mcc = make_scorer(matthews_corrcoef)
# cross_mcc_all_NB = cross_val_score(estimator=classifier_NB, X=x_train, y=y_train, cv=folds, scoring=mcc)
#
# cross_AUC_all_NB = cross_val_score(estimator=classifier_NB, X=x_train, y=y_train, cv=folds, scoring='roc_auc')
#
# cross_validation_NB = [round((cross_accuracy_all_NB.mean()), 3), round((cross_recall_all_NB.mean()), 3),
#                        round((cross_precision_all_NB.mean()), 3), round((cross_f1_all_NB.mean()), 3),
#                        round((cross_mcc_all_NB.mean()), 3), round((cross_AUC_all_NB.mean()), 3)]
# print(cross_validation_NB)
#
# # ------------------------------------------------------------------------------------------------------------------------------------
#
# ## 15.2 ML model 2 - KNN Classifier
#
# # KNN
# print('KNN')
# ## How to choose the best number of neighbours? Let's create a range and see it!
#
# k_values = range(1, 10)
# KNN_MCC = []
#
# for n in k_values:
#     classifier_KNN = KNeighborsClassifier(n_neighbors=n)
#     model_KNN = classifier_KNN.fit(x_train, y_train)
#
#     # Predict the classifier's responses for the Test dataset
#     y_pred_KNN = model_KNN.predict(x_test)
#
#     # Evaluate using MCC:
#     KNN_MCC.append(float(round(matthews_corrcoef(y_test, y_pred_KNN), 3)))
#
# print(KNN_MCC)
#
# ##Visualise how the MCC metric varies with different values of Neighbors:
# plt.plot(k_values, KNN_MCC)
# plt.xlabel("Number of Neighbours")
# plt.ylabel("MCC Performance")
#
# # Get the number of neighbours of the maximum MCC score:
# selected_N = KNN_MCC.index(max(KNN_MCC)) + 1  # earlier returned 3, now 9
#
# # Train KNN with optimum k value
#
# classifier_KNN_new = KNeighborsClassifier(n_neighbors=selected_N)  # (n_neighbors = max(KNN_MCC))
# classifier_KNN_new.fit(x_train, y_train)
#
# # Predict the classifier's responses for the Test dataset
# y_pred_KNN_new = classifier_KNN_new.predict(x_test)
#
# ## Evaluate the Performance of blind test
# blind_cm_KNN = confusion_matrix(y_test, y_pred_KNN_new)  # cm for confusion matrix , len(y_test) - 22206
# print(blind_cm_KNN)
#
# blind_acc_KNN = float(round(balanced_accuracy_score(y_test, y_pred_KNN_new), 3))
# print(blind_acc_KNN)
#
# blind_recall_KNN = float(round(recall_score(y_test, y_pred_KNN_new), 3))  # tp / (tp + fn)
# print(blind_recall_KNN)
#
# blind_precision_KNN = float(round(precision_score(y_test, y_pred_KNN_new), 3))  # tp / (tp + fp)
# print(blind_precision_KNN)
#
# blind_f1_KNN = float(round(f1_score(y_test, y_pred_KNN_new),
#                            3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# # It is primarily used to compare the performance of two classifiers.
# # Suppose that classifier A has a higher recall, and classifier B has higher precision.
# # In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
# print(blind_f1_KNN)
#
# blind__mcc_KNN = float(round(matthews_corrcoef(y_test, y_pred_KNN_new),
#                              3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
# print(blind__mcc_KNN)
#
# blind_AUC_KNN = float(round(roc_auc_score(y_test, (classifier_KNN_new.predict_proba(x_test)[:, 1])), 3))
# print(blind_AUC_KNN)
# # area under the ROC curve, which is the curve having False Positive Rate on the x-axis and True Positive Rate on the y-axis at all classification thresholds.
#
# blind_test_KNN = [blind_acc_KNN, blind_recall_KNN, blind_precision_KNN, blind_f1_KNN, blind__mcc_KNN, blind_AUC_KNN]
# print(blind_test_KNN)
#
# # Number of Folds to split the data:
# # folds = 10 # not stratified
#
# folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # defined earlier under Naive Bayes
#
# # Call the function of cross-validation passing the parameters: # this returned 10 accuracies, and at the next step, we took the mean of this.
# cross_accuracy_all_KNN = cross_val_score(estimator=classifier_KNN_new, X=x_train, y=y_train, cv=folds,
#                                          scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.
#
# cross_precision_all_KNN = cross_val_score(estimator=classifier_KNN_new, X=x_train, y=y_train, cv=folds,
#                                           scoring='precision')
#
# cross_recall_all_KNN = cross_val_score(estimator=classifier_KNN_new, X=x_train, y=y_train, cv=folds, scoring='recall')
#
# cross_f1_all_KNN = cross_val_score(estimator=classifier_KNN_new, X=x_train, y=y_train, cv=folds, scoring='f1')
#
# # no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
# mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
# cross_mcc_all_KNN = cross_val_score(estimator=classifier_KNN_new, X=x_train, y=y_train, cv=folds, scoring=mcc)
#
# cross_AUC_all_KNN = cross_val_score(estimator=classifier_KNN_new, X=x_train, y=y_train, cv=folds, scoring='roc_auc')
#
# cross_validation_KNN = [round((cross_accuracy_all_KNN.mean()), 3), round((cross_recall_all_KNN.mean()), 3),
#                         round((cross_precision_all_KNN.mean()), 3), round((cross_f1_all_KNN.mean()), 3),
#                         round((cross_mcc_all_KNN.mean()), 3), round((cross_AUC_all_KNN.mean()), 3)]
# print(cross_validation_KNN)
#
# # ------------------------------------------------------------------------------------------------------------------------------------
# # ## 15.3 SVM
# #
# # print('SVM')
# #
# # # Training the SVM:
# # classifier_DT = DecisionTreeClassifier(criterion='entropy', random_state=0)
# #
# # classifier_SVM = SVC(kernel = 'linear', probability=True, random_state = 0)
# #
# # # criterion - The function to measure the quality of a split.
# # # Gini index and entropy is the criterion for calculating information gain. Decision tree algorithms use information gain to split a node.
# # # Both gini and entropy are measures of impurity of a node. A node having multiple classes is impure whereas a node having only one class is pure.  Entropy in statistics is analogous to entropy in thermodynamics where it signifies disorder. If there are multiple classes in a node, there is disorder in that node.
# #
# # classifier_SVM.fit(x_train, y_train)
# #
# # # Predict the classifier response for the Test dataset:
# # y_pred_DT = classifier_SVM.predict(x_test)
# #
# # ## Evaluate the Performance of blind test
# # blind_cm_SVM = confusion_matrix(y_test, y_pred_DT)  # cm for confusion matrix , len(y_test) - 22206
# # print(blind_cm_SVM)
# #
# # blind_acc_SVM = float(
# #     round(balanced_accuracy_score(y_test, y_pred_DT), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
# # print(blind_acc_SVM)
# #
# # blind_recall_SVM = float(round(recall_score(y_test, y_pred_DT), 3))  # tp / (tp + fn)
# # print(blind_recall_SVM)
# #
# # blind_precision_SVM = float(round(precision_score(y_test, y_pred_DT), 3))  # tp / (tp + fp)
# # print(blind_precision_SVM)
# #
# # blind_f1_SVM = float(round(f1_score(y_test, y_pred_DT),
# #                           3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# # # It is primarily used to compare the performance of two classifiers.
# # # Suppose that classifier A has a higher recall, and classifier B has higher precision.
# # # In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
# # print(blind_f1_SVM)
# #
# # blind_mcc_SVM = float(round(matthews_corrcoef(y_test, y_pred_DT),
# #                             3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
# # print(blind_mcc_SVM)
# #
# # blind_AUC_SVM = float(round(roc_auc_score(y_test, (classifier_SVM.predict_proba(x_test)[:, 1])), 3))
# # print(blind_AUC_SVM)
# #
# # blind_test_SVM = [blind_acc_SVM, blind_recall_SVM, blind_precision_SVM, blind_f1_SVM, blind_mcc_SVM, blind_AUC_SVM]
# # print(blind_test_SVM)
# #
# # # Number of Folds to split the data:
# # # folds = 10 # not stratified
# #
# # folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes
# #
# # # Call the function of cross-validation passing the parameters:
# # cross_accuracy_all_SVM = cross_val_score(estimator=classifier_SVM, X=x_train, y=y_train, cv=folds,
# #                                         scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.
# #
# # cross_precision_all_SVM = cross_val_score(estimator=classifier_SVM, X=x_train, y=y_train, cv=folds, scoring='precision')
# #
# # cross_recall_all_SVM = cross_val_score(estimator=classifier_SVM, X=x_train, y=y_train, cv=folds, scoring='recall')
# #
# # cross_f1_all_SVM = cross_val_score(estimator=classifier_SVM, X=x_train, y=y_train, cv=folds, scoring='f1')
# #
# # # no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
# # mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
# # cross_mcc_all_SVM = cross_val_score(estimator=classifier_SVM, X=x_train, y=y_train, cv=folds, scoring=mcc)
# #
# # cross_AUC_all_SVM = cross_val_score(estimator=classifier_SVM, X=x_train, y=y_train, cv=folds, scoring='roc_auc')
# #
# # cross_validation_SVM = [round((cross_accuracy_all_SVM.mean()), 3), round((cross_recall_all_SVM.mean()), 3),
# #                        round((cross_precision_all_SVM.mean()), 3), round((cross_f1_all_SVM.mean()), 3),
# #                        round((cross_mcc_all_SVM.mean()), 3), round((cross_AUC_all_SVM.mean()), 3)]
# # print(cross_validation_SVM)
#
# # ------------------------------------------------------------------------------------------------------------------------------------
#
# ## 15.3 Decision trees
#
# print('Decision trees')
#
# # Training the Decision trees:
# classifier_DT = DecisionTreeClassifier(criterion='entropy', random_state=0)
# # criterion - The function to measure the quality of a split.
# # Gini index and entropy is the criterion for calculating information gain. Decision tree algorithms use information gain to split a node.
# # Both gini and entropy are measures of impurity of a node. A node having multiple classes is impure whereas a node having only one class is pure.  Entropy in statistics is analogous to entropy in thermodynamics where it signifies disorder. If there are multiple classes in a node, there is disorder in that node.
#
# classifier_DT.fit(x_train, y_train)
#
# # Predict the classifier response for the Test dataset:
# y_pred_DT = classifier_DT.predict(x_test)
#
# ## Evaluate the Performance of blind test
# blind_cm_DT = confusion_matrix(y_test, y_pred_DT)  # cm for confusion matrix , len(y_test) - 22206
# print(blind_cm_DT)
#
# blind_acc_DT = float(
#     round(balanced_accuracy_score(y_test, y_pred_DT), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
# print(blind_acc_DT)
#
# blind_recall_DT = float(round(recall_score(y_test, y_pred_DT), 3))  # tp / (tp + fn)
# print(blind_recall_DT)
#
# blind_precision_DT = float(round(precision_score(y_test, y_pred_DT), 3))  # tp / (tp + fp)
# print(blind_precision_DT)
#
# blind_f1_DT = float(round(f1_score(y_test, y_pred_DT),
#                           3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# # It is primarily used to compare the performance of two classifiers.
# # Suppose that classifier A has a higher recall, and classifier B has higher precision.
# # In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
# print(blind_f1_DT)
#
# blind__mcc_DT = float(round(matthews_corrcoef(y_test, y_pred_DT),
#                             3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
# print(blind__mcc_DT)
#
# blind_AUC_DT = float(round(roc_auc_score(y_test, (classifier_DT.predict_proba(x_test)[:, 1])), 3))
# print(blind_AUC_DT)
#
# blind_test_DT = [blind_acc_DT, blind_recall_DT, blind_precision_DT, blind_f1_DT, blind__mcc_DT, blind_AUC_DT]
# print(blind_test_DT)
#
# # Number of Folds to split the data:
# # folds = 10 # not stratified
#
# folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes
#
# # Call the function of cross-validation passing the parameters:
# cross_accuracy_all_DT = cross_val_score(estimator=classifier_DT, X=x_train, y=y_train, cv=folds,
#                                         scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.
#
# cross_precision_all_DT = cross_val_score(estimator=classifier_DT, X=x_train, y=y_train, cv=folds, scoring='precision')
#
# cross_recall_all_DT = cross_val_score(estimator=classifier_DT, X=x_train, y=y_train, cv=folds, scoring='recall')
#
# cross_f1_all_DT = cross_val_score(estimator=classifier_DT, X=x_train, y=y_train, cv=folds, scoring='f1')
#
# # no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
# mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
# cross_mcc_all_DT = cross_val_score(estimator=classifier_DT, X=x_train, y=y_train, cv=folds, scoring=mcc)
#
# cross_AUC_all_DT = cross_val_score(estimator=classifier_DT, X=x_train, y=y_train, cv=folds, scoring='roc_auc')
#
# cross_validation_DT = [round((cross_accuracy_all_DT.mean()), 3), round((cross_recall_all_DT.mean()), 3),
#                        round((cross_precision_all_DT.mean()), 3), round((cross_f1_all_DT.mean()), 3),
#                        round((cross_mcc_all_DT.mean()), 3), round((cross_AUC_all_DT.mean()), 3)]
# print(cross_validation_DT)
#
# # ------------------------------------------------------------------------------------------------------------------------------------
#
# ## 15.4 Random forest
#
# print('Random forest')
#
# classifier_RF = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
# # n_estimators - No. of trees in the forest. Try n_estimators = 100 (default value) also to check whether the accuracy is improving.
# # criterion{“gini”, “entropy”}, default=”gini” . This is the function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
# # random_state is for RandomState instance or None, default=None. Controls the randomness of the estimator.
#
# # criterion?
# # A node is 100% impure when a node is split evenly 50/50 and 100% pure when all of its data belongs to a single class.
#
# classifier_RF.fit(x_train, y_train)
#
# # Predict the classifier response for the Test dataset:
# y_pred_RF = classifier_RF.predict(x_test)
#
# ## Evaluate the Performance of blind test
# blind_cm_RF = confusion_matrix(y_test, y_pred_RF)  # cm for confusion matrix , len(y_test) - 22206
# print(blind_cm_RF)
#
# blind_acc_RF = float(
#     round(balanced_accuracy_score(y_test, y_pred_RF), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
# print(blind_acc_RF)
#
# blind_recall_RF = float(round(recall_score(y_test, y_pred_RF), 3))  # tp / (tp + fn)
# print(blind_recall_RF)
#
# blind_precision_RF = float(round(precision_score(y_test, y_pred_RF), 3))  # tp / (tp + fp)
# print(blind_precision_RF)
#
# blind_f1_RF = float(round(f1_score(y_test, y_pred_RF),
#                           3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# # It is primarily used to compare the performance of two classifiers.
# # Suppose that classifier A has a higher recall, and classifier B has higher precision.
# # In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
# print(blind_f1_RF)
#
# blind__mcc_RF = float(round(matthews_corrcoef(y_test, y_pred_RF),
#                             3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
# print(blind__mcc_RF)
#
# blind_AUC_RF = float(round(roc_auc_score(y_test, (classifier_RF.predict_proba(x_test)[:, 1])), 3))
# print(blind_AUC_RF)
#
# blind_test_RF = [blind_acc_RF, blind_recall_RF, blind_precision_RF, blind_f1_RF, blind__mcc_RF, blind_AUC_RF]
# print(blind_test_RF)
#
# # Number of Folds to split the data:
# # folds = 10 # not stratified
#
# folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes
#
# # Call the function of cross-validation passing the parameters:
# cross_accuracy_all_RF = cross_val_score(estimator=classifier_RF, X=x_train, y=y_train, cv=folds,
#                                         scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.
#
# cross_precision_all_RF = cross_val_score(estimator=classifier_RF, X=x_train, y=y_train, cv=folds, scoring='precision')
#
# cross_recall_all_RF = cross_val_score(estimator=classifier_RF, X=x_train, y=y_train, cv=folds, scoring='recall')
#
# cross_f1_all_RF = cross_val_score(estimator=classifier_RF, X=x_train, y=y_train, cv=folds, scoring='f1')
#
# # no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
# mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
# cross_mcc_all_RF = cross_val_score(estimator=classifier_RF, X=x_train, y=y_train, cv=folds, scoring=mcc)
#
# cross_AUC_all_RF = cross_val_score(estimator=classifier_RF, X=x_train, y=y_train, cv=folds, scoring='roc_auc')
#
# cross_validation_RF = [round((cross_accuracy_all_RF.mean()), 3), round((cross_recall_all_RF.mean()), 3),
#                        round((cross_precision_all_RF.mean()), 3), round((cross_f1_all_RF.mean()), 3),
#                        round((cross_mcc_all_RF.mean()), 3), round((cross_AUC_all_RF.mean()), 3)]
# print(cross_validation_RF)
#
# # ------------------------------------------------------------------------------------------------------------------------------------
#
# print("Random forest - with paramater 'class_weight'")
#
# ## 15.5 Random forest - with paramater 'class_weight'
#
# classifier_RF_cw = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0,
#                                           class_weight='balanced')
# # n_estimators - No. of trees in the forest. Try n_estimators = 100 (default value) also to check whether the accuracy is improving.
# # criterion{“gini”, “entropy”}, default=”gini” . This is the function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
# # random_state is for RandomState instance or None, default=None. Controls the randomness of the estimator.
#
# # The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as,
# # n_samples / (n_classes * np.bincount(y))
#
# # Unlike the oversampling and under-sampling methods, the balanced weights methods do not modify the minority and majority class ratio.
# # Instead, it penalizes the wrong predictions on the minority class by giving more weight to the loss function.
#
# classifier_RF_cw.fit(x_train, y_train)
#
# # Predict the classifier response for the Test dataset:
# y_pred_RF_cw = classifier_RF_cw.predict(x_test)
#
# ## Evaluate the Performance of blind test
# blind_cm_RF_cw = confusion_matrix(y_test, y_pred_RF_cw)  # cm for confusion matrix , len(y_test) - 22206
# print(blind_cm_RF)
#
# blind_acc_RF_cw = float(
#     round(balanced_accuracy_score(y_test, y_pred_RF_cw), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
# print(blind_acc_RF)
#
# blind_recall_RF_cw = float(round(recall_score(y_test, y_pred_RF_cw), 3))  # tp / (tp + fn)
# print(blind_recall_RF)
#
# blind_precision_RF_cw = float(round(precision_score(y_test, y_pred_RF_cw), 3))  # tp / (tp + fp)
# print(blind_precision_RF)
#
# blind_f1_RF_cw = float(round(f1_score(y_test, y_pred_RF_cw),
#                              3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# # It is primarily used to compare the performance of two classifiers.
# # Suppose that classifier A has a higher recall, and classifier B has higher precision.
# # In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
# print(blind_f1_RF)
#
# blind__mcc_RF_cw = float(round(matthews_corrcoef(y_test, y_pred_RF_cw),
#                                3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
# print(blind__mcc_RF)
#
# blind_AUC_RF_cw = float(round(roc_auc_score(y_test, (classifier_RF_cw.predict_proba(x_test)[:, 1])), 3))
# print(blind_AUC_RF_cw)
#
# blind_test_RF_cw = [blind_acc_RF_cw, blind_recall_RF_cw, blind_precision_RF_cw, blind_f1_RF_cw, blind__mcc_RF_cw,
#                     blind_AUC_RF_cw]
# print(blind_test_RF_cw)
#
# # Number of Folds to split the data:
# # folds = 10 # not stratified
#
# folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes
#
# # Call the function of cross-validation passing the parameters:
# cross_accuracy_all_RF_cw = cross_val_score(estimator=classifier_RF_cw, X=x_train, y=y_train, cv=folds,
#                                            scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.
#
# cross_precision_all_RF_cw = cross_val_score(estimator=classifier_RF_cw, X=x_train, y=y_train, cv=folds,
#                                             scoring='precision')
#
# cross_recall_all_RF_cw = cross_val_score(estimator=classifier_RF_cw, X=x_train, y=y_train, cv=folds, scoring='recall')
#
# cross_f1_all_RF_cw = cross_val_score(estimator=classifier_RF_cw, X=x_train, y=y_train, cv=folds, scoring='f1')
#
# # no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
# mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
# cross_mcc_all_RF_cw = cross_val_score(estimator=classifier_RF_cw, X=x_train, y=y_train, cv=folds, scoring=mcc)
#
# cross_AUC_all_RF_cw = cross_val_score(estimator=classifier_RF_cw, X=x_train, y=y_train, cv=folds, scoring='roc_auc')
#
# cross_validation_RF_cw = [round((cross_accuracy_all_RF_cw.mean()), 3), round((cross_recall_all_RF_cw.mean()), 3),
#                           round((cross_precision_all_RF_cw.mean()), 3), round((cross_f1_all_RF_cw.mean()), 3),
#                           round((cross_mcc_all_RF_cw.mean()), 3), round((cross_AUC_all_RF_cw.mean()), 3)]
# print(cross_validation_RF_cw)
#
# # ------------------------------------------------------------------------------------------------------------------------------------
# print("AdaBoostClassifier")
#
# ## 15.6 AdaBoostClassifier
#
# # a boosting technique
# # focus on the areas where the system is not perfoming well
#
# # This classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the
# # same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.
#
# # Create adaboost classifer object
# classifier_AdaB = AdaBoostClassifier(n_estimators=50, learning_rate=1)
#
# # base_estimator: It is a weak learner used to train the model. It uses DecisionTreeClassifier as default weak learner for training purpose. You can also specify different machine learning algorithms.
# # n_estimators: Number of weak learners to train iteratively.
# # learning_rate: It contributes to the weights of weak learners. It uses 1 as a default value.
#
# # Train Adaboost Classifer
# classifier_AdaB.fit(x_train, y_train)
#
# # Predict the response for test dataset
# y_pred_AdaB = classifier_AdaB.predict(x_test)
#
# ## Evaluate the Performance of blind test
# blind_cm_AdaB = confusion_matrix(y_test, y_pred_AdaB)  # cm for confusion matrix , len(y_test) - 22206
# print(blind_cm_AdaB)
#
# blind_acc_AdaB = float(
#     round(balanced_accuracy_score(y_test, y_pred_AdaB), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
# print(blind_acc_AdaB)
#
# blind_recall_AdaB = float(round(recall_score(y_test, y_pred_AdaB), 3))  # tp / (tp + fn)
# print(blind_recall_AdaB)
#
# blind_precision_AdaB = float(round(precision_score(y_test, y_pred_AdaB), 3))  # tp / (tp + fp)
# print(blind_precision_AdaB)
#
# blind_f1_AdaB = float(round(f1_score(y_test, y_pred_AdaB),
#                             3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# # It is primarily used to compare the performance of two classifiers.
# # Suppose that classifier A has a higher recall, and classifier B has higher precision.
# # In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
# print(blind_f1_AdaB)
#
# blind__mcc_AdaB = float(round(matthews_corrcoef(y_test, y_pred_AdaB),
#                               3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
# print(blind__mcc_AdaB)
#
# blind_AUC_AdaB = float(round(roc_auc_score(y_test, (classifier_AdaB.predict_proba(x_test)[:, 1])), 3))
# print(blind_AUC_AdaB)
#
# blind_test_AdaB = [blind_acc_AdaB, blind_recall_AdaB, blind_precision_AdaB, blind_f1_AdaB, blind__mcc_AdaB,
#                    blind_AUC_AdaB]
# print(blind_test_AdaB)
#
# # Number of Folds to split the data:
# # folds = 10 # not stratified
#
# folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes
#
# # Call the function of cross-validation passing the parameters:
# cross_accuracy_all_AdaB = cross_val_score(estimator=classifier_AdaB, X=x_train, y=y_train, cv=folds,
#                                           scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.
#
# cross_precision_all_AdaB = cross_val_score(estimator=classifier_AdaB, X=x_train, y=y_train, cv=folds,
#                                            scoring='precision')
#
# cross_recall_all_AdaB = cross_val_score(estimator=classifier_AdaB, X=x_train, y=y_train, cv=folds, scoring='recall')
#
# cross_f1_all_AdaB = cross_val_score(estimator=classifier_AdaB, X=x_train, y=y_train, cv=folds, scoring='f1')
#
# # no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
# mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
# cross_mcc_all_AdaB = cross_val_score(estimator=classifier_AdaB, X=x_train, y=y_train, cv=folds, scoring=mcc)
#
# cross_AUC_all_AdaB = cross_val_score(estimator=classifier_AdaB, X=x_train, y=y_train, cv=folds, scoring='roc_auc')
#
# cross_validation_AdaB = [round((cross_accuracy_all_AdaB.mean()), 3), round((cross_recall_all_AdaB.mean()), 3),
#                          round((cross_precision_all_AdaB.mean()), 3), round((cross_f1_all_AdaB.mean()), 3),
#                          round((cross_mcc_all_AdaB.mean()), 3), round((cross_AUC_all_AdaB.mean()), 3)]
# print(cross_validation_AdaB)
#
# # ------------------------------------------------------------------------------------------------------------------------------------
#
# ## 15.7 XGBoost (Extreme Gradient Boosting)
#
# print("XGBoost")
#
# # Create XGBoost classifer object
# # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
#
# # default parameter values - https://stackoverflow.com/questions/34674797/xgboost-xgbclassifier-defaults-in-python
# # default - max_depth=3 , learning_rate=0.1 , n_estimators=100 , objective='binary:logistic'
#
# classifier_XGB = xgb.XGBClassifier(objective="binary:logistic", max_depth=3, learning_rate=0.1, n_estimators=100,
#                                    random_state=0)  # random_state = 42
#
# # Train Adaboost Classifer
# classifier_XGB.fit(x_train, y_train)
#
# # Predict the response for test dataset
# y_pred_XGB = classifier_XGB.predict(x_test)
#
# ## Evaluate the Performance of blind test
# blind_cm_XGB = confusion_matrix(y_test, y_pred_XGB)  # cm for confusion matrix , len(y_test) - 22206
# print(blind_cm_XGB)
#
# blind_acc_XGB = float(
#     round(balanced_accuracy_score(y_test, y_pred_XGB), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
# print(blind_acc_XGB)
#
# blind_recall_XGB = float(round(recall_score(y_test, y_pred_XGB), 3))  # tp / (tp + fn)
# print(blind_recall_XGB)
#
# blind_precision_XGB = float(round(precision_score(y_test, y_pred_XGB), 3))  # tp / (tp + fp)
# print(blind_precision_XGB)
#
# blind_f1_XGB = float(round(f1_score(y_test, y_pred_XGB),
#                            3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# # It is primarily used to compare the performance of two classifiers.
# # Suppose that classifier A has a higher recall, and classifier B has higher precision.
# # In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
# print(blind_f1_XGB)
#
# blind__mcc_XGB = float(round(matthews_corrcoef(y_test, y_pred_XGB),
#                              3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
# print(blind__mcc_XGB)
#
# blind_AUC_XGB = float(round(roc_auc_score(y_test, (classifier_XGB.predict_proba(x_test)[:, 1])), 3))
# print(blind_AUC_XGB)
#
# blind_test_XGB = [blind_acc_XGB, blind_recall_XGB, blind_precision_XGB, blind_f1_XGB, blind__mcc_XGB, blind_AUC_XGB]
# print(blind_test_XGB)
#
# # Number of Folds to split the data:
# # folds = 10 # not stratified
#
# folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes
#
# # Call the function of cross-validation passing the parameters:
# cross_accuracy_all_XGB = cross_val_score(estimator=classifier_XGB, X=x_train, y=y_train, cv=folds,
#                                          scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.
#
# cross_precision_all_XGB = cross_val_score(estimator=classifier_XGB, X=x_train, y=y_train, cv=folds, scoring='precision')
#
# cross_recall_all_XGB = cross_val_score(estimator=classifier_XGB, X=x_train, y=y_train, cv=folds, scoring='recall')
#
# cross_f1_all_XGB = cross_val_score(estimator=classifier_XGB, X=x_train, y=y_train, cv=folds, scoring='f1')
#
# # no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
# mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
# cross_mcc_all_XGB = cross_val_score(estimator=classifier_XGB, X=x_train, y=y_train, cv=folds, scoring=mcc)
#
# cross_AUC_all_XGB = cross_val_score(estimator=classifier_XGB, X=x_train, y=y_train, cv=folds, scoring='roc_auc')
#
# cross_validation_XGB = [round((cross_accuracy_all_XGB.mean()), 3), round((cross_recall_all_XGB.mean()), 3),
#                         round((cross_precision_all_XGB.mean()), 3), round((cross_f1_all_XGB.mean()), 3),
#                         round((cross_mcc_all_XGB.mean()), 3), round((cross_AUC_all_XGB.mean()), 3)]
# print(cross_validation_XGB)
#
# # ------------------------------------------------------------------------------------------------------------------------------------

# # 15.8 LGBMClassifier (Extreme Gradient Boosting)

print("LGBMClassifier")

# Create LGBM classifier object

classifier_LGBM = LGBMClassifier(random_state=0)  # random_state=42

# model = LGBMClassifier(colsample_bytree=0.61, min_child_samples=321, min_child_weight=0.01, n_estimators=100, num_leaves=45, reg_alpha=0.1, reg_lambda=1, subsample=0.56)

# 2 ways to import libraries when create training object
# import lightgbm
# clf = lightgbm.LGBMClassifier()

# from lightgbm import LGBMClassifier
# classifier_LGBM = LGBMClassifier()

# Train Adaboost Classifier
classifier_LGBM.fit(x_train, y_train)

# Predict the response for test dataset
y_pred_LGBM = classifier_LGBM.predict(x_test)

######### start new

# Predict the response for train dataset
y_pred_LGBM_train = classifier_LGBM.predict(x_train)

# Create a DataFrame with labels - train data set
label_train_actual_and_predicted = pd.DataFrame({
    'hadm_id': hadm_id_train,
    'HIT_label_actual': y_train,
    'HIT_label_predicted': y_pred_LGBM_train,
})

# Create a DataFrame with labels - train data set
label_test_actual_and_predicted = pd.DataFrame({
    'hadm_id': hadm_id_test,
    'HIT_label_actual': y_test,
    'HIT_label_predicted': y_pred_LGBM,
})

output_result_dir = '/Users/psenevirathn/Desktop/PhD/Coding/Python/output_csv_files'

save_train_data = os.path.join(output_result_dir,
                               'label_train_actual_and_predicted_with_updated_ground_truth_New_TP_to_TN_as_baseline.csv')  # This Returns a path. os.path.join - https://www.geeksforgeeks.org/python-os-path-join-method/

label_train_actual_and_predicted.to_csv(save_train_data, float_format='%.0f')

save_test_data = os.path.join(output_result_dir,
                              'label_test_actual_and_predicted_with_updated_ground_truth_New_TP_to_TN_as_baseline.csv')  # This Returns a path. os.path.join - https://www.geeksforgeeks.org/python-os-path-join-method/

label_test_actual_and_predicted.to_csv(save_test_data, float_format='%.0f')
print(label_test_actual_and_predicted)
######### close new

## Evaluate the Performance of blind test
blind_cm_LGBM = confusion_matrix(y_test, y_pred_LGBM)  # cm for confusion matrix , len(y_test) - 22206
print(blind_cm_LGBM)

#[[2226   67]
# [ 299   91]]


blind_acc_LGBM = float(
    round(balanced_accuracy_score(y_test, y_pred_LGBM), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
print(blind_acc_LGBM)

blind_recall_LGBM = float(round(recall_score(y_test, y_pred_LGBM), 3))  # tp / (tp + fn)
print(blind_recall_LGBM)

blind_precision_LGBM = float(round(precision_score(y_test, y_pred_LGBM), 3))  # tp / (tp + fp)
print(blind_precision_LGBM)

blind_f1_LGBM = float(round(f1_score(y_test, y_pred_LGBM),
                            3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# It is primarily used to compare the performance of two classifiers.
# Suppose that classifier A has a higher recall, and classifier B has higher precision.
# In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
print(blind_f1_LGBM)

blind__mcc_LGBM = float(round(matthews_corrcoef(y_test, y_pred_LGBM),
                              3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
print(blind__mcc_LGBM)

blind_AUC_LGBM = float(round(roc_auc_score(y_test, (classifier_LGBM.predict_proba(x_test)[:, 1])), 3))
print(blind_AUC_LGBM)

blind_test_LGBM = [blind_acc_LGBM, blind_recall_LGBM, blind_precision_LGBM, blind_f1_LGBM, blind__mcc_LGBM,
                   blind_AUC_LGBM]
print(blind_test_LGBM) # [0.602, 0.233, 0.576, 0.332, 0.306, 0.787]

# Number of Folds to split the data:
# folds = 10 # not stratified

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes

# Call the function of cross-validation passing the parameters:
cross_accuracy_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds,
                                          scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# for each metric (for example, let'a consider Precisoin (PPV), we can calculate the metric in 2 ways. In both ways, it return the metric value for each fold, as in a list)

# METHOD 1 - Using scoring='precision'. This approach uses a predefined scoring metric directly from scikit-learn. It returns an array of precision scores for each fold. The precision is calculated based on the predictions made by the model on the validation set for each fold.
# This is straightforward and generally works well for most cases. However, it assumes that the positive class is the one labeled as 1

cross_precision_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds,
                                           scoring='precision')

# METHOD 2 - Using make_scorer(precision_score).This approach explicitly uses the precision_score function wrapped with make_scorer. Similar to the first method, it returns an array of precision scores for each fold.
# Using make_scorer allows you to specify additional parameters if needed (e.g., average method for multi-class classification), or to customize the scoring function in any way. This makes it a more versatile option if your precision calculation needs to accommodate specific scenarios.

# for eaxmple, we can define which class should be positive.
# For calculate PPV (precision)
# cross_precision_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds, scoring=make_scorer(precision_score)))

# NPV
#-------

# or else, You can use make_scorer to pass in pos_label=0 to the precision score function (metrics.precision_score) to get NPV. Like this:
# npv = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds, scoring=make_scorer(precision_score, pos_label=0))
# print(npv1)
# This returned NPV for each folder - # [0.89067202 0.87860697 0.89101917 0.87573964 0.89249493 0.89180991 0.88473205 0.88188188 0.88358209 0.89013225]

# Or else, can calulate NPV directly as below, which returened a single value (considering all 10 predictions from 10 folds)

# Get predictions to calculate NPV
y_pred = cross_val_predict(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds)  # cross_val_predict - In 10-fold cross-validation using cross_val_predict, the function does indeed return a single array of predictions for the entire dataset, but it does this by aggregating predictions made during each of the folds

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()

# Calculate NPV
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(y_pred)  # [0 0 0 ... 0 0 0]
print(y_pred.shape)  # (10732,)
print(confusion_matrix(y_train, y_pred).ravel()) # [8929  280 1146  377]
print(tp)  # 377
print(tn)  # 8929

print(fp)  # 280
print(fn)  # 1146
print(npv)  # 0.8862531017369727

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

cross_recall_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds, scoring='recall')

cross_f1_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds, scoring='f1')

# no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
cross_mcc_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds, scoring=mcc)

cross_AUC_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds, scoring='roc_auc')

cross_validation_LGBM = [round((cross_accuracy_all_LGBM.mean()), 3), round((cross_recall_all_LGBM.mean()), 3),
                         round((cross_precision_all_LGBM.mean()), 3), round((cross_f1_all_LGBM.mean()), 3),
                         round((cross_mcc_all_LGBM.mean()), 3), round((cross_AUC_all_LGBM.mean()), 3)]

print(cross_validation_LGBM) # [0.609, 0.247, 0.575, 0.345, 0.316, 0.805]

## ------------------------------------------------------------------------------------------------------------------------------------

# # # 15.9 GradientBoost Classifier
#
# print("GB")
#
# # Create GradientBoost classifier object
#
# classifier_GB = GradientBoostingClassifier(n_estimators=300, random_state=0)  # random_state=1
#
# # model = LGBMClassifier(colsample_bytree=0.61, min_child_samples=321, min_child_weight=0.01, n_estimators=100, num_leaves=45, reg_alpha=0.1, reg_lambda=1, subsample=0.56)
#
# # 2 ways to import libraries when create training object
# # import lightgbm
# # clf = lightgbm.LGBMClassifier()
#
# # from lightgbm import LGBMClassifier
# # classifier_LGBM = LGBMClassifier()
#
# # Train Adaboost Classifier
# classifier_GB.fit(x_train, y_train)
#
# # Predict the response for test dataset
# y_pred_GB = classifier_GB.predict(x_test)
#
# ## Evaluate the Performance of blind test
# blind_cm_GB = confusion_matrix(y_test, y_pred_GB)  # cm for confusion matrix , len(y_test) - 22206
# print(blind_cm_GB)
#
# blind_acc_GB = float(
#     round(balanced_accuracy_score(y_test, y_pred_GB), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
# print(blind_acc_GB)
#
# blind_recall_GB = float(round(recall_score(y_test, y_pred_GB), 3))  # tp / (tp + fn)
# print(blind_recall_GB)
#
# blind_precision_GB = float(round(precision_score(y_test, y_pred_GB), 3))  # tp / (tp + fp)
# print(blind_precision_GB)
#
# blind_f1_GB = float(round(f1_score(y_test, y_pred_GB),
#                           3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# # It is primarily used to compare the performance of two classifiers.
# # Suppose that classifier A has a higher recall, and classifier B has higher precision.
# # In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
# print(blind_f1_GB)
#
# blind__mcc_GB = float(round(matthews_corrcoef(y_test, y_pred_GB),
#                             3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
# print(blind__mcc_GB)
#
# blind_AUC_GB = float(round(roc_auc_score(y_test, (classifier_GB.predict_proba(x_test)[:, 1])), 3))
# print(blind_AUC_GB)
#
# blind_test_GB = [blind_acc_GB, blind_recall_GB, blind_precision_GB, blind_f1_GB, blind__mcc_GB,
#                  blind_AUC_GB]
# print(blind_test_GB)
#
# # Number of Folds to split the data:
# # folds = 10 # not stratified
#
# folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes
#
# # Call the function of cross-validation passing the parameters:
# cross_accuracy_all_GB = cross_val_score(estimator=classifier_GB, X=x_train, y=y_train, cv=folds,
#                                         scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.
#
# cross_precision_all_GB = cross_val_score(estimator=classifier_GB, X=x_train, y=y_train, cv=folds,
#                                          scoring='precision')
#
# cross_recall_all_GB = cross_val_score(estimator=classifier_GB, X=x_train, y=y_train, cv=folds, scoring='recall')
#
# cross_f1_all_GB = cross_val_score(estimator=classifier_GB, X=x_train, y=y_train, cv=folds, scoring='f1')
#
# # no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
# mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
# cross_mcc_all_GB = cross_val_score(estimator=classifier_GB, X=x_train, y=y_train, cv=folds, scoring=mcc)
#
# cross_AUC_all_GB = cross_val_score(estimator=classifier_GB, X=x_train, y=y_train, cv=folds, scoring='roc_auc')
#
# cross_validation_GB = [round((cross_accuracy_all_GB.mean()), 3), round((cross_recall_all_GB.mean()), 3),
#                        round((cross_precision_all_GB.mean()), 3), round((cross_f1_all_GB.mean()), 3),
#                        round((cross_mcc_all_GB.mean()), 3), round((cross_AUC_all_GB.mean()), 3)]
# print(cross_validation_GB)
#
# # ------------------------------------------------------------------------------------------------------------------------------------

# 18. Compare results of different ML models

# comparison_ML_models = pd.DataFrame({
#     'BT_NB': blind_test_NB,
#     'CV_NB': cross_validation_NB,
#     'BT_KNN': blind_test_KNN,
#     'CV_KNN': cross_validation_KNN,
#     'BT_DT': blind_test_DT,
#     'CV_DT': cross_validation_DT,
#     'BT_RF': blind_test_RF,
#     'CV_RF': cross_validation_RF,
#     'BT_RF(weighted)': blind_test_RF_cw,
#     'CV_RF(weighted)': cross_validation_RF_cw,
#     'BT_AdaB': blind_test_AdaB,
#     'CV_AdaB': cross_validation_AdaB,
#     'BT_XGB': blind_test_XGB,
#     'CV_XGB': cross_validation_XGB,
#     'BT_LGBM': blind_test_LGBM,
#     'CV_LGBM': cross_validation_LGBM,
#     'BT_GB': blind_test_GB,
#     'CV_GB': cross_validation_GB
# },
#     index=['balanced_accuracy', 'recall', 'precision', 'f1', 'MCC', 'AUC'])
#
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
#
# print(comparison_ML_models)

#------------------------------------------------------------------------------------------------------------
# python /Users/psenevirathn/PycharmProjects/myproject2/current_codes/10_retrain_classifer_with_updated_labels.py /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/train_data_before_preprocessing.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/test_data_before_preprocessing.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/After_clustering/updated_labels.csv
# python /Users/psenevirathn/PycharmProjects/myproject2/current_codes/10_retrain_classifer_with_updated_labels.py /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/train_data_before_preprocessing.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/test_data_before_preprocessing.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/After_clustering/updated_labels_random_TP_to_TN_as_baseline.csv
# python /Users/psenevirathn/PycharmProjects/myproject2/current_codes/10_retrain_classifer_with_updated_labels.py /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/train_data_before_preprocessing.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/test_data_before_preprocessing.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/After_clustering/updated_labels_new_TP_to_TN_ratio_as_baseline.csv


# python /Users/psenevirathn/PycharmProjects/myproject2/current_codes/10_retrain_classifer_with_updated_labels.py /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/train_data_before_preprocessing.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/test_data_before_preprocessing.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/After_clustering/updated_labels_new_TP_to_TN_ratio_as_baseline_Aug_20.csv
