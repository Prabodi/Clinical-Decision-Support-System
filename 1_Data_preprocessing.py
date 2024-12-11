# all heparin. platelet count, lab tests, vital signs - both icu and non icu (hospital) records should be included.
# If the patient doesn't have any icu record, exclude thise patients.

# Two condotions to evaluate ground truth of HIT - https://www.uptodate.com/contents/clinical-presentation-and-diagnosis-of-heparin-induced-thrombocytopenia?search=heparin%20induced%20thrombocytopenia&source=search_result&selectedTitle=1%7E150&usage_type=default&display_rank=1

#1. New onset of thrombocytopenia (ie, platelet count <150,000/microL)
#2. A decrease in platelet count by 50 percent or more, even if the platelet count exceeds 150,000/microL

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
from sklearn.preprocessing import LabelEncoder
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE  # Recursive feature elimination
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# ------------------------------------------------------------------------------------------------------------------------------------

# pd.set_option('display.max_columns', 10)
# pd.set_option('display.max_rows', 50)  # None

# 2. Import files

df_1_hep_admin_demographics = pd.read_csv(
    sys.argv[1])  # first_hep details, hep_type (T/P), treatment_type (UFH/LMWH), dermographics

df_2_platelets = pd.read_csv(sys.argv[2])  # Platelet_count_full_list

df3_vitalsigns = pd.read_csv(sys.argv[3])  # vitalsigns - lab, bg, bg_art, vitalsigns, GCS

# Rounding-off values of selected columns (which have a lot of decimal points) into 2 decimal points

df3_vitalsigns = df3_vitalsigns.round(
    {'heart_rate_mean': 2, 'sbp_mean': 2, 'dbp_mean': 2, 'mbp_mean': 2, 'resp_rate_mean': 2, 'temperature_mean': 2,
     'spo2_mean': 2, 'glucose_mean': 2})
df3_vitalsigns = df3_vitalsigns.round(
    {'heart_rate_mean': 2, 'sbp_mean': 2, 'dbp_mean': 2, 'mbp_mean': 2, 'resp_rate_mean': 2, 'temperature_mean': 2,
     'spo2_mean': 2, 'glucose_mean': 2})

# ------------------------------------------------------------------------------------------------------------------------------------

full_cohort_1 = df_1_hep_admin_demographics

## ------------------------------------------------------------------------------------------------------------------------------------

# 4. Platelet_record details - Set Baseline

# df_2_platelets - We considered all platelete count records - either in icu or non-icu setting

# Task 1 - Time difference from each platelet count record to first hep_admin time of that patient

# Step 1 - Join p_count records with first_hep_admin_time of each patinet

merge_platelets_first_hep = pd.merge(full_cohort_1, df_2_platelets, on='hadm_id', how='left')

# Step 3 - # calculating time (in hrs) from each p_charttime to first_hep_time of each admission

# merge_platelets_first_hep['diff']= merge_platelets_first_hep['hep_start'] - merge_platelets_first_hep['p_charttime'] # if positive, p_record is before first_hep_admin

merge_platelets_first_hep['p_charttime'] = pd.to_datetime(merge_platelets_first_hep['p_charttime'], dayfirst=True)
merge_platelets_first_hep['hep_start'] = pd.to_datetime(merge_platelets_first_hep['hep_start'], dayfirst=True)

merge_platelets_first_hep['diff'] = merge_platelets_first_hep['p_charttime'] - merge_platelets_first_hep[
    'hep_start']  # if positive, p_record is after first_hep_admin

# merge_platelets_first_hep['diff_hrs'] = round(merge_platelets_first_hep['diff'] / np.timedelta64(1, 'h'))

merge_platelets_first_hep['diff_hrs'] = (merge_platelets_first_hep['diff'] / np.timedelta64(1, 'h'))

# Step 4 - set baselines
# If have platelet_count details before first hep (i.e, diff_hrs > 0), use the p_count closest and prior to first_hep
# For it, need to get all the P_count records, before first hep_admin (same hadm_id may have multiple records)

# All_platelets_before_first_hep = merge_platelets_first_hep[merge_platelets_first_hep['diff_hrs']>=0]

All_platelets_before_first_hep = merge_platelets_first_hep[merge_platelets_first_hep['diff_hrs'] <= 0]

# if have platelet_count details before first hep (i.e, diff_hrs > 0), use the p_count closest and prior to first_hep (baseline1)

baseline1 = All_platelets_before_first_hep.loc[
    All_platelets_before_first_hep.groupby('hadm_id').diff_hrs.idxmax()].reset_index(drop=True)

# Rename columns of baseline_platelet details for better understaning

baseline1.rename(columns={'p_charttime': 'base_p_charttime1', 'platelet': 'base_platelets1'}, inplace=True)

# Join baseline_p info with hadm_ids who had at least one platelet record before first hep admin

merge_platelets_first_hep_baseline1 = pd.merge(merge_platelets_first_hep,
                                               baseline1[['hadm_id', 'base_p_charttime1', 'base_platelets1']],
                                               on='hadm_id', how='left')

# Take the patients who don't have baseline1, means the patients who don't have any P_count before first_hep

thresh = 24
All_platelets_after_first_hep = merge_platelets_first_hep_baseline1[
    (merge_platelets_first_hep_baseline1['base_p_charttime1'].isnull()) & (merge_platelets_first_hep_baseline1[
                                                                               'diff_hrs'] < thresh)]  # .merge_platelets_first_hep_baseline1.loc[merge_platelets_first_hep_baseline1.groupby('hadm_id').diff_hrs.idxmin()].reset_index(drop=True)

baseline2 = All_platelets_after_first_hep.loc[
    All_platelets_after_first_hep.groupby('hadm_id').diff_hrs.idxmin()].reset_index(drop=True)

baseline2.rename(columns={'p_charttime': 'base_p_charttime2', 'platelet': 'base_platelets2'}, inplace=True)

merge_platelets_first_hep_baseline2 = pd.merge(merge_platelets_first_hep_baseline1,
                                               baseline2[['hadm_id', 'base_p_charttime2', 'base_platelets2']],
                                               on='hadm_id', how='left')

print('count1')
print(len(merge_platelets_first_hep_baseline2[merge_platelets_first_hep_baseline2['base_p_charttime1'].notnull()][
              'hadm_id'].unique()))  # 10792
print(len(merge_platelets_first_hep_baseline2[merge_platelets_first_hep_baseline2['base_p_charttime2'].notnull()][
              'hadm_id'].unique()))  # 2548
print(len(merge_platelets_first_hep_baseline2[(merge_platelets_first_hep_baseline2['base_p_charttime1'].isnull()) & (
    merge_platelets_first_hep_baseline2['base_p_charttime2'].isnull())]['hadm_id'].unique()))  # 76

merge_platelets_first_hep_baseline2['base_p_charttime'] = np.where(
    (merge_platelets_first_hep_baseline2['base_p_charttime1'].notnull()),
    merge_platelets_first_hep_baseline2['base_p_charttime1'], merge_platelets_first_hep_baseline2['base_p_charttime2'])

merge_platelets_first_hep_baseline2['base_platelets'] = np.where(
    (merge_platelets_first_hep_baseline2['base_platelets1'].notnull()),
    merge_platelets_first_hep_baseline2['base_platelets1'], merge_platelets_first_hep_baseline2['base_platelets2'])

# Select only the required fields

merge_platelets_first_hep_baseline_full = merge_platelets_first_hep_baseline2[
    ['hadm_id', 'p_charttime', 'platelet', 'hep_start', 'diff', 'diff_hrs', 'base_p_charttime', 'base_platelets']]

print(merge_platelets_first_hep_baseline_full.shape)  # (179252, 8)

# merge baseline platelets details wth the main table

full_cohort_3 = pd.merge(full_cohort_1, merge_platelets_first_hep_baseline_full[
    ['hadm_id', 'base_p_charttime', 'base_platelets']].drop_duplicates(), on='hadm_id', how='left')

print('count2')

full_cohort_3['hep_start'] = pd.to_datetime(full_cohort_3['hep_start'], dayfirst=True)
full_cohort_3['base_p_charttime'] = pd.to_datetime(full_cohort_3['base_p_charttime'], dayfirst=True)

print(full_cohort_3.shape)  # (13416, 23)
print(len(full_cohort_3[full_cohort_3['hep_start'] >= full_cohort_3['base_p_charttime']]))  # 10792
print(len(full_cohort_3[full_cohort_3['hep_start'] < full_cohort_3['base_p_charttime']]))  # 2548
print(len(full_cohort_3[full_cohort_3['base_p_charttime'].isnull()]))  # 76

# ------------------------------------------------------------------------------------------------------------------------------------


# 5. Check First Criteria for thrombo - P_count drop below 150k after admin of Heparin

merge_platelets_first_hep_baseline_full['HIT_c1_150k'] = np.where(((merge_platelets_first_hep_baseline_full[
                                                                        'p_charttime'] >
                                                                    merge_platelets_first_hep_baseline_full[
                                                                        'hep_start']) & (
                                                                           merge_platelets_first_hep_baseline_full[
                                                                               'platelet'] < 150)), 1, 0)  # previous <= 150

# Take all p_records where p_count < 150. One hadm_id may have multiple records

only_150k = merge_platelets_first_hep_baseline_full[merge_platelets_first_hep_baseline_full['HIT_c1_150k'] == 1]
only_150k

# check the first time p<=150k happened for each hadm_id, after hep admin. Here, one hadm_id only has one record

only_150k['p_charttime'] = pd.to_datetime(only_150k['p_charttime'], dayfirst=True)

first_hit_P_150k_record = only_150k.loc[only_150k.groupby('hadm_id').p_charttime.idxmin()].reset_index(
    drop=True)
first_hit_P_150k_record.rename(columns={'p_charttime': 'c1_150k_charttime', 'platelet': 'c1_150k_platelets'},
                               inplace=True)

full_cohort_4 = pd.merge(full_cohort_3,
                         first_hit_P_150k_record[['hadm_id', 'c1_150k_charttime', 'c1_150k_platelets', 'HIT_c1_150k']],
                         on='hadm_id', how='left')

# ------------------------------------------------------------------------------------------------------------------------------------

# 6. Check Second Criteria for thrombo - P_count drop >= 50%, compared to baseline

# check condition 2 for HIT

merge_platelets_first_hep_baseline_full['HIT_c2_50%_drop'] = np.where(((merge_platelets_first_hep_baseline_full[
                                                                            'p_charttime'] >
                                                                        merge_platelets_first_hep_baseline_full[
                                                                            'hep_start']) & ((
                                                                                                     merge_platelets_first_hep_baseline_full[
                                                                                                         'platelet'] /
                                                                                                     merge_platelets_first_hep_baseline_full[
                                                                                                         'base_platelets']) <= 0.5)),
                                                                      1, 0)

only_50_per_drop = merge_platelets_first_hep_baseline_full[
    merge_platelets_first_hep_baseline_full['HIT_c2_50%_drop'] == 1]
only_50_per_drop

# check the first time p_drop >= 50% (compared to baseline) happened for each hadm_id, after hep admin. Here, one hadm_id only has one record

only_50_per_drop['p_charttime'] = pd.to_datetime(only_50_per_drop['p_charttime'], dayfirst=True)

first_hit_P_50_per_drop_record = only_50_per_drop.loc[
    only_50_per_drop.groupby('hadm_id').p_charttime.idxmin()].reset_index(drop=True)  # (1122, 10)

first_hit_P_50_per_drop_record.rename(columns={'p_charttime': 'c2_50per_charttime', 'platelet': 'c2_50per_platelets'},
                                      inplace=True)

full_cohort_5 = pd.merge(full_cohort_4, first_hit_P_50_per_drop_record[
    ['hadm_id', 'c2_50per_charttime', 'c2_50per_platelets', 'HIT_c2_50%_drop']], on='hadm_id', how='left')

full_cohort_5['HIT_both_c1&c2'] = np.where(
    ((full_cohort_5['HIT_c1_150k'] == 1) | (full_cohort_5['HIT_c2_50%_drop'] == 1)), 1, 0)

# c1&c2_first_p_charttime
full_cohort_5['c1&c2_first_p_charttime'] = (
    full_cohort_5[['c1_150k_charttime', 'c2_50per_charttime']].min(axis=1)).where(
    (full_cohort_5['HIT_both_c1&c2'] == 1), pd.NaT)
full_cohort_5['first_hep_to_first_HIT'] = pd.to_datetime(full_cohort_5['c1&c2_first_p_charttime'],
                                                         dayfirst=True) - pd.to_datetime(
    full_cohort_5['hep_start'], dayfirst=True)

# Convert days to hours
full_cohort_5['first_hep_to_first_HIT_hrs'] = round(full_cohort_5['first_hep_to_first_HIT'] / np.timedelta64(1, 'h'))

# ------------------------------------------------------------------------------------------------------------------------------------

# 7. Check HIT from 5 days to 10 days

# Get all the P_counts of each hadm_id within 3 - 10 days from first_hep_admin
# Need to change code blockes #8 & #9
lower = 120  # in hrs, so 5 days # old - 72
upper = 240  # in hrs, so 10 days

merge_platelets_first_hep_baseline_full_copy1 = merge_platelets_first_hep_baseline_full

# Evaluate criteria 1 for HIT (p<150), Considering P_record between 3- 10 days, from first hep

merge_platelets_first_hep_baseline_full_copy1['HIT_c1_150k_5_10_days'] = np.where(((
                                                                                           merge_platelets_first_hep_baseline_full_copy1[
                                                                                               'p_charttime'] >
                                                                                           merge_platelets_first_hep_baseline_full_copy1[
                                                                                               'hep_start']) &
                                                                                   (
                                                                                           merge_platelets_first_hep_baseline_full_copy1[
                                                                                               'diff_hrs'] >= lower) & (
                                                                                           merge_platelets_first_hep_baseline_full_copy1[
                                                                                               'diff_hrs'] <= upper) &
                                                                                   (
                                                                                           merge_platelets_first_hep_baseline_full_copy1[
                                                                                               'platelet'] < 150)), # previous <= 150
                                                                                  1, 0)

# diff = p_charttime - first_hep

only_150k_5_10_days = merge_platelets_first_hep_baseline_full_copy1[
    merge_platelets_first_hep_baseline_full_copy1['HIT_c1_150k_5_10_days'] == 1]

only_150k_5_10_days['p_charttime'] = pd.to_datetime(only_150k_5_10_days['p_charttime'], dayfirst=True)

first_hit_P_150k_record5_10_days = only_150k_5_10_days.loc[
    only_150k_5_10_days.groupby('hadm_id').p_charttime.idxmin()].reset_index(drop=True)
first_hit_P_150k_record5_10_days.rename(
    columns={'p_charttime': 'c1_150k_charttime_5_10_days', 'platelet': 'c1_150k_platelets_5_10_days'}, inplace=True)

full_cohort_6 = pd.merge(full_cohort_5, first_hit_P_150k_record5_10_days[
    ['hadm_id', 'c1_150k_charttime_5_10_days', 'c1_150k_platelets_5_10_days', 'HIT_c1_150k_5_10_days']], on='hadm_id',
                         how='left')

# Evaluate criteria 2 for HIT (p_derop>=50%), Considering P_record between 3- 10 days, from first hep


# print(full_cohort_7[(full_cohort_7['c1_150k_charttime_3_10_days'] >= full_cohort_7['c2_50per_charttime_3_10_days']) & (full_cohort_7['base_p_charttime'] >= full_cohort_7['c2_50per_charttime_3_10_days'])][['hadm_id', 'hep_start','base_p_charttime', 'base_platelets', 'c1_150k_charttime_3_10_days', 'c1_150k_platelets_3_10_days', 'HIT_c1_150k_3_10_days', 'c2_50per_charttime_3_10_days', 'c2_50per_platelets_3_10_days', 'HIT_c2_50%_drop_3_10_days', 'HIT_both_c1&c2_3_10_days', 'c1&c2_first_p_time_3_10_days']])


merge_platelets_first_hep_baseline_full_copy1['HIT_c2_50per_drop_5_10_days'] = np.where(((
                                                                                                 merge_platelets_first_hep_baseline_full_copy1[
                                                                                                     'p_charttime'] >
                                                                                                 merge_platelets_first_hep_baseline_full_copy1[
                                                                                                     'hep_start']) &
                                                                                         (
                                                                                                 merge_platelets_first_hep_baseline_full_copy1[
                                                                                                     'p_charttime'] >
                                                                                                 merge_platelets_first_hep_baseline_full_copy1[
                                                                                                     'base_p_charttime']) &
                                                                                         (
                                                                                                 merge_platelets_first_hep_baseline_full_copy1[
                                                                                                     'diff_hrs'] >= lower) &
                                                                                         (
                                                                                                 merge_platelets_first_hep_baseline_full_copy1[
                                                                                                     'diff_hrs'] <= upper) &
                                                                                         ((
                                                                                                  merge_platelets_first_hep_baseline_full_copy1[
                                                                                                      'platelet'] /
                                                                                                  merge_platelets_first_hep_baseline_full_copy1[
                                                                                                      'base_platelets']) <= 0.5)),
                                                                                        1, 0)

merge_platelets_first_hep_baseline_full_copy1

only_50_per_drop_5_10_days = merge_platelets_first_hep_baseline_full_copy1[
    merge_platelets_first_hep_baseline_full_copy1['HIT_c2_50per_drop_5_10_days'] == 1]
only_50_per_drop_5_10_days

# check the first time p_drop >= 50% (compared to baseline) happened for each hadm_id, after hep admin. Here, one hadm_id only has one record
only_50_per_drop_5_10_days['p_charttime'] = pd.to_datetime(only_50_per_drop_5_10_days['p_charttime'], dayfirst=True)

first_hit_P_50_per_drop_record_5_10_days = only_50_per_drop_5_10_days.loc[
    only_50_per_drop_5_10_days.groupby('hadm_id').p_charttime.idxmin()].reset_index(drop=True)
first_hit_P_50_per_drop_record_5_10_days.rename(
    columns={'p_charttime': 'c2_50per_charttime_5_10_days', 'platelet': 'c2_50per_platelets_5_10_days'}, inplace=True)

full_cohort_7 = pd.merge(full_cohort_6, first_hit_P_50_per_drop_record_5_10_days[
    ['hadm_id', 'c2_50per_charttime_5_10_days', 'c2_50per_platelets_5_10_days', 'HIT_c2_50per_drop_5_10_days']],
                         on='hadm_id', how='left')

full_cohort_7['HIT_both_c1_and_c2_5_10_days'] = np.where(
    ((full_cohort_7['HIT_c1_150k_5_10_days'] == 1) | (full_cohort_7['HIT_c2_50per_drop_5_10_days'] == 1)), 1, 0)
full_cohort_7['c1&c2_first_p_time_5_10_days'] = (
    full_cohort_7[['c1_150k_charttime_5_10_days', 'c2_50per_charttime_5_10_days']].min(axis=1)).where(
    (full_cohort_7['HIT_both_c1_and_c2_5_10_days'] == 1), pd.NaT)
full_cohort_7['first_hep_to_HIT_5_10_days'] = pd.to_datetime(
    full_cohort_7['c1&c2_first_p_time_5_10_days'], dayfirst=True) - pd.to_datetime(full_cohort_7['hep_start'],
                                                                                   dayfirst=True)

# Convert days to hours
full_cohort_7['first_hep_to_HIT_hrs_5_10_days'] = round(
    full_cohort_7['first_hep_to_HIT_5_10_days'] / np.timedelta64(1, 'h'))

# check stats

print("check stats")

print(len(full_cohort_7[full_cohort_7['HIT_c1_150k'] == 1]['hadm_id']))  # 5720
print(len(full_cohort_7[full_cohort_7['HIT_c2_50%_drop'] == 1]))  # 1045
print(len(full_cohort_7[full_cohort_7['HIT_both_c1&c2'] == 1]))  # 5820
print(len(full_cohort_7[(full_cohort_7['HIT_c1_150k'] == 1) & (full_cohort_7['HIT_c2_50%_drop'] == 1)]))  # 945

# Below numbers may change, as we will apply outliers removeal, in latter section (Section 12).
# The final class distribution can be found from section 13.

print(len(full_cohort_7[full_cohort_7['HIT_c1_150k_5_10_days'] == 1]))  # 1931
print(len(full_cohort_7[full_cohort_7['HIT_c2_50per_drop_5_10_days'] == 1]))  # 477
print(len(full_cohort_7[full_cohort_7['HIT_both_c1_and_c2_5_10_days'] == 1]))  # 1989
print(len(full_cohort_7[(full_cohort_7['HIT_c1_150k_5_10_days'] == 1) & (
        full_cohort_7['HIT_c2_50per_drop_5_10_days'] == 1)]))  # 419

print(full_cohort_7.columns.tolist())
# ['subject_id', 'hadm_id', 'stay_id', 'hep_start', 'icu_in_time_first_hep', 'icu_out_time_first_hep', 'first_careunit', 'last_careunit', 'admittime', 'dischtime', 'treatment_types', 'hep_types', 'event_txt', 'drug', 'admittime_1', 'dischtime_1', 'admission_type', 'admission_location', 'hospital_expire_flag', 'gender', 'anchor_age', 'base_p_charttime', 'base_platelets', 'c1_150k_charttime', 'c1_150k_platelets', 'HIT_c1_150k', 'c2_50per_charttime', 'c2_50per_platelets', 'HIT_c2_50%_drop', 'HIT_both_c1&c2', 'c1&c2_first_p_charttime', 'first_hep_to_first_HIT', 'first_hep_to_first_HIT_hrs', 'c1_150k_charttime_5_10_days', 'c1_150k_platelets_5_10_days', 'HIT_c1_150k_5_10_days', 'c2_50per_charttime_5_10_days', 'c2_50per_platelets_5_10_days', 'HIT_c2_50per_drop_5_10_days', 'HIT_both_c1_and_c2_5_10_days', 'c1&c2_first_p_time_5_10_days', 'first_hep_to_HIT_5_10_days', 'first_hep_to_HIT_hrs_5_10_days']

# ------------------------------------------------------------------------------------------------------------------------------------

# 10. Merge Vital signs

full_cohort_10 = full_cohort_7

# select vital signs only and exclude status column for each vital sign

vital_signs_all_without_status = df3_vitalsigns[
    ["stay_id", 'heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean', 'dbp_min',
     'dbp_max', 'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean', 'resp_rate_min', 'resp_rate_max', 'resp_rate_mean',
     'spo2_min', 'spo2_max', 'spo2_mean', 'temperature_vital_min', 'temperature_vital_max', 'temperature_vital_mean',
     'glucose_vital_min', 'glucose_vital_max', 'glucose_vital_mean', 'hematocrit_bg_min', 'hematocrit_bg_max',
     'hemoglobin_bg_min', 'hemoglobin_bg_max', 'bicarbonate_bg_min', 'bicarbonate_bg_max', 'calcium_bg_min',
     'calcium_bg_max', 'chloride_bg_min', 'chloride_bg_max', 'sodium_bg_min', 'sodium_bg_max', 'potassium_bg_min',
     'potassium_bg_max', 'temperature_bg_min', 'temperature_bg_max', 'glucose_bg_min', 'glucose_bg_max', 'lactate_min',
     'lactate_max', 'ph_min', 'ph_max', 'baseexcess_min', 'baseexcess_max', 'carboxyhemoglobin_min',
     'carboxyhemoglobin_max', 'methemoglobin_min', 'methemoglobin_max', 'hematocrit_lab_min', 'hematocrit_lab_max',
     'hemoglobin_lab_min', 'hemoglobin_lab_max', 'bicarbonate_lab_min', 'bicarbonate_lab_max', 'calcium_lab_min',
     'calcium_lab_max', 'chloride_lab_min', 'chloride_lab_max', 'sodium_lab_min', 'sodium_lab_max', 'potassium_lab_min',
     'potassium_lab_max', 'glucose_lab_min', 'glucose_lab_max', 'platelets_min', 'platelets_max', 'wbc_min', 'wbc_max',
     'albumin_min', 'albumin_max', 'globulin_min', 'globulin_max', 'total_protein_min', 'total_protein_max',
     'aniongap_min', 'aniongap_max', 'bun_min', 'bun_max', 'creatinine_min', 'creatinine_max', 'abs_basophils_min',
     'abs_basophils_max', 'abs_eosinophils_min', 'abs_eosinophils_max', 'abs_lymphocytes_min', 'abs_lymphocytes_max',
     'abs_monocytes_min', 'abs_monocytes_max', 'abs_neutrophils_min', 'abs_neutrophils_max', 'atyps_min', 'atyps_max',
     'bands_min', 'bands_max', 'imm_granulocytes_min', 'imm_granulocytes_max', 'metas_min', 'metas_max', 'nrbc_min',
     'nrbc_max', 'd_dimer_min', 'd_dimer_max', 'fibrinogen_min', 'fibrinogen_max', 'thrombin_min', 'thrombin_max',
     'inr_min', 'inr_max', 'pt_min', 'pt_max', 'ptt_min', 'ptt_max', 'alt_min', 'alt_max', 'alp_min', 'alp_max',
     'ast_min', 'ast_max', 'amylase_min', 'amylase_max', 'bilirubin_total_min', 'bilirubin_total_max',
     'bilirubin_direct_min', 'bilirubin_direct_max', 'bilirubin_indirect_min', 'bilirubin_indirect_max', 'ck_cpk_min',
     'ck_cpk_max', 'ck_mb_min', 'ck_mb_max', 'ggt_min', 'ggt_max', 'ld_ldh_min', 'ld_ldh_max', 'so2_bg_min',
     'so2_bg_max', 'po2_bg_min', 'po2_bg_max', 'pco2_bg_min', 'pco2_bg_max', 'aado2_bg_min', 'aado2_bg_max',
     'fio2_bg_min', 'fio2_bg_max', 'totalco2_bg_min', 'totalco2_bg_max', 'so2_bg_art_min', 'so2_bg_art_max',
     'po2_bg_art_min', 'po2_bg_art_max', 'pco2_bg_art_min', 'pco2_bg_art_max', 'aado2_bg_art_min', 'aado2_bg_art_max',
     'fio2_bg_art_min', 'fio2_bg_art_max', 'totalco2_bg_art_min', 'totalco2_bg_art_max', 'gcs_min']]

full_cohort_11 = pd.merge(full_cohort_10, vital_signs_all_without_status, on='stay_id', how='left')

# ------------------------------------------------------------------------------------------------------------------------------------

# 11. Missing values count of each row

print('count3')
print(full_cohort_11.shape)  # (13416, 206)

print(len(full_cohort_11[full_cohort_11['hep_start'] >= full_cohort_11['base_p_charttime']]))  # 10792

print(len(full_cohort_11[full_cohort_11['hep_start'] < full_cohort_11['base_p_charttime']]))  # 2548

print(len(full_cohort_11[full_cohort_11['base_p_charttime'].isnull()]))  # 76

# Define new feature - 'base_platelets'
# check counts for base_platelets

full_cohort_11['base_platelets'] = np.where(full_cohort_11['hep_start'] > full_cohort_11['base_p_charttime'],
                                            full_cohort_11['base_platelets'], np.NaN)


print(len(full_cohort_11[full_cohort_11['hep_start'] >= full_cohort_11['base_p_charttime']][['hadm_id', 'hep_start', 'base_p_charttime']]))  # 10792
print(len(full_cohort_11[full_cohort_11['hep_start'] > full_cohort_11['base_p_charttime']][['hadm_id', 'hep_start', 'base_p_charttime']]))  # 10767
print(len(full_cohort_11[full_cohort_11['hep_start'] == full_cohort_11['base_p_charttime']][['hadm_id', 'hep_start', 'base_p_charttime']]))  # 25

# #print(len(full_cohort_11[full_cohort_11['base_platelets'] > 150 & full_cohort_11['base_platelets'] <= 400]))

print('bbb')
print(len(full_cohort_11[full_cohort_11['base_platelets'] < 150]))  # 3057
print(len(full_cohort_11[full_cohort_11['base_platelets'] >= 150]))  # 7710
print(len(full_cohort_11[full_cohort_11['base_platelets'] == 400]))  # 6
print(len(full_cohort_11[full_cohort_11['base_platelets'] > 400]))  # 471
print(len(full_cohort_11[full_cohort_11['base_platelets'].isnull()].hadm_id.unique()))  # 2649

features_set1_with_all_attributes = full_cohort_11[
    # When we give the fIrst heparin dose, We should make the decision whether the patient will get HIT within next 5 to days, or not.
    # Therefore, as features, we should only use the features those ara availble by the time of first heparin dose.

    ['hadm_id', 'stay_id', 'HIT_both_c1_and_c2_5_10_days', 'first_careunit', 'admission_type', 'admission_location', 'gender',
     'anchor_age', 'base_platelets', 'hep_types', 'treatment_types', 'heart_rate_min', 'heart_rate_max',
     'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean', 'dbp_min', 'dbp_max', 'dbp_mean', 'mbp_min', 'mbp_max',
     'mbp_mean', 'resp_rate_min', 'resp_rate_max', 'resp_rate_mean', 'spo2_min', 'spo2_max', 'spo2_mean',
     'temperature_vital_min', 'temperature_vital_max', 'temperature_vital_mean', 'glucose_vital_min',
     'glucose_vital_max', 'glucose_vital_mean', 'hematocrit_bg_min', 'hematocrit_bg_max', 'hemoglobin_bg_min',
     'hemoglobin_bg_max', 'bicarbonate_bg_min', 'bicarbonate_bg_max', 'calcium_bg_min', 'calcium_bg_max',
     'chloride_bg_min', 'chloride_bg_max', 'sodium_bg_min', 'sodium_bg_max', 'potassium_bg_min', 'potassium_bg_max',
     'temperature_bg_min', 'temperature_bg_max', 'glucose_bg_min', 'glucose_bg_max', 'lactate_min', 'lactate_max',
     'ph_min', 'ph_max', 'baseexcess_min', 'baseexcess_max', 'carboxyhemoglobin_min', 'carboxyhemoglobin_max',
     'methemoglobin_min', 'methemoglobin_max', 'hematocrit_lab_min', 'hematocrit_lab_max', 'hemoglobin_lab_min',
     'hemoglobin_lab_max', 'bicarbonate_lab_min', 'bicarbonate_lab_max', 'calcium_lab_min', 'calcium_lab_max',
     'chloride_lab_min', 'chloride_lab_max', 'sodium_lab_min', 'sodium_lab_max', 'potassium_lab_min',
     'potassium_lab_max', 'glucose_lab_min', 'glucose_lab_max', 'platelets_min', 'platelets_max', 'wbc_min', 'wbc_max',
     'albumin_min', 'albumin_max', 'globulin_min', 'globulin_max', 'total_protein_min', 'total_protein_max',
     'aniongap_min', 'aniongap_max', 'bun_min', 'bun_max', 'creatinine_min', 'creatinine_max', 'abs_basophils_min',
     'abs_basophils_max', 'abs_eosinophils_min', 'abs_eosinophils_max', 'abs_lymphocytes_min', 'abs_lymphocytes_max',
     'abs_monocytes_min', 'abs_monocytes_max', 'abs_neutrophils_min', 'abs_neutrophils_max', 'atyps_min', 'atyps_max',
     'bands_min', 'bands_max', 'imm_granulocytes_min', 'imm_granulocytes_max', 'metas_min', 'metas_max', 'nrbc_min',
     'nrbc_max', 'd_dimer_min', 'd_dimer_max', 'fibrinogen_min', 'fibrinogen_max', 'thrombin_min', 'thrombin_max',
     'inr_min', 'inr_max', 'pt_min', 'pt_max', 'ptt_min', 'ptt_max', 'alt_min', 'alt_max', 'alp_min', 'alp_max',
     'ast_min', 'ast_max', 'amylase_min', 'amylase_max', 'bilirubin_total_min', 'bilirubin_total_max',
     'bilirubin_direct_min', 'bilirubin_direct_max', 'bilirubin_indirect_min', 'bilirubin_indirect_max', 'ck_cpk_min',
     'ck_cpk_max', 'ck_mb_min', 'ck_mb_max', 'ggt_min', 'ggt_max', 'ld_ldh_min', 'ld_ldh_max', 'so2_bg_min',
     'so2_bg_max', 'po2_bg_min', 'po2_bg_max', 'pco2_bg_min', 'pco2_bg_max', 'aado2_bg_min', 'aado2_bg_max',
     'fio2_bg_min', 'fio2_bg_max', 'totalco2_bg_min', 'totalco2_bg_max', 'so2_bg_art_min', 'so2_bg_art_max',
     'po2_bg_art_min', 'po2_bg_art_max', 'pco2_bg_art_min', 'pco2_bg_art_max', 'aado2_bg_art_min', 'aado2_bg_art_max',
     'fio2_bg_art_min', 'fio2_bg_art_max', 'totalco2_bg_art_min', 'totalco2_bg_art_max', 'gcs_min']]

# remove base platelet records, if base platelet records was recorded after first heparin dose
# Base platelet count records should be the prior, and the closes platelet count record before first heparin dose.
# # but for some patients, there were no platelet count records before first heparin dose.
# For such patients, the closest p_count record after, but within the first 24hrs since first heparin dose(in full_cohort_3) was considered.
# But here when we select features in 'features_set1_with_all_attributes', we remove such base p_count records, as all the features shpuld be availble by the time of first hepairn dose.


# remove less frequent / less important vitals sings, those had high % of missing values.
print(features_set1_with_all_attributes.shape)  # (13416, 173)

features_set1 = features_set1_with_all_attributes.drop(
    ["imm_granulocytes_max", "imm_granulocytes_min", "abs_basophils_max", "abs_basophils_min", "abs_monocytes_min",
     "abs_monocytes_max", "abs_lymphocytes_max", "abs_lymphocytes_min", "abs_neutrophils_max", "abs_neutrophils_min",
     "po2_bg_max", "po2_bg_min", "abs_eosinophils_max", "abs_eosinophils_min", "baseexcess_min", "baseexcess_max",
     "so2_bg_max", "so2_bg_min", "fio2_bg_max", "fio2_bg_min", "bilirubin_indirect_max", "bilirubin_indirect_min",
     "metas_max", "metas_min", "aado2_bg_min", "aado2_bg_max", ], axis=1)

print(features_set1.shape)  # (13416, 147)


# Number of missing values per each column

# Remove columns if the column has missing values more than the thresh value

thresh_for_missing_values = 0.8  # AT least 80% of the values of each feature should be non-missing, in order to stay
print('Before deletion of missing values' + str(
    features_set1.shape))  # (13487, 147)

# Before deletion of missing values(13416, 147) - includung 'stay_id' and 'HIT_both_c1_and_c2_5_10_days'

pd.set_option('display.max_rows', None)

print(features_set1.isnull().sum().sort_values(ascending=False).reset_index(drop=False))

#                             index      0
# 0                    thrombin_max  13403
# 1                    thrombin_min  13403
# 2                     d_dimer_max  13349
# 3                     d_dimer_min  13349
# 4                         ggt_min  13279
# 5                         ggt_max  13279
# 6                    globulin_min  13229
# 7                    globulin_max  13229
# 8               total_protein_min  13152
# 9               total_protein_max  13152
# 10              methemoglobin_min  13101
# 11              methemoglobin_max  13101
# 12                    amylase_max  13098
# 13                    amylase_min  13098
# 14          carboxyhemoglobin_min  13084
# 15          carboxyhemoglobin_max  13084
# 16               aado2_bg_art_min  13070
# 17               aado2_bg_art_max  13070
# 18           bilirubin_direct_max  12881
# 19           bilirubin_direct_min  12881
# 20             bicarbonate_bg_max  12578
# 21             bicarbonate_bg_min  12578
# 22                fio2_bg_art_max  12494
# 23                fio2_bg_art_min  12494
# 24                       nrbc_min  12398
# 25                       nrbc_max  12398
# 26                 so2_bg_art_min  12065
# 27                 so2_bg_art_max  12065
# 28                      atyps_max  11926
# 29                      atyps_min  11926
# 30                 fibrinogen_min  11871
# 31                 fibrinogen_max  11871
# 32                      bands_max  11803
# 33                      bands_min  11803
# 34              hemoglobin_bg_max  11718
# 35              hemoglobin_bg_min  11718
# 36              hematocrit_bg_max  11718
# 37              hematocrit_bg_min  11718
# 38             temperature_bg_min  11508
# 39             temperature_bg_max  11508
# 40                chloride_bg_min  11369
# 41                chloride_bg_max  11369
# 42                  sodium_bg_min  10962
# 43                  sodium_bg_max  10962
# 44                 glucose_bg_min  10739
# 45                 glucose_bg_max  10739
# 46                     ck_cpk_max  10305
# 47                     ck_cpk_min  10305
# 48                      ck_mb_min  10135
# 49                      ck_mb_max  10135
# 50                     ld_ldh_max   9890
# 51                     ld_ldh_min   9890
# 52                 calcium_bg_min   9886
# 53                 calcium_bg_max   9886
# 54                pco2_bg_art_min   9750
# 55                 po2_bg_art_min   9750
# 56                 po2_bg_art_max   9750
# 57                pco2_bg_art_max   9750
# 58            totalco2_bg_art_max   9750
# 59            totalco2_bg_art_min   9750
# 60               potassium_bg_max   9686
# 61               potassium_bg_min   9686
# 62                    albumin_min   7158
# 63                    albumin_max   7158
# 64                    pco2_bg_max   5728
# 65                    pco2_bg_min   5728
# 66                totalco2_bg_min   5726
# 67                totalco2_bg_max   5726
# 68            bilirubin_total_max   5713
# 69            bilirubin_total_min   5713
# 70                        alt_min   5647
# 71                        alt_max   5647
# 72                        alp_max   5602
# 73                        alp_min   5602
# 74                        ast_min   5547
# 75                        ast_max   5547
# 76                         ph_max   5487
# 77                         ph_min   5487
# 78                    lactate_min   4641
# 79                    lactate_max   4641
# 80                 base_platelets   2649
# 81                        ptt_min   2292
# 82                        ptt_max   2292
# 83              glucose_vital_min   2200
# 84              glucose_vital_max   2200
# 85             glucose_vital_mean   2200
# 86                         pt_max   2195
# 87                         pt_min   2195
# 88                        inr_max   2195
# 89                        inr_min   2195
# 90                calcium_lab_max   1427
# 91                calcium_lab_min   1427
# 92          temperature_vital_min    722
# 93         temperature_vital_mean    722
# 94          temperature_vital_max    722
# 95                       dbp_mean    555
# 96                        dbp_max    555
# 97                        dbp_min    555
# 98                       sbp_mean    554
# 99                        sbp_max    554
# 100                       sbp_min    554
# 101                 resp_rate_max    478
# 102                 resp_rate_min    478
# 103                resp_rate_mean    478
# 104                       mbp_min    466
# 105                       mbp_max    466
# 106                      mbp_mean    466
# 107                      spo2_min    458
# 108                     spo2_mean    458
# 109                      spo2_max    458
# 110                heart_rate_max    436
# 111                heart_rate_min    436
# 112               heart_rate_mean    436
# 113                       gcs_min    424
# 114               glucose_lab_max    421
# 115               glucose_lab_min    421
# 116                  aniongap_min    404
# 117                  aniongap_max    404
# 118           bicarbonate_lab_max    397
# 119           bicarbonate_lab_min    397
# 120              chloride_lab_max    393
# 121              chloride_lab_min    393
# 122             potassium_lab_max    390
# 123                sodium_lab_max    390
# 124                sodium_lab_min    390
# 125             potassium_lab_min    390
# 126            hemoglobin_lab_max    325
# 127            hemoglobin_lab_min    325
# 128                 platelets_max    324
# 129                 platelets_min    324
# 130                       wbc_max    314
# 131                       wbc_min    314
# 132                creatinine_min    312
# 133                creatinine_max    312
# 134            hematocrit_lab_max    301
# 135            hematocrit_lab_min    301
# 136                       bun_min    285
# 137                       bun_max    285
# 138  HIT_both_c1_and_c2_5_10_days      0
# 139               treatment_types      0
# 140                     hep_types      0
# 141                    anchor_age      0
# 142                        gender      0
# 143            admission_location      0
# 144                admission_type      0
# 145                first_careunit      0
# 146                       stay_id      0

# -----------------------------------------------------

thresh = len(features_set1) * thresh_for_missing_values
features_set1.dropna(axis=1, thresh=thresh, inplace=True)  # thresh - Require that many non-NA values
print('After deletion of missing values' + str(
    features_set1.shape))
# After deletion of missing values(13416,, 67) - includung 'stay_id' and 'HIT_both_c1_and_c2_5_10_days'

# check the new feature set(after deletion of features with higher number of missing values compared to the given threshold

print(features_set1.isnull().sum().sort_values(ascending=False).reset_index(drop=False))  # (replace = 'True')

#                            index     0
# 0                 base_platelets  2649
# 1                        ptt_max  2292
# 2                        ptt_min  2292
# 3             glucose_vital_mean  2200
# 4              glucose_vital_min  2200
# 5              glucose_vital_max  2200
# 6                        inr_max  2195
# 7                        inr_min  2195
# 8                         pt_max  2195
# 9                         pt_min  2195
# 10               calcium_lab_max  1427
# 11               calcium_lab_min  1427
# 12        temperature_vital_mean   722
# 13         temperature_vital_max   722
# 14         temperature_vital_min   722
# 15                       dbp_max   555
# 16                      dbp_mean   555
# 17                       dbp_min   555
# 18                      sbp_mean   554
# 19                       sbp_max   554
# 20                       sbp_min   554
# 21                 resp_rate_min   478
# 22                 resp_rate_max   478
# 23                resp_rate_mean   478
# 24                       mbp_max   466
# 25                      mbp_mean   466
# 26                       mbp_min   466
# 27                      spo2_min   458
# 28                      spo2_max   458
# 29                     spo2_mean   458
# 30               heart_rate_mean   436
# 31                heart_rate_max   436
# 32                heart_rate_min   436
# 33                       gcs_min   424
# 34               glucose_lab_max   421
# 35               glucose_lab_min   421
# 36                  aniongap_max   404
# 37                  aniongap_min   404
# 38           bicarbonate_lab_min   397
# 39           bicarbonate_lab_max   397
# 40              chloride_lab_min   393
# 41              chloride_lab_max   393
# 42                sodium_lab_min   390
# 43                sodium_lab_max   390
# 44             potassium_lab_min   390
# 45             potassium_lab_max   390
# 46            hemoglobin_lab_min   325
# 47            hemoglobin_lab_max   325
# 48                 platelets_max   324
# 49                 platelets_min   324
# 50                       wbc_min   314
# 51                       wbc_max   314
# 52                creatinine_min   312
# 53                creatinine_max   312
# 54            hematocrit_lab_min   301
# 55            hematocrit_lab_max   301
# 56                       bun_min   285
# 57                       bun_max   285
# 58               treatment_types     0
# 59  HIT_both_c1_and_c2_5_10_days     0
# 60                     hep_types     0
# 61                    anchor_age     0
# 62                        gender     0
# 63            admission_location     0
# 64                admission_type     0
# 65                first_careunit     0
# 66                       stay_id     0

pd.set_option('display.max_rows', 10)

#-------------------------------------------------------------------------------------------------

# merge 'status' columns corresponding to features with missing values > 20%

vital_signs_status_list = df3_vitalsigns[
    ["stay_id", 'heart_rate_min_status', 'heart_rate_max_status', 'heart_rate_mean_status', 'sbp_min_status',
     'sbp_max_status', 'sbp_mean_status', 'dbp_min_status', 'dbp_max_status', 'dbp_mean_status', 'mbp_min_status',
     'mbp_max_status', 'mbp_mean_status', 'resp_rate_min_status', 'resp_rate_max_status', 'resp_rate_mean_status',
     'spo2_min_status', 'spo2_max_status', 'spo2_mean_status', 'temperature_vital_min_status',
     'temperature_vital_max_status', 'temperature_vital_mean_status', 'glucose_vital_min_status',
     'glucose_vital_max_status', 'glucose_vital_mean_status', 'hematocrit_bg_min_status', 'hematocrit_bg_max_status',
     'hemoglobin_bg_min_status', 'hemoglobin_bg_max_status', 'bicarbonate_bg_min_status', 'bicarbonate_bg_max_status',
     'calcium_bg_min_status', 'calcium_bg_max_status', 'chloride_bg_min_status', 'chloride_bg_max_status',
     'sodium_bg_min_status', 'sodium_bg_max_status', 'potassium_bg_min_status', 'potassium_bg_max_status',
     'temperature_bg_min_status', 'temperature_bg_max_status', 'glucose_bg_min_status', 'glucose_bg_max_status',
     'lactate_min_status', 'lactate_max_status', 'ph_min_status', 'ph_max_status', 'baseexcess_min_status',
     'baseexcess_max_status', 'carboxyhemoglobin_min_status', 'carboxyhemoglobin_max_status',
     'methemoglobin_min_status', 'methemoglobin_max_status', 'hematocrit_lab_min_status', 'hematocrit_lab_max_status',
     'hemoglobin_lab_min_status', 'hemoglobin_lab_max_status', 'bicarbonate_lab_min_status',
     'bicarbonate_lab_max_status', 'calcium_lab_min_status', 'calcium_lab_max_status', 'chloride_lab_min_status',
     'chloride_lab_max_status', 'sodium_lab_min_status', 'sodium_lab_max_status', 'potassium_lab_min_status',
     'potassium_lab_max_status', 'glucose_lab_min_status', 'glucose_lab_max_status', 'platelets_min_status',
     'platelets_max_status', 'wbc_min_status', 'wbc_max_status', 'albumin_min_status', 'albumin_max_status',
     'globulin_min_status', 'globulin_max_status', 'total_protein_min_status', 'total_protein_max_status',
     'aniongap_min_status', 'aniongap_max_status', 'bun_min_status', 'bun_max_status', 'creatinine_min_status',
     'creatinine_max_status', 'abs_basophils_min_status', 'abs_basophils_max_status', 'abs_eosinophils_min_status',
     'abs_eosinophils_max_status', 'abs_lymphocytes_min_status', 'abs_lymphocytes_max_status',
     'abs_monocytes_min_status', 'abs_monocytes_max_status', 'abs_neutrophils_min_status', 'abs_neutrophils_max_status',
     'atyps_min_status', 'atyps_max_status', 'bands_min_status', 'bands_max_status', 'imm_granulocytes_min_status',
     'imm_granulocytes_max_status', 'metas_min_status', 'metas_max_status', 'nrbc_min_status', 'nrbc_max_status',
     'd_dimer_min_status', 'd_dimer_max_status', 'fibrinogen_min_status', 'fibrinogen_max_status',
     'thrombin_min_status', 'thrombin_max_status', 'inr_min_status', 'inr_max_status', 'pt_min_status', 'pt_max_status',
     'ptt_min_status', 'ptt_max_status', 'alt_min_status', 'alt_max_status', 'alp_min_status', 'alp_max_status',
     'ast_min_status', 'ast_max_status', 'amylase_min_status', 'amylase_max_status', 'bilirubin_total_min_status',
     'bilirubin_total_max_status', 'bilirubin_direct_min_status', 'bilirubin_direct_max_status',
     'bilirubin_indirect_min_status', 'bilirubin_indirect_max_status', 'ck_cpk_min_status', 'ck_cpk_max_status',
     'ck_mb_min_status', 'ck_mb_max_status', 'ggt_min_status', 'ggt_max_status', 'ld_ldh_min_status',
     'ld_ldh_max_status', 'so2_bg_min_status', 'so2_bg_max_status', 'po2_bg_min_status', 'po2_bg_max_status',
     'pco2_bg_min_status', 'pco2_bg_max_status', 'aado2_bg_min_status', 'aado2_bg_max_status', 'fio2_bg_min_status',
     'fio2_bg_max_status', 'totalco2_bg_min_status', 'totalco2_bg_max_status', 'so2_bg_art_min_status',
     'so2_bg_art_max_status', 'po2_bg_art_min_status', 'po2_bg_art_max_status', 'pco2_bg_art_min_status',
     'pco2_bg_art_max_status', 'aado2_bg_art_min_status', 'aado2_bg_art_max_status', 'fio2_bg_art_min_status',
     'fio2_bg_art_max_status', 'totalco2_bg_art_min_status', 'totalco2_bg_art_max_status', 'gcs_min_status']]

# case 1 - considering completed cases (omit features with missing) only - 13 features - 43(after completing categorical to numerical)
# x1 = features_set1[['admission_type', 'admission_location', 'gender', 'anchor_age', 'hep_types', 'treatment_types', 'heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'spo2_min', 'spo2_max', 'spo2_mean', 'first_careunit']]

# case 2 - selecting all features, with missing values no more than given threshold - 63 features

# x1 = features_set1.drop(['HIT_both_c1_and_c2_5_10_days', 'stay_id'], axis = 1)

# case 3 - 65 features -  features in case #2 + vital_sign_status(categorical) features those have non-missing values between 70%-80%

# x1 = (pd.merge(features_set1 , vital_signs_status_list[['stay_id','lactate_min_status', 'lactate_max_status']], on= 'stay_id' , how = 'left')).drop(['HIT_both_c1_and_c2_5_10_days', 'stay_id'], axis = 1)

# case 4 - 79 features -  features in case #2 + vital_sign_status(categorical) features those have non-missing values between 60%-80%

# x1 = (pd.merge(features_set1 , vital_signs_status_list[['stay_id','lactate_min_status', 'lactate_max_status', "ph_min_status", "ph_max_status", "totalco2_bg_min_status", "totalco2_bg_max_status", "pco2_bg_min_status", "pco2_bg_max_status", "ast_min_status", "ast_max_status", "alp_min_status", "alp_max_status", "alt_min_status", "alt_max_status", "bilirubin_total_min_status", "bilirubin_total_max_status"]], on= 'stay_id' , how = 'left')).drop(['HIT_both_c1_and_c2_5_10_days', 'stay_id'], axis = 1)

# case 5 - 81 features -  features in case #2 + vital_sign_status(categorical) features those have non-missing values between 40%-80%
# No features for missing values between 50%-60%. That't why directly went from 60%-80% to 40%-80%

# x1 = (pd.merge(features_set1 , vital_signs_status_list[['stay_id','lactate_min_status', 'lactate_max_status', "ph_min_status", "ph_max_status", "totalco2_bg_min_status", "totalco2_bg_max_status", "pco2_bg_min_status", "pco2_bg_max_status", "ast_min_status", "ast_max_status", "alp_min_status", "alp_max_status", "alt_min_status", "alt_max_status", "bilirubin_total_min_status", "bilirubin_total_max_status", "albumin_min_status", "albumin_max_status"]], on= 'stay_id' , how = 'left')).drop(['HIT_both_c1_and_c2_5_10_days', 'stay_id'], axis = 1)

# case 6 - 89 features -  features in case #2 + vital_sign_status(categorical) features those have non-missing values between 30%-80%

# x1 = (pd.merge(features_set1 , vital_signs_status_list[['stay_id','lactate_min_status', 'lactate_max_status', "ph_min_status", "ph_max_status", "totalco2_bg_min_status", "totalco2_bg_max_status", "pco2_bg_min_status", "pco2_bg_max_status", "ast_min_status", "ast_max_status", "alp_min_status", "alp_max_status", "alt_min_status", "alt_max_status", "bilirubin_total_min_status", "bilirubin_total_max_status", "albumin_min_status", "albumin_max_status", "pco2_bg_art_min_status", "pco2_bg_art_max_status", "po2_bg_art_max_status", "po2_bg_art_min_status", "totalco2_bg_art_max_status", "totalco2_bg_art_min_status", "ld_ldh_max_status", "ld_ldh_min_status"]], on= 'stay_id' , how = 'left')).drop(['HIT_both_c1_and_c2_5_10_days', 'stay_id'], axis = 1)

# case 7 - 93 features -  features in case #2 + vital_sign_status(categorical) features those have non-missing values between 20%-80%

# x1 = (pd.merge(features_set1 , vital_signs_status_list[['stay_id','lactate_min_status', 'lactate_max_status', "ph_min_status", "ph_max_status", "totalco2_bg_min_status", "totalco2_bg_max_status", "pco2_bg_min_status", "pco2_bg_max_status", "ast_min_status", "ast_max_status", "alp_min_status", "alp_max_status", "alt_min_status", "alt_max_status", "bilirubin_total_min_status", "bilirubin_total_max_status", "albumin_min_status", "albumin_max_status", "pco2_bg_art_min_status", "pco2_bg_art_max_status", "po2_bg_art_max_status", "po2_bg_art_min_status", "totalco2_bg_art_max_status", "totalco2_bg_art_min_status", "ld_ldh_max_status", "ld_ldh_min_status", 'ck_cpk_min_status', 'ck_cpk_max_status', 'ck_mb_min_status', 'ck_mb_max_status']], on= 'stay_id' , how = 'left')).drop(['HIT_both_c1_and_c2_5_10_days', 'stay_id'], axis = 1)

# case 8 - 99 features -  features in case #2 + vital_sign_status(categorical) features those have non-missing values between 10%-80%

# x1 = (pd.merge(features_set1 , vital_signs_status_list[['stay_id','lactate_min_status', 'lactate_max_status', "ph_min_status", "ph_max_status", "totalco2_bg_min_status", "totalco2_bg_max_status", "pco2_bg_min_status", "pco2_bg_max_status", "ast_min_status", "ast_max_status", "alp_min_status", "alp_max_status", "alt_min_status", "alt_max_status", "bilirubin_total_min_status", "bilirubin_total_max_status", "albumin_min_status", "albumin_max_status", "pco2_bg_art_min_status", "pco2_bg_art_max_status", "po2_bg_art_max_status", "po2_bg_art_min_status", "totalco2_bg_art_max_status", "totalco2_bg_art_min_status", "ld_ldh_max_status", "ld_ldh_min_status", 'ck_cpk_min_status', 'ck_cpk_max_status', 'ck_mb_min_status', 'ck_mb_max_status', "fio2_bg_art_max_status", "fio2_bg_art_min_status", "so2_bg_art_max_status", "so2_bg_art_min_status", "fibrinogen_max_status", "fibrinogen_min_status"]], on= 'stay_id' , how = 'left')).drop(['HIT_both_c1_and_c2_5_10_days', 'stay_id'], axis = 1)

# case 9 - 125 features -  features in case #2 (63 features) + vital_sign_status(categorical) features (62 features) those have non-missing values between 0%-80% (all features)

print(features_set1.columns.tolist())
# ['stay_id', 'HIT_both_c1_and_c2_5_10_days', 'first_careunit', 'admission_type', 'admission_location', 'gender', 'anchor_age', 'base_platelets', 'hep_types', 'treatment_types', 'heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean', 'dbp_min', 'dbp_max', 'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean', 'resp_rate_min', 'resp_rate_max', 'resp_rate_mean', 'spo2_min', 'spo2_max', 'spo2_mean', 'temperature_vital_min', 'temperature_vital_max', 'temperature_vital_mean', 'glucose_vital_min', 'glucose_vital_max', 'glucose_vital_mean', 'hematocrit_lab_min', 'hematocrit_lab_max', 'hemoglobin_lab_min', 'hemoglobin_lab_max', 'bicarbonate_lab_min', 'bicarbonate_lab_max', 'calcium_lab_min', 'calcium_lab_max', 'chloride_lab_min', 'chloride_lab_max', 'sodium_lab_min', 'sodium_lab_max', 'potassium_lab_min', 'potassium_lab_max', 'glucose_lab_min', 'glucose_lab_max', 'platelets_min', 'platelets_max', 'wbc_min', 'wbc_max', 'aniongap_min', 'aniongap_max', 'bun_min', 'bun_max', 'creatinine_min', 'creatinine_max', 'inr_min', 'inr_max', 'pt_min', 'pt_max', 'ptt_min', 'ptt_max', 'gcs_min']

# # ------------------------------------------------------------------------------------------------------------------------------------

full_feature_list_with_labels = (pd.merge(features_set1, vital_signs_status_list[
    ['stay_id', 'thrombin_min_status', 'thrombin_max_status', 'd_dimer_max_status', 'd_dimer_min_status',
     'methemoglobin_min_status', 'methemoglobin_max_status', 'ggt_min_status', 'ggt_max_status', 'globulin_min_status',
     'globulin_max_status', 'total_protein_min_status', 'total_protein_max_status', 'atyps_max_status',
     'atyps_min_status', 'carboxyhemoglobin_min_status', 'carboxyhemoglobin_max_status', 'amylase_max_status',
     'amylase_min_status', 'aado2_bg_art_max_status', 'aado2_bg_art_min_status', 'bilirubin_direct_min_status',
     'bilirubin_direct_max_status', 'bicarbonate_bg_min_status', 'bicarbonate_bg_max_status', 'fio2_bg_art_min_status',
     'fio2_bg_art_max_status', 'nrbc_max_status', 'nrbc_min_status', 'bands_min_status', 'bands_max_status',
     'so2_bg_art_min_status', 'so2_bg_art_max_status', 'fibrinogen_max_status', 'fibrinogen_min_status',
     'hematocrit_bg_min_status', 'hematocrit_bg_max_status', 'hemoglobin_bg_min_status', 'hemoglobin_bg_max_status',
     'temperature_bg_max_status', 'temperature_bg_min_status', 'chloride_bg_min_status', 'chloride_bg_max_status',
     'sodium_bg_max_status', 'sodium_bg_min_status', 'glucose_bg_max_status', 'glucose_bg_min_status',
     'ck_cpk_max_status', 'ck_cpk_min_status', 'ck_mb_max_status', 'ck_mb_min_status', 'ld_ldh_max_status',
     'ld_ldh_min_status', 'calcium_bg_max_status', 'calcium_bg_min_status', 'pco2_bg_art_min_status',
     'po2_bg_art_max_status', 'totalco2_bg_art_max_status', 'totalco2_bg_art_min_status', 'pco2_bg_art_max_status',
     'po2_bg_art_min_status', 'potassium_bg_min_status', 'potassium_bg_max_status', 'albumin_max_status',
     'albumin_min_status', 'bilirubin_total_min_status', 'bilirubin_total_max_status', 'alt_max_status',
     'alt_min_status', 'alp_max_status', 'alp_min_status', 'ast_min_status', 'ast_max_status', 'pco2_bg_max_status',
     'pco2_bg_min_status', 'totalco2_bg_min_status', 'totalco2_bg_max_status', 'ph_min_status', 'ph_max_status',
     'lactate_min_status', 'lactate_max_status']],
                                          on='stay_id', how='left'))

# # ------------------------------------------------------------------------------------------------------------------------------------

# 12. Handling outliers
#
# check for outliers, and remove them.

# We chck outliers only for 'Numerical features'.

# stats of final 'Numerical' features
# features_set1.columns.tolist() returns the list of NUMERICAL features

pd.set_option('display.max_columns', None)
print(full_feature_list_with_labels[features_set1.columns.tolist()].describe())

# by running above line code, we checked for any abnormal value for 'max', for each column.
# we oberved, only for one column, ( for stay_id = 39817570), 'glucose_vital_max' = 999999.0, beacuse of this abnormal values, 'glucose_vital_mean' was also abnormal, for that row.
# We had 2 simple methods to deal with this - Either entirely remove this row, or to impute these abnormal values.
# We removed the entire row, where the stay_id = 39817570

# To check abnormal values for 'glucose_vital_max'. We sort the entire data frame by column 'glucose_vital_max', to check abnormal values for ''glucose_vital_max'. We oberved that only abnormal value was 999999, and thh second highest value was ok.
print(full_feature_list_with_labels.sort_values(by='glucose_vital_max', ascending=False).head(10))

pd.set_option('display.max_columns', 10)

# check for the stay_id with outlier values

print(full_feature_list_with_labels[full_feature_list_with_labels['glucose_vital_max'] == 999999.0])  # stay_id = 39817570

pd.set_option('display.max_columns', 10)

# check for the record with outlier

print('with outlier')

print(full_feature_list_with_labels.shape)  # (13416, 147)- INCLUDING columns for 'stay_id' and label

# remove the entire row with outliers, from full data set - 'full_feature_list_with_labels'

full_feature_list_with_labels = full_feature_list_with_labels.drop(full_feature_list_with_labels[full_feature_list_with_labels['glucose_vital_max'] == 999999.0].index) # drop the entire row with the outlier

print('without outlier')

print(full_feature_list_with_labels.shape)  # (13415, 147) - INCLUDING columns for 'stay_id' and label - only one row was removed.

# # ------------------------------------------------------------------------------------------------------------------------------------

# 13. Assign 'X' and 'Y' and Distribution of classes (hit / no hit)

#x1 = full_feature_list_with_labels.drop(['HIT_both_c1_and_c2_5_10_days', 'stay_id'], axis=1)
#x1 = full_feature_list_with_labels.drop(['HIT_both_c1_and_c2_5_10_days', 'hadm_id', 'stay_id'], axis=1)
x1 = full_feature_list_with_labels.drop(['HIT_both_c1_and_c2_5_10_days', 'stay_id'], axis=1) # hadm_id is still there, will drop it later

print(x1.columns.tolist())

# ['first_careunit', 'admission_type', 'admission_location', 'gender', 'anchor_age', 'base_platelets', 'hep_types', 'treatment_types', 'heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean', 'dbp_min', 'dbp_max', 'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean', 'resp_rate_min', 'resp_rate_max', 'resp_rate_mean', 'spo2_min', 'spo2_max', 'spo2_mean', 'temperature_vital_min', 'temperature_vital_max', 'temperature_vital_mean', 'glucose_vital_min', 'glucose_vital_max', 'glucose_vital_mean', 'hematocrit_lab_min', 'hematocrit_lab_max', 'hemoglobin_lab_min', 'hemoglobin_lab_max', 'bicarbonate_lab_min', 'bicarbonate_lab_max', 'calcium_lab_min', 'calcium_lab_max', 'chloride_lab_min', 'chloride_lab_max', 'sodium_lab_min', 'sodium_lab_max', 'potassium_lab_min', 'potassium_lab_max', 'glucose_lab_min', 'glucose_lab_max', 'platelets_min', 'platelets_max', 'wbc_min', 'wbc_max', 'aniongap_min', 'aniongap_max', 'bun_min', 'bun_max', 'creatinine_min', 'creatinine_max', 'inr_min', 'inr_max', 'pt_min', 'pt_max', 'ptt_min', 'ptt_max', 'gcs_min', 'thrombin_min_status', 'thrombin_max_status', 'd_dimer_max_status', 'd_dimer_min_status', 'methemoglobin_min_status', 'methemoglobin_max_status', 'ggt_min_status', 'ggt_max_status', 'globulin_min_status', 'globulin_max_status', 'total_protein_min_status', 'total_protein_max_status', 'atyps_max_status', 'atyps_min_status', 'carboxyhemoglobin_min_status', 'carboxyhemoglobin_max_status', 'amylase_max_status', 'amylase_min_status', 'aado2_bg_art_max_status', 'aado2_bg_art_min_status', 'bilirubin_direct_min_status', 'bilirubin_direct_max_status', 'bicarbonate_bg_min_status', 'bicarbonate_bg_max_status', 'fio2_bg_art_min_status', 'fio2_bg_art_max_status', 'nrbc_max_status', 'nrbc_min_status', 'bands_min_status', 'bands_max_status', 'so2_bg_art_min_status', 'so2_bg_art_max_status', 'fibrinogen_max_status', 'fibrinogen_min_status', 'hematocrit_bg_min_status', 'hematocrit_bg_max_status', 'hemoglobin_bg_min_status', 'hemoglobin_bg_max_status', 'temperature_bg_max_status', 'temperature_bg_min_status', 'chloride_bg_min_status', 'chloride_bg_max_status', 'sodium_bg_max_status', 'sodium_bg_min_status', 'glucose_bg_max_status', 'glucose_bg_min_status', 'ck_cpk_max_status', 'ck_cpk_min_status', 'ck_mb_max_status', 'ck_mb_min_status', 'ld_ldh_max_status', 'ld_ldh_min_status', 'calcium_bg_max_status', 'calcium_bg_min_status', 'pco2_bg_art_min_status', 'po2_bg_art_max_status', 'totalco2_bg_art_max_status', 'totalco2_bg_art_min_status', 'pco2_bg_art_max_status', 'po2_bg_art_min_status', 'potassium_bg_min_status', 'potassium_bg_max_status', 'albumin_max_status', 'albumin_min_status', 'bilirubin_total_min_status', 'bilirubin_total_max_status', 'alt_max_status', 'alt_min_status', 'alp_max_status', 'alp_min_status', 'ast_min_status', 'ast_max_status', 'pco2_bg_max_status', 'pco2_bg_min_status', 'totalco2_bg_min_status', 'totalco2_bg_max_status', 'ph_min_status', 'ph_max_status', 'lactate_min_status', 'lactate_max_status']
y1 = full_feature_list_with_labels['HIT_both_c1_and_c2_5_10_days']

print(x1.shape)  # (13415, 145)
print(y1.shape)  # (13415,)

# count class ditribution - HIT / No HIT

label_count = Counter(y1)
print(label_count)  # Counter({0: 11426, 1: 1989}) # previous - # Counter({0: 11426, 1: 1989})

#------------------------------------------------------------------------------------------------------------------------------------


# can consider base_platelets as a features(as for some hadm_id, it is after first hep)


# # ------------------------------------------------------------------------------------------------------------------------------------
#
# 14. Stratified train - test split

# We can keep 80% of the data to Train the model and the remaining 20% for Testing.

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, stratify=y1, random_state=0)
print("Check the subsets size: x_train:{}, y_train:{}, x_test:{}, y_test:{}. \n\n".format(x_train.shape, y_train.shape,
                                                                                          x_test.shape, y_test.shape))
# Check the subsets size: x_train:(10732, 145), y_train:(10732,), x_test:(2683, 145), y_test:(2683,).

print(x_train.columns.tolist())

# ['first_careunit', 'admission_type', 'admission_location', 'gender', 'anchor_age', 'base_platelets', 'hep_types', 'treatment_types', 'heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean', 'dbp_min', 'dbp_max', 'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean', 'resp_rate_min', 'resp_rate_max', 'resp_rate_mean', 'spo2_min', 'spo2_max', 'spo2_mean', 'temperature_vital_min', 'temperature_vital_max', 'temperature_vital_mean', 'glucose_vital_min', 'glucose_vital_max', 'glucose_vital_mean', 'hematocrit_lab_min', 'hematocrit_lab_max', 'hemoglobin_lab_min', 'hemoglobin_lab_max', 'bicarbonate_lab_min', 'bicarbonate_lab_max', 'calcium_lab_min', 'calcium_lab_max', 'chloride_lab_min', 'chloride_lab_max', 'sodium_lab_min', 'sodium_lab_max', 'potassium_lab_min', 'potassium_lab_max', 'glucose_lab_min', 'glucose_lab_max', 'platelets_min', 'platelets_max', 'wbc_min', 'wbc_max', 'aniongap_min', 'aniongap_max', 'bun_min', 'bun_max', 'creatinine_min', 'creatinine_max', 'inr_min', 'inr_max', 'pt_min', 'pt_max', 'ptt_min', 'ptt_max', 'gcs_min', 'thrombin_min_status', 'thrombin_max_status', 'd_dimer_max_status', 'd_dimer_min_status', 'methemoglobin_min_status', 'methemoglobin_max_status', 'ggt_min_status', 'ggt_max_status', 'globulin_min_status', 'globulin_max_status', 'total_protein_min_status', 'total_protein_max_status', 'atyps_max_status', 'atyps_min_status', 'carboxyhemoglobin_min_status', 'carboxyhemoglobin_max_status', 'amylase_max_status', 'amylase_min_status', 'aado2_bg_art_max_status', 'aado2_bg_art_min_status', 'bilirubin_direct_min_status', 'bilirubin_direct_max_status', 'bicarbonate_bg_min_status', 'bicarbonate_bg_max_status', 'fio2_bg_art_min_status', 'fio2_bg_art_max_status', 'nrbc_max_status', 'nrbc_min_status', 'bands_min_status', 'bands_max_status', 'so2_bg_art_min_status', 'so2_bg_art_max_status', 'fibrinogen_max_status', 'fibrinogen_min_status', 'hematocrit_bg_min_status', 'hematocrit_bg_max_status', 'hemoglobin_bg_min_status', 'hemoglobin_bg_max_status', 'temperature_bg_max_status', 'temperature_bg_min_status', 'chloride_bg_min_status', 'chloride_bg_max_status', 'sodium_bg_max_status', 'sodium_bg_min_status', 'glucose_bg_max_status', 'glucose_bg_min_status', 'ck_cpk_max_status', 'ck_cpk_min_status', 'ck_mb_max_status', 'ck_mb_min_status', 'ld_ldh_max_status', 'ld_ldh_min_status', 'calcium_bg_max_status', 'calcium_bg_min_status', 'pco2_bg_art_min_status', 'po2_bg_art_max_status', 'totalco2_bg_art_max_status', 'totalco2_bg_art_min_status', 'pco2_bg_art_max_status', 'po2_bg_art_min_status', 'potassium_bg_min_status', 'potassium_bg_max_status', 'albumin_max_status', 'albumin_min_status', 'bilirubin_total_min_status', 'bilirubin_total_max_status', 'alt_max_status', 'alt_min_status', 'alp_max_status', 'alp_min_status', 'ast_min_status', 'ast_max_status', 'pco2_bg_max_status', 'pco2_bg_min_status', 'totalco2_bg_min_status', 'totalco2_bg_max_status', 'ph_min_status', 'ph_max_status', 'lactate_min_status', 'lactate_max_status']

# ------------------------------------------------------------------------------------------------------------------------------------
# # #
# # # save results as csv files
# #
# # # save cohort with first heparin time, label, and HIT diagnosed time
# #
# # # Save train and test data separately, BEFORE pre-processing.
#
# # To preserve desired datetime format:
#
# full_cohort_7['hep_start'] = pd.to_datetime(full_cohort_7['hep_start'], dayfirst=True)
# full_cohort_7['icu_in_time_first_hep'] = pd.to_datetime(full_cohort_7['icu_in_time_first_hep'], dayfirst=True)
# full_cohort_7['icu_out_time_first_hep'] = pd.to_datetime(full_cohort_7['icu_out_time_first_hep'], dayfirst=True)
# full_cohort_7['admittime'] = pd.to_datetime(full_cohort_7['admittime'], dayfirst=True)
# full_cohort_7['dischtime'] = pd.to_datetime(full_cohort_7['dischtime'], dayfirst=True)
# full_cohort_7['base_p_charttime'] = pd.to_datetime(full_cohort_7['base_p_charttime'], dayfirst=True)
# full_cohort_7['c1_150k_charttime_5_10_days'] = pd.to_datetime(full_cohort_7['c1_150k_charttime_5_10_days'], dayfirst=True)
# full_cohort_7['c2_50per_charttime_5_10_days'] = pd.to_datetime(full_cohort_7['c2_50per_charttime_5_10_days'], dayfirst=True)
# full_cohort_7['c1&c2_first_p_time_5_10_days'] = pd.to_datetime(full_cohort_7['c1&c2_first_p_time_5_10_days'], dayfirst=True)
#
# #print(full_cohort_7.columns.tolist())
# #['subject_id', 'hadm_id', 'stay_id', 'hep_start', 'icu_in_time_first_hep', 'icu_out_time_first_hep', 'first_careunit', 'last_careunit', 'admittime', 'dischtime', 'treatment_types', 'hep_types', 'admittime_1', 'dischtime_1', 'admission_type', 'admission_location', 'hospital_expire_flag', 'gender', 'anchor_age', 'base_p_charttime', 'base_platelets', 'c1_150k_charttime', 'c1_150k_platelets', 'HIT_c1_150k', 'c2_50per_charttime', 'c2_50per_platelets', 'HIT_c2_50%_drop', 'HIT_both_c1&c2', 'c1&c2_first_p_charttime', 'first_hep_to_first_HIT', 'first_hep_to_first_HIT_hrs', 'c1_150k_charttime_5_10_days', 'c1_150k_platelets_5_10_days', 'HIT_c1_150k_5_10_days', 'c2_50per_charttime_5_10_days', 'c2_50per_platelets_5_10_days', 'HIT_c2_50per_drop_5_10_days', 'HIT_both_c1_and_c2_5_10_days', 'c1&c2_first_p_time_5_10_days', 'first_hep_to_HIT_5_10_days', 'first_hep_to_HIT_hrs_5_10_days']

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# save output results into csv files

output_result_dir = '/Users/psenevirathn/Desktop/PhD/Coding/Python/output_csv_files'

save_full_list = os.path.join(output_result_dir,
                              'cohort_with_firstheptime_and_HITtime.csv')  # This Returns a path. os.path.join - https://www.geeksforgeeks.org/python-os-path-join-method/

data_to_save_with_outliers = full_cohort_7[['subject_id', 'hadm_id', 'stay_id', 'hep_start', 'icu_in_time_first_hep', 'icu_out_time_first_hep', 'first_careunit', 'last_careunit', 'admittime', 'dischtime', 'treatment_types', 'hep_types', 'admission_type', 'admission_location', 'hospital_expire_flag', 'gender', 'anchor_age', 'base_p_charttime', 'base_platelets', 'c1_150k_charttime_5_10_days', 'c1_150k_platelets_5_10_days', 'HIT_c1_150k_5_10_days', 'c2_50per_charttime_5_10_days', 'c2_50per_platelets_5_10_days', 'HIT_c2_50per_drop_5_10_days', 'HIT_both_c1_and_c2_5_10_days', 'c1&c2_first_p_time_5_10_days', 'first_hep_to_HIT_5_10_days', 'first_hep_to_HIT_hrs_5_10_days']]


# drop outlier data point (stay_id = 39817570)

data_after_dropping_outliers = data_to_save_with_outliers.drop(data_to_save_with_outliers[data_to_save_with_outliers['stay_id'] == 39817570].index)

print(data_to_save_with_outliers.shape)  # (13416, 29)
print(data_after_dropping_outliers.shape)  # (13415, 29)

#data_after_dropping_outliers.to_csv(save_full_list)

# Save train and test data separately, BEFORE pre-processing.

output_result_dir = '/Users/psenevirathn/Desktop/PhD/Coding/Python/output_csv_files'

save_full_list = os.path.join(output_result_dir,
                              'full_data_before_preprocessing.csv')  # This Returns a path. os.path.join - https://www.geeksforgeeks.org/python-os-path-join-method/

#full_feature_list_with_labels.to_csv(save_full_list)

### new

concat_train = pd.concat([x_train, y_train], axis=1)

output_result_dir = '/Users/psenevirathn/Desktop/PhD/Coding/Python/output_csv_files'

save_full_list = os.path.join(output_result_dir,
                              'train_data_before_preprocessing.csv')  # This Returns a path. os.path.join - https://www.geeksforgeeks.org/python-os-path-join-method/

#concat_train.to_csv(save_full_list)

concat_test = pd.concat([x_test, y_test], axis=1)

output_result_dir = '/Users/psenevirathn/Desktop/PhD/Coding/Python/output_csv_files'

save_full_list = os.path.join(output_result_dir,
                              'test_data_before_preprocessing.csv')  # This Returns a path. os.path.join - https://www.geeksforgeeks.org/python-os-path-join-method/

#concat_test.to_csv(save_full_list)
#
# # ------------------------------------------------------------------------------------------------------------------------------------

# 12. Convert Catgorical variables to numerical

# # first_careunit - 9 (distinct types)
# # admission_type - 9
# # admission_location - 11
# # gender - 2
# # hep_types - 2
# # treatment_types - 2

# Label encoding - columns -> Gender
# convert column 'Gender'to numeric (binary)

x_train = x_train.drop('hadm_id', axis=1)
x_test = x_test.drop('hadm_id', axis=1)


# OneHotEncoding - other categorical features columns
# first_careunit -0, admission_type - 1 , admission_location - 2, hep_types - 6, treatment_types - 7

# case #1 , #2
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['first_careunit','admission_type','admission_location','hep_types','treatment_types'])], remainder='passthrough') # transformers=[( kind of transformation-here encoding'', what kind of transformation we need to do, indexed of columns we need to encode)], what happen to remining columns

# case #3
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['first_careunit','admission_type','admission_location','hep_types','treatment_types','lactate_min_status', 'lactate_max_status'])], remainder='passthrough') # transformers=[( kind of transformation-here encoding'', what kind of transformation we need to do, indexed of columns we need to encode)], what happen to remining columns

# case #4
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['first_careunit','admission_type','admission_location','hep_types','treatment_types','lactate_min_status', 'lactate_max_status', "ph_min_status", "ph_max_status", "totalco2_bg_min_status", "totalco2_bg_max_status", "pco2_bg_min_status", "pco2_bg_max_status", "ast_min_status", "ast_max_status", "alp_min_status", "alp_max_status", "alt_min_status", "alt_max_status", "bilirubin_total_min_status", "bilirubin_total_max_status"])], remainder='passthrough') # transformers=[( kind of transformation-here encoding'', what kind of transformation we need to do, indexed of columns we need to encode)], what happen to remining columns

# case #5
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['first_careunit','admission_type','admission_location','hep_types','treatment_types','lactate_min_status', 'lactate_max_status', "ph_min_status", "ph_max_status", "totalco2_bg_min_status", "totalco2_bg_max_status", "pco2_bg_min_status", "pco2_bg_max_status", "ast_min_status", "ast_max_status", "alp_min_status", "alp_max_status", "alt_min_status", "alt_max_status", "bilirubin_total_min_status", "bilirubin_total_max_status", "albumin_min_status", "albumin_max_status"])], remainder='passthrough') # transformers=[( kind of transformation-here encoding'', what kind of transformation we need to do, indexed of columns we need to encode)], what happen to remining columns

# case 6
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['first_careunit','admission_type','admission_location','hep_types','treatment_types','lactate_min_status', 'lactate_max_status', "ph_min_status", "ph_max_status", "totalco2_bg_min_status", "totalco2_bg_max_status", "pco2_bg_min_status", "pco2_bg_max_status", "ast_min_status", "ast_max_status", "alp_min_status", "alp_max_status", "alt_min_status", "alt_max_status", "bilirubin_total_min_status", "bilirubin_total_max_status", "albumin_min_status", "albumin_max_status", "pco2_bg_art_min_status", "pco2_bg_art_max_status", "po2_bg_art_min_status", "po2_bg_art_max_status", "totalco2_bg_art_min_status", "totalco2_bg_art_max_status", "ld_ldh_min_status", "ld_ldh_max_status"])], remainder='passthrough') # transformers=[( kind of transformation-here encoding'', what kind of transformation we need to do, indexed of columns we need to encode)], what happen to remining columns

# case 7
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['first_careunit','admission_type','admission_location','hep_types','treatment_types','lactate_min_status', 'lactate_max_status', "ph_min_status", "ph_max_status", "totalco2_bg_min_status", "totalco2_bg_max_status", "pco2_bg_min_status", "pco2_bg_max_status", "ast_min_status", "ast_max_status", "alp_min_status", "alp_max_status", "alt_min_status", "alt_max_status", "bilirubin_total_min_status", "bilirubin_total_max_status", "albumin_min_status", "albumin_max_status", "pco2_bg_art_min_status", "pco2_bg_art_max_status", "po2_bg_art_min_status", "po2_bg_art_max_status", "totalco2_bg_art_min_status", "totalco2_bg_art_max_status", "ld_ldh_min_status", "ld_ldh_max_status", 'ck_cpk_min_status', 'ck_cpk_max_status', 'ck_mb_min_status', 'ck_mb_max_status'])], remainder='passthrough') # transformers=[( kind of transformation-here encoding'', what kind of transformation we need to do, indexed of columns we need to encode)], what happen to remining columns

# case 8
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['first_careunit','admission_type','admission_location','hep_types','treatment_types','lactate_min_status', 'lactate_max_status', "ph_min_status", "ph_max_status", "totalco2_bg_min_status", "totalco2_bg_max_status", "pco2_bg_min_status", "pco2_bg_max_status", "ast_min_status", "ast_max_status", "alp_min_status", "alp_max_status", "alt_min_status", "alt_max_status", "bilirubin_total_min_status", "bilirubin_total_max_status", "albumin_min_status", "albumin_max_status", "pco2_bg_art_min_status", "pco2_bg_art_max_status", "po2_bg_art_min_status", "po2_bg_art_max_status", "totalco2_bg_art_min_status", "totalco2_bg_art_max_status", "ld_ldh_min_status", "ld_ldh_max_status", 'ck_cpk_min_status', 'ck_cpk_max_status', 'ck_mb_min_status', 'ck_mb_max_status', "fio2_bg_art_min_status", "fio2_bg_art_max_status", "so2_bg_art_min_status", "so2_bg_art_max_status", "fibrinogen_min_status", "fibrinogen_max_status"])], remainder='passthrough') # transformers=[( kind of transformation-here encoding'', what kind of transformation we need to do, indexed of columns we need to encode)], what happen to remining columns

# case 9

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'),
                   ['first_careunit', 'admission_type', 'admission_location', 'gender', 'hep_types', 'treatment_types',
                    'thrombin_min_status', 'thrombin_max_status', 'd_dimer_max_status', 'd_dimer_min_status',
                    'methemoglobin_min_status', 'methemoglobin_max_status', 'ggt_min_status', 'ggt_max_status',
                    'globulin_min_status', 'globulin_max_status', 'total_protein_min_status',
                    'total_protein_max_status', 'atyps_max_status', 'atyps_min_status', 'carboxyhemoglobin_min_status',
                    'carboxyhemoglobin_max_status', 'amylase_max_status', 'amylase_min_status',
                    'aado2_bg_art_max_status', 'aado2_bg_art_min_status', 'bilirubin_direct_min_status',
                    'bilirubin_direct_max_status', 'bicarbonate_bg_min_status', 'bicarbonate_bg_max_status',
                    'fio2_bg_art_min_status', 'fio2_bg_art_max_status', 'nrbc_max_status', 'nrbc_min_status',
                    'bands_min_status', 'bands_max_status', 'so2_bg_art_min_status', 'so2_bg_art_max_status',
                    'fibrinogen_max_status', 'fibrinogen_min_status', 'hematocrit_bg_min_status',
                    'hematocrit_bg_max_status', 'hemoglobin_bg_min_status', 'hemoglobin_bg_max_status',
                    'temperature_bg_max_status', 'temperature_bg_min_status', 'chloride_bg_min_status',
                    'chloride_bg_max_status', 'sodium_bg_max_status', 'sodium_bg_min_status', 'glucose_bg_max_status',
                    'glucose_bg_min_status', 'ck_cpk_max_status', 'ck_cpk_min_status', 'ck_mb_max_status',
                    'ck_mb_min_status', 'ld_ldh_max_status', 'ld_ldh_min_status', 'calcium_bg_max_status',
                    'calcium_bg_min_status', 'pco2_bg_art_min_status', 'po2_bg_art_max_status',
                    'totalco2_bg_art_max_status', 'totalco2_bg_art_min_status', 'pco2_bg_art_max_status',
                    'po2_bg_art_min_status', 'potassium_bg_min_status', 'potassium_bg_max_status', 'albumin_max_status',
                    'albumin_min_status', 'bilirubin_total_min_status', 'bilirubin_total_max_status', 'alt_max_status',
                    'alt_min_status', 'alp_max_status', 'alp_min_status', 'ast_min_status', 'ast_max_status',
                    'pco2_bg_max_status', 'pco2_bg_min_status', 'totalco2_bg_min_status', 'totalco2_bg_max_status',
                    'ph_min_status', 'ph_max_status', 'lactate_min_status', 'lactate_max_status'])],
    remainder='passthrough')  # transformers=[( kind of transformation-here encoding'', what kind of transformation we need to do, indexed of columns we need to encode)], what happen to remining columns

# We can use - OneHotEncoder(handle_unknown = 'ignore'), or ‘infrequent_if_exist’ - to ignore new values in test set, which were not in training set.
# Only one column had this issue.  column = 'admission_type'  / attribute = 'AMBULATORY OBSERVATION'
x_train = np.array(ct.fit_transform(
    x_train))  # here 'np' (NumPy) was added because, fit_transform itself doesn't return output in np array, so in order to train future machine learning models, np is added.

print(x_train.shape)  # (10732, 281)

x_test = np.array(ct.transform(x_test))  # handle_unknown = 'ignore'
print(x_test.shape)  # (2683, 281)

# print('new2')
x_axis_original = ct.get_feature_names_out().tolist()  # numerical strats from index 222
print(x_axis_original)

# ['encoder__first_careunit_Coronary Care Unit (CCU)', 'encoder__first_careunit_Medical Intensive Care Unit (MICU)', 'encoder__first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)', 'encoder__first_careunit_Neuro Intermediate', 'encoder__first_careunit_Neuro Stepdown', 'encoder__first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)', 'encoder__first_careunit_Surgical Intensive Care Unit (SICU)', 'encoder__first_careunit_Trauma SICU (TSICU)', 'encoder__admission_type_DIRECT OBSERVATION', 'encoder__admission_type_ELECTIVE', 'encoder__admission_type_EU OBSERVATION', 'encoder__admission_type_EW EMER.', 'encoder__admission_type_OBSERVATION ADMIT', 'encoder__admission_type_SURGICAL SAME DAY ADMISSION', 'encoder__admission_type_URGENT', 'encoder__admission_location_CLINIC REFERRAL', 'encoder__admission_location_EMERGENCY ROOM', 'encoder__admission_location_INFORMATION NOT AVAILABLE', 'encoder__admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'encoder__admission_location_PACU', 'encoder__admission_location_PHYSICIAN REFERRAL', 'encoder__admission_location_PROCEDURE SITE', 'encoder__admission_location_TRANSFER FROM HOSPITAL', 'encoder__admission_location_TRANSFER FROM SKILLED NURSING FACILITY', 'encoder__admission_location_WALK-IN/SELF REFERRAL', 'encoder__gender_M', 'encoder__hep_types_UFH', 'encoder__treatment_types_T', 'encoder__thrombin_min_status_normal', 'encoder__thrombin_min_status_not ordered', 'encoder__thrombin_max_status_normal', 'encoder__thrombin_max_status_not ordered', 'encoder__d_dimer_max_status_normal', 'encoder__d_dimer_max_status_not ordered', 'encoder__d_dimer_min_status_normal', 'encoder__d_dimer_min_status_not ordered', 'encoder__methemoglobin_min_status_normal', 'encoder__methemoglobin_min_status_not ordered', 'encoder__methemoglobin_max_status_normal', 'encoder__methemoglobin_max_status_not ordered', 'encoder__ggt_min_status_low', 'encoder__ggt_min_status_normal', 'encoder__ggt_min_status_not ordered', 'encoder__ggt_max_status_low', 'encoder__ggt_max_status_normal', 'encoder__ggt_max_status_not ordered', 'encoder__globulin_min_status_low', 'encoder__globulin_min_status_normal', 'encoder__globulin_min_status_not ordered', 'encoder__globulin_max_status_low', 'encoder__globulin_max_status_normal', 'encoder__globulin_max_status_not ordered', 'encoder__total_protein_min_status_low', 'encoder__total_protein_min_status_normal', 'encoder__total_protein_min_status_not ordered', 'encoder__total_protein_max_status_low', 'encoder__total_protein_max_status_normal', 'encoder__total_protein_max_status_not ordered', 'encoder__atyps_max_status_normal', 'encoder__atyps_max_status_not ordered', 'encoder__atyps_min_status_normal', 'encoder__atyps_min_status_not ordered', 'encoder__carboxyhemoglobin_min_status_normal', 'encoder__carboxyhemoglobin_min_status_not ordered', 'encoder__carboxyhemoglobin_max_status_normal', 'encoder__carboxyhemoglobin_max_status_not ordered', 'encoder__amylase_max_status_normal', 'encoder__amylase_max_status_not ordered', 'encoder__amylase_min_status_normal', 'encoder__amylase_min_status_not ordered', 'encoder__aado2_bg_art_max_status_not ordered', 'encoder__aado2_bg_art_min_status_not ordered', 'encoder__bilirubin_direct_min_status_normal', 'encoder__bilirubin_direct_min_status_not ordered', 'encoder__bilirubin_direct_max_status_normal', 'encoder__bilirubin_direct_max_status_not ordered', 'encoder__bicarbonate_bg_min_status_low', 'encoder__bicarbonate_bg_min_status_normal', 'encoder__bicarbonate_bg_min_status_not ordered', 'encoder__bicarbonate_bg_max_status_low', 'encoder__bicarbonate_bg_max_status_normal', 'encoder__bicarbonate_bg_max_status_not ordered', 'encoder__fio2_bg_art_min_status_not ordered', 'encoder__fio2_bg_art_max_status_not ordered', 'encoder__nrbc_max_status_normal', 'encoder__nrbc_max_status_not ordered', 'encoder__nrbc_min_status_normal', 'encoder__nrbc_min_status_not ordered', 'encoder__bands_min_status_normal', 'encoder__bands_min_status_not ordered', 'encoder__bands_max_status_normal', 'encoder__bands_max_status_not ordered', 'encoder__so2_bg_art_min_status_not ordered', 'encoder__so2_bg_art_max_status_not ordered', 'encoder__fibrinogen_max_status_low', 'encoder__fibrinogen_max_status_normal', 'encoder__fibrinogen_max_status_not ordered', 'encoder__fibrinogen_min_status_low', 'encoder__fibrinogen_min_status_normal', 'encoder__fibrinogen_min_status_not ordered', 'encoder__hematocrit_bg_min_status_not ordered', 'encoder__hematocrit_bg_max_status_not ordered', 'encoder__hemoglobin_bg_min_status_low', 'encoder__hemoglobin_bg_min_status_normal', 'encoder__hemoglobin_bg_min_status_not ordered', 'encoder__hemoglobin_bg_max_status_low', 'encoder__hemoglobin_bg_max_status_normal', 'encoder__hemoglobin_bg_max_status_not ordered', 'encoder__temperature_bg_max_status_not ordered', 'encoder__temperature_bg_min_status_not ordered', 'encoder__chloride_bg_min_status_low', 'encoder__chloride_bg_min_status_normal', 'encoder__chloride_bg_min_status_not ordered', 'encoder__chloride_bg_max_status_low', 'encoder__chloride_bg_max_status_normal', 'encoder__chloride_bg_max_status_not ordered', 'encoder__sodium_bg_max_status_low', 'encoder__sodium_bg_max_status_normal', 'encoder__sodium_bg_max_status_not ordered', 'encoder__sodium_bg_min_status_low', 'encoder__sodium_bg_min_status_normal', 'encoder__sodium_bg_min_status_not ordered', 'encoder__glucose_bg_max_status_low', 'encoder__glucose_bg_max_status_normal', 'encoder__glucose_bg_max_status_not ordered', 'encoder__glucose_bg_min_status_low', 'encoder__glucose_bg_min_status_normal', 'encoder__glucose_bg_min_status_not ordered', 'encoder__ck_cpk_max_status_low', 'encoder__ck_cpk_max_status_normal', 'encoder__ck_cpk_max_status_not ordered', 'encoder__ck_cpk_min_status_low', 'encoder__ck_cpk_min_status_normal', 'encoder__ck_cpk_min_status_not ordered', 'encoder__ck_mb_max_status_normal', 'encoder__ck_mb_max_status_not ordered', 'encoder__ck_mb_min_status_normal', 'encoder__ck_mb_min_status_not ordered', 'encoder__ld_ldh_max_status_low', 'encoder__ld_ldh_max_status_normal', 'encoder__ld_ldh_max_status_not ordered', 'encoder__ld_ldh_min_status_low', 'encoder__ld_ldh_min_status_normal', 'encoder__ld_ldh_min_status_not ordered', 'encoder__calcium_bg_max_status_low', 'encoder__calcium_bg_max_status_normal', 'encoder__calcium_bg_max_status_not ordered', 'encoder__calcium_bg_min_status_low', 'encoder__calcium_bg_min_status_normal', 'encoder__calcium_bg_min_status_not ordered', 'encoder__pco2_bg_art_min_status_low', 'encoder__pco2_bg_art_min_status_normal', 'encoder__pco2_bg_art_min_status_not ordered', 'encoder__po2_bg_art_max_status_low', 'encoder__po2_bg_art_max_status_normal', 'encoder__po2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_max_status_low', 'encoder__totalco2_bg_art_max_status_normal', 'encoder__totalco2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_min_status_low', 'encoder__totalco2_bg_art_min_status_normal', 'encoder__totalco2_bg_art_min_status_not ordered', 'encoder__pco2_bg_art_max_status_low', 'encoder__pco2_bg_art_max_status_normal', 'encoder__pco2_bg_art_max_status_not ordered', 'encoder__po2_bg_art_min_status_low', 'encoder__po2_bg_art_min_status_normal', 'encoder__po2_bg_art_min_status_not ordered', 'encoder__potassium_bg_min_status_low', 'encoder__potassium_bg_min_status_normal', 'encoder__potassium_bg_min_status_not ordered', 'encoder__potassium_bg_max_status_low', 'encoder__potassium_bg_max_status_normal', 'encoder__potassium_bg_max_status_not ordered', 'encoder__albumin_max_status_low', 'encoder__albumin_max_status_normal', 'encoder__albumin_max_status_not ordered', 'encoder__albumin_min_status_low', 'encoder__albumin_min_status_normal', 'encoder__albumin_min_status_not ordered', 'encoder__bilirubin_total_min_status_normal', 'encoder__bilirubin_total_min_status_not ordered', 'encoder__bilirubin_total_max_status_normal', 'encoder__bilirubin_total_max_status_not ordered', 'encoder__alt_max_status_normal', 'encoder__alt_max_status_not ordered', 'encoder__alt_min_status_normal', 'encoder__alt_min_status_not ordered', 'encoder__alp_max_status_low', 'encoder__alp_max_status_normal', 'encoder__alp_max_status_not ordered', 'encoder__alp_min_status_low', 'encoder__alp_min_status_normal', 'encoder__alp_min_status_not ordered', 'encoder__ast_min_status_normal', 'encoder__ast_min_status_not ordered', 'encoder__ast_max_status_normal', 'encoder__ast_max_status_not ordered', 'encoder__pco2_bg_max_status_low', 'encoder__pco2_bg_max_status_normal', 'encoder__pco2_bg_max_status_not ordered', 'encoder__pco2_bg_min_status_low', 'encoder__pco2_bg_min_status_normal', 'encoder__pco2_bg_min_status_not ordered', 'encoder__totalco2_bg_min_status_low', 'encoder__totalco2_bg_min_status_normal', 'encoder__totalco2_bg_min_status_not ordered', 'encoder__totalco2_bg_max_status_low', 'encoder__totalco2_bg_max_status_normal', 'encoder__totalco2_bg_max_status_not ordered', 'encoder__ph_min_status_low', 'encoder__ph_min_status_normal', 'encoder__ph_min_status_not ordered', 'encoder__ph_max_status_low', 'encoder__ph_max_status_normal', 'encoder__ph_max_status_not ordered', 'encoder__lactate_min_status_low', 'encoder__lactate_min_status_normal', 'encoder__lactate_min_status_not ordered', 'encoder__lactate_max_status_low', 'encoder__lactate_max_status_normal', 'encoder__lactate_max_status_not ordered', 'remainder__anchor_age', 'remainder__base_platelets', 'remainder__heart_rate_min', 'remainder__heart_rate_max', 'remainder__heart_rate_mean', 'remainder__sbp_min', 'remainder__sbp_max', 'remainder__sbp_mean', 'remainder__dbp_min', 'remainder__dbp_max', 'remainder__dbp_mean', 'remainder__mbp_min', 'remainder__mbp_max', 'remainder__mbp_mean', 'remainder__resp_rate_min', 'remainder__resp_rate_max', 'remainder__resp_rate_mean', 'remainder__spo2_min', 'remainder__spo2_max', 'remainder__spo2_mean', 'remainder__temperature_vital_min', 'remainder__temperature_vital_max', 'remainder__temperature_vital_mean', 'remainder__glucose_vital_min', 'remainder__glucose_vital_max', 'remainder__glucose_vital_mean', 'remainder__hematocrit_lab_min', 'remainder__hematocrit_lab_max', 'remainder__hemoglobin_lab_min', 'remainder__hemoglobin_lab_max', 'remainder__bicarbonate_lab_min', 'remainder__bicarbonate_lab_max', 'remainder__calcium_lab_min', 'remainder__calcium_lab_max', 'remainder__chloride_lab_min', 'remainder__chloride_lab_max', 'remainder__sodium_lab_min', 'remainder__sodium_lab_max', 'remainder__potassium_lab_min', 'remainder__potassium_lab_max', 'remainder__glucose_lab_min', 'remainder__glucose_lab_max', 'remainder__platelets_min', 'remainder__platelets_max', 'remainder__wbc_min', 'remainder__wbc_max', 'remainder__aniongap_min', 'remainder__aniongap_max', 'remainder__bun_min', 'remainder__bun_max', 'remainder__creatinine_min', 'remainder__creatinine_max', 'remainder__inr_min', 'remainder__inr_max', 'remainder__pt_min', 'remainder__pt_max', 'remainder__ptt_min', 'remainder__ptt_max', 'remainder__gcs_min']#
#------------------------------------------------------------------------------------------------------------------------------------
#
# 13. Impute Missing values
## here all missing values are numerical features. Non of the categorical feaures had missing values.

imputer = KNNImputer(n_neighbors=2)
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# (imputer.fit_transform(X_s[i][greedy_23]), columns = greedy_23)

# # ------------------------------------------------------------------------------------------------------------------------------------
#
# # # Feature scaling
# #
# # # first 242 columns - converted from categorical features -> 1-241 (index 0-240) -> from OneHotEncoding , 242th column (Gender) , i.e. index = 241 - LabelEnconding
# # #
# # sc = StandardScaler()
# #
# # x_train[:, 242:] = sc.fit_transform(x_train[:, 242:])  # Here feature scaling not applied to dummy columns(first 3 columns), i.e. for France = 100,Spain=010 and Germany=001, because those column values are alread in between -3 and 3, and also, if feature scaling do to these columns, abnormal values may return
# # # Here 'fit method' calculate ,mean and the standard devation of each feature. 'Transform method' apply equation, { Xstand=[x-mean(x)]/standard devation(x) , where x -feature, here have to categoroed for x, which is salary and ange. which called 'Standarization'}, for each feature.
# #
# # x_test[:, 242:] = sc.transform(x_test[:,
# #                                242:])  # Here, when do feature scaling in test set, test set should be scaled by using the same parameters used in training set.
# # # Also, x_test is the input for the prediction function got from training set. That's why here only transform method is using instead fit_transform.
# # # Means, here when apply standarization to each of two features (age and salary), the mean and the standard deviation used is the values got from training data. >> Xstand_test=[x_test-mean(x_train)]/standard devation(x_train)
#
# # ------------------------------------------------------------------------------------------------------------------------------------
# ### test - checking whether feature scaling is required or not
#
# dataset1 = pd.DataFrame(x_train[:, 242:])
# print(dataset1.columns.tolist())
#
# # summarize
#
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)  # None
#
# print("new line")
# print(dataset1.describe())
#
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# commented from here

# In x_train and x_test, first 222 (index - 0:221) columns came from oneHot Encoding of categorical features.
# There we don't do feature scaling for them, as they are already binary
sc = MinMaxScaler()

print(x_axis_original[220])  # encoder__lactate_max_status_normal
print(x_axis_original[221])  # encoder__lactate_max_status_not ordered
print(x_axis_original[222])  # remainder__anchor_age

x_train[:, 222:] = sc.fit_transform(x_train[:,
                                    222:])  # Here feature scaling not applied to dummy columns(first 3 columns), i.e. for France = 100,Spain=010 and Germany=001, because those column values are alread in between -3 and 3, and also, if feature scaling do to these columns, abnormal values may return
# Here 'fit method' calculate ,mean and the standard devation of each feature. 'Transform method' apply equation, { Xstand=[x-mean(x)]/standard devation(x) , where x -feature, here have to categoroed for x, which is salary and ange. which called 'Standarization'}, for each feature.

x_test[:, 222:] = sc.transform(x_test[:,
                               222:])  # Here, when do feature scaling in test set, test set should be scaled by using the same parameters used in training set.
# Also, x_test is the input for the prediction function got from training set. That's why here only transform method is using instead fit_transform.
# Means, here when apply standarization to each of two features (age and salary), the mean and the standard deviation used is the values got from training data. >> Xstand_test=[x_test-mean(x_train)]/standard devation(x_train)

# # ----------------------------------------------------------------------------------------------------------------------------------
# Save train and test data separately, AFTER pre-processing.

col_names = ['encoder__first_careunit_Coronary Care Unit (CCU)', 'encoder__first_careunit_Medical Intensive Care Unit (MICU)', 'encoder__first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)', 'encoder__first_careunit_Neuro Intermediate', 'encoder__first_careunit_Neuro Stepdown', 'encoder__first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)', 'encoder__first_careunit_Surgical Intensive Care Unit (SICU)', 'encoder__first_careunit_Trauma SICU (TSICU)', 'encoder__admission_type_DIRECT OBSERVATION', 'encoder__admission_type_ELECTIVE', 'encoder__admission_type_EU OBSERVATION', 'encoder__admission_type_EW EMER.', 'encoder__admission_type_OBSERVATION ADMIT', 'encoder__admission_type_SURGICAL SAME DAY ADMISSION', 'encoder__admission_type_URGENT', 'encoder__admission_location_CLINIC REFERRAL', 'encoder__admission_location_EMERGENCY ROOM', 'encoder__admission_location_INFORMATION NOT AVAILABLE', 'encoder__admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'encoder__admission_location_PACU', 'encoder__admission_location_PHYSICIAN REFERRAL', 'encoder__admission_location_PROCEDURE SITE', 'encoder__admission_location_TRANSFER FROM HOSPITAL', 'encoder__admission_location_TRANSFER FROM SKILLED NURSING FACILITY', 'encoder__admission_location_WALK-IN/SELF REFERRAL', 'encoder__gender_M', 'encoder__hep_types_UFH', 'encoder__treatment_types_T', 'encoder__thrombin_min_status_normal', 'encoder__thrombin_min_status_not ordered', 'encoder__thrombin_max_status_normal', 'encoder__thrombin_max_status_not ordered', 'encoder__d_dimer_max_status_normal', 'encoder__d_dimer_max_status_not ordered', 'encoder__d_dimer_min_status_normal', 'encoder__d_dimer_min_status_not ordered', 'encoder__methemoglobin_min_status_normal', 'encoder__methemoglobin_min_status_not ordered', 'encoder__methemoglobin_max_status_normal', 'encoder__methemoglobin_max_status_not ordered', 'encoder__ggt_min_status_low', 'encoder__ggt_min_status_normal', 'encoder__ggt_min_status_not ordered', 'encoder__ggt_max_status_low', 'encoder__ggt_max_status_normal', 'encoder__ggt_max_status_not ordered', 'encoder__globulin_min_status_low', 'encoder__globulin_min_status_normal', 'encoder__globulin_min_status_not ordered', 'encoder__globulin_max_status_low', 'encoder__globulin_max_status_normal', 'encoder__globulin_max_status_not ordered', 'encoder__total_protein_min_status_low', 'encoder__total_protein_min_status_normal', 'encoder__total_protein_min_status_not ordered', 'encoder__total_protein_max_status_low', 'encoder__total_protein_max_status_normal', 'encoder__total_protein_max_status_not ordered', 'encoder__atyps_max_status_normal', 'encoder__atyps_max_status_not ordered', 'encoder__atyps_min_status_normal', 'encoder__atyps_min_status_not ordered', 'encoder__carboxyhemoglobin_min_status_normal', 'encoder__carboxyhemoglobin_min_status_not ordered', 'encoder__carboxyhemoglobin_max_status_normal', 'encoder__carboxyhemoglobin_max_status_not ordered', 'encoder__amylase_max_status_normal', 'encoder__amylase_max_status_not ordered', 'encoder__amylase_min_status_normal', 'encoder__amylase_min_status_not ordered', 'encoder__aado2_bg_art_max_status_not ordered', 'encoder__aado2_bg_art_min_status_not ordered', 'encoder__bilirubin_direct_min_status_normal', 'encoder__bilirubin_direct_min_status_not ordered', 'encoder__bilirubin_direct_max_status_normal', 'encoder__bilirubin_direct_max_status_not ordered', 'encoder__bicarbonate_bg_min_status_low', 'encoder__bicarbonate_bg_min_status_normal', 'encoder__bicarbonate_bg_min_status_not ordered', 'encoder__bicarbonate_bg_max_status_low', 'encoder__bicarbonate_bg_max_status_normal', 'encoder__bicarbonate_bg_max_status_not ordered', 'encoder__fio2_bg_art_min_status_not ordered', 'encoder__fio2_bg_art_max_status_not ordered', 'encoder__nrbc_max_status_normal', 'encoder__nrbc_max_status_not ordered', 'encoder__nrbc_min_status_normal', 'encoder__nrbc_min_status_not ordered', 'encoder__bands_min_status_normal', 'encoder__bands_min_status_not ordered', 'encoder__bands_max_status_normal', 'encoder__bands_max_status_not ordered', 'encoder__so2_bg_art_min_status_not ordered', 'encoder__so2_bg_art_max_status_not ordered', 'encoder__fibrinogen_max_status_low', 'encoder__fibrinogen_max_status_normal', 'encoder__fibrinogen_max_status_not ordered', 'encoder__fibrinogen_min_status_low', 'encoder__fibrinogen_min_status_normal', 'encoder__fibrinogen_min_status_not ordered', 'encoder__hematocrit_bg_min_status_not ordered', 'encoder__hematocrit_bg_max_status_not ordered', 'encoder__hemoglobin_bg_min_status_low', 'encoder__hemoglobin_bg_min_status_normal', 'encoder__hemoglobin_bg_min_status_not ordered', 'encoder__hemoglobin_bg_max_status_low', 'encoder__hemoglobin_bg_max_status_normal', 'encoder__hemoglobin_bg_max_status_not ordered', 'encoder__temperature_bg_max_status_not ordered', 'encoder__temperature_bg_min_status_not ordered', 'encoder__chloride_bg_min_status_low', 'encoder__chloride_bg_min_status_normal', 'encoder__chloride_bg_min_status_not ordered', 'encoder__chloride_bg_max_status_low', 'encoder__chloride_bg_max_status_normal', 'encoder__chloride_bg_max_status_not ordered', 'encoder__sodium_bg_max_status_low', 'encoder__sodium_bg_max_status_normal', 'encoder__sodium_bg_max_status_not ordered', 'encoder__sodium_bg_min_status_low', 'encoder__sodium_bg_min_status_normal', 'encoder__sodium_bg_min_status_not ordered', 'encoder__glucose_bg_max_status_low', 'encoder__glucose_bg_max_status_normal', 'encoder__glucose_bg_max_status_not ordered', 'encoder__glucose_bg_min_status_low', 'encoder__glucose_bg_min_status_normal', 'encoder__glucose_bg_min_status_not ordered', 'encoder__ck_cpk_max_status_low', 'encoder__ck_cpk_max_status_normal', 'encoder__ck_cpk_max_status_not ordered', 'encoder__ck_cpk_min_status_low', 'encoder__ck_cpk_min_status_normal', 'encoder__ck_cpk_min_status_not ordered', 'encoder__ck_mb_max_status_normal', 'encoder__ck_mb_max_status_not ordered', 'encoder__ck_mb_min_status_normal', 'encoder__ck_mb_min_status_not ordered', 'encoder__ld_ldh_max_status_low', 'encoder__ld_ldh_max_status_normal', 'encoder__ld_ldh_max_status_not ordered', 'encoder__ld_ldh_min_status_low', 'encoder__ld_ldh_min_status_normal', 'encoder__ld_ldh_min_status_not ordered', 'encoder__calcium_bg_max_status_low', 'encoder__calcium_bg_max_status_normal', 'encoder__calcium_bg_max_status_not ordered', 'encoder__calcium_bg_min_status_low', 'encoder__calcium_bg_min_status_normal', 'encoder__calcium_bg_min_status_not ordered', 'encoder__pco2_bg_art_min_status_low', 'encoder__pco2_bg_art_min_status_normal', 'encoder__pco2_bg_art_min_status_not ordered', 'encoder__po2_bg_art_max_status_low', 'encoder__po2_bg_art_max_status_normal', 'encoder__po2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_max_status_low', 'encoder__totalco2_bg_art_max_status_normal', 'encoder__totalco2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_min_status_low', 'encoder__totalco2_bg_art_min_status_normal', 'encoder__totalco2_bg_art_min_status_not ordered', 'encoder__pco2_bg_art_max_status_low', 'encoder__pco2_bg_art_max_status_normal', 'encoder__pco2_bg_art_max_status_not ordered', 'encoder__po2_bg_art_min_status_low', 'encoder__po2_bg_art_min_status_normal', 'encoder__po2_bg_art_min_status_not ordered', 'encoder__potassium_bg_min_status_low', 'encoder__potassium_bg_min_status_normal', 'encoder__potassium_bg_min_status_not ordered', 'encoder__potassium_bg_max_status_low', 'encoder__potassium_bg_max_status_normal', 'encoder__potassium_bg_max_status_not ordered', 'encoder__albumin_max_status_low', 'encoder__albumin_max_status_normal', 'encoder__albumin_max_status_not ordered', 'encoder__albumin_min_status_low', 'encoder__albumin_min_status_normal', 'encoder__albumin_min_status_not ordered', 'encoder__bilirubin_total_min_status_normal', 'encoder__bilirubin_total_min_status_not ordered', 'encoder__bilirubin_total_max_status_normal', 'encoder__bilirubin_total_max_status_not ordered', 'encoder__alt_max_status_normal', 'encoder__alt_max_status_not ordered', 'encoder__alt_min_status_normal', 'encoder__alt_min_status_not ordered', 'encoder__alp_max_status_low', 'encoder__alp_max_status_normal', 'encoder__alp_max_status_not ordered', 'encoder__alp_min_status_low', 'encoder__alp_min_status_normal', 'encoder__alp_min_status_not ordered', 'encoder__ast_min_status_normal', 'encoder__ast_min_status_not ordered', 'encoder__ast_max_status_normal', 'encoder__ast_max_status_not ordered', 'encoder__pco2_bg_max_status_low', 'encoder__pco2_bg_max_status_normal', 'encoder__pco2_bg_max_status_not ordered', 'encoder__pco2_bg_min_status_low', 'encoder__pco2_bg_min_status_normal', 'encoder__pco2_bg_min_status_not ordered', 'encoder__totalco2_bg_min_status_low', 'encoder__totalco2_bg_min_status_normal', 'encoder__totalco2_bg_min_status_not ordered', 'encoder__totalco2_bg_max_status_low', 'encoder__totalco2_bg_max_status_normal', 'encoder__totalco2_bg_max_status_not ordered', 'encoder__ph_min_status_low', 'encoder__ph_min_status_normal', 'encoder__ph_min_status_not ordered', 'encoder__ph_max_status_low', 'encoder__ph_max_status_normal', 'encoder__ph_max_status_not ordered', 'encoder__lactate_min_status_low', 'encoder__lactate_min_status_normal', 'encoder__lactate_min_status_not ordered', 'encoder__lactate_max_status_low', 'encoder__lactate_max_status_normal', 'encoder__lactate_max_status_not ordered', 'remainder__anchor_age', 'remainder__base_platelets', 'remainder__heart_rate_min', 'remainder__heart_rate_max', 'remainder__heart_rate_mean', 'remainder__sbp_min', 'remainder__sbp_max', 'remainder__sbp_mean', 'remainder__dbp_min', 'remainder__dbp_max', 'remainder__dbp_mean', 'remainder__mbp_min', 'remainder__mbp_max', 'remainder__mbp_mean', 'remainder__resp_rate_min', 'remainder__resp_rate_max', 'remainder__resp_rate_mean', 'remainder__spo2_min', 'remainder__spo2_max', 'remainder__spo2_mean', 'remainder__temperature_vital_min', 'remainder__temperature_vital_max', 'remainder__temperature_vital_mean', 'remainder__glucose_vital_min', 'remainder__glucose_vital_max', 'remainder__glucose_vital_mean', 'remainder__hematocrit_lab_min', 'remainder__hematocrit_lab_max', 'remainder__hemoglobin_lab_min', 'remainder__hemoglobin_lab_max', 'remainder__bicarbonate_lab_min', 'remainder__bicarbonate_lab_max', 'remainder__calcium_lab_min', 'remainder__calcium_lab_max', 'remainder__chloride_lab_min', 'remainder__chloride_lab_max', 'remainder__sodium_lab_min', 'remainder__sodium_lab_max', 'remainder__potassium_lab_min', 'remainder__potassium_lab_max', 'remainder__glucose_lab_min', 'remainder__glucose_lab_max', 'remainder__platelets_min', 'remainder__platelets_max', 'remainder__wbc_min', 'remainder__wbc_max', 'remainder__aniongap_min', 'remainder__aniongap_max', 'remainder__bun_min', 'remainder__bun_max', 'remainder__creatinine_min', 'remainder__creatinine_max', 'remainder__inr_min', 'remainder__inr_max', 'remainder__pt_min', 'remainder__pt_max', 'remainder__ptt_min', 'remainder__ptt_max', 'remainder__gcs_min', 'label']
# save train set

y1_train = np.array([y_train])
y2_train = np.transpose(y1_train)
concat_xtrain_and_ytrain = np.concatenate((x_train, y2_train), axis=1)

train_set = pd.DataFrame(concat_xtrain_and_ytrain, columns=col_names)

output_result_dir = '/Users/psenevirathn/Desktop/PhD/Coding/Python/output_csv_files'

save_train_loc = os.path.join(output_result_dir,
                              'train_set_AFTER_preprocessing.csv')  # This Returns a path. os.path.join - https://www.geeksforgeeks.org/python-os-path-join-method/

#train_set.to_csv(save_train_loc)

# save test set

y1_test = np.array([y_test])  # (1, 2698)
y2_test = np.transpose(y1_test)  # (2698, 1)
concat_xtest_and_ytest = np.concatenate((x_test, y2_test), axis=1)

test_set = pd.DataFrame(concat_xtest_and_ytest, columns=col_names)

# output_result_dir = '/Users/psenevirathn/Desktop/PhD/Coding/Python/output_csv_files'

save_test_loc = os.path.join(output_result_dir,
                             'test_set_AFTER_preprocessing.csv')  # This Returns a path. os.path.join - https://www.geeksforgeeks.org/python-os-path-join-method/

#test_set.to_csv(save_test_loc)

# # ------------------------------------------------------------------------------------------------------------------------------------
#
# # 15. ML Predictors
#
# # ------------------------------------------------------------------------------------------------------------------------------------
#

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(full_cohort_11[full_cohort_11['hadm_id'] == 20415003])

## 15.1. ML model 1 - Naive-Bayes

print('Naive-Bayes')
# Training the Naive-Bayes:
classifier_NB = GaussianNB()
classifier_NB.fit(x_train, y_train)

# Predict the classifier response for the Test dataset:
y_pred_NB = classifier_NB.predict(x_test)

# len(y_test) - 22206
print(len(y_pred_NB[y_pred_NB == 0]))
print(len(y_test[y_test == 0]))

## Evaluate the Performance of blind test
blind_cm_NB = confusion_matrix(y_test, y_pred_NB)  # cm for confusion matrix , len(y_test) - 22206
print(blind_cm_NB)

blind_acc_NB = float(
    round(balanced_accuracy_score(y_test, y_pred_NB), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
print(blind_acc_NB)

blind_recall_NB = float(round(recall_score(y_test, y_pred_NB), 3))  # tp / (tp + fn)
print(blind_recall_NB)

blind_precision_NB = float(round(precision_score(y_test, y_pred_NB), 3))  # tp / (tp + fp)
print(blind_precision_NB)

blind_f1_NB = float(round(f1_score(y_test, y_pred_NB),
                          3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# It is primarily used to compare the performance of two classifiers.
# Suppose that classifier A has a higher recall, and classifier B has higher precision.
# In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
print(blind_f1_NB)

blind__mcc_NB = float(round(matthews_corrcoef(y_test, y_pred_NB),
                            3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
print(blind__mcc_NB)

blind_AUC_NB = float(round(roc_auc_score(y_test, (classifier_NB.predict_proba(x_test)[:, 1])), 3))
print(blind_AUC_NB)
# area under the ROC curve, which is the curve having False Positive Rate on the x-axis and True Positive Rate on the y-axis at all classification thresholds.

blind_test_NB = [blind_acc_NB, blind_recall_NB, blind_precision_NB, blind_f1_NB, blind__mcc_NB, blind_AUC_NB]
print(blind_test_NB)

# roc

y_pred_proba_NB = classifier_NB.predict_proba(x_test)[::,
                  1]  # Start at the beginning, end when it ends, walk in steps of 1 , # first col is prob of y=0, while 2nd col is prob of y=1 . https://dev.to/rajat_naegi/simply-explained-predictproba-263i

# roc_auc_score(y, clf.predict_proba(X)[:, 1])
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_NB)
auc = roc_auc_score(y_test, y_pred_proba_NB)

# plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
# plt.legend(loc=4)
# plt.show()

# Number of Folds to split the data:
# folds = 10 # not stratified

folds = StratifiedKFold(n_splits=10, shuffle=True,
                        random_state=0)  # why 'shuffle' parameter - https://stackoverflow.com/questions/63236831/shuffle-parameter-in-sklearn-model-selection-stratifiedkfold

# Call the function of cross-validation passing the parameters:
cross_accuracy_all_NB = cross_val_score(estimator=classifier_NB, X=x_train, y=y_train, cv=folds,
                                        scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.

cross_precision_all_NB = cross_val_score(estimator=classifier_NB, X=x_train, y=y_train, cv=folds, scoring='precision')

cross_recall_all_NB = cross_val_score(estimator=classifier_NB, X=x_train, y=y_train, cv=folds, scoring='recall')

cross_f1_all_NB = cross_val_score(estimator=classifier_NB, X=x_train, y=y_train, cv=folds, scoring='f1')

# no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
mcc = make_scorer(matthews_corrcoef)
cross_mcc_all_NB = cross_val_score(estimator=classifier_NB, X=x_train, y=y_train, cv=folds, scoring=mcc)

cross_AUC_all_NB = cross_val_score(estimator=classifier_NB, X=x_train, y=y_train, cv=folds, scoring='roc_auc')

cross_validation_NB = [round((cross_accuracy_all_NB.mean()), 3), round((cross_recall_all_NB.mean()), 3),
                       round((cross_precision_all_NB.mean()), 3), round((cross_f1_all_NB.mean()), 3),
                       round((cross_mcc_all_NB.mean()), 3), round((cross_AUC_all_NB.mean()), 3)]
print(cross_validation_NB)

# ------------------------------------------------------------------------------------------------------------------------------------

## 15.2 ML model 2 - KNN Classifier

# KNN
print('KNN')
## How to choose the best number of neighbours? Let's create a range and see it!

k_values = range(1, 10)
KNN_MCC = []

for n in k_values:
    classifier_KNN = KNeighborsClassifier(n_neighbors=n)
    model_KNN = classifier_KNN.fit(x_train, y_train)

    # Predict the classifier's responses for the Test dataset
    y_pred_KNN = model_KNN.predict(x_test)

    # Evaluate using MCC:
    KNN_MCC.append(float(round(matthews_corrcoef(y_test, y_pred_KNN), 3)))

print(KNN_MCC)

##Visualise how the MCC metric varies with different values of Neighbors:
plt.plot(k_values, KNN_MCC)
plt.xlabel("Number of Neighbours")
plt.ylabel("MCC Performance")

# Get the number of neighbours of the maximum MCC score:
selected_N = KNN_MCC.index(max(KNN_MCC)) + 1  # earlier returned 3, now 9

# Train KNN with optimum k value

classifier_KNN_new = KNeighborsClassifier(n_neighbors=selected_N)  # (n_neighbors = max(KNN_MCC))
classifier_KNN_new.fit(x_train, y_train)

# Predict the classifier's responses for the Test dataset
y_pred_KNN_new = classifier_KNN_new.predict(x_test)

## Evaluate the Performance of blind test
blind_cm_KNN = confusion_matrix(y_test, y_pred_KNN_new)  # cm for confusion matrix , len(y_test) - 22206
print(blind_cm_KNN)

blind_acc_KNN = float(round(balanced_accuracy_score(y_test, y_pred_KNN_new), 3))
print(blind_acc_KNN)

blind_recall_KNN = float(round(recall_score(y_test, y_pred_KNN_new), 3))  # tp / (tp + fn)
print(blind_recall_KNN)

blind_precision_KNN = float(round(precision_score(y_test, y_pred_KNN_new), 3))  # tp / (tp + fp)
print(blind_precision_KNN)

blind_f1_KNN = float(round(f1_score(y_test, y_pred_KNN_new),
                           3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# It is primarily used to compare the performance of two classifiers.
# Suppose that classifier A has a higher recall, and classifier B has higher precision.
# In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
print(blind_f1_KNN)

blind__mcc_KNN = float(round(matthews_corrcoef(y_test, y_pred_KNN_new),
                             3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
print(blind__mcc_KNN)

blind_AUC_KNN = float(round(roc_auc_score(y_test, (classifier_KNN_new.predict_proba(x_test)[:, 1])), 3))
print(blind_AUC_KNN)
# area under the ROC curve, which is the curve having False Positive Rate on the x-axis and True Positive Rate on the y-axis at all classification thresholds.

blind_test_KNN = [blind_acc_KNN, blind_recall_KNN, blind_precision_KNN, blind_f1_KNN, blind__mcc_KNN, blind_AUC_KNN]
print(blind_test_KNN)

# Number of Folds to split the data:
# folds = 10 # not stratified

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # defined earlier under Naive Bayes

# Call the function of cross-validation passing the parameters: # this returned 10 accuracies, and at the next step, we took the mean of this.
cross_accuracy_all_KNN = cross_val_score(estimator=classifier_KNN_new, X=x_train, y=y_train, cv=folds,
                                         scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.

cross_precision_all_KNN = cross_val_score(estimator=classifier_KNN_new, X=x_train, y=y_train, cv=folds,
                                          scoring='precision')

cross_recall_all_KNN = cross_val_score(estimator=classifier_KNN_new, X=x_train, y=y_train, cv=folds, scoring='recall')

cross_f1_all_KNN = cross_val_score(estimator=classifier_KNN_new, X=x_train, y=y_train, cv=folds, scoring='f1')

# no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
cross_mcc_all_KNN = cross_val_score(estimator=classifier_KNN_new, X=x_train, y=y_train, cv=folds, scoring=mcc)

cross_AUC_all_KNN = cross_val_score(estimator=classifier_KNN_new, X=x_train, y=y_train, cv=folds, scoring='roc_auc')

cross_validation_KNN = [round((cross_accuracy_all_KNN.mean()), 3), round((cross_recall_all_KNN.mean()), 3),
                        round((cross_precision_all_KNN.mean()), 3), round((cross_f1_all_KNN.mean()), 3),
                        round((cross_mcc_all_KNN.mean()), 3), round((cross_AUC_all_KNN.mean()), 3)]
print(cross_validation_KNN)

# ------------------------------------------------------------------------------------------------------------------------------------

## 15.3 Decision trees

print('Decision trees')

# Training the Decision trees:
classifier_DT = DecisionTreeClassifier(criterion='entropy', random_state=0)
# criterion - The function to measure the quality of a split.
# Gini index and entropy is the criterion for calculating information gain. Decision tree algorithms use information gain to split a node.
# Both gini and entropy are measures of impurity of a node. A node having multiple classes is impure whereas a node having only one class is pure.  Entropy in statistics is analogous to entropy in thermodynamics where it signifies disorder. If there are multiple classes in a node, there is disorder in that node.

classifier_DT.fit(x_train, y_train)

# Predict the classifier response for the Test dataset:
y_pred_DT = classifier_DT.predict(x_test)

## Evaluate the Performance of blind test
blind_cm_DT = confusion_matrix(y_test, y_pred_DT)  # cm for confusion matrix , len(y_test) - 22206
print(blind_cm_DT)

blind_acc_DT = float(
    round(balanced_accuracy_score(y_test, y_pred_DT), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
print(blind_acc_DT)

blind_recall_DT = float(round(recall_score(y_test, y_pred_DT), 3))  # tp / (tp + fn)
print(blind_recall_DT)

blind_precision_DT = float(round(precision_score(y_test, y_pred_DT), 3))  # tp / (tp + fp)
print(blind_precision_DT)

blind_f1_DT = float(round(f1_score(y_test, y_pred_DT),
                          3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# It is primarily used to compare the performance of two classifiers.
# Suppose that classifier A has a higher recall, and classifier B has higher precision.
# In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
print(blind_f1_DT)

blind__mcc_DT = float(round(matthews_corrcoef(y_test, y_pred_DT),
                            3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
print(blind__mcc_DT)

blind_AUC_DT = float(round(roc_auc_score(y_test, (classifier_DT.predict_proba(x_test)[:, 1])), 3))
print(blind_AUC_DT)

blind_test_DT = [blind_acc_DT, blind_recall_DT, blind_precision_DT, blind_f1_DT, blind__mcc_DT, blind_AUC_DT]
print(blind_test_DT)

# Number of Folds to split the data:
# folds = 10 # not stratified

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes

# Call the function of cross-validation passing the parameters:
cross_accuracy_all_DT = cross_val_score(estimator=classifier_DT, X=x_train, y=y_train, cv=folds,
                                        scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.

cross_precision_all_DT = cross_val_score(estimator=classifier_DT, X=x_train, y=y_train, cv=folds, scoring='precision')

cross_recall_all_DT = cross_val_score(estimator=classifier_DT, X=x_train, y=y_train, cv=folds, scoring='recall')

cross_f1_all_DT = cross_val_score(estimator=classifier_DT, X=x_train, y=y_train, cv=folds, scoring='f1')

# no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
cross_mcc_all_DT = cross_val_score(estimator=classifier_DT, X=x_train, y=y_train, cv=folds, scoring=mcc)

cross_AUC_all_DT = cross_val_score(estimator=classifier_DT, X=x_train, y=y_train, cv=folds, scoring='roc_auc')

cross_validation_DT = [round((cross_accuracy_all_DT.mean()), 3), round((cross_recall_all_DT.mean()), 3),
                       round((cross_precision_all_DT.mean()), 3), round((cross_f1_all_DT.mean()), 3),
                       round((cross_mcc_all_DT.mean()), 3), round((cross_AUC_all_DT.mean()), 3)]
print(cross_validation_DT)

# ------------------------------------------------------------------------------------------------------------------------------------

## 15.4 Random forest

print('Random forest')

classifier_RF = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
# n_estimators - No. of trees in the forest. Try n_estimators = 100 (default value) also to check whether the accuracy is improving.
# criterion{“gini”, “entropy”}, default=”gini” . This is the function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
# random_state is for RandomState instance or None, default=None. Controls the randomness of the estimator.

# criterion?
# A node is 100% impure when a node is split evenly 50/50 and 100% pure when all of its data belongs to a single class.

classifier_RF.fit(x_train, y_train)

# Predict the classifier response for the Test dataset:
y_pred_RF = classifier_RF.predict(x_test)

## Evaluate the Performance of blind test
blind_cm_RF = confusion_matrix(y_test, y_pred_RF)  # cm for confusion matrix , len(y_test) - 22206
print(blind_cm_RF)

blind_acc_RF = float(
    round(balanced_accuracy_score(y_test, y_pred_RF), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
print(blind_acc_RF)

blind_recall_RF = float(round(recall_score(y_test, y_pred_RF), 3))  # tp / (tp + fn)
print(blind_recall_RF)

blind_precision_RF = float(round(precision_score(y_test, y_pred_RF), 3))  # tp / (tp + fp)
print(blind_precision_RF)

blind_f1_RF = float(round(f1_score(y_test, y_pred_RF),
                          3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# It is primarily used to compare the performance of two classifiers.
# Suppose that classifier A has a higher recall, and classifier B has higher precision.
# In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
print(blind_f1_RF)

blind__mcc_RF = float(round(matthews_corrcoef(y_test, y_pred_RF),
                            3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
print(blind__mcc_RF)

blind_AUC_RF = float(round(roc_auc_score(y_test, (classifier_RF.predict_proba(x_test)[:, 1])), 3))
print(blind_AUC_RF)

blind_test_RF = [blind_acc_RF, blind_recall_RF, blind_precision_RF, blind_f1_RF, blind__mcc_RF, blind_AUC_RF]
print(blind_test_RF)

# Number of Folds to split the data:
# folds = 10 # not stratified

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes

# Call the function of cross-validation passing the parameters:
cross_accuracy_all_RF = cross_val_score(estimator=classifier_RF, X=x_train, y=y_train, cv=folds,
                                        scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.

cross_precision_all_RF = cross_val_score(estimator=classifier_RF, X=x_train, y=y_train, cv=folds, scoring='precision')

cross_recall_all_RF = cross_val_score(estimator=classifier_RF, X=x_train, y=y_train, cv=folds, scoring='recall')

cross_f1_all_RF = cross_val_score(estimator=classifier_RF, X=x_train, y=y_train, cv=folds, scoring='f1')

# no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
cross_mcc_all_RF = cross_val_score(estimator=classifier_RF, X=x_train, y=y_train, cv=folds, scoring=mcc)

cross_AUC_all_RF = cross_val_score(estimator=classifier_RF, X=x_train, y=y_train, cv=folds, scoring='roc_auc')

cross_validation_RF = [round((cross_accuracy_all_RF.mean()), 3), round((cross_recall_all_RF.mean()), 3),
                       round((cross_precision_all_RF.mean()), 3), round((cross_f1_all_RF.mean()), 3),
                       round((cross_mcc_all_RF.mean()), 3), round((cross_AUC_all_RF.mean()), 3)]
print(cross_validation_RF)

# ------------------------------------------------------------------------------------------------------------------------------------

print("Random forest - with paramater 'class_weight'")

## 15.5 Random forest - with paramater 'class_weight'

classifier_RF_cw = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0,
                                          class_weight='balanced')
# n_estimators - No. of trees in the forest. Try n_estimators = 100 (default value) also to check whether the accuracy is improving.
# criterion{“gini”, “entropy”}, default=”gini” . This is the function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
# random_state is for RandomState instance or None, default=None. Controls the randomness of the estimator.

# The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as,
# n_samples / (n_classes * np.bincount(y))

# Unlike the oversampling and under-sampling methods, the balanced weights methods do not modify the minority and majority class ratio.
# Instead, it penalizes the wrong predictions on the minority class by giving more weight to the loss function.

classifier_RF_cw.fit(x_train, y_train)

# Predict the classifier response for the Test dataset:
y_pred_RF_cw = classifier_RF_cw.predict(x_test)

## Evaluate the Performance of blind test
blind_cm_RF_cw = confusion_matrix(y_test, y_pred_RF_cw)  # cm for confusion matrix , len(y_test) - 22206
print(blind_cm_RF)

blind_acc_RF_cw = float(
    round(balanced_accuracy_score(y_test, y_pred_RF_cw), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
print(blind_acc_RF)

blind_recall_RF_cw = float(round(recall_score(y_test, y_pred_RF_cw), 3))  # tp / (tp + fn)
print(blind_recall_RF)

blind_precision_RF_cw = float(round(precision_score(y_test, y_pred_RF_cw), 3))  # tp / (tp + fp)
print(blind_precision_RF)

blind_f1_RF_cw = float(round(f1_score(y_test, y_pred_RF_cw),
                             3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# It is primarily used to compare the performance of two classifiers.
# Suppose that classifier A has a higher recall, and classifier B has higher precision.
# In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
print(blind_f1_RF)

blind__mcc_RF_cw = float(round(matthews_corrcoef(y_test, y_pred_RF_cw),
                               3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
print(blind__mcc_RF)

blind_AUC_RF_cw = float(round(roc_auc_score(y_test, (classifier_RF_cw.predict_proba(x_test)[:, 1])), 3))
print(blind_AUC_RF_cw)

blind_test_RF_cw = [blind_acc_RF_cw, blind_recall_RF_cw, blind_precision_RF_cw, blind_f1_RF_cw, blind__mcc_RF_cw,
                    blind_AUC_RF_cw]
print(blind_test_RF_cw)

# Number of Folds to split the data:
# folds = 10 # not stratified

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes

# Call the function of cross-validation passing the parameters:
cross_accuracy_all_RF_cw = cross_val_score(estimator=classifier_RF_cw, X=x_train, y=y_train, cv=folds,
                                           scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.

cross_precision_all_RF_cw = cross_val_score(estimator=classifier_RF_cw, X=x_train, y=y_train, cv=folds,
                                            scoring='precision')

cross_recall_all_RF_cw = cross_val_score(estimator=classifier_RF_cw, X=x_train, y=y_train, cv=folds, scoring='recall')

cross_f1_all_RF_cw = cross_val_score(estimator=classifier_RF_cw, X=x_train, y=y_train, cv=folds, scoring='f1')

# no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
cross_mcc_all_RF_cw = cross_val_score(estimator=classifier_RF_cw, X=x_train, y=y_train, cv=folds, scoring=mcc)

cross_AUC_all_RF_cw = cross_val_score(estimator=classifier_RF_cw, X=x_train, y=y_train, cv=folds, scoring='roc_auc')

cross_validation_RF_cw = [round((cross_accuracy_all_RF_cw.mean()), 3), round((cross_recall_all_RF_cw.mean()), 3),
                          round((cross_precision_all_RF_cw.mean()), 3), round((cross_f1_all_RF_cw.mean()), 3),
                          round((cross_mcc_all_RF_cw.mean()), 3), round((cross_AUC_all_RF_cw.mean()), 3)]
print(cross_validation_RF_cw)

# ------------------------------------------------------------------------------------------------------------------------------------
print("AdaBoostClassifier")

## 15.6 AdaBoostClassifier

# a boosting technique
# focus on the areas where the system is not perfoming well

# This classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the
# same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

# Create adaboost classifer object
classifier_AdaB = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# base_estimator: It is a weak learner used to train the model. It uses DecisionTreeClassifier as default weak learner for training purpose. You can also specify different machine learning algorithms.
# n_estimators: Number of weak learners to train iteratively.
# learning_rate: It contributes to the weights of weak learners. It uses 1 as a default value.

# Train Adaboost Classifer
classifier_AdaB.fit(x_train, y_train)

# Predict the response for test dataset
y_pred_AdaB = classifier_AdaB.predict(x_test)

## Evaluate the Performance of blind test
blind_cm_AdaB = confusion_matrix(y_test, y_pred_AdaB)  # cm for confusion matrix , len(y_test) - 22206
print(blind_cm_AdaB)

blind_acc_AdaB = float(
    round(balanced_accuracy_score(y_test, y_pred_AdaB), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
print(blind_acc_AdaB)

blind_recall_AdaB = float(round(recall_score(y_test, y_pred_AdaB), 3))  # tp / (tp + fn)
print(blind_recall_AdaB)

blind_precision_AdaB = float(round(precision_score(y_test, y_pred_AdaB), 3))  # tp / (tp + fp)
print(blind_precision_AdaB)

blind_f1_AdaB = float(round(f1_score(y_test, y_pred_AdaB),
                            3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# It is primarily used to compare the performance of two classifiers.
# Suppose that classifier A has a higher recall, and classifier B has higher precision.
# In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
print(blind_f1_AdaB)

blind__mcc_AdaB = float(round(matthews_corrcoef(y_test, y_pred_AdaB),
                              3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
print(blind__mcc_AdaB)

blind_AUC_AdaB = float(round(roc_auc_score(y_test, (classifier_AdaB.predict_proba(x_test)[:, 1])), 3))
print(blind_AUC_AdaB)

blind_test_AdaB = [blind_acc_AdaB, blind_recall_AdaB, blind_precision_AdaB, blind_f1_AdaB, blind__mcc_AdaB,
                   blind_AUC_AdaB]
print(blind_test_AdaB)

# Number of Folds to split the data:
# folds = 10 # not stratified

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes

# Call the function of cross-validation passing the parameters:
cross_accuracy_all_AdaB = cross_val_score(estimator=classifier_AdaB, X=x_train, y=y_train, cv=folds,
                                          scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.

cross_precision_all_AdaB = cross_val_score(estimator=classifier_AdaB, X=x_train, y=y_train, cv=folds,
                                           scoring='precision')

cross_recall_all_AdaB = cross_val_score(estimator=classifier_AdaB, X=x_train, y=y_train, cv=folds, scoring='recall')

cross_f1_all_AdaB = cross_val_score(estimator=classifier_AdaB, X=x_train, y=y_train, cv=folds, scoring='f1')

# no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
cross_mcc_all_AdaB = cross_val_score(estimator=classifier_AdaB, X=x_train, y=y_train, cv=folds, scoring=mcc)

cross_AUC_all_AdaB = cross_val_score(estimator=classifier_AdaB, X=x_train, y=y_train, cv=folds, scoring='roc_auc')

cross_validation_AdaB = [round((cross_accuracy_all_AdaB.mean()), 3), round((cross_recall_all_AdaB.mean()), 3),
                         round((cross_precision_all_AdaB.mean()), 3), round((cross_f1_all_AdaB.mean()), 3),
                         round((cross_mcc_all_AdaB.mean()), 3), round((cross_AUC_all_AdaB.mean()), 3)]
print(cross_validation_AdaB)

# ------------------------------------------------------------------------------------------------------------------------------------

## 15.7 XGBoost (Extreme Gradient Boosting)

print("XGBoost")

# Create XGBoost classifer object
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier

# default parameter values - https://stackoverflow.com/questions/34674797/xgboost-xgbclassifier-defaults-in-python
# default - max_depth=3 , learning_rate=0.1 , n_estimators=100 , objective='binary:logistic'

classifier_XGB = xgb.XGBClassifier(objective="binary:logistic", max_depth=3, learning_rate=0.1, n_estimators=100,
                                   random_state=0)  # random_state = 42

# Train Adaboost Classifer
classifier_XGB.fit(x_train, y_train)

# Predict the response for test dataset
y_pred_XGB = classifier_XGB.predict(x_test)

## Evaluate the Performance of blind test
blind_cm_XGB = confusion_matrix(y_test, y_pred_XGB)  # cm for confusion matrix , len(y_test) - 22206
print(blind_cm_XGB)

blind_acc_XGB = float(
    round(balanced_accuracy_score(y_test, y_pred_XGB), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
print(blind_acc_XGB)

blind_recall_XGB = float(round(recall_score(y_test, y_pred_XGB), 3))  # tp / (tp + fn)
print(blind_recall_XGB)

blind_precision_XGB = float(round(precision_score(y_test, y_pred_XGB), 3))  # tp / (tp + fp)
print(blind_precision_XGB)

blind_f1_XGB = float(round(f1_score(y_test, y_pred_XGB),
                           3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# It is primarily used to compare the performance of two classifiers.
# Suppose that classifier A has a higher recall, and classifier B has higher precision.
# In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
print(blind_f1_XGB)

blind__mcc_XGB = float(round(matthews_corrcoef(y_test, y_pred_XGB),
                             3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
print(blind__mcc_XGB)

blind_AUC_XGB = float(round(roc_auc_score(y_test, (classifier_XGB.predict_proba(x_test)[:, 1])), 3))
print(blind_AUC_XGB)

blind_test_XGB = [blind_acc_XGB, blind_recall_XGB, blind_precision_XGB, blind_f1_XGB, blind__mcc_XGB, blind_AUC_XGB]
print(blind_test_XGB)

# Number of Folds to split the data:
# folds = 10 # not stratified

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes

# Call the function of cross-validation passing the parameters:
cross_accuracy_all_XGB = cross_val_score(estimator=classifier_XGB, X=x_train, y=y_train, cv=folds,
                                         scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.

cross_precision_all_XGB = cross_val_score(estimator=classifier_XGB, X=x_train, y=y_train, cv=folds, scoring='precision')

cross_recall_all_XGB = cross_val_score(estimator=classifier_XGB, X=x_train, y=y_train, cv=folds, scoring='recall')

cross_f1_all_XGB = cross_val_score(estimator=classifier_XGB, X=x_train, y=y_train, cv=folds, scoring='f1')

# no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
cross_mcc_all_XGB = cross_val_score(estimator=classifier_XGB, X=x_train, y=y_train, cv=folds, scoring=mcc)

cross_AUC_all_XGB = cross_val_score(estimator=classifier_XGB, X=x_train, y=y_train, cv=folds, scoring='roc_auc')

cross_validation_XGB = [round((cross_accuracy_all_XGB.mean()), 3), round((cross_recall_all_XGB.mean()), 3),
                        round((cross_precision_all_XGB.mean()), 3), round((cross_f1_all_XGB.mean()), 3),
                        round((cross_mcc_all_XGB.mean()), 3), round((cross_AUC_all_XGB.mean()), 3)]
print(cross_validation_XGB)

# ------------------------------------------------------------------------------------------------------------------------------------

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

## Evaluate the Performance of blind test
blind_cm_LGBM = confusion_matrix(y_test, y_pred_LGBM)  # cm for confusion matrix , len(y_test) - 22206
print(blind_cm_LGBM)

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
print(blind_test_LGBM)

# Number of Folds to split the data:
# folds = 10 # not stratified

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes

# Call the function of cross-validation passing the parameters:
cross_accuracy_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds,
                                          scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.

cross_precision_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds,
                                           scoring='precision')

cross_recall_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds, scoring='recall')

cross_f1_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds, scoring='f1')

# no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
cross_mcc_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds, scoring=mcc)

cross_AUC_all_LGBM = cross_val_score(estimator=classifier_LGBM, X=x_train, y=y_train, cv=folds, scoring='roc_auc')

cross_validation_LGBM = [round((cross_accuracy_all_LGBM.mean()), 3), round((cross_recall_all_LGBM.mean()), 3),
                         round((cross_precision_all_LGBM.mean()), 3), round((cross_f1_all_LGBM.mean()), 3),
                         round((cross_mcc_all_LGBM.mean()), 3), round((cross_AUC_all_LGBM.mean()), 3)]
print(cross_validation_LGBM)

# ------------------------------------------------------------------------------------------------------------------------------------


# # 15.9 GradientBoost Classifier

print("GB")

# Create GradientBoost classifier object

classifier_GB = GradientBoostingClassifier(n_estimators=300, random_state=0)  # random_state=1

# model = LGBMClassifier(colsample_bytree=0.61, min_child_samples=321, min_child_weight=0.01, n_estimators=100, num_leaves=45, reg_alpha=0.1, reg_lambda=1, subsample=0.56)

# 2 ways to import libraries when create training object
# import lightgbm
# clf = lightgbm.LGBMClassifier()

# from lightgbm import LGBMClassifier
# classifier_LGBM = LGBMClassifier()

# Train Adaboost Classifier
classifier_GB.fit(x_train, y_train)

# Predict the response for test dataset
y_pred_GB = classifier_GB.predict(x_test)

## Evaluate the Performance of blind test
blind_cm_GB = confusion_matrix(y_test, y_pred_GB)  # cm for confusion matrix , len(y_test) - 22206
print(blind_cm_GB)

blind_acc_GB = float(
    round(balanced_accuracy_score(y_test, y_pred_GB), 3))  # balanced_accuracy_score = 0.5 ((tp/p) + (tn/n))
print(blind_acc_GB)

blind_recall_GB = float(round(recall_score(y_test, y_pred_GB), 3))  # tp / (tp + fn)
print(blind_recall_GB)

blind_precision_GB = float(round(precision_score(y_test, y_pred_GB), 3))  # tp / (tp + fp)
print(blind_precision_GB)

blind_f1_GB = float(round(f1_score(y_test, y_pred_GB),
                          3))  # The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
# It is primarily used to compare the performance of two classifiers.
# Suppose that classifier A has a higher recall, and classifier B has higher precision.
# In this case, the F1-scores for both the classifiers can be used to determine which one produces better results.
print(blind_f1_GB)

blind__mcc_GB = float(round(matthews_corrcoef(y_test, y_pred_GB),
                            3))  # Matthews correlation coefficient,  C = 1 -> perfect agreement, C = 0 -> random, and C = -1 -> total disagreement between prediction and observation
print(blind__mcc_GB)

blind_AUC_GB = float(round(roc_auc_score(y_test, (classifier_GB.predict_proba(x_test)[:, 1])), 3))
print(blind_AUC_GB)

blind_test_GB = [blind_acc_GB, blind_recall_GB, blind_precision_GB, blind_f1_GB, blind__mcc_GB,
                 blind_AUC_GB]
print(blind_test_GB)

# Number of Folds to split the data:
# folds = 10 # not stratified

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # # defined earlier under Naive Bayes

# Call the function of cross-validation passing the parameters:
cross_accuracy_all_GB = cross_val_score(estimator=classifier_GB, X=x_train, y=y_train, cv=folds,
                                        scoring='balanced_accuracy')  # can replace scoring string by = ‘f1’, ‘accuracy’, 'balanced_accuracy'.

cross_precision_all_GB = cross_val_score(estimator=classifier_GB, X=x_train, y=y_train, cv=folds,
                                         scoring='precision')

cross_recall_all_GB = cross_val_score(estimator=classifier_GB, X=x_train, y=y_train, cv=folds, scoring='recall')

cross_f1_all_GB = cross_val_score(estimator=classifier_GB, X=x_train, y=y_train, cv=folds, scoring='f1')

# no direct scorer to calculate mcc in cross validation. hence convert metric 'matthews_corrcoef' to a scorer using make_scorer
mcc = make_scorer(matthews_corrcoef)  # defined earlier under Naive Bayes
cross_mcc_all_GB = cross_val_score(estimator=classifier_GB, X=x_train, y=y_train, cv=folds, scoring=mcc)

cross_AUC_all_GB = cross_val_score(estimator=classifier_GB, X=x_train, y=y_train, cv=folds, scoring='roc_auc')

cross_validation_GB = [round((cross_accuracy_all_GB.mean()), 3), round((cross_recall_all_GB.mean()), 3),
                       round((cross_precision_all_GB.mean()), 3), round((cross_f1_all_GB.mean()), 3),
                       round((cross_mcc_all_GB.mean()), 3), round((cross_AUC_all_GB.mean()), 3)]
print(cross_validation_GB)

# ------------------------------------------------------------------------------------------------------------------------------------

# 18. Compare results of different ML models

comparison_ML_models = pd.DataFrame({
    'BT_NB': blind_test_NB,
    'CV_NB': cross_validation_NB,
    'BT_KNN': blind_test_KNN,
    'CV_KNN': cross_validation_KNN,
    'BT_DT': blind_test_DT,
    'CV_DT': cross_validation_DT,
    'BT_RF': blind_test_RF,
    'CV_RF': cross_validation_RF,
    'BT_RF(weighted)': blind_test_RF_cw,
    'CV_RF(weighted)': cross_validation_RF_cw,
    'BT_AdaB': blind_test_AdaB,
    'CV_AdaB': cross_validation_AdaB,
    'BT_XGB': blind_test_XGB,
    'CV_XGB': cross_validation_XGB,
    'BT_LGBM': blind_test_LGBM,
    'CV_LGBM': cross_validation_LGBM,
    'BT_GB': blind_test_GB,
    'CV_GB': cross_validation_GB
},
    index=['balanced_accuracy', 'recall', 'precision', 'f1', 'MCC', 'AUC'])


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(comparison_ML_models)

# python /Users/psenevirathn/PycharmProjects/myproject2/current_codes/1_Jan_3_preprocessing.py /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/first_hep_with_hep_type_treatment_type_dermographics.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/Platelet_counts.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/Vitalsigns.csv
