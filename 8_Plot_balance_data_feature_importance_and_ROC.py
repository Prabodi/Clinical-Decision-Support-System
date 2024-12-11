
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
    f1_score, auc, roc_curve, roc_auc_score, balanced_accuracy_score, precision_recall_curve, average_precision_score
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
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from numpy import interp
from scipy.interpolate import interp1d
from matplotlib.pyplot import figure

# ------------------------------------------------------------------------------------------------------------------------------------
# selected categorical features

# ['first_careunit', 'admission_location', 'gender', 'lactate_min_status', 'lactate_max_status', 'ph_min_status',
#      'ph_max_status', 'totalco2_bg_min_status', 'totalco2_bg_max_status', 'pco2_bg_min_status', 'pco2_bg_max_status',
#      'ast_min_status', 'ast_max_status', 'alp_min_status', 'alp_max_status', 'alt_min_status', 'alt_max_status',
#      'bilirubin_total_min_status', 'bilirubin_total_max_status', 'albumin_min_status', 'albumin_max_status',
#      'pco2_bg_art_min_status', 'pco2_bg_art_max_status', 'po2_bg_art_min_status', 'po2_bg_art_max_status',
#      'totalco2_bg_art_min_status', 'totalco2_bg_art_max_status', 'ld_ldh_min_status', 'ld_ldh_max_status',
#      'ck_cpk_min_status', 'ck_cpk_max_status', 'ck_mb_min_status', 'ck_mb_max_status', 'fibrinogen_min_status',
#      'fibrinogen_max_status', 'bilirubin_direct_min_status', 'bilirubin_direct_max_status', 'nrbc_min_status',
#      'nrbc_max_status', 'bands_min_status', 'bands_max_status']

# selected numerical features

# LGBM - first - 14 features
# ['platelets_min', 'inr_max', 'aniongap_max', 'platelets_max', 'potassium_max', 'dbp_min', 'sbp_mean', 'aniongap_min', 'calcium_min', 'bun_min', 'temperature_min', 'bicarbonate_max', 'ptt_min', 'base_platelets']]

# 2. Import files
# These files contain raw data, but already spliited as train-test sets. We need to make out data balance now. Now we make balance the training set only.

df_train_all = pd.read_csv(sys.argv[1])  # after prepocessing - training set only
df_test_all = pd.read_csv(sys.argv[2])  # after prepocessing - test set only

# --------------------------------------------------------------------------------------------------------------

# categorical feature selection

# training dataset

numerical_features_selected = ['platelets_min', 'inr_max', 'aniongap_min', 'glucose_vital_min', 'heart_rate_max',
                               'creatinine_min', 'bicarbonate_lab_min', 'spo2_min', 'calcium_lab_min',
                               'glucose_lab_max', 'mbp_mean', 'sodium_lab_max', 'platelets_max', 'spo2_max',
                               'temperature_vital_mean', 'pt_max', 'hematocrit_lab_max', 'sbp_min', 'anchor_age',
                               'resp_rate_max', 'base_platelets']

cat_features_selected = ['first_careunit', 'admission_location', 'gender', 'bilirubin_direct_min_status',
                         'bilirubin_direct_max_status',
                         'nrbc_max_status', 'nrbc_min_status', 'bands_min_status', 'bands_max_status',
                         'fibrinogen_max_status',
                         'fibrinogen_min_status', 'hematocrit_bg_min_status', 'hematocrit_bg_max_status',
                         'hemoglobin_bg_min_status',
                         'hemoglobin_bg_max_status', 'temperature_bg_max_status', 'temperature_bg_min_status',
                         'sodium_bg_max_status',
                         'sodium_bg_min_status', 'glucose_bg_max_status', 'glucose_bg_min_status', 'ck_cpk_max_status',
                         'ck_cpk_min_status',
                         'ck_mb_max_status', 'ck_mb_min_status', 'ld_ldh_max_status', 'ld_ldh_min_status',
                         'calcium_bg_max_status',
                         'calcium_bg_min_status', 'pco2_bg_art_min_status', 'po2_bg_art_max_status',
                         'totalco2_bg_art_max_status',
                         'totalco2_bg_art_min_status', 'pco2_bg_art_max_status', 'po2_bg_art_min_status',
                         'potassium_bg_min_status',
                         'potassium_bg_max_status', 'albumin_max_status', 'albumin_min_status',
                         'bilirubin_total_min_status',
                         'bilirubin_total_max_status', 'alt_max_status', 'alt_min_status', 'alp_max_status',
                         'alp_min_status',
                         'ast_min_status', 'ast_max_status', 'pco2_bg_max_status', 'pco2_bg_min_status',
                         'totalco2_bg_min_status',
                         'totalco2_bg_max_status', 'ph_min_status', 'ph_max_status', 'lactate_min_status',
                         'lactate_max_status']

df_train_numerical_selected = df_train_all[numerical_features_selected]

df_train_categorical_selected = df_train_all[cat_features_selected]

df_train_cat_selected_numerical_selected = pd.concat([df_train_numerical_selected, df_train_categorical_selected],
                                                     axis=1)

print(df_train_numerical_selected.shape)  # (10090, 14)
print(df_train_categorical_selected.shape)  # (10090, 41)
print(df_train_cat_selected_numerical_selected.shape)  # (10090, 55)

# ------------------------------------------------------------------------------------------------------------------------------------

# testing data set

df_test_numerical_selected = df_test_all[numerical_features_selected]

df_test_categorical_selected = df_test_all[cat_features_selected]

df_test_cat_selected_numerical_selected = pd.concat([df_test_numerical_selected, df_test_categorical_selected, ],
                                                    axis=1)

print(df_test_numerical_selected.shape)  # (2523, 14)
print(df_test_categorical_selected.shape)  # (2523, 41)
print(df_test_cat_selected_numerical_selected.shape)  # (2523, 55)

# ------------------------------------------------------------------------------------------------------------------------------------

# Convert Categorical features to Numerical - Dummy variable encoding # (18440, 209) (18440, 169)

# train set

ct1 = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'),
                   cat_features_selected)],
    remainder='passthrough')

df_train_cat_selected_numerical_selected = np.array(ct1.fit_transform(
    df_train_cat_selected_numerical_selected))  # here 'np' (NumPy) was added because, fit_transform itself doesn't return output in np array, so in order to train future machine learning models, np is added.

print(df_train_cat_selected_numerical_selected.shape)  # (10090, 131)

# ------------------------------------------------------------------------------------------------------------------------------------
# Missing value imputation

ImputerKNN = KNNImputer(n_neighbors=2)
df_train_cat_selected_numerical_selected = ImputerKNN.fit_transform(df_train_cat_selected_numerical_selected)

print(df_train_cat_selected_numerical_selected)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Here we have 2 concerns
# 1. We caanot have missing values when do class balancing (otherewise it pops-up the error - 'ValueError: Input contains NaN'
# 2. When apply SMOTE, we artificially create new sample. For that, it's better to keep categorical feature as it is.(rather than convert them into numerical features)

# Handle imbalance data - make the data to balance with HIT:NO HIT = 1:2

# SMOTE with Random UnderSampling - https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/#:~:text=Manually%20Combine%20SMOTE%20and%20Random%20Undersampling,-We%20are%20not&text=The%20authors%20of%20the%20technique,better%20than%20plain%20under%2Dsampling.

print(df_train_cat_selected_numerical_selected)

numerical_features_completed_data = pd.DataFrame(df_train_cat_selected_numerical_selected[:, 151:],
                                                 columns=numerical_features_selected)

x_train_numerical_completed_and_categorical_without_encoding = pd.concat(
    [numerical_features_completed_data, df_train_categorical_selected], axis=1)

pd.set_option('display.max_columns', None)

print(x_train_numerical_completed_and_categorical_without_encoding)

y_train_raw = df_train_all['label']
counter1 = Counter(y_train_raw)
print(counter1)  # Counter({0: 8762, 1: 1328})

# first 14 columns (index - 0 to 13) - numerical features
# next 41 columns (index - 14 to 54) - cat features
cat_index = list(range(21, 76))

over = SMOTENC(categorical_features=cat_index,
               random_state=3,
               sampling_strategy=0.2)  # Oversampling methods duplicate or create new synthetic examples in the minority clas
# https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html

under = RandomUnderSampler(
    sampling_strategy=0.5, random_state=3)  # undersampling methods delete or merge examples in the majority class.

steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)

# transform the dataset

# pd.set_option('display.max_columns', None)
# print(x_train_numerical_completed_and_categorical_without_encoding.head(1))

x_train, y_train = pipeline.fit_resample(
    x_train_numerical_completed_and_categorical_without_encoding, y_train_raw)

# summarize the new class distribution
counter2 = Counter(y_train)
print(counter2)  # Counter({0: 3504, 1: 1752})

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# download balance training data set
#
# train_balanced = pd.concat([x_train, y_train], axis=1)
#
# output_result_dir = '/Users/psenevirathn/Desktop/PhD/Coding/Python/output_csv_files/Dec_1'
#
# save_train_loc = os.path.join(output_result_dir,
#                               'training_set_half_preprocessd_balanced_data.csv')  # This Returns a path. os.path.join - https://www.geeksforgeeks.org/python-os-path-join-method/
#
# train_balanced.to_csv(save_train_loc)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Convert Categorical features to Numerical - Dummy variable encoding # (18440, 209) (18440, 169)

# train set

ct2 = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'),
                   cat_features_selected)],
    remainder='passthrough')

x_train = np.array(ct2.fit_transform(
    x_train))  # here 'np' (NumPy) was added because, fit_transform itself doesn't return output in np array, so in order to train future machine learning models, np is added.

print(x_train.shape)  # (5256, 131)

# sometimes, this has shape (5256, 130) - it's beacuse all the records 'encoder__lactate_max_status_low' removed when do undersampling.
# sometimes, this has shape (5256, 129) - it's beacuse all the record 'encoder__lactate_max_status_low' AND 'encoder__admission_location_INTERNAL TRANSFER TO OR FROM PSYCH' removed when do undersampling.
# If the shape is not (5256, 131), then it will generate an error, because shape of CT2 will not match with test set, to 'fit'.


# test set

x_test = np.array(
    ct2.transform(df_test_cat_selected_numerical_selected))  # handle_unknown = 'ignore'

x_axis_original = ct2.get_feature_names_out().tolist()
print(x_axis_original)

# ['encoder__first_careunit_Coronary Care Unit (CCU)', 'encoder__first_careunit_Medical Intensive Care Unit (MICU)', 'encoder__first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)', 'encoder__first_careunit_Neuro Intermediate', 'encoder__first_careunit_Neuro Stepdown', 'encoder__first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)', 'encoder__first_careunit_Surgical Intensive Care Unit (SICU)', 'encoder__first_careunit_Trauma SICU (TSICU)', 'encoder__admission_location_CLINIC REFERRAL', 'encoder__admission_location_EMERGENCY ROOM', 'encoder__admission_location_INFORMATION NOT AVAILABLE', 'encoder__admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'encoder__admission_location_PACU', 'encoder__admission_location_PHYSICIAN REFERRAL', 'encoder__admission_location_PROCEDURE SITE', 'encoder__admission_location_TRANSFER FROM HOSPITAL', 'encoder__admission_location_TRANSFER FROM SKILLED NURSING FACILITY', 'encoder__admission_location_WALK-IN/SELF REFERRAL', 'encoder__gender_M', 'encoder__lactate_min_status_low', 'encoder__lactate_min_status_normal', 'encoder__lactate_min_status_not ordered', 'encoder__lactate_max_status_low', 'encoder__lactate_max_status_normal', 'encoder__lactate_max_status_not ordered', 'encoder__ph_min_status_low', 'encoder__ph_min_status_normal', 'encoder__ph_min_status_not ordered', 'encoder__ph_max_status_low', 'encoder__ph_max_status_normal', 'encoder__ph_max_status_not ordered', 'encoder__totalco2_bg_min_status_low', 'encoder__totalco2_bg_min_status_normal', 'encoder__totalco2_bg_min_status_not ordered', 'encoder__totalco2_bg_max_status_low', 'encoder__totalco2_bg_max_status_normal', 'encoder__totalco2_bg_max_status_not ordered', 'encoder__pco2_bg_min_status_low', 'encoder__pco2_bg_min_status_normal', 'encoder__pco2_bg_min_status_not ordered', 'encoder__pco2_bg_max_status_low', 'encoder__pco2_bg_max_status_normal', 'encoder__pco2_bg_max_status_not ordered', 'encoder__ast_min_status_normal', 'encoder__ast_min_status_not ordered', 'encoder__ast_max_status_normal', 'encoder__ast_max_status_not ordered', 'encoder__alp_min_status_low', 'encoder__alp_min_status_normal', 'encoder__alp_min_status_not ordered', 'encoder__alp_max_status_low', 'encoder__alp_max_status_normal', 'encoder__alp_max_status_not ordered', 'encoder__alt_min_status_normal', 'encoder__alt_min_status_not ordered', 'encoder__alt_max_status_normal', 'encoder__alt_max_status_not ordered', 'encoder__bilirubin_total_min_status_normal', 'encoder__bilirubin_total_min_status_not ordered', 'encoder__bilirubin_total_max_status_normal', 'encoder__bilirubin_total_max_status_not ordered', 'encoder__albumin_min_status_low', 'encoder__albumin_min_status_normal', 'encoder__albumin_min_status_not ordered', 'encoder__albumin_max_status_low', 'encoder__albumin_max_status_normal', 'encoder__albumin_max_status_not ordered', 'encoder__pco2_bg_art_min_status_low', 'encoder__pco2_bg_art_min_status_normal', 'encoder__pco2_bg_art_min_status_not ordered', 'encoder__pco2_bg_art_max_status_low', 'encoder__pco2_bg_art_max_status_normal', 'encoder__pco2_bg_art_max_status_not ordered', 'encoder__po2_bg_art_min_status_low', 'encoder__po2_bg_art_min_status_normal', 'encoder__po2_bg_art_min_status_not ordered', 'encoder__po2_bg_art_max_status_low', 'encoder__po2_bg_art_max_status_normal', 'encoder__po2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_min_status_low', 'encoder__totalco2_bg_art_min_status_normal', 'encoder__totalco2_bg_art_min_status_not ordered', 'encoder__totalco2_bg_art_max_status_low', 'encoder__totalco2_bg_art_max_status_normal', 'encoder__totalco2_bg_art_max_status_not ordered', 'encoder__ld_ldh_min_status_low', 'encoder__ld_ldh_min_status_normal', 'encoder__ld_ldh_min_status_not ordered', 'encoder__ld_ldh_max_status_low', 'encoder__ld_ldh_max_status_normal', 'encoder__ld_ldh_max_status_not ordered', 'encoder__ck_cpk_min_status_low', 'encoder__ck_cpk_min_status_normal', 'encoder__ck_cpk_min_status_not ordered', 'encoder__ck_cpk_max_status_low', 'encoder__ck_cpk_max_status_normal', 'encoder__ck_cpk_max_status_not ordered', 'encoder__ck_mb_min_status_normal', 'encoder__ck_mb_min_status_not ordered', 'encoder__ck_mb_max_status_normal', 'encoder__ck_mb_max_status_not ordered', 'encoder__fibrinogen_min_status_low', 'encoder__fibrinogen_min_status_normal', 'encoder__fibrinogen_min_status_not ordered', 'encoder__fibrinogen_max_status_low', 'encoder__fibrinogen_max_status_normal', 'encoder__fibrinogen_max_status_not ordered', 'encoder__bilirubin_direct_min_status_normal', 'encoder__bilirubin_direct_min_status_not ordered', 'encoder__bilirubin_direct_max_status_normal', 'encoder__bilirubin_direct_max_status_not ordered', 'encoder__nrbc_min_status_not ordered', 'encoder__nrbc_max_status_not ordered', 'encoder__bands_min_status_normal', 'encoder__bands_min_status_not ordered', 'encoder__bands_max_status_normal', 'encoder__bands_max_status_not ordered', 'remainder__platelets_min', 'remainder__inr_max', 'remainder__aniongap_max', 'remainder__platelets_max', 'remainder__potassium_max', 'remainder__dbp_min', 'remainder__sbp_mean', 'remainder__aniongap_min', 'remainder__calcium_min', 'remainder__bun_min', 'remainder__temperature_min', 'remainder__bicarbonate_max', 'remainder__ptt_min', 'remainder__base_platelets']
print(len(x_axis_original))  # 131

print(x_axis_original[
          150])  # encoder__bands_max_status_not ordered - Therefore, upto (including) index = 116 column, are from ctaegorical features encoding.
print(x_axis_original[151])  # remainder__platelets_min

# Missing values imputation for test set - no nedd to do missing value imputation on trainig set, coz, df 'x_train_numerical_completed_and_categorical_without_encoding' contained completed numerical features, and cat features anyway don't have missing values, as converted from OneHot Encoding.
x_test = ImputerKNN.transform(x_test)  # ISSUE - X has 130 features, but KNNImputer is expecting 131 features as input.

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# feature scaling - only for numerical(continuous features)


# In train_x_y , train_x_y:
#   first 17 column - Numerical features
#   from there to end - cat features

sc = MinMaxScaler()

x_train[:, 151:] = sc.fit_transform(x_train[:,
                                    151:])  # Here feature scaling not applied to dummy columns(first 3 columns), i.e. for France = 100,Spain=010 and Germany=001, because those column values are alread in between -3 and 3, and also, if feature scaling do to these columns, abnormal values may return
# Here 'fit method' calculate ,mean and the standard devation of each feature. 'Transform method' apply equation, { Xstand=[x-mean(x)]/standard devation(x) , where x -feature, here have to categoroed for x, which is salary and ange. which called 'Standarization'}, for each feature.

x_test[:, 151:] = sc.transform(x_test[:,
                               151:])  # Here, when do feature scaling in test set, test set should be scaled by using the same parameters used in training set.
# Also, x_test is the input for the prediction function got from training set. That's why here only transform method is using instead fit_transform.
# Means, here when apply standarization to each of two features (age and salary), the mean and the standard deviation used is the values got from training data. >> Xstand_test=[x_test-mean(x_train)]/standard devation(x_train)


# # ------------------------------------------------------------------------------------------------------------------------------------

y_test = df_test_all['label']

counter2 = Counter(y_test)
print(counter2) # Counter({0: 2191, 1: 332})

# y_train = y_train.to_numpy()
# y_test = y_test.to_numpy()

# # ------------------------------------------------------------------------------------------------------------------------------------
# 15. ML Predictors

## 15.1. ML model 1 - Naive-Bayes

# print('Naive-Bayes')
#
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

#
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

## ------------------------------------------------------------------------------------------------------------------------------------
#
# # 18. Compare results of different ML models
#
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
#
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
#
#
# print(comparison_ML_models)

# # ## # ------------------------------------------------------------------------------------------------------------------------------------
#
# # # feature importance
# #
# [0.688, 0.491, 0.392, 0.436, 0.342, 0.804]
# [0.752, 0.624, 0.721, 0.669, 0.524, 0.853]

importance = classifier_LGBM.feature_importances_  # classifier_GB classifier_DT

# print(len(x_axis_original))
# ['encoder__first_careunit_Coronary Care Unit (CCU)', 'encoder__first_careunit_Medical Intensive Care Unit (MICU)', 'encoder__first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)', 'encoder__first_careunit_Neuro Intermediate', 'encoder__first_careunit_Neuro Stepdown', 'encoder__first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)', 'encoder__first_careunit_Surgical Intensive Care Unit (SICU)', 'encoder__first_careunit_Trauma SICU (TSICU)', 'encoder__admission_location_CLINIC REFERRAL', 'encoder__admission_location_EMERGENCY ROOM', 'encoder__admission_location_INFORMATION NOT AVAILABLE', 'encoder__admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'encoder__admission_location_PACU', 'encoder__admission_location_PHYSICIAN REFERRAL', 'encoder__admission_location_PROCEDURE SITE', 'encoder__admission_location_TRANSFER FROM HOSPITAL', 'encoder__admission_location_TRANSFER FROM SKILLED NURSING FACILITY', 'encoder__admission_location_WALK-IN/SELF REFERRAL', 'encoder__gender_M', 'encoder__bilirubin_direct_min_status_normal', 'encoder__bilirubin_direct_min_status_not ordered', 'encoder__bilirubin_direct_max_status_normal', 'encoder__bilirubin_direct_max_status_not ordered', 'encoder__nrbc_max_status_not ordered', 'encoder__nrbc_min_status_not ordered', 'encoder__bands_min_status_normal', 'encoder__bands_min_status_not ordered', 'encoder__bands_max_status_normal', 'encoder__bands_max_status_not ordered', 'encoder__fibrinogen_max_status_low', 'encoder__fibrinogen_max_status_normal', 'encoder__fibrinogen_max_status_not ordered', 'encoder__fibrinogen_min_status_low', 'encoder__fibrinogen_min_status_normal', 'encoder__fibrinogen_min_status_not ordered', 'encoder__hematocrit_bg_min_status_not ordered', 'encoder__hematocrit_bg_max_status_not ordered', 'encoder__hemoglobin_bg_min_status_low', 'encoder__hemoglobin_bg_min_status_normal', 'encoder__hemoglobin_bg_min_status_not ordered', 'encoder__hemoglobin_bg_max_status_low', 'encoder__hemoglobin_bg_max_status_normal', 'encoder__hemoglobin_bg_max_status_not ordered', 'encoder__temperature_bg_max_status_not ordered', 'encoder__temperature_bg_min_status_not ordered', 'encoder__sodium_bg_max_status_low', 'encoder__sodium_bg_max_status_normal', 'encoder__sodium_bg_max_status_not ordered', 'encoder__sodium_bg_min_status_low', 'encoder__sodium_bg_min_status_normal', 'encoder__sodium_bg_min_status_not ordered', 'encoder__glucose_bg_max_status_low', 'encoder__glucose_bg_max_status_normal', 'encoder__glucose_bg_max_status_not ordered', 'encoder__glucose_bg_min_status_low', 'encoder__glucose_bg_min_status_normal', 'encoder__glucose_bg_min_status_not ordered', 'encoder__ck_cpk_max_status_low', 'encoder__ck_cpk_max_status_normal', 'encoder__ck_cpk_max_status_not ordered', 'encoder__ck_cpk_min_status_low', 'encoder__ck_cpk_min_status_normal', 'encoder__ck_cpk_min_status_not ordered', 'encoder__ck_mb_max_status_normal', 'encoder__ck_mb_max_status_not ordered', 'encoder__ck_mb_min_status_normal', 'encoder__ck_mb_min_status_not ordered', 'encoder__ld_ldh_max_status_low', 'encoder__ld_ldh_max_status_normal', 'encoder__ld_ldh_max_status_not ordered', 'encoder__ld_ldh_min_status_low', 'encoder__ld_ldh_min_status_normal', 'encoder__ld_ldh_min_status_not ordered', 'encoder__calcium_bg_max_status_low', 'encoder__calcium_bg_max_status_normal', 'encoder__calcium_bg_max_status_not ordered', 'encoder__calcium_bg_min_status_low', 'encoder__calcium_bg_min_status_normal', 'encoder__calcium_bg_min_status_not ordered', 'encoder__pco2_bg_art_min_status_low', 'encoder__pco2_bg_art_min_status_normal', 'encoder__pco2_bg_art_min_status_not ordered', 'encoder__po2_bg_art_max_status_low', 'encoder__po2_bg_art_max_status_normal', 'encoder__po2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_max_status_low', 'encoder__totalco2_bg_art_max_status_normal', 'encoder__totalco2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_min_status_low', 'encoder__totalco2_bg_art_min_status_normal', 'encoder__totalco2_bg_art_min_status_not ordered', 'encoder__pco2_bg_art_max_status_low', 'encoder__pco2_bg_art_max_status_normal', 'encoder__pco2_bg_art_max_status_not ordered', 'encoder__po2_bg_art_min_status_low', 'encoder__po2_bg_art_min_status_normal', 'encoder__po2_bg_art_min_status_not ordered', 'encoder__potassium_bg_min_status_low', 'encoder__potassium_bg_min_status_normal', 'encoder__potassium_bg_min_status_not ordered', 'encoder__potassium_bg_max_status_low', 'encoder__potassium_bg_max_status_normal', 'encoder__potassium_bg_max_status_not ordered', 'encoder__albumin_max_status_low', 'encoder__albumin_max_status_normal', 'encoder__albumin_max_status_not ordered', 'encoder__albumin_min_status_low', 'encoder__albumin_min_status_normal', 'encoder__albumin_min_status_not ordered', 'encoder__bilirubin_total_min_status_normal', 'encoder__bilirubin_total_min_status_not ordered', 'encoder__bilirubin_total_max_status_normal', 'encoder__bilirubin_total_max_status_not ordered', 'encoder__alt_max_status_normal', 'encoder__alt_max_status_not ordered', 'encoder__alt_min_status_normal', 'encoder__alt_min_status_not ordered', 'encoder__alp_max_status_low', 'encoder__alp_max_status_normal', 'encoder__alp_max_status_not ordered', 'encoder__alp_min_status_low', 'encoder__alp_min_status_normal', 'encoder__alp_min_status_not ordered', 'encoder__ast_min_status_normal', 'encoder__ast_min_status_not ordered', 'encoder__ast_max_status_normal', 'encoder__ast_max_status_not ordered', 'encoder__pco2_bg_max_status_low', 'encoder__pco2_bg_max_status_normal', 'encoder__pco2_bg_max_status_not ordered', 'encoder__pco2_bg_min_status_low', 'encoder__pco2_bg_min_status_normal', 'encoder__pco2_bg_min_status_not ordered', 'encoder__totalco2_bg_min_status_low', 'encoder__totalco2_bg_min_status_normal', 'encoder__totalco2_bg_min_status_not ordered', 'encoder__totalco2_bg_max_status_low', 'encoder__totalco2_bg_max_status_normal', 'encoder__totalco2_bg_max_status_not ordered', 'encoder__ph_min_status_low', 'encoder__ph_min_status_normal', 'encoder__ph_min_status_not ordered', 'encoder__ph_max_status_low', 'encoder__ph_max_status_normal', 'encoder__ph_max_status_not ordered', 'encoder__lactate_min_status_low', 'encoder__lactate_min_status_normal', 'encoder__lactate_min_status_not ordered', 'encoder__lactate_max_status_low', 'encoder__lactate_max_status_normal', 'encoder__lactate_max_status_not ordered', 'remainder__platelets_min', 'remainder__inr_max', 'remainder__aniongap_min', 'remainder__glucose_vital_min', 'remainder__heart_rate_max', 'remainder__creatinine_min', 'remainder__bicarbonate_lab_min', 'remainder__spo2_min', 'remainder__calcium_lab_min', 'remainder__glucose_lab_max', 'remainder__mbp_mean', 'remainder__sodium_lab_max', 'remainder__platelets_max', 'remainder__spo2_max', 'remainder__temperature_vital_mean', 'remainder__pt_max', 'remainder__hematocrit_lab_max', 'remainder__sbp_min', 'remainder__anchor_age', 'remainder__resp_rate_max', 'remainder__base_platelets']

x_axis = ['encoder__first_careunit_Coronary Care Unit (CCU)', 'encoder__first_careunit_Medical Intensive Care Unit (MICU)', 'encoder__first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)', 'encoder__first_careunit_Neuro Intermediate', 'encoder__first_careunit_Neuro Stepdown', 'encoder__first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)', 'encoder__first_careunit_Surgical Intensive Care Unit (SICU)', 'encoder__first_careunit_Trauma SICU (TSICU)', 'encoder__admission_location_CLINIC REFERRAL', 'encoder__admission_location_EMERGENCY ROOM', 'encoder__admission_location_INFORMATION NOT AVAILABLE', 'encoder__admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'encoder__admission_location_PACU', 'encoder__admission_location_PHYSICIAN REFERRAL', 'encoder__admission_location_PROCEDURE SITE', 'encoder__admission_location_TRANSFER FROM HOSPITAL', 'encoder__admission_location_TRANSFER FROM SKILLED NURSING FACILITY', 'encoder__admission_location_WALK-IN/SELF REFERRAL', 'encoder__gender_M', 'encoder__bilirubin_direct_min_status_normal', 'encoder__bilirubin_direct_min_status_not ordered', 'encoder__bilirubin_direct_max_status_normal', 'encoder__bilirubin_direct_max_status_not ordered', 'encoder__nrbc_max_status_not ordered', 'encoder__nrbc_min_status_not ordered', 'encoder__bands_min_status_normal', 'encoder__bands_min_status_not ordered', 'encoder__bands_max_status_normal', 'encoder__bands_max_status_not ordered', 'encoder__fibrinogen_max_status_low', 'encoder__fibrinogen_max_status_normal', 'encoder__fibrinogen_max_status_not ordered', 'encoder__fibrinogen_min_status_low', 'encoder__fibrinogen_min_status_normal', 'encoder__fibrinogen_min_status_not ordered', 'encoder__hematocrit_bg_min_status_not ordered', 'encoder__hematocrit_bg_max_status_not ordered', 'encoder__hemoglobin_bg_min_status_low', 'encoder__hemoglobin_bg_min_status_normal', 'encoder__hemoglobin_bg_min_status_not ordered', 'encoder__hemoglobin_bg_max_status_low', 'encoder__hemoglobin_bg_max_status_normal', 'encoder__hemoglobin_bg_max_status_not ordered', 'encoder__temperature_bg_max_status_not ordered', 'encoder__temperature_bg_min_status_not ordered', 'encoder__sodium_bg_max_status_low', 'encoder__sodium_bg_max_status_normal', 'encoder__sodium_bg_max_status_not ordered', 'encoder__sodium_bg_min_status_low', 'encoder__sodium_bg_min_status_normal', 'encoder__sodium_bg_min_status_not ordered', 'encoder__glucose_bg_max_status_low', 'encoder__glucose_bg_max_status_normal', 'encoder__glucose_bg_max_status_not ordered', 'encoder__glucose_bg_min_status_low', 'encoder__glucose_bg_min_status_normal', 'encoder__glucose_bg_min_status_not ordered', 'encoder__ck_cpk_max_status_low', 'encoder__ck_cpk_max_status_normal', 'encoder__ck_cpk_max_status_not ordered', 'encoder__ck_cpk_min_status_low', 'encoder__ck_cpk_min_status_normal', 'encoder__ck_cpk_min_status_not ordered', 'encoder__ck_mb_max_status_normal', 'encoder__ck_mb_max_status_not ordered', 'encoder__ck_mb_min_status_normal', 'encoder__ck_mb_min_status_not ordered', 'encoder__ld_ldh_max_status_low', 'encoder__ld_ldh_max_status_normal', 'encoder__ld_ldh_max_status_not ordered', 'encoder__ld_ldh_min_status_low', 'encoder__ld_ldh_min_status_normal', 'encoder__ld_ldh_min_status_not ordered', 'encoder__calcium_bg_max_status_low', 'encoder__calcium_bg_max_status_normal', 'encoder__calcium_bg_max_status_not ordered', 'encoder__calcium_bg_min_status_low', 'encoder__calcium_bg_min_status_normal', 'encoder__calcium_bg_min_status_not ordered', 'encoder__pco2_bg_art_min_status_low', 'encoder__pco2_bg_art_min_status_normal', 'encoder__pco2_bg_art_min_status_not ordered', 'encoder__po2_bg_art_max_status_low', 'encoder__po2_bg_art_max_status_normal', 'encoder__po2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_max_status_low', 'encoder__totalco2_bg_art_max_status_normal', 'encoder__totalco2_bg_art_max_status_not ordered', 'encoder__totalco2_bg_art_min_status_low', 'encoder__totalco2_bg_art_min_status_normal', 'encoder__totalco2_bg_art_min_status_not ordered', 'encoder__pco2_bg_art_max_status_low', 'encoder__pco2_bg_art_max_status_normal', 'encoder__pco2_bg_art_max_status_not ordered', 'encoder__po2_bg_art_min_status_low', 'encoder__po2_bg_art_min_status_normal', 'encoder__po2_bg_art_min_status_not ordered', 'encoder__potassium_bg_min_status_low', 'encoder__potassium_bg_min_status_normal', 'encoder__potassium_bg_min_status_not ordered', 'encoder__potassium_bg_max_status_low', 'encoder__potassium_bg_max_status_normal', 'encoder__potassium_bg_max_status_not ordered', 'encoder__albumin_max_status_low', 'encoder__albumin_max_status_normal', 'encoder__albumin_max_status_not ordered', 'encoder__albumin_min_status_low', 'encoder__albumin_min_status_normal', 'encoder__albumin_min_status_not ordered', 'encoder__bilirubin_total_min_status_normal', 'encoder__bilirubin_total_min_status_not ordered', 'encoder__bilirubin_total_max_status_normal', 'encoder__bilirubin_total_max_status_not ordered', 'encoder__alt_max_status_normal', 'encoder__alt_max_status_not ordered', 'encoder__alt_min_status_normal', 'encoder__alt_min_status_not ordered', 'encoder__alp_max_status_low', 'encoder__alp_max_status_normal', 'encoder__alp_max_status_not ordered', 'encoder__alp_min_status_low', 'encoder__alp_min_status_normal', 'encoder__alp_min_status_not ordered', 'encoder__ast_min_status_normal', 'encoder__ast_min_status_not ordered', 'encoder__ast_max_status_normal', 'encoder__ast_max_status_not ordered', 'encoder__pco2_bg_max_status_low', 'encoder__pco2_bg_max_status_normal', 'encoder__pco2_bg_max_status_not ordered', 'encoder__pco2_bg_min_status_low', 'encoder__pco2_bg_min_status_normal', 'encoder__pco2_bg_min_status_not ordered', 'encoder__totalco2_bg_min_status_low', 'encoder__totalco2_bg_min_status_normal', 'encoder__totalco2_bg_min_status_not ordered', 'encoder__totalco2_bg_max_status_low', 'encoder__totalco2_bg_max_status_normal', 'encoder__totalco2_bg_max_status_not ordered', 'encoder__ph_min_status_low', 'encoder__ph_min_status_normal', 'encoder__ph_min_status_not ordered', 'encoder__ph_max_status_low', 'encoder__ph_max_status_normal', 'encoder__ph_max_status_not ordered', 'encoder__lactate_min_status_low', 'encoder__lactate_min_status_normal', 'encoder__lactate_min_status_not ordered', 'encoder__lactate_max_status_low', 'encoder__lactate_max_status_normal', 'encoder__lactate_max_status_not ordered', 'remainder__platelets_min', 'remainder__inr_max', 'remainder__aniongap_min', 'remainder__glucose_vital_min', 'remainder__heart_rate_max', 'remainder__creatinine_min', 'remainder__bicarbonate_lab_min', 'remainder__spo2_min', 'remainder__calcium_lab_min', 'remainder__glucose_lab_max', 'remainder__mbp_mean', 'remainder__sodium_lab_max', 'remainder__platelets_max', 'remainder__spo2_max', 'remainder__temperature_vital_mean', 'remainder__pt_max', 'remainder__hematocrit_lab_max', 'remainder__sbp_min', 'remainder__anchor_age', 'remainder__resp_rate_max', 'remainder__base_platelets']

y_axis = importance
print(y_axis.tolist())

# [6, 13, 6, 1, 3, 1, 7, 14, 0, 5, 0, 0, 0, 3, 0, 12, 2, 2, 14, 0, 6, 5, 0, 5, 0, 0, 1, 4, 0, 1, 1, 1, 0, 2, 0, 1, 0, 3, 0, 0, 4, 0, 0, 5, 0, 1, 0, 2, 2, 4, 0, 0, 0, 3, 0, 3, 0, 0, 0, 6, 9, 2, 4, 0, 5, 5, 1, 0, 2, 3, 0, 4, 7, 0, 2, 8, 1, 8, 2, 0, 3, 1, 2, 1, 6, 0, 3, 1, 2, 3, 4, 0, 5, 2, 1, 4, 0, 2, 2, 7, 3, 3, 3, 9, 2, 6, 17, 2, 4, 6, 8, 3, 3, 8, 3, 13, 2, 0, 11, 3, 0, 4, 6, 4, 1, 11, 1, 2, 3, 13, 7, 5, 4, 3, 4, 0, 2, 10, 11, 23, 7, 9, 13, 11, 5, 0, 6, 10, 0, 5, 1, 139, 62, 97, 127, 143, 127, 131, 95, 93, 111, 155, 111, 128, 47, 156, 103, 136, 127, 152, 100, 109]

print(len(y_axis))  # 172

y_axis, x_axis = zip(*sorted(zip(y_axis, x_axis), reverse=False))  # reverse sort data on 'y_axis'

print(y_axis)
# (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11, 11, 11, 12, 13, 13, 13, 13, 14, 14, 17, 23, 47, 62, 93, 95, 97, 100, 103, 109, 111, 111, 127, 127, 127, 128, 131, 136, 139, 143, 152, 155, 156)

print(x_axis)
# ('encoder__admission_location_CLINIC REFERRAL', 'encoder__admission_location_INFORMATION NOT AVAILABLE', 'encoder__admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'encoder__admission_location_PACU', 'encoder__admission_location_PROCEDURE SITE', 'encoder__alp_max_status_low', 'encoder__alp_min_status_low', 'encoder__bands_max_status_not ordered', 'encoder__bands_min_status_normal', 'encoder__bilirubin_direct_max_status_not ordered', 'encoder__bilirubin_direct_min_status_normal', 'encoder__calcium_bg_max_status_low', 'encoder__ck_cpk_max_status_low', 'encoder__ck_cpk_max_status_normal', 'encoder__ck_mb_max_status_normal', 'encoder__fibrinogen_min_status_low', 'encoder__fibrinogen_min_status_not ordered', 'encoder__glucose_bg_max_status_low', 'encoder__glucose_bg_max_status_normal', 'encoder__glucose_bg_min_status_low', 'encoder__glucose_bg_min_status_not ordered', 'encoder__hematocrit_bg_max_status_not ordered', 'encoder__hemoglobin_bg_max_status_normal', 'encoder__hemoglobin_bg_max_status_not ordered', 'encoder__hemoglobin_bg_min_status_normal', 'encoder__hemoglobin_bg_min_status_not ordered', 'encoder__lactate_max_status_low', 'encoder__lactate_min_status_low', 'encoder__ld_ldh_max_status_low', 'encoder__ld_ldh_min_status_low', 'encoder__nrbc_min_status_not ordered', 'encoder__pco2_bg_art_max_status_low', 'encoder__pco2_bg_art_min_status_low', 'encoder__po2_bg_art_min_status_not ordered', 'encoder__sodium_bg_max_status_normal', 'encoder__sodium_bg_min_status_not ordered', 'encoder__temperature_bg_min_status_not ordered', 'encoder__totalco2_bg_art_max_status_low', 'encoder__totalco2_bg_min_status_not ordered', 'encoder__ast_max_status_not ordered', 'encoder__ast_min_status_not ordered', 'encoder__bands_min_status_not ordered', 'encoder__calcium_bg_min_status_low', 'encoder__ck_mb_min_status_not ordered', 'encoder__fibrinogen_max_status_low', 'encoder__fibrinogen_max_status_normal', 'encoder__fibrinogen_max_status_not ordered', 'encoder__first_careunit_Neuro Intermediate', 'encoder__first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)', 'encoder__hematocrit_bg_min_status_not ordered', 'encoder__lactate_max_status_not ordered', 'encoder__pco2_bg_art_min_status_not ordered', 'encoder__po2_bg_art_max_status_normal', 'encoder__po2_bg_art_min_status_low', 'encoder__sodium_bg_max_status_low', 'encoder__totalco2_bg_art_max_status_not ordered', 'encoder__admission_location_TRANSFER FROM SKILLED NURSING FACILITY', 'encoder__admission_location_WALK-IN/SELF REFERRAL', 'encoder__albumin_max_status_normal', 'encoder__albumin_min_status_normal', 'encoder__alt_min_status_not ordered', 'encoder__calcium_bg_max_status_normal', 'encoder__calcium_bg_min_status_not ordered', 'encoder__ck_cpk_min_status_normal', 'encoder__fibrinogen_min_status_normal', 'encoder__ld_ldh_max_status_normal', 'encoder__pco2_bg_art_max_status_not ordered', 'encoder__pco2_bg_max_status_low', 'encoder__po2_bg_art_max_status_low', 'encoder__potassium_bg_min_status_low', 'encoder__potassium_bg_min_status_normal', 'encoder__sodium_bg_max_status_not ordered', 'encoder__sodium_bg_min_status_low', 'encoder__totalco2_bg_art_min_status_low', 'encoder__totalco2_bg_max_status_low', 'encoder__admission_location_PHYSICIAN REFERRAL', 'encoder__alp_max_status_not ordered', 'encoder__alt_max_status_not ordered', 'encoder__bilirubin_total_max_status_normal', 'encoder__bilirubin_total_max_status_not ordered', 'encoder__first_careunit_Neuro Stepdown', 'encoder__glucose_bg_max_status_not ordered', 'encoder__glucose_bg_min_status_normal', 'encoder__hemoglobin_bg_min_status_low', 'encoder__ld_ldh_max_status_not ordered', 'encoder__pco2_bg_art_min_status_normal', 'encoder__pco2_bg_max_status_normal', 'encoder__potassium_bg_max_status_low', 'encoder__potassium_bg_max_status_normal', 'encoder__potassium_bg_max_status_not ordered', 'encoder__totalco2_bg_art_max_status_normal', 'encoder__totalco2_bg_art_min_status_normal', 'encoder__totalco2_bg_min_status_low', 'encoder__albumin_min_status_not ordered', 'encoder__alp_min_status_normal', 'encoder__ast_min_status_normal', 'encoder__bands_max_status_normal', 'encoder__ck_cpk_min_status_not ordered', 'encoder__hemoglobin_bg_max_status_low', 'encoder__ld_ldh_min_status_normal', 'encoder__pco2_bg_min_status_not ordered', 'encoder__po2_bg_art_min_status_normal', 'encoder__sodium_bg_min_status_normal', 'encoder__totalco2_bg_art_min_status_not ordered', 'encoder__totalco2_bg_min_status_normal', 'encoder__admission_location_EMERGENCY ROOM', 'encoder__bilirubin_direct_max_status_normal', 'encoder__ck_mb_max_status_not ordered', 'encoder__ck_mb_min_status_normal', 'encoder__lactate_max_status_normal', 'encoder__nrbc_max_status_not ordered', 'encoder__pco2_bg_art_max_status_normal', 'encoder__pco2_bg_min_status_normal', 'encoder__ph_max_status_not ordered', 'encoder__temperature_bg_max_status_not ordered', 'encoder__albumin_max_status_not ordered', 'encoder__alp_min_status_not ordered', 'encoder__bilirubin_direct_min_status_not ordered', 'encoder__bilirubin_total_min_status_normal', 'encoder__ck_cpk_max_status_not ordered', 'encoder__first_careunit_Coronary Care Unit (CCU)', 'encoder__first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)', 'encoder__lactate_min_status_normal', 'encoder__po2_bg_art_max_status_not ordered', 'encoder__first_careunit_Surgical Intensive Care Unit (SICU)', 'encoder__ld_ldh_min_status_not ordered', 'encoder__pco2_bg_min_status_low', 'encoder__ph_min_status_normal', 'encoder__potassium_bg_min_status_not ordered', 'encoder__alt_max_status_normal', 'encoder__bilirubin_total_min_status_not ordered', 'encoder__calcium_bg_max_status_not ordered', 'encoder__calcium_bg_min_status_normal', 'encoder__albumin_max_status_low', 'encoder__ck_cpk_min_status_low', 'encoder__ph_min_status_not ordered', 'encoder__lactate_min_status_not ordered', 'encoder__totalco2_bg_max_status_normal', 'encoder__alp_max_status_normal', 'encoder__ast_max_status_normal', 'encoder__ph_max_status_normal', 'encoder__totalco2_bg_max_status_not ordered', 'encoder__admission_location_TRANSFER FROM HOSPITAL', 'encoder__alt_min_status_normal', 'encoder__first_careunit_Medical Intensive Care Unit (MICU)', 'encoder__pco2_bg_max_status_not ordered', 'encoder__ph_max_status_low', 'encoder__first_careunit_Trauma SICU (TSICU)', 'encoder__gender_M', 'encoder__albumin_min_status_low', 'encoder__ph_min_status_low', 'remainder__spo2_max', 'remainder__inr_max', 'remainder__calcium_lab_min', 'remainder__spo2_min', 'remainder__aniongap_min', 'remainder__resp_rate_max', 'remainder__pt_max', 'remainder__base_platelets', 'remainder__glucose_lab_max', 'remainder__sodium_lab_max', 'remainder__creatinine_min', 'remainder__glucose_vital_min', 'remainder__sbp_min', 'remainder__platelets_max', 'remainder__bicarbonate_lab_min', 'remainder__hematocrit_lab_max', 'remainder__platelets_min', 'remainder__heart_rate_max', 'remainder__anchor_age', 'remainder__mbp_mean', 'remainder__temperature_vital_mean')

# # plot first 21 features which have the highest feature imporatanc
#

x_axis = x_axis[-21:]
y_axis = y_axis[-21:]

print(x_axis) # ('remainder__spo2_max', 'remainder__inr_max', 'remainder__calcium_lab_min', 'remainder__spo2_min', 'remainder__aniongap_min', 'remainder__resp_rate_max', 'remainder__pt_max', 'remainder__base_platelets', 'remainder__glucose_lab_max', 'remainder__sodium_lab_max', 'remainder__creatinine_min', 'remainder__glucose_vital_min', 'remainder__sbp_min', 'remainder__platelets_max', 'remainder__bicarbonate_lab_min', 'remainder__hematocrit_lab_max', 'remainder__platelets_min', 'remainder__heart_rate_max', 'remainder__anchor_age', 'remainder__mbp_mean', 'remainder__temperature_vital_mean')

print(y_axis) # (47, 62, 93, 95, 97, 100, 103, 109, 111, 111, 127, 127, 127, 128, 131, 136, 139, 143, 152, 155, 156)


# plt.figure()
# plt.barh(x_axis, y_axis)
# # plt.yticks(rotation=90)
# # plt.xticks(rotation=90)
# # plt.xticks(fontsize=5)
# plt.yticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.xlabel('Feature Importance', fontsize=12)
# plt.ylabel('Feature name', fontsize=12)
# plt.title('Feature importance (HIT positive:HIT negative = 15:85)', fontsize=12)
# plt.savefig('imbalance_data_feature_importance.png')
# plt.show()

# # ------------------------------------------------------------------------------------------------------------------------------------
# #
# #plot ROC for each fold in CV in same plot
# #refered a comment in - https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
#
#
# def draw_cv_roc_curve(classifier, folds, x_train, y_train):
#     """
#     Draw a Cross Validated ROC Curve.
#     Args:
#         classifier: Classifier Object
#         cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
#         X: Feature Pandas DataFrame
#         y: Response Pandas Series
#     Example largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
#     """
#     # Creating ROC Curve with Cross Validation
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#
#     i = 0
#
#     # folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # new
#
#     for train, test in folds.split(x_train, y_train):
#         # classifier_DT = DecisionTreeClassifier(criterion='entropy', random_state=0)  # new - define classifier
#         probas_ = classifier.fit(x_train[train], y_train.iloc[train]).predict_proba(x_train[test])
#
#         #     probas_ = classifier_DT.fit(x_train[train], np.array(y_train.iloc[train])).predict_proba(
#         #         x_train[test])  # classifier_GB classifier
#
#         # Compute ROC curve and area the curve
#         fpr, tpr, thresholds = roc_curve(y_train.iloc[test], probas_[:, 1])
#         tprs.append(interp(mean_fpr, fpr, tpr))
#
#         tprs[-1][0] = 0.0
#         roc_auc = auc(fpr, tpr)
#         aucs.append(roc_auc)
#         plt.plot(fpr, tpr, lw=1, alpha=0.3,
#                  label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#
#         i += 1
#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#              label='Random chance', alpha=.8)
#
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)
#     plt.plot(mean_fpr, mean_tpr, color='b',
#              label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#              lw=2, alpha=.8)
#
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                      label=r'$\pm$ 1 std. dev.')
#
#     plt.xlim([-0.01, 1.01])
#     plt.ylim([-0.01, 1.01])
#     plt.xlabel('False Positive Rate', fontsize=20)
#     plt.ylabel('True Positive Rate', fontsize=20)
#     plt.title('Cross-Validation ROC curve - Balanced data set (HIT positive/negative = 1:2)', fontsize=20)
#     plt.legend(loc="lower right", prop={'size': 12})
#     # plt.savefig('balance_data_ROC.png')
#     plt.show()
#
#
# # clf = DecisionTreeClassifier(criterion='entropy', random_state=0)  # new - define classifier
# clf = LGBMClassifier(random_state=0)
#
# folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # new
#
# draw_cv_roc_curve(clf, folds, x_train, y_train)  # call the function
# # ---------------------------------------------------------------------------
# # plot precision-recall cuve
# # refered a comment in - https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
#
# def draw_cv_pr_curve(classifier, folds, x_train, y_train):
#     """
#     Draw a Cross Validated PR Curve.
#     Keyword Args:
#         classifier: Classifier Object
#         cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
#         X: Feature Pandas DataFrame
#         y: Response Pandas Series
#
#     Largely taken from: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
#     """
#     y_real = []
#     y_proba = []
#
#     i = 0
#     for train, test in folds.split(x_train, y_train):
#         probas_ = classifier.fit(x_train[train], y_train.iloc[train]).predict_proba(x_train[test])
#         # Compute ROC curve and area the curve
#         precision, recall, _ = precision_recall_curve(y_train.iloc[test], probas_[:, 1])
#
#         # Plotting each individual PR Curve
#         plt.plot(recall, precision, lw=1, alpha=0.3,
#                  label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y_train.iloc[test], probas_[:, 1])))
#
#         y_real.append(y_train.iloc[test])
#         y_proba.append(probas_[:, 1])
#
#         i += 1
#
#     y_real = np.concatenate(y_real)
#     y_proba = np.concatenate(y_proba)
#
#     precision, recall, _ = precision_recall_curve(y_real, y_proba)
#
#     plt.plot(recall, precision, color='b',
#              label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)),
#              lw=2, alpha=.8)
#
#     plt.xlim([-0.01, 1.01])
#     plt.ylim([-0.01, 1.01])
#     plt.xlabel('False Positive Rate', fontsize=20)
#     plt.ylabel('True Positive Rate', fontsize=20)
#     plt.title('Cross-Validation Precision-Recall curve - Balanced data set (HIT positive/negative = 1:2)', fontsize=20)
#     plt.legend(loc="upper right", prop={'size': 12})
#     # plt.savefig('balance_data_ROC.png')
#     plt.show()
#
#
# # clf = DecisionTreeClassifier(criterion='entropy', random_state=0)  # new - define classifier
# clf = LGBMClassifier(random_state=0)
#
# folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # new
#
# draw_cv_pr_curve(clf, folds, x_train, y_train)  # call the function
#
#
# # ------------------------------------------------------------------------------------------------------------------------------------
# #
# # # Leaf plot - values calculated from - http://araw.mede.uic.edu/cgi-bin/testcalc.pl
# #
# # x = np.array([0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1])
# # y1 = np.array([0, 0.49, 0.60, 0.68, 0.79, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.00, 1])
# # y2 = np.array([0, 0.08, 0.11, 0.15, 0.24, 0.33, 0.42, 0.52, 0.63, 0.74, 0.87, 0.99, 1])
# #
# # x_new = np.linspace(x.min(), x.max(), 500)
# #
# # f1 = interp1d(x, y1, kind='quadratic')
# # y1_smooth = f1(x_new)
# # plt.plot(x_new, y1_smooth, color='r')
# #
# # f2 = interp1d(x, y2, kind='quadratic')
# # y2_smooth = f2(x_new)
# # plt.plot(x_new, y2_smooth, color='b')
# #
# # y3 = x_new
# # plt.plot(x_new, y3, linestyle='dashed', color='g')
# # plt.xlabel('Pre-test probability', fontsize=10)
# # plt.ylabel('Post-test probability', fontsize=10)
# # plt.legend(['Positive test (LR+ = 8.59)', 'Negative test (LR- = 0.73)', 'Without test'])
# # plt.title("The leaf plot - Original data (HIT positive/negative = 15:85)", fontsize=10)
# # plt.show()
# # # ------------------------------------------------------------------------------------------------------------------------------------
#
#
# # python 7_Jan_3_balance_data_feature_importance_and_ROC.py /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/train_data_before_preprocessing.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/test_data_before_preprocessing.csv
#
#

