# 1. Import Libraries

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------------------------------------------------------------

df_train_all = pd.read_csv(sys.argv[1])  # after prepocessing - training set only
df_test_all = pd.read_csv(sys.argv[2])  # after prepocessing - test set only

# selected cat fetaure

# ['first_careunit', 'admission_location', 'gender', 'lactate_min_status', 'lactate_max_status', 'ph_min_status',
#      'ph_max_status', 'totalco2_bg_min_status', 'totalco2_bg_max_status', 'pco2_bg_min_status', 'pco2_bg_max_status',
#      'ast_min_status', 'ast_max_status', 'alp_min_status', 'alp_max_status', 'alt_min_status', 'alt_max_status',
#      'bilirubin_total_min_status', 'bilirubin_total_max_status', 'albumin_min_status', 'albumin_max_status',
#      'pco2_bg_art_min_status', 'pco2_bg_art_max_status', 'po2_bg_art_min_status', 'po2_bg_art_max_status',
#      'totalco2_bg_art_min_status', 'totalco2_bg_art_max_status', 'ld_ldh_min_status', 'ld_ldh_max_status',
#      'ck_cpk_min_status', 'ck_cpk_max_status', 'ck_mb_min_status', 'ck_mb_max_status', 'fibrinogen_min_status',
#      'fibrinogen_max_status', 'bilirubin_direct_min_status', 'bilirubin_direct_max_status', 'nrbc_min_status',
#      'nrbc_max_status', 'bands_min_status', 'bands_max_status']

# slected numerical features (LGBM - 14 features)

# ['platelets_min', 'inr_max', 'aniongap_max', 'platelets_max', 'potassium_max', 'dbp_min', 'sbp_mean', 'aniongap_min', 'calcium_min', 'bun_min', 'temperature_min', 'bicarbonate_max', 'ptt_min', 'base_platelets']]
# ------------------------------------------------------------------------------------------------------------------------------------

df_train_all['label'] = np.where((df_train_all['label'] == 1), 'HIT', 'No HIT')

# box plots for numerical features

# 1. plot a feature distibution separately for HIT and no HIT

num_features = df_train_all[['platelets_min', 'inr_max', 'anchor_age']]
#
# sns.boxplot(data=df_train_all, x='label', y='platelets_min')
# plt.show()
#
# # ------------------------------------------------------------------------------------------------------------------------------------
#
# plt.figure()
#
# # 2. plot the entire numeric feature regardless of the label
#
# # initialize figure with 4 subplots in a row
# fig, ax = plt.subplots(1, 3, figsize=(10, 6))
#
# # add padding between the subplots
# plt.subplots_adjust(wspace=0.5)
#
# # draw boxplot for age in the 1st subplot
# sns.boxplot(data=df_train_all['platelets_min'], ax=ax[0], color='brown', )
# ax[0].set_xlabel('platelets_min')
#
# # draw boxplot for station_distance in the 2nd subplot
# sns.boxplot(data=df_train_all['inr_max'], ax=ax[1], color='g')
# ax[1].set_xlabel('inr_max')
#
# # draw boxplot for stores_count in the 3rd subplot
# sns.boxplot(data=df_train_all['anchor_age'], ax=ax[2], color='y')
# ax[2].set_xlabel('anchor_age')
#
# # by default, you'll see x-tick label set to 0 in each subplot
# # remove it by setting it to empty list
# for subplot in ax:
#     subplot.set_xticklabels([])
#
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------
# # 3. multiple box plots considering the label too
#
# # refer- https://proclusacademy.com/blog/quicktip/boxplot-separate-yaxis/

# plt.figure()

# # 'platelets_min', 'inr_max', 'anchor_age'
# fig, ax = plt.subplots(1, 3, figsize=(6, 6))
#
# # colour options - https://sites.google.com/view/paztronomer/blog/basic/python-colors
#
# # draw boxplot for age in the 1st subplot
# sns.boxplot(data=df_train_all, x='label', y='anchor_age', ax=ax[0], color='orange', )
# ax[0].set_xlabel('label')
#
# sns.boxplot(data=df_train_all, x='label', y='platelets_min', ax=ax[1], color='cornflowerblue', )
# ax[1].set_xlabel('label')
#
# sns.boxplot(data=df_train_all, x='label', y='inr_max', ax=ax[2], color='olivedrab', )
# ax[2].set_xlabel('label')
#
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------
# plot categorical feature distribution

# gender, lactate_max_status, ast_max_status

# refer - https://seaborn.pydata.org/tutorial/categorical.html

# initialize figure with 4 subplots in a row
# fig, axes = plt.subplots(2, 2, figsize=(6, 6))
#
# # add padding between the subplots
# plt.subplots_adjust(wspace=0.5)
# #
# # # draw boxplot for age in the 1st subplot
#
# sns.histplot(data=df_train_all, ax=axes[0, 0], x="label", hue="gender", stat="count", multiple="dodge", shrink=.8)
# axes[0, 0].set_xlabel('label')
# #
# sns.histplot(data=df_train_all, ax=axes[0, 1], x="label", hue="lactate_max_status", stat="count", multiple="dodge", shrink=.8)
# axes[0, 1].set_xlabel('label')
#
# sns.histplot(data=df_train_all, ax=axes[1, 0], x="label", hue="ast_max_status", stat="count", multiple="dodge", shrink=.8)
# axes[1, 0].set_xlabel('label')
#
# sns.histplot(data=df_train_all, ax=axes[1, 1], x="label", hue="alt_max_status", stat="count", multiple="dodge", shrink=.8)
# axes[1, 1].set_xlabel('label')
#
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------

# initialize figure with 4 subplots in a row
fig, axes = plt.subplots(2, 3, figsize=(6, 6))

# add padding between the subplots
plt.subplots_adjust(wspace=0.4)
plt.subplots_adjust(hspace=0.4)

sns.boxplot(data=df_train_all, ax=axes[0, 0], x='label', y='anchor_age') #, color='orange', )
axes[0, 0].set_xlabel('label', fontsize=20)
axes[0, 0].set_ylabel('Patient count', fontsize=20)
axes[0, 0].yaxis.set_tick_params(labelsize=20)
axes[0, 0].xaxis.set_tick_params(labelsize=20)

sns.boxplot(data=df_train_all, ax=axes[0, 1], x='label', y='platelets_min') #, color='cornflowerblue', )
axes[0, 1].set_xlabel('label', fontsize=20)
axes[0, 1].set_ylabel('label', fontsize=20)
axes[0, 1].yaxis.set_tick_params(labelsize=20)
axes[0, 1].xaxis.set_tick_params(labelsize=20)

sns.boxplot(data=df_train_all, ax=axes[0, 2], x='label', y='inr_max') #, color='olivedrab', )
axes[0, 2].set_xlabel('label', fontsize=20)
axes[0, 2].set_ylabel('label', fontsize=20)
axes[0, 2].yaxis.set_tick_params(labelsize=20)
axes[0, 2].xaxis.set_tick_params(labelsize=20)

gfg = sns.histplot(data=df_train_all, ax=axes[1, 0], x="label", hue="gender", stat="count", multiple="dodge", shrink=.8)
axes[1, 0].set_xlabel('label', fontsize=20)
axes[1, 0].set_ylabel('Patient count', fontsize=20)
axes[1, 0].yaxis.set_tick_params(labelsize=20)
axes[1, 0].xaxis.set_tick_params(labelsize=20)
#axes[1, 0].legend(fontsize=20)
#
sns.histplot(data=df_train_all, ax=axes[1, 1], x="label", hue="lactate_max_status", stat="count", multiple="dodge", shrink=.8)
axes[1, 1].set_xlabel('label', fontsize=20)
axes[1, 1].set_ylabel('Patient count', fontsize=20)
axes[1, 1].yaxis.set_tick_params(labelsize=20)
axes[1, 1].xaxis.set_tick_params(labelsize=20)


axes[1, 2].legend(fontsize=50)
sns.histplot(data=df_train_all, ax=axes[1, 2], x="label", hue="ast_max_status", stat="count", multiple="dodge", shrink=.8)
axes[1, 2].set_xlabel('label', fontsize=20)
axes[1, 2].set_ylabel('Patient count', fontsize=20)
axes[1, 2].yaxis.set_tick_params(labelsize=20)
axes[1, 2].xaxis.set_tick_params(labelsize=20)

plt.show()

#sns.catplot(data=df_train_all, x="label", y=Counter["gender"], kind="bar")

#sns.countplot(x ='gender', data = df_train_all)

# sns.histplot(data=tips, x="day", hue="sex", multiple="dodge", shrink=.8) # https://seaborn.pydata.org/generated/seaborn.histplot.html

# sns.histplot(binwidth=0.5, x="Playing_Role", hue="Bought_By", data=df, stat="count", multiple="stack") #https://datascience.stackexchange.com/questions/89692/plot-two-categorical-variables
#
# palette1 = ["#fee090", "#fdae61", "#4575b4", "#313695"]
#
# sns.histplot(data=df_train_all, x="label", hue="lactate_max_status", stat="count", multiple="dodge", shrink=.8) #palette=palette1
#
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------
# python 9_Jan_3_Feature_distribution_of_selected-features.py /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/train_data_before_preprocessing.csv /Users/psenevirathn/Desktop/PhD/Coding/Python/input_csv_files/test_data_before_preprocessing.csv
