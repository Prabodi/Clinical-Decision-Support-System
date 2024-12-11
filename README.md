Clinical Decision Support System

This repository contains a Clinical Decision Support System (CDSS) — a Machine Learning-based classifier designed to predict Heparin-Induced Thrombocytopenia (HIT) onset at the first dose of heparin. The goal of this project is to provide a set of Python scripts for data preprocessing, feature selection, model training, and evaluation using data from the MIMIC-IV database.
Files and Scripts
1. Cohort Extraction from MIMIC-IV

    cohort_extraction.sql: SQL script for extracting the cohort from the MIMIC-IV database.

2. Data Preprocessing

    1_Data_preprocessing.py: Data preprocessing script, which handles the initial cleaning and preparation of the dataset.

3. Categorical Feature Selection

    2_Categorical_feature_selection.py: This script performs categorical feature selection using the chi-squared proportional test. It selects the most relevant categorical features for the model.

4. Preprocessing for Numerical Feature Selection

    3_Preprocessing_for_numerical_feature_selection.py: After selecting categorical features, this script preprocesses the data required for the next step: continuous feature selection.

5. Model Training After Feature Selection

    5_Train_model_after_feature_selection.py: Trains the model after feature selection. It evaluates different classification algorithms and validates both internally and externally.

6. Plot ROC and Feature Importance (Original Data)

    6_Plot_original_data_feature_importance_and_ROC.py: This script plots the ROC curve and feature importance for the original dataset (before balancing).

7. Balancing Data and Training Model

    7_Balance_data_Train_model_after_feature_selection.py: Balances the dataset by adjusting the ratio of positive to negative classes (1:2) using SMOTE, and then trains the model using different classifiers, validated internally and externally.

8. Plot ROC and Feature Importance (Balanced Data)

    8_Plot_balance_data_feature_importance_and_ROC.py: Plots the ROC curve and feature importance for the balanced dataset created by the previous script (7).

9. Plot LR Curve and Feature Importance

    9_Plot_LR_curve_and_Feature_Importance.py: This script plots Likelihood Ratio (LR) curves and feature importance for both the original and balanced datasets.

10. Plot Feature Distribution of Selected Features

    10_Plot_feature_distribution_of_selected_features.py: Plots the distribution of only the selected features from the feature selection step.

How to Run the Project

    Clone the repository:

git clone https://github.com/Prabodi/Clinical-Decision-Support-System.git

Install dependencies:

    Ensure you have Python installed.
    Install the necessary libraries by running:

    pip install -r requirements.txt

Run the individual Python scripts as needed:

    Each script is designed to be run sequentially based on the process flow (data preprocessing, feature selection, model training, etc.).
    Example:

        python 1_Data_preprocessing.py

    Ensure you have access to the MIMIC-IV dataset for cohort extraction and preprocessing.

Dependencies

    Python 3.x
    Required Python libraries (listed in requirements.txt):
        pandas
        scikit-learn
        matplotlib
        seaborn
        imbalanced-learn
        numpy
        scipy
        sqlalchemy
        SMOTE

Project Overview

The goal of this project is to develop a machine learning classifier to predict Heparin-Induced Thrombocytopenia (HIT), which can be a life-threatening condition triggered by the administration of heparin. The system works with data extracted from the MIMIC-IV database, where various steps are involved, including cohort extraction, data preprocessing, feature selection, model training, and performance evaluation. The model uses different machine learning techniques and validates its performance both internally and externally.

Additionally, this project explores the impact of data balancing on model performance by using SMOTE to balance the dataset before training. It also includes detailed evaluation through ROC curves, feature importance, and Likelihood Ratio curves.

Additional Notes

    If you are using the MIMIC-IV dataset, ensure that you have the necessary permissions and access to the database.
    The scripts assume that the data is pre-processed and stored in a format compatible with the code.
