# Insurance Claim Prediction 

This project aims to predict whether a building will have at least **one insurance claim during the insured period** using machine learning techniques. The dataset contains features related to buildings, policies, and historical claims. Ensemble models like **Gradient Boosting** and **Random Forest** were applied to identify high-risk buildings and support insurance risk management decisions.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Problem Statement](#problem-statement)  
3. [Dataset Description](#dataset-description)  
4. [Data Exploration & Preprocessing](#data-exploration--preprocessing)  
5. [Modeling Approach](#modeling-approach)  
6. [Evaluation Metrics](#evaluation-metrics)  
7. [Results and Interpretation](#results-and-interpretation)  
8. [Hyperparameter Tuning](#hyperparameter-tuning)  

---

## Project Overview

Insurance companies face financial risk if claim-prone buildings are not identified accurately. This project uses **machine learning classification models** to predict buildings that are likely to experience at least one claim during their insured period. The model assists in **risk assessment, premium pricing, and loss mitigation**.

---

## Problem Statement

Given historical building and policy data:

- **Input:** Features such as building type, age, location, coverage, and prior claims.  
- **Output:** Binary classification:
  - `0` → No claim during the insured period  
  - `1` → At least one claim during the insured period  

Challenges addressed:

- **Imbalanced dataset** (claims are rare)
- Complex interactions between risk factors



## Dataset Description

Customer Id	          Identification number for the Policy holder
YearOfObservation	    year of observation for the insured policy
Insured_Period	      duration of insurance policy in Olusola Insurance. (Ex: Full year insurance, Policy Duration = 1; 6 months = 0.5
Residential	          is the building a residential building or not
Building_Painted	    is the building painted or not (N-Painted, V-Not Painted)
Building_Fenced	      is the building fence or not (N-Fenced, V-Not Fenced)
Garden building       has garden or not (V-has garden; O-no garden)
Settlement	          Area where the building is located. (R- rural area; U- urban area)
Building Dimension	  Size of the insured building in m2
Building_Type	        The type of building (Type 1, 2, 3, 4)
Date_of_Occupancy	    date building was first occupied
NumberOfWindows	      number of windows in the building
Geo Code	            Geographical Code of the Insured building
Claim	target variable. (0: no claim, 1: at least one claim over insured period).

---

## Data Exploration & Preprocessing

Key steps performed:

1. **Handling Missing Values** – Imputed or removed missing entries.  
2. **Feature Encoding** – Converted categorical variables into numeric format using one-hot or label encoding.  
3. **Feature Scaling** – Applied StandardScaler for models sensitive to feature magnitude.  
4. **Class Imbalance Analysis** –
   Checked distribution:
   df['Claim'].value_counts(normalize=True) * 100

   Result: Minority class (Claim = 1) is underrepresented (~23%).

Checked Claim rate over insured period
   Bin insured period
    df['Insured_bin'] = pd.cut(df['Insured_Period'], bins=10)
    claim_rate = df.groupby('Insured_bin')['Claim'].mean()
    Result: Buildings insured for a short duration tend to have lower claim probabilities.

Checked Building Dimension vs Insured Period
    result:Buildings of various sizes are insured for both short and long periods. A few large buildings appear as outliers, especially at longer insured periods.


## Modeling Approach

Models evaluated:

**Logistic Regression** – Baseline model for linear relationships.

**Random Forest Classifier**– Handles non-linear interactions, robust to overfitting.

**Gradient Boosting Classifier** – Best-performing ensemble, emphasizes misclassified examples.

**Key techniques used:**

Sample weighting to address imbalance

Threshold tuning to improve minority-class recall

Cross-validation with stratified splits

## Evaluation Metrics

Because of class imbalance, standard accuracy is not enough. Metrics used:

**Metric	Description**
Accuracy	Percentage of correct predictions
Precision	Correct positive predictions / all predicted positives
Recall	Correct positive predictions / all actual positives
F1-score	Harmonic mean of precision and recall
ROC-AUC	Probability-based measure of discriminative power
Confusion Matrix	True positives, false positives, true negatives, false negatives

## Results and Interpretation

Gradient Boosting Classifier final performance:

Class	Precision	Recall	F1-score	Support
0	0.84	0.75	0.79	1105
1	0.38	0.53	0.44	327
Accuracy			0.70	1432
ROC-AUC			0.692	

**Interpretation:**

Strong performance on majority class (no claims)

Moderate sensitivity to claims (minority class)

Fair overall discriminative ability (ROC-AUC ≈ 0.69)

Threshold tuning and sample weighting improved minority-class detection

Confusion Matrix:

[[826 279]
 [154 173]]

## Hyperparameter Tuning

Used GridSearchCV for tuning parameters:

n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample

Sample weighting applied for minority class

Decision threshold lowered (e.g., 0.35) to increase recall for claim cases

Improved Class 1 recall from ~0.50 → ~0.60 without major loss in Class 0 performance

