# Insurance Claim Prediction Project

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
9. [Usage Instructions](#usage-instructions)  
10. [Future Improvements](#future-improvements)  
11. [Repository Structure](#repository-structure)  
12. [License](#license)  
13. [References](#references)

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
4. **Class Imbalance Analysis** – Checked distribution:
   ```python
   df['Claim'].value_counts(normalize=True) * 100
