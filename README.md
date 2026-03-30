# Predicting Kick Cars at Auto Auctions

## Overview
A supervised machine learning project focused on predicting whether auction vehicles are likely to become problematic purchases, commonly referred to as kick cars.

The project combines classification modeling and explainability to address a high-value business problem in automotive remarketing and risk screening.

## Business Problem
Auto auctions involve pricing and purchasing decisions under uncertainty. Buying a defective or high-risk vehicle can lead to downstream repair costs, margin erosion, and inventory inefficiency.

The objective is to identify likely bad buys before purchase using historical auction and vehicle data.

## Solution
This project develops a classification pipeline that:
- preprocesses historical vehicle and auction records
- compares multiple candidate models
- evaluates performance using classification metrics
- interprets model decisions with SHAP

## Methodology

### Data Preparation
Prepared auction vehicle data by cleaning records, encoding variables, and handling missingness.

### Modeling
Tested classification approaches and selected Random Forest as the strongest performer.

### Evaluation
Measured model performance using accuracy and ROC-AUC.

### Interpretability
Used SHAP to identify which features most influenced high-risk predictions.

## Results
- Random Forest achieved 93.12% accuracy
- ROC-AUC reached 0.9634
- SHAP analysis improved interpretability for decision support

## Tech Stack
Python, scikit-learn, Pandas, NumPy, SHAP

## Repository Structure
```text
Predicting-Kick-Cars-at-Auto-Auctions/
├── notebooks/
├── data/
├── outputs/
└── README.md
```

## How to Run
	1.	load the source dataset into the project directory
	2.	run preprocessing and model training notebooks or scripts
	3.	evaluate output metrics and interpret SHAP plots

## Future Improvements
	•	package preprocessing and modeling into modular scripts
	•	introduce calibration and threshold tuning
	•	compare gradient boosting methods
	•	deploy an inference interface for analyst use

## Author
Ashlesha Kadam
