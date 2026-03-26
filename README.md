## Auto Auctions Risk Prediction

### A Machine Learning Approach to Identifying High-Risk Vehicle Purchases

#### Overview

Auto auctions involve purchasing vehicles under uncertainty, often with limited inspection and incomplete information. This project builds an end-to-end machine learning pipeline to predict whether a vehicle is likely to be a “bad buy,” helping reduce financial risk and improve procurement decisions.

The problem is framed as a binary classification task, where the model predicts whether a vehicle will result in downstream loss.

#### Problem Statement

Each vehicle is classified as:
	•	1 (Bad Buy): likely to incur loss due to condition or pricing inefficiencies
	•	0 (Safe Buy): expected to perform within acceptable cost and resale margins

The main challenge lies in class imbalance and nonlinear relationships between pricing signals, vehicle attributes, and market benchmarks.

#### Data & Features

The dataset includes:
	•	Vehicle characteristics (age, odometer, make/model)
	•	Acquisition price and auction details
	•	Market benchmarks (MMR acquisition and retail values)
	•	Condition and transaction metadata

Feature engineering focused on extracting meaningful economic signals:
	•	Price differences between acquisition and market value
	•	Depreciation patterns across age and mileage
	•	Interaction effects between pricing and condition

Categorical features were one-hot encoded, and missing values were handled using median (numerical) and mode (categorical) imputation.

#### Methodology

##### Class Imbalance Handling

SMOTE (Synthetic Minority Oversampling Technique) was used to balance the dataset and improve detection of “bad buys.”

##### Models Trained
	•	Logistic Regression (baseline)
	•	Random Forest
	•	XGBoost
	•	LightGBM

##### Evaluation Metrics
	•	ROC-AUC (primary metric)
	•	Accuracy
	•	Precision, Recall, F1-score


#### Results

Model	Accuracy	ROC-AUC
Logistic Regression	83.94%	0.8959
Random Forest	93.12%	0.9634
XGBoost	92.16%	0.9579
LightGBM	91.94%	0.9559

Best Model: Random Forest
	•	Accuracy: 93.12%
	•	ROC-AUC: 0.9634
	•	Precision (Bad Buy): 0.99
	•	Recall (Bad Buy): 0.87
	•	F1-score (Bad Buy): 0.93

Tree-based ensemble models significantly outperformed the linear baseline, capturing nonlinear relationships between pricing and vehicle attributes.

#### Model Interpretation

SHAP (SHapley Additive Explanations) was used to interpret model predictions.

##### Key insights:
	•	Large gaps between acquisition price and market value increase risk
	•	Pricing inconsistencies are strong predictors of bad buys
	•	Nonlinear interactions between mileage, age, and cost are important

#### Business Impact
	•	Reduces risk of high-loss vehicle purchases
	•	Improves procurement decision-making
	•	Enables better inventory quality and turnover
	•	Supports data-driven auction strategies

With a ROC-AUC of 0.96+, the model provides a strong signal for identifying risky inventory before purchase.

#### Tech Stack
	•	Python (pandas, NumPy)
	•	scikit-learn, XGBoost, LightGBM
	•	imbalanced-learn (SMOTE)
	•	matplotlib, seaborn
	•	SHAP
	•	Jupyter Notebook

#### How to Run
- git clone https://github.com/your-username/auto-auction-risk-prediction.git
- cd auto-auction-risk-prediction
- pip install -r requirements.txt
- jupyter notebook

#### Project Structure

##### ├── data/
##### ├── notebooks/
##### ├── outputs/
##### ├── models/
##### ├── README.md
##### └── requirements.txt

#### Future Work
	•	Hyperparameter tuning (Optuna)
	•	Cost-sensitive modeling
	•	Real-time inference pipeline
	•	Integration with auction decision systems

#### Summary

This project demonstrates a complete machine learning workflow for risk prediction in auction environments, combining feature engineering, class imbalance handling, ensemble models, and interpretability. The resulting system is both accurate and practical for real-world deployment.
