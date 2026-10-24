# Customer Churn Prediction

## Project Overview
Customer Churn Prediction is a machine learning project that predicts whether a telecom customer is likely to leave the company or continue using its services.

The project analyzes customer information and compares different machine learning models to identify customers who may leave.

## Objectives
- Analyze customer data to understand churn patterns.
- Perform Exploratory Data Analysis (EDA) using graphs.
- Train and evaluate machine learning models.
- Predict whether customers are likely to stay or leave.
- Compare model performance using evaluation metrics.

## Dataset
The project uses the **IBM Telco Customer Churn dataset**, containing 7,043 customer records and 33 columns.

The dataset includes information such as:
- Customer tenure
- Contract type
- Internet service
- Monthly charges
- Payment method
- Technical support
- Churn status

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn (where supported)
- Scikit-learn
- XGBoost

## Machine Learning Models
The following models were trained and evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

## Exploratory Data Analysis
The project includes visualizations to study:
- Customer churn distribution
- Churn by contract type
- Customer tenure
- Internet service and churn
- Monthly charges
- Payment methods

## Model Evaluation Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 76.3% | 54.1% | 70.3% | 61.2% | 0.835 |
| Random Forest | 76.4% | 54.2% | 70.6% | 61.3% | 0.834 |
| XGBoost | 77.4% | 55.7% | 71.4% | 62.6% | 0.842 |

XGBoost achieved the highest test accuracy and F1-score in this run.

## Project Files
- `main.py` — Performs data exploration and creates visualizations.
- `model.py` — Prepares data, trains models, and evaluates performance.
- `Telco_customer_churn.csv` — Dataset used for analysis.
- `customer_churn_predictions.csv` — Contains actual churn labels, predicted labels, churn probabilities, and model information.

## How to Run the Project

1. Install Python.
2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib scikit-learn xgboost
   ```

3. Place the dataset in the project folder.
4. Open a terminal in the appropriate project directory.
5. Run the exploratory analysis:

   ```bash
   python main.py
   ```

6. Run the machine learning models:

   ```bash
   python model.py
   ```

## Conclusion
This project demonstrates how machine learning can help telecom companies identify customers who may leave their services. By analyzing customer information and comparing different models, businesses can identify potential churn and plan customer-retention strategies.

## Future Improvements
- Improve model performance through hyperparameter tuning.
- Test additional machine learning algorithms.
- Develop an interactive dashboard for churn predictions.
- Use the model to score new customer records.
