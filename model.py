
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==================================================
# 1. LOAD AND PREPARE DATA
# ==================================================

df = pd.read_csv("Customer-churn-Prediction/Telco_customer_churn.csv")

features = [
    "Contract",
    "Tenure Months",
    "Internet Service",
    "Monthly Charges",
    "Payment Method",
    "Paperless Billing",
    "Tech Support",
    "Online Security"
]

X = df[features].copy()
y = df["Churn Label"].map({"No": 0, "Yes": 1})

# Remove rows with missing target labels, if any
valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows].astype(int)

# Convert categorical columns into numeric columns
X = pd.get_dummies(X, drop_first=True)

# Split off the final test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create a separate validation set for threshold selection
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("Data preparation complete!")
print("Training data:", X_fit.shape)
print("Validation data:", X_val.shape)
print("Testing data:", X_test.shape)


# ==================================================
# 2. CREATE THE THREE MODELS
# ==================================================

# Logistic Regression
lr_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# XGBoost: give more weight to the churn class
negative_count = int((y_fit == 0).sum())
positive_count = int((y_fit == 1).sum())
class_ratio = negative_count / max(positive_count, 1)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=class_ratio,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}


# ==================================================
# 3. TRAIN MODELS AND SELECT THRESHOLDS
#    USING VALIDATION DATA ONLY
# ==================================================

thresholds = {}

print("\n--- Validation: Threshold Selection ---")

for name, model in models.items():

    model.fit(X_fit, y_fit)

    val_prob = model.predict_proba(X_val)[:, 1]

    # Select the threshold giving the highest validation F1-score
    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(0.20, 0.71, 0.02):

        val_pred = (val_prob >= threshold).astype(int)

        score = f1_score(
            y_val,
            val_pred,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    thresholds[name] = best_threshold

    print(
        f"{name}: threshold = {best_threshold:.2f}, "
        f"validation F1 = {best_f1:.3f}"
    )


# ==================================================
# 4. RETRAIN ON ALL TRAINING DATA
# ==================================================

for model in models.values():
    model.fit(X_train, y_train)


# ==================================================
# 5. EVALUATE ALL THREE MODELS
# ==================================================

results = []
test_predictions = {}
test_probabilities = {}

for name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    # Use the threshold selected using validation data
    predictions = (
        probabilities >= thresholds[name]
    ).astype(int)

    test_probabilities[name] = probabilities
    test_predictions[name] = predictions

    results.append({
        "Model": name,
        "Threshold": thresholds[name],
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

    print(f"\n--- {name} Classification Report ---")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Stayed", "Left"],
        zero_division=0
    ))


# Display the model comparison
results_df = pd.DataFrame(results)

print("\n--- Final Model Comparison ---")
print(results_df.round(3).to_string(index=False))


# ==================================================
# 6. SELECT THE BEST MODEL USING VALIDATION F1
# ==================================================

validation_results = []

for name, model in models.items():
    val_prob = model.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= thresholds[name]).astype(int)

    validation_results.append({
        "Model": name,
        "Validation F1": f1_score(
            y_val, val_pred, zero_division=0
        )
    })

validation_df = pd.DataFrame(validation_results)

print("\n--- Validation Model Comparison ---")
print(validation_df.round(3).to_string(index=False))

best_index = validation_df["Validation F1"].idxmax()
best_model_name = validation_df.loc[best_index, "Model"]

print("\nSelected model:", best_model_name)
print("Selected threshold:", thresholds[best_model_name])




# Confusion matrix for the selected model
best_predictions = test_predictions[best_model_name]

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Left"]
)

disp.plot()
plt.title(f"{best_model_name} Confusion Matrix")
plt.tight_layout()
plt.show()


# ==================================================
# 8. SAVE CUSTOMER-LEVEL PREDICTIONS
# ==================================================

best_probabilities = test_probabilities[best_model_name]

prediction_results = pd.DataFrame({
    "Actual Churn": y_test.map({
        0: "Stayed",
        1: "Left"
    }),
    "Predicted Churn": pd.Series(
        best_predictions,
        index=y_test.index
    ).map({
        0: "Likely to Stay",
        1: "Likely to Leave"
    }),
    "Churn Probability (%)": (
        best_probabilities * 100
    ).round(2),
    "Model Used": best_model_name,
    "Decision Threshold": thresholds[best_model_name]
}, index=y_test.index)

print("\n--- Customer Churn Predictions ---")
print(prediction_results.head(20).to_string())

prediction_results.to_csv(
    "customer_churn_predictions.csv",
    index=True,
    index_label="Customer Row Index"
)

print("\nPredictions saved to customer_churn_predictions.csv")
print("\nProject evaluation completed!")
