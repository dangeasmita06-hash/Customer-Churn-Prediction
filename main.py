
import pandas as pd

# Load the dataset
df = pd.read_csv("Customer-churn-Prediction/Telco_customer_churn.csv")

# Display the first 5 rows
print(df.head())

# Display the dataset size
print("Dataset shape:", df.shape)

# Count customers who left and stayed
print(df["Churn Label"].value_counts())



# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())




# Create the customer churn graph
import matplotlib.pyplot as plt

churn_counts = df["Churn Label"].value_counts()

ax = churn_counts.plot(kind="bar")

# Display exact values above each bar
for bar in ax.patches:
    ax.annotate(
        str(int(bar.get_height())),
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="bottom"
    )

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Label")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()


# Graph 2: Churn by Contract Type

contract_churn = df.groupby(
    ["Contract", "Churn Label"]
).size().unstack(fill_value=0)

ax = contract_churn.plot(kind="bar")
for container in ax.containers:
    ax.bar_label(container)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn Label")
plt.tight_layout()



# Graph 3: Churn by Tenure

df["Tenure Group"] = pd.cut(
    df["Tenure Months"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"],
    include_lowest=True
)

tenure_churn = df.groupby(
    ["Tenure Group", "Churn Label"],
    observed=False
).size().unstack(fill_value=0)

ax = tenure_churn.plot(kind="bar")

# Display exact values on the bars
for container in ax.containers:
    ax.bar_label(container)

plt.title("Customer Churn by Tenure")
plt.xlabel("Customer Tenure")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn Label")
plt.tight_layout()




# Graph 4: Churn by Internet Service

internet_churn = df.groupby(
    ["Internet Service", "Churn Label"]
).size().unstack(fill_value=0)

ax = internet_churn.plot(kind="bar")

# Display exact values on the bars
for container in ax.containers:
    ax.bar_label(container)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn Label")
plt.tight_layout()



# Graph 5: Churn by Monthly Charges


# Graph 5: Monthly Charges by Churn (Violin Plot)

no_churn = df[df["Churn Label"] == "No"]["Monthly Charges"]
yes_churn = df[df["Churn Label"] == "Yes"]["Monthly Charges"]

plt.figure(figsize=(8, 5))

plt.violinplot(
    [no_churn, yes_churn],
    showmedians=True
)

plt.title("Monthly Charges by Customer Churn")
plt.xlabel("Churn Label")
plt.ylabel("Monthly Charges")
plt.xticks([1, 2], ["No - Stayed", "Yes - Left"])
plt.tight_layout()



# Graph 6: Churn by Payment Method

payment_churn = df.groupby(
    ["Payment Method", "Churn Label"]
).size().unstack(fill_value=0)

ax = payment_churn.plot(kind="bar")

# Display exact values on the bars
for container in ax.containers:
    ax.bar_label(container)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=25, ha="right")
plt.legend(title="Churn Label")
plt.tight_layout()

plt.show()
