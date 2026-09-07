#=============================
# Step 48.28 : Generate Final Visualizations
#=============================
print("\n========== Step 48.28 : GENERATE FINAL VISUALIZATIONS ========== ")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


#=============================
# Step 48.28 : Load Final CSV
#=============================
print("\n========== Step 48.28 : LOAD FINAL CSV ========== ")

input_path = Path(
    "reports/final_invoice_output.csv"
)

data = pd.read_csv(
    input_path
)

print(
    "\nFinal CSV loaded successfully."
)

print(
    "Rows:",
    len(data)
)

print(
    "Columns:",
    len(data.columns)
)


#=============================
# Step 48.28 : Prepare Visualization Folder
#=============================
print("\n========== Step 48.28 : PREPARE VISUALIZATION FOLDER ========== ")

output_dir = Path(
    "reports/visualizations"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

print(
    "\nVisualization folder ready:",
    output_dir
)


#=============================
# Step 48.28 : Prepare Numeric Data
#=============================
print("\n========== Step 48.28 : PREPARE NUMERIC DATA ========== ")

numeric_columns = [
    "Quantity",
    "Unit Price",
    "Net Worth",
    "VAT",
    "Gross Worth",
    "Anomaly Score"
]

for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

print(
    "\nNumeric columns prepared successfully."
)


#=============================
# Step 48.28 : Unit Price Box Plot
#=============================
print("\n========== Step 48.28 : UNIT PRICE BOX PLOT ========== ")

plt.figure(
    figsize=(8, 5)
)

plt.boxplot(
    data["Unit Price"].dropna()
)

plt.title(
    "Unit Price Distribution"
)

plt.ylabel(
    "Unit Price"
)

plt.tight_layout()

plt.savefig(
    output_dir / "unit_price_boxplot.png",
    dpi=300
)

plt.close()


#=============================
# Step 48.28 : Net Worth Box Plot
#=============================
print("\n========== Step 48.28 : NET WORTH BOX PLOT ========== ")

plt.figure(
    figsize=(8, 5)
)

plt.boxplot(
    data["Net Worth"].dropna()
)

plt.title(
    "Net Worth Distribution"
)

plt.ylabel(
    "Net Worth"
)

plt.tight_layout()

plt.savefig(
    output_dir / "net_worth_boxplot.png",
    dpi=300
)

plt.close()


#=============================
# Step 48.28 : Gross Worth Box Plot
#=============================
print("\n========== Step 48.28 : GROSS WORTH BOX PLOT ========== ")

plt.figure(
    figsize=(8, 5)
)

plt.boxplot(
    data["Gross Worth"].dropna()
)

plt.title(
    "Gross Worth Distribution"
)

plt.ylabel(
    "Gross Worth"
)

plt.tight_layout()

plt.savefig(
    output_dir / "gross_worth_boxplot.png",
    dpi=300
)

plt.close()


#=============================
# Step 48.28 : Net Worth vs Gross Worth
#=============================
print("\n========== Step 48.28 : NET WORTH VS GROSS WORTH ========== ")

normal_records = data[
    data["IsAnomaly"] == False
]

anomaly_records = data[
    data["IsAnomaly"] == True
]

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    normal_records["Net Worth"],
    normal_records["Gross Worth"],
    alpha=0.5,
    label="Normal"
)

plt.scatter(
    anomaly_records["Net Worth"],
    anomaly_records["Gross Worth"],
    alpha=0.7,
    label="Anomaly"
)

plt.title(
    "Net Worth vs Gross Worth"
)

plt.xlabel(
    "Net Worth"
)

plt.ylabel(
    "Gross Worth"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    output_dir / "net_worth_vs_gross_worth_anomalies.png",
    dpi=300
)

plt.close()


#=============================
# Step 48.28 : Anomaly Score vs Net Worth
#=============================
print("\n========== Step 48.28 : ANOMALY SCORE VS NET WORTH ========== ")

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    data["Net Worth"],
    data["Anomaly Score"],
    alpha=0.5
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title(
    "Anomaly Score vs Net Worth"
)

plt.xlabel(
    "Net Worth"
)

plt.ylabel(
    "Anomaly Score"
)

plt.tight_layout()

plt.savefig(
    output_dir / "anomaly_score_vs_net_worth.png",
    dpi=300
)

plt.close()


#=============================
# Step 48.28 : Anomaly Distribution
#=============================
print("\n========== Step 48.28 : ANOMALY DISTRIBUTION ========== ")

normal_count = (
    data["IsAnomaly"] == False
).sum()

anomaly_count = (
    data["IsAnomaly"] == True
).sum()

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    ["Normal", "Anomaly"],
    [
        normal_count,
        anomaly_count
    ]
)

plt.title(
    "Anomaly Distribution"
)

plt.ylabel(
    "Number of Records"
)

plt.tight_layout()

plt.savefig(
    output_dir / "anomaly_distribution.png",
    dpi=300
)

plt.close()


#=============================
# Step 48.28 : Financial Validation Status
#=============================
print("\n========== Step 48.28 : FINANCIAL VALIDATION STATUS ========== ")

valid_count = (
    data["Validation Result"] == True
).sum()

invalid_count = (
    data["Validation Result"] == False
).sum()

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    ["Valid", "Invalid"],
    [
        valid_count,
        invalid_count
    ]
)

plt.title(
    "Financial Validation Status"
)

plt.ylabel(
    "Number of Records"
)

plt.tight_layout()

plt.savefig(
    output_dir / "anomaly_validation_status.png",
    dpi=300
)

plt.close()


#=============================
# Step 48.28 : Display Final Results
#=============================
print("\n========== Step 48.28 : DISPLAY FINAL RESULTS ========== ")

print(
    "\nTotal records:",
    len(data)
)

print(
    "Normal records:",
    normal_count
)

print(
    "Anomaly records:",
    anomaly_count
)

print(
    "Valid financial records:",
    valid_count
)

print(
    "Invalid financial records:",
    invalid_count
)

print(
    "\nFinal visualization files:"
)

for file_path in sorted(
    output_dir.glob("*.png")
):

    print(
        file_path
    )


print(
    "\n========== STEP 48.28 COMPLETED ========== "
)