#=============================
# Step 48.7 : Build Anomaly Detection Module
#=============================
print("\n========== Step 48.7 : BUILD ANOMALY DETECTION MODULE ========== ")

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


FEATURE_COLUMNS = [
    "Quantity",
    "Unit Price",
    "Net Worth",
    "Gross Worth"
]


def detect_anomalies(dataframe):

    data = dataframe[FEATURE_COLUMNS].copy()

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42
    )

    predictions = model.fit_predict(scaled_data)

    anomaly_scores = model.decision_function(scaled_data)

    result = dataframe.copy()

    result["Prediction"] = predictions
    result["Anomaly Score"] = anomaly_scores
    result["IsAnomaly"] = predictions == -1

    return result


print("\nAnomaly detection module loaded successfully.")
print("Function available: detect_anomalies()")

print("\n========== STEP 48.7 COMPLETED ========== ")