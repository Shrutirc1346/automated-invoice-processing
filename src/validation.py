#=============================
# Step 48.8 : Build Validation Module
#=============================
print("\n========== Step 48.8 : BUILD VALIDATION MODULE ========== ")

import pandas as pd


def validate_financial_data(dataframe):

    result = dataframe.copy()

    result["Calculated Net Worth"] = (
        result["Quantity"] * result["Unit Price"]
    )

    result["Net Worth Valid"] = (
        (result["Calculated Net Worth"] - result["Net Worth"]).abs()
        <= 0.01
    )

    result["Calculated Gross Worth"] = (
        result["Net Worth"] * (1 + result["VAT"] / 100)
    )

    result["Gross Worth Valid"] = (
        (result["Calculated Gross Worth"] - result["Gross Worth"]).abs()
        <= 0.01
    )

    result["Validation Result"] = (
        result["Net Worth Valid"] &
        result["Gross Worth Valid"]
    )

    return result


print("\nValidation module loaded successfully.")
print("Function available: validate_financial_data()")

print("\n========== STEP 48.8 COMPLETED ========== ")