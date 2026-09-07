#=============================
# Step 48.23 : Re-run Modular Pipeline
#=============================
print("\n========== Step 48.23 : RE-RUN MODULAR PIPELINE ========== ")

import os
import pandas as pd

from src.validation import validate_financial_data
from src.anomaly_detector import detect_anomalies
from src.ocr_engine import extract_text

from src.parser import (
    extract_invoice_information,
    extract_items,
    extract_all_ocr_financial_rows
)


#=============================
# Step 48.22 : Connect Financial Parsers
#=============================
print("\n========== Step 48.22 : CONNECT FINANCIAL PARSERS ========== ")


def process_invoice(image_path):

    ocr_text = extract_text(
        image_path
    )

    invoice_information = extract_invoice_information(
        ocr_text
    )

    items = extract_items(
        ocr_text
    )

    financial_records = extract_all_ocr_financial_rows(
        ocr_text
    )

    records = []

    for index, item in enumerate(items):

        financial_record = (
            financial_records[index]
            if index < len(financial_records)
            else {}
        )

        record = {

            "Image Name": os.path.basename(
                image_path
            ),

            "Invoice Number": invoice_information[
                "Invoice Number"
            ],

            "Invoice Date": invoice_information[
                "Invoice Date"
            ],

            "Vendor Name": invoice_information[
                "Vendor Name"
            ],

            "Total Amount": invoice_information[
                "Total Amount"
            ],

            "Item Number": item[
                "Item Number"
            ],

            "Description": item[
                "Description"
            ],

            "Quantity": item[
                "Quantity"
            ],

            "Unit Price": financial_record.get(
                "Unit Price"
            ),

            "Net Worth": financial_record.get(
                "Net Worth"
            ),

            "VAT": financial_record.get(
                "VAT"
            ),

            "Gross Worth": financial_record.get(
                "Gross Worth"
            )

        }

        records.append(
            record
        )

    records_dataframe = pd.DataFrame(
        records
    )

    validated_records = validate_financial_data(
        records_dataframe
    )

    return validated_records.to_dict(
        orient="records"
    )


#=============================
# Step 48.12F : Fix Invoice Folder Integration
#=============================
print("\n========== Step 48.12F : FIX INVOICE FOLDER INTEGRATION ========== ")


def process_invoice_folder(invoice_folder):

    all_records = []

    image_files = [
        file_name
        for file_name in os.listdir(
            invoice_folder
        )
        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    for index, file_name in enumerate(
        image_files,
        start=1
    ):

        image_path = os.path.join(
            invoice_folder,
            file_name
        )

        print(
            f"\nProcessing invoice {index}/{len(image_files)}: {file_name}"
        )

        try:

            records = process_invoice(
                image_path
            )

            all_records.extend(
                records
            )

        except Exception as error:

            print(
                f"Error processing {file_name}: {error}"
            )


    dataframe = pd.DataFrame(
        all_records
    )

    if dataframe.empty:

        return dataframe


    anomaly_result = detect_anomalies(
        dataframe
    )

    return anomaly_result


print("\nIntegrated production pipeline loaded successfully.")
print("Functions available:")
print("process_invoice()")
print("process_invoice_folder()")

print("\n========== STEP 48.23 READY ========== ")