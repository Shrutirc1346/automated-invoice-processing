#=============================
# Step 48.11B : Fix Integrated Production Pipeline
#=============================
print("\n========== Step 48.11B : FIX INTEGRATED PRODUCTION PIPELINE ========== ")

import os
import pandas as pd

from src.ocr_engine import extract_text

print("\n========== Step 48.16A : CONNECT SPECIAL-SEPARATED FINANCIAL PARSER ========== ")

from src.parser import (
    extract_invoice_information,
    extract_items,
    extract_quantities,
    extract_descriptions,
    extract_financial_values
)

def process_invoice(image_path):

    ocr_text = extract_text(image_path)

    invoice_information = extract_invoice_information(
        ocr_text
    )

    items = extract_items(
        ocr_text
    )

    quantities = extract_quantities(
    ocr_text
    )
    descriptions = extract_descriptions(
        ocr_text
    )

    financial_records = extract_financial_values(ocr_text)

    records = []

    record_count = min(
        len(items),
        len(descriptions),
        len(financial_records)
    )

    for index in range(record_count):

        record = {
            "Image Name": os.path.basename(image_path),
            "Invoice Number": invoice_information["Invoice Number"],
            "Invoice Date": invoice_information["Invoice Date"],
            "Vendor Name": invoice_information["Vendor Name"],
            "Total Amount": invoice_information["Total Amount"],
            "Item Number": items[index]["Item Number"],
            "Description": descriptions[index]["Description"],
            "Quantity": quantities[index] if index < len(quantities) else None,            
            "Unit Price": financial_records[index]["Unit Price"],
            "Net Worth": financial_records[index]["Net Worth"],
            "VAT": financial_records[index]["VAT"],
            "Gross Worth": financial_records[index]["Gross Worth"]
        }

        records.append(record)

    return records


#=============================
# Step 48.12F : Fix Invoice Folder Integration
#=============================
print("\n========== Step 48.12F : FIX INVOICE FOLDER INTEGRATION ========== ")


def process_invoice_folder(invoice_folder):

    all_records = []

    image_files = [
        file_name
        for file_name in os.listdir(invoice_folder)
        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    for index, file_name in enumerate(image_files, start=1):

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

    return pd.DataFrame(
        all_records
    )


print("\nIntegrated production pipeline loaded successfully.")
print("Functions available:")
print("process_invoice()")
print("process_invoice_folder()")

print("\n========== STEP 48.11B COMPLETED ========== ")