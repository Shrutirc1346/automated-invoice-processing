#=============================
# Step 48.24E : Clean and Unify Invoice Parser
#=============================
print("\n========== Step 48.24E : CLEAN AND UNIFY INVOICE PARSER ========== ")

import re


def convert_number(value):

    value = value.strip()
    value = value.replace("$", "")
    value = value.replace(" ", "")
    value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return None


def extract_invoice_information(ocr_text):

    invoice_number_match = re.search(
        r"Invoice\s*(?:no|number|#)\s*[:\-]?\s*([A-Za-z0-9\-]+)",
        ocr_text,
        re.IGNORECASE
    )

    invoice_date_match = re.search(
        r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",
        ocr_text
    )

    total_amount_matches = re.findall(
        r"\$\s*([\d\s]+,\d{2})",
        ocr_text
    )

    total_amount = (
        convert_number(total_amount_matches[-1])
        if total_amount_matches
        else None
    )

    vendor_name = None

    vendor_match = re.search(
        r"Seller:\s*(?:Client:\s*)?(.+?)\s+\d+\s+[A-Za-z]",
        ocr_text,
        re.IGNORECASE
    )

    if vendor_match:
        vendor_name = vendor_match.group(1).strip()

    return {
        "Invoice Number": (
            invoice_number_match.group(1)
            if invoice_number_match
            else None
        ),
        "Invoice Date": (
            invoice_date_match.group(1)
            if invoice_date_match
            else None
        ),
        "Vendor Name": vendor_name,
        "Total Amount": total_amount
    }


#=============================
# Step 48.24G : Fix Item Description Extraction
#=============================
print("\n========== Step 48.24G : FIX ITEM DESCRIPTION EXTRACTION ========== ")

#=============================
# Step 48.24H : Fix Column-Interleaved Description Extraction
#=============================
print("\n========== Step 48.24H : FIX COLUMN-INTERLEAVED DESCRIPTION EXTRACTION ========== ")

def extract_items(ocr_text):

    items = []

    items_section = re.search(
        r"ITEMS.*?(?=SUMMARY)",
        ocr_text,
        re.IGNORECASE | re.DOTALL
    )

    if not items_section:
        return items

    items_text = items_section.group(0)

    items_text = re.sub(
        r"^.*?Gross worth",
        "",
        items_text,
        count=1,
        flags=re.IGNORECASE | re.DOTALL
    )

    quantity_pattern = re.compile(
        r"\d+(?:[,.]\d{1,2})?\s+each\b",
        re.IGNORECASE
    )

    financial_pattern = re.compile(
        r"([\d\s]+[,.]\d{2})\s+"
        r"([\d\s]+[,.]\d{2})\s+"
        r"(\d+(?:[,.]\d+)?)%\s+"
        r"([\d\s]+[,.]\d{2})",
        re.IGNORECASE
    )

    quantity_matches = list(quantity_pattern.finditer(items_text))
    financial_matches = list(financial_pattern.finditer(items_text))

    for index, quantity_match in enumerate(quantity_matches):

        description_start = 0

        if index > 0 and index - 1 < len(financial_matches):
            description_start = financial_matches[index - 1].end()

        description = items_text[
            description_start:quantity_match.start()
        ].strip()

        description = re.sub(
            r"^\s*\d+\.\s*",
            "",
            description
        )

        description = re.sub(
            r"^\s*\d+\s+",
            "",
            description
        )

        description = re.sub(
            r"\s+",
            " ",
            description
        ).strip()

        if description:
            items.append({
                "Item Number": index + 1,
                "Description": description
            })

    return items

print("\nColumn-interleaved description extraction fixed successfully.")
print("Function available: extract_items()")
print("\n========== STEP 48.24H COMPLETED ========== ")

print("\nItem description extraction fixed successfully.")
print("Function available: extract_items()")
print("\n========== STEP 48.24G COMPLETED ========== ")


def extract_quantities(ocr_text):

    quantity_matches = re.findall(
        r"\b(\d+(?:[,.]\d{1,2})?)\s+each\b",
        ocr_text,
        re.IGNORECASE
    )

    quantities = [
        float(value.replace(",", "."))
        for value in quantity_matches
    ]

    return quantities


def extract_descriptions(ocr_text):

    items = extract_items(ocr_text)

    descriptions = []

    for item in items:

        descriptions.append({
            "Item Number": item["Item Number"],
            "Description": item["Description"]
        })

    return descriptions


def extract_financial_values(ocr_text):

    financial_records = []

    items_section = re.search(
        r"ITEMS.*?(?=SUMMARY)",
        ocr_text,
        re.IGNORECASE | re.DOTALL
    )

    if not items_section:
        return financial_records

    items_text = items_section.group(0)

    financial_pattern = re.compile(
        r"\d+(?:[,.]\d{1,2})?\s+each\s+"
        r"([\d\s]+[,.]\d{2})\s+"
        r"([\d\s]+[,.]\d{2})\s+"
        r"(\d+(?:[,.]\d+)?)%\s+"
        r"([\d\s]+[,.]\d{2})",
        re.IGNORECASE
    )

    matches = financial_pattern.findall(items_text)

    for match in matches:

        unit_price = convert_number(match[0])
        net_worth = convert_number(match[1])
        vat = convert_number(match[2])
        gross_worth = convert_number(match[3])

        financial_records.append({
            "Unit Price": unit_price,
            "Net Worth": net_worth,
            "VAT": vat,
            "Gross Worth": gross_worth
        })

    return financial_records


print("\nInvoice parser module loaded successfully.")
print("Functions available:")
print("extract_invoice_information()")
print("extract_items()")
print("extract_quantities()")
print("extract_descriptions()")
print("extract_financial_values()")

print("\n========== STEP 48.24E COMPLETED ==========")