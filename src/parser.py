#=============================
# Step 48.24 : Fix PDF Financial Layout Parser
#=============================
print("\n========== Step 48.24 : FIX PDF FINANCIAL LAYOUT PARSER ========== ")

import re


#=============================
# Step 48.21 : Convert OCR Number
#=============================
print("\n========== Step 48.21 : CONVERT OCR NUMBER ========== ")


def convert_number(value):

    value = value.strip()
    value = value.replace("$", "")
    value = value.replace(" ", "")
    value = value.replace(",", ".")

    try:

        return float(value)

    except ValueError:

        return None


def convert_ocr_number(value):

    cleaned_value = (
        value
        .replace("$", "")
        .replace(" ", "")
        .strip()
    )

    if "," in cleaned_value and "." in cleaned_value:

        if cleaned_value.rfind(",") > cleaned_value.rfind("."):

            cleaned_value = (
                cleaned_value
                .replace(".", "")
                .replace(",", ".")
            )

        else:

            cleaned_value = (
                cleaned_value
                .replace(",", "")
            )

    elif "," in cleaned_value:

        cleaned_value = (
            cleaned_value
            .replace(",", ".")
        )

    try:

        return float(cleaned_value)

    except ValueError:

        return None


#=============================
# Step 48.22 : Connect Item Extraction Functions
#=============================
print("\n========== Step 48.22 : CONNECT ITEM EXTRACTION FUNCTIONS ========== ")


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
        convert_number(
            total_amount_matches[-1]
        )
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

        vendor_name = (
            vendor_match.group(1).strip()
        )

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


def extract_descriptions_and_quantities(ocr_text):

    extracted_items = []

    items_match = re.search(
        r"ITEMS(.*?)(?=SUMMARY)",
        ocr_text,
        re.DOTALL
    )

    if not items_match:

        return extracted_items

    items_text = items_match.group(1)

    item_blocks = re.split(
        r"\n(?=\d+\.\s)",
        items_text
    )

    for block in item_blocks:

        block = block.strip()

        if not block:

            continue

        item_number_match = re.match(
            r"(\d+)\.\s",
            block
        )

        if not item_number_match:

            continue

        item_number = int(
            item_number_match.group(1)
        )

        quantity_match = re.search(
            r"(\d+[,.]\d{2})\s*(?:each|eac)?",
            block
        )

        if quantity_match:

            quantity = float(
                quantity_match.group(1).replace(
                    ",",
                    "."
                )
            )

        else:

            quantity = None

        description = re.sub(
            r"^\d+\.\s*",
            "",
            block
        )

        description = re.split(
            r"\d+[,.]\d{2}",
            description
        )[0]

        description = " ".join(
            description.split()
        )

        extracted_items.append({

            "Item Number": item_number,

            "Description": description,

            "Quantity": quantity

        })

    return extracted_items


def extract_unnumbered_descriptions(ocr_text):

    descriptions = []

    description_section_match = re.search(
        r"ITEMS\s+Description\s+(.*?)SUMMARY",
        ocr_text,
        re.DOTALL | re.IGNORECASE
    )

    if not description_section_match:

        return descriptions

    description_text = (
        description_section_match.group(1)
    )

    description_blocks = re.split(
        r"\n\s*\n",
        description_text
    )

    for block in description_blocks:

        cleaned_description = " ".join(
            block.split()
        )

        if cleaned_description:

            descriptions.append(
                cleaned_description
            )

    return descriptions


def extract_separate_quantities(ocr_text):

    quantity_section = re.search(
        r"Total\s+(.*?)VAT\s*\[%\]",
        ocr_text,
        re.DOTALL
    )

    quantities = []

    if quantity_section:

        quantity_text = (
            quantity_section.group(1)
        )

        quantity_matches = re.findall(
            r"\b\d+[,.]\d{2}\b",
            quantity_text
        )

        quantities = [

            float(
                value.replace(
                    ",",
                    "."
                )
            )

            for value in quantity_matches

        ]

    return quantities


def extract_all_items(ocr_text):

    extracted_items = (
        extract_descriptions_and_quantities(
            ocr_text
        )
    )

    if not extracted_items:

        unnumbered_descriptions = (
            extract_unnumbered_descriptions(
                ocr_text
            )
        )

        for i, description in enumerate(
            unnumbered_descriptions,
            start=1
        ):

            extracted_items.append({

                "Item Number": i,

                "Description": description,

                "Quantity": None

            })

    separate_quantities = (
        extract_separate_quantities(
            ocr_text
        )
    )

    if separate_quantities:

        for i, item in enumerate(
            extracted_items
        ):

            if item["Quantity"] is None:

                if i < len(
                    separate_quantities
                ):

                    item["Quantity"] = (
                        separate_quantities[i]
                    )

    return extracted_items


def extract_items(ocr_text):

    return extract_all_items(
        ocr_text
    )


def extract_quantities(ocr_text):

    extracted_items = (
        extract_all_items(
            ocr_text
        )
    )

    return [

        item["Quantity"]

        for item in extracted_items

    ]


def extract_descriptions(ocr_text):

    extracted_items = (
        extract_all_items(
            ocr_text
        )
    )

    return [

        {
            "Description": item[
                "Description"
            ]
        }

        for item in extracted_items

    ]


print("\nItem extraction functions connected successfully.")
print("Functions available:")
print("extract_items()")
print("extract_quantities()")
print("extract_descriptions()")
print("extract_all_items()")


#=============================
# Step 48.21 : Detect Financial Table Sections
#=============================
print("\n========== Step 48.21 : DETECT FINANCIAL TABLE SECTIONS ========== ")


def extract_financial_sections(ocr_text):

    lines = [

        line.strip()

        for line in ocr_text.splitlines()

        if line.strip()

    ]

    section_positions = {

        "net_price": None,

        "net_worth": None,

        "gross_worth": None

    }

    for index, line in enumerate(lines):

        line_lower = line.lower()

        if (
            "net price" in line_lower
            and "net worth" in line_lower
        ):

            section_positions[
                "net_price"
            ] = index

        elif (
            line_lower == "net worth"
            and section_positions[
                "net_worth"
            ] is None
        ):

            section_positions[
                "net_worth"
            ] = index

        elif (
            line_lower == "gross worth"
            and section_positions[
                "gross_worth"
            ] is None
        ):

            section_positions[
                "gross_worth"
            ] = index

    sections = {}

    return (
        section_positions,
        sections
    )


#=============================
# Step 48.21 : Extract Single-Line Financial Values
#=============================
print("\n========== Step 48.21 : EXTRACT SINGLE-LINE FINANCIAL VALUES ========== ")


def extract_single_line_financial_values(
    ocr_text
):

    lines = [

        line.strip()

        for line in ocr_text.splitlines()

        if line.strip()

    ]

    records = []

    financial_pattern = re.compile(

        r"each\s+"

        r"([\d\s]+[,.]\d{2})\s+"

        r"([\d\s]+[,.]\d{2})\s+"

        r"(\d+(?:[,.]\d+)?)%\s+"

        r"([\d\s]+[,.]\d{2})",

        re.IGNORECASE

    )

    for line in lines:

        match = (
            financial_pattern.search(
                line
            )
        )

        if not match:

            continue

        unit_price = (
            convert_ocr_number(
                match.group(1)
            )
        )

        net_worth = (
            convert_ocr_number(
                match.group(2)
            )
        )

        vat = (
            convert_ocr_number(
                match.group(3)
            )
        )

        gross_worth = (
            convert_ocr_number(
                match.group(4)
            )
        )

        records.append({

            "Unit Price": unit_price,

            "Net Worth": net_worth,

            "VAT": vat,

            "Gross Worth": gross_worth,

            "Extraction Layout": "Single-Line"

        })

    return records


#=============================
# Step 48.21 : Extract Separated Financial Values
#=============================
print("\n========== Step 48.21 : EXTRACT SEPARATED FINANCIAL VALUES ========== ")


def extract_separated_financial_values(
    ocr_text
):

    lines = [

        line.strip()

        for line in ocr_text.splitlines()

        if line.strip()

    ]

    net_values = []

    vat_values = []

    gross_values = []

    net_section_start = None

    gross_section_positions = []

    for index, line in enumerate(lines):

        lower_line = line.lower()

        if (
            "net price" in lower_line
            and "net worth" in lower_line
        ):

            net_section_start = index

        if lower_line == "gross worth":

            gross_section_positions.append(
                index
            )

        elif (
            lower_line == "gross"
            and index + 1 < len(lines)
            and lines[index + 1].lower()
            == "worth"
        ):

            gross_section_positions.append(
                index
            )

    if net_section_start is None:

        return []

    net_end = len(lines)

    for index in range(
        net_section_start + 1,
        len(lines)
    ):

        lower_line = lines[index].lower()

        if lower_line in (
            "gross worth",
            "gross"
        ):

            net_end = index

            break

    net_lines = lines[
        net_section_start + 1:
        net_end
    ]

    for line in net_lines:

        financial_match = re.fullmatch(

            r"\$?\s*"
            r"([\d\s]+[,.]\d{2})\s+"
            r"([\d\s]+[,.]\d{2})",

            line

        )

        if financial_match:

            unit_price = (
                convert_ocr_number(
                    financial_match.group(1)
                )
            )

            net_worth = (
                convert_ocr_number(
                    financial_match.group(2)
                )
            )

            net_values.append({

                "Unit Price": unit_price,

                "Net Worth": net_worth

            })

            continue

        vat_match = re.fullmatch(

            r"(\d+(?:[,.]\d+)?)%",

            line

        )

        if vat_match:

            vat_values.append(

                convert_ocr_number(
                    vat_match.group(1)
                )

            )

    if gross_section_positions:

        first_gross_position = (
            gross_section_positions[0]
        )

        if len(
            gross_section_positions
        ) >= 2:

            second_gross_position = (
                gross_section_positions[1]
            )

        else:

            second_gross_position = len(
                lines
            )

        gross_start = (
            first_gross_position + 1
        )

        if (
            lines[first_gross_position].lower()
            == "gross"
        ):

            gross_start += 1

        gross_lines = lines[
            gross_start:
            second_gross_position
        ]

        for line in gross_lines:

            if re.fullmatch(
                r"\$?\s*[\d\s]+[,.]\d{2}",
                line
            ):

                value = (
                    convert_ocr_number(
                        line
                    )
                )

                if value is not None:

                    gross_values.append(
                        value
                    )

    item_count = min(

        len(net_values),

        len(vat_values),

        len(gross_values)

    )

    records = []

    for index in range(
        item_count
    ):

        records.append({

            "Unit Price": net_values[
                index
            ]["Unit Price"],

            "Net Worth": net_values[
                index
            ]["Net Worth"],

            "VAT": vat_values[
                index
            ],

            "Gross Worth": gross_values[
                index
            ],

            "Extraction Layout":
                "Separated-Financial"

        })

    return records


#=============================
# Step 48.21 : Extract Split Gross Worth Values
#=============================
print("\n========== Step 48.21 : EXTRACT SPLIT GROSS WORTH VALUES ========== ")


def extract_item_gross_worth_values_v2(
    ocr_text
):

    lines = [

        line.strip()

        for line in ocr_text.splitlines()

        if line.strip()

    ]

    gross_section_positions = []

    index = 0

    while index < len(lines):

        current_line = (
            lines[index].lower()
        )

        if current_line == "gross worth":

            gross_section_positions.append(
                index
            )

        elif (
            current_line == "gross"
            and index + 1 < len(lines)
            and lines[index + 1].lower()
            == "worth"
        ):

            gross_section_positions.append(
                index
            )

        index += 1

    if not gross_section_positions:

        return []

    if len(gross_section_positions) >= 2:

        first_position = (
            gross_section_positions[0]
        )

        second_position = (
            gross_section_positions[1]
        )

        start_position = (
            first_position + 1
        )

        if (
            lines[first_position].lower()
            == "gross"
        ):

            start_position += 1

        end_position = second_position

    else:

        first_position = (
            gross_section_positions[0]
        )

        start_position = (
            first_position + 1
        )

        if (
            lines[first_position].lower()
            == "gross"
        ):

            start_position += 1

        end_position = len(lines)

    gross_values = []

    for line in lines[
        start_position:end_position
    ]:

        value_text = (

            line
            .replace("$", "")
            .replace(" ", "")

        )

        if not re.match(
            r"^\d+[,.]\d{2}$",
            value_text
        ):

            continue

        value = (
            convert_ocr_number(
                value_text
            )
        )

        if (
            value is not None
            and value > 0
        ):

            gross_values.append(
                value
            )

    return gross_values


#=============================
# Step 48.24 : Extract PDF Separated Financial Layout
#=============================
print("\n========== Step 48.24 : EXTRACT PDF SEPARATED FINANCIAL LAYOUT ========== ")


def extract_pdf_separated_financial_values(
    ocr_text
):

    lines = [

        line.strip()

        for line in ocr_text.splitlines()

        if line.strip()

    ]

    # Detect the special PDF layout.
    has_net_price_header = any(
        line.lower() == "net price"
        for line in lines
    )

    has_net_worth_vat_header = any(
        line.lower() == "net worth vat [%]"
        for line in lines
    )

    has_gross_worth_header = any(
        line.lower() == "gross worth"
        for line in lines
    )

    if not (
        has_net_price_header
        and has_net_worth_vat_header
        and has_gross_worth_header
    ):

        return []


    #=============================
    # Step 48.24 : Extract PDF Unit Prices
    #=============================
    print(
        "\n========== "
        "Step 48.24 : EXTRACT PDF UNIT PRICES "
        "========== "
    )

    net_price_start = None
    net_worth_total_start = None

    for index, line in enumerate(lines):

        lower_line = line.lower()

        if lower_line == "net price":

            net_price_start = index

        elif (
            lower_line == "net worth"
            and net_worth_total_start is None
        ):

            net_worth_total_start = index

    unit_prices = []

    if (
        net_price_start is not None
        and net_worth_total_start is not None
    ):

        net_price_lines = lines[
            net_price_start + 1:
            net_worth_total_start
        ]

        for line in net_price_lines:

            value_text = (
                line
                .replace("$", "")
                .replace(" ", "")
                .strip()
            )

            if not re.fullmatch(
                r"\d+[,.]\d{2}",
                value_text
            ):

                continue

            value = (
                convert_ocr_number(
                    value_text
                )
            )

            if value is not None:

                unit_prices.append(
                    value
                )


    #=============================
    # Step 48.24 : Extract Net Worth and VAT
    #=============================
    print(
        "\n========== "
        "Step 48.24 : EXTRACT PDF NET WORTH AND VAT "
        "========== "
    )

    net_worth_vat_start = None

    for index, line in enumerate(lines):

        if line.lower() == "net worth vat [%]":

            net_worth_vat_start = index

            break

    net_worth_values = []
    vat_values = []

    if net_worth_vat_start is not None:

        end_position = len(lines)

        for index in range(
            net_worth_vat_start + 1,
            len(lines)
        ):

            if lines[index].lower() in (
                "vat",
                "gross",
                "gross worth"
            ):

                end_position = index

                break

        net_worth_vat_lines = lines[
            net_worth_vat_start + 1:
            end_position
        ]

        for line in net_worth_vat_lines:

            match = re.fullmatch(

                r"\$?\s*"
                r"([\d\s]+[,.]\d{2})\s+"
                r"(\d+(?:[,.]\d+)?)%",

                line

            )

            if not match:

                continue

            net_worth = (
                convert_ocr_number(
                    match.group(1)
                )
            )

            vat = (
                convert_ocr_number(
                    match.group(2)
                )
            )

            if net_worth is not None:

                net_worth_values.append(
                    net_worth
                )

            if vat is not None:

                vat_values.append(
                    vat
                )


    #=============================
    # Step 48.24 : Extract PDF Gross Worth
    #=============================
    print(
        "\n========== "
        "Step 48.24 : EXTRACT PDF GROSS WORTH "
        "========== "
    )

    gross_worth_start = None

    for index, line in enumerate(lines):

        if line.lower() == "gross worth":

            gross_worth_start = index

            break

    gross_worth_values = []

    if gross_worth_start is not None:

        for line in lines[
            gross_worth_start + 1:
        ]:

            value_text = (
                line
                .replace("$", "")
                .replace(" ", "")
                .strip()
            )

            if not re.fullmatch(
                r"\d+[,.]\d{2}",
                value_text
            ):

                continue

            value = (
                convert_ocr_number(
                    value_text
                )
            )

            if value is not None:

                gross_worth_values.append(
                    value
                )


    #=============================
    # Step 48.24 : Determine Item Count
    #=============================
    print(
        "\n========== "
        "Step 48.24 : DETERMINE PDF ITEM COUNT "
        "========== "
    )

    quantities = extract_separate_quantities(
        ocr_text
    )

    item_count_candidates = [

        len(unit_prices),

        len(net_worth_values),

        len(vat_values),

        len(gross_worth_values)

    ]

    if quantities:

        item_count_candidates.append(
            len(quantities)
        )

    item_count = min(
        item_count_candidates
    )


    # Gross worth contains the invoice total
    # after the item-level gross values.
    #
    # Therefore, use only the first item_count
    # values and ignore the final invoice total.

    records = []

    for index in range(
        item_count
    ):

        records.append({

            "Unit Price": unit_prices[
                index
            ],

            "Net Worth": net_worth_values[
                index
            ],

            "VAT": vat_values[
                index
            ],

            "Gross Worth": gross_worth_values[
                index
            ],

            "Extraction Layout":
                "PDF-Separated-Financial"

        })

    return records


#=============================
# Step 48.21 : Combine Financial Parsers
#=============================
print("\n========== Step 48.21 : COMBINE FINANCIAL PARSERS ========== ")


def extract_all_ocr_financial_rows(
    ocr_text
):

    #=============================
    # Step 48.24 : Try PDF Separated Layout
    #=============================
    print(
        "\n========== "
        "Step 48.24 : TRY PDF SEPARATED LAYOUT "
        "========== "
    )

    pdf_separated_rows = (
        extract_pdf_separated_financial_values(
            ocr_text
        )
    )

    if pdf_separated_rows:

        return pdf_separated_rows


    #=============================
    # Step 48.21 : Try Single-Line Layout
    #=============================
    print(
        "\n========== "
        "Step 48.21 : TRY SINGLE-LINE LAYOUT "
        "========== "
    )

    single_line_rows = (
        extract_single_line_financial_values(
            ocr_text
        )
    )

    if single_line_rows:

        return single_line_rows


    #=============================
    # Step 48.21 : Try Separated Financial Layout
    #=============================
    print(
        "\n========== "
        "Step 48.21 : TRY SEPARATED FINANCIAL LAYOUT "
        "========== "
    )

    separated_rows = (
        extract_separated_financial_values(
            ocr_text
        )
    )

    if separated_rows:

        return separated_rows


    return []


print("\nFinancial parser functions connected successfully.")
print("Function available:")
print("extract_all_ocr_financial_rows()")

print("\n========== STEP 48.24 COMPLETED ========== ")