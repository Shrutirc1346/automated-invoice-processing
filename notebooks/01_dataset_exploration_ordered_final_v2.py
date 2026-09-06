import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd =  r"C:\Program Files\Tesseract-OCR\tesseract.exe\tesseract.exe"
print("Tesseract Version:")
print(pytesseract.get_tesseract_version())


#====================================================
# STEP 1 : LOADIND AND EXPLORATION OF DATASET
#====================================================

print("\n============= LOADIND AND EXPLORATION OF DATASET==============")

# Load the dataset
csv_path = 'data/raw/invoice_dataset/batch1_1.csv'
df=pd.read_csv(csv_path)

print(df.head())

# Number of rows 
print("Number of rows in the dataset:", df.shape[0])

# Columns in the dataset
print("Columns in the dataset:", df.columns)
print("Columns in the dataset:", df.columns.tolist())

# number of rows and columns
print("Number of rows and columns in the dataset:", df.shape)

# last 5 records
print("Last 5 records in the dataset:",df.tail())

# data types of each column
print("Data types of each column in the dataset:", df.dtypes)

# Display complete information about the dataset, including:column names, number of non-null values, data types, memory usage
print("Complete information about the dataset:")
print(df.info())

# number of missing values in each column
print("Number of missing values in each column:", df.isnull().sum())  

#number of duplicate records in the dataset
print("Number of duplicate records in the dataset:", df.duplicated().sum())

# numbwer of unique invoce image names
print("Number of Unique invoice image names in the dataset:", df['File Name'].nunique())

# check if there are missing records in OCR Text column
print("Number of missing records in OCR Text column:", df['OCRed Text'].isnull().sum())

# inspect one JSON record from the dataset
# compare json and OCR text to see if they match

print("Sample JSON data",df.loc[0,"Json Data"])
print("Sample OCR Text data",df.loc[0,"OCRed Text"])


#===============================================
# STEP 2 : GROUND TRUTH / OCR VALIDATION
#===============================================

print("\n=========== GROUND TRUTH / OCR VALIDATION ================")

# Parse JSON data
import json
json_data = json.loads(df.loc[0,"Json Data"])

print("\n Type of JSON data:", type(json_data))
print("\n Parse JSON data:",json_data)

# Extract specific information from the JSON data
invoice_number = json_data["invoice"]["invoice_number"]
invoice_date = json_data["invoice"]["invoice_date"]
vendor_name = json_data["invoice"]["seller_name"]
total_amount = json_data["subtotal"]["total"]

print("\n----Invoice Information----")
print("Invoice Number:", invoice_number)
print("Invoice Date:", invoice_date)        
print("Vendor Name:", vendor_name)
print("Total Amount:", total_amount)

json_items = json_data["items"]
print("\n----Invoice Items----")

print("Number of items in the invoice:", len(json_items))

for item in json_items:
    print("Description:", item["description"])
    print("Quantity:", item["quantity"])
    print("total Price:", item["total_price"])

df.loc[0,"OCRed Text"]

#=====================================================
# STEP 3 : IMAGE PREPROCESSING USING OPENCV
# =========================================

print("\n================IMAGE PREPROCESSING USING OPENCV===================")
import cv2
import matplotlib.pyplot as plt
import pytesseract


# Path of the invoice image
image_path = "C:/Shruti/automated_invoice_processing/data/raw/invoice_dataset/batch1_1/batch1-0498.jpg"


# Function for invoice image preprocessing
def preprocess_image(image_path):

    # Load image
    image = cv2.imread(image_path)

    # Check whether image was loaded successfully
    if image is None:
        print("Error: Image could not be loaded")
        return None

    else:
        print("Image loaded successfully")
        print("Original Image Shape:", image.shape)


    # Convert color image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Grayscale Image Shape:", gray_image.shape)


    # Display grayscale image
    plt.figure(figsize=(10, 12))
    plt.imshow(gray_image, cmap="gray")
    plt.title("Grayscale Invoice")
    plt.axis("off")
    # plt.show()


    # Noise reduction using Gaussian Blur
    denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    print("Noise reduction completed")


    # Display noise reduced image
    plt.figure(figsize=(10, 12))
    plt.imshow(denoised_image, cmap="gray")
    plt.title("Noise Reduced Invoice")
    plt.axis("off")
    # plt.show()


    # Binarization using Otsu's Thresholding
    threshold_value, binary_image = cv2.threshold(
        denoised_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print("Automatically selected threshold:", threshold_value)
    print("Binarization Completed")


    # Display binarized image
    plt.figure(figsize=(10, 12))
    plt.imshow(binary_image, cmap="gray")
    plt.title("Binarized Invoice")
    plt.axis("off")
    # plt.show()

    #=============================
    # Step 39.2 : Release Matplotlib Memory
    #=============================
    print("\n========== Step 39.2 : Release Matplotlib Memory ========== ")

    plt.close("all")

   
    # Return the final preprocessed image
    return binary_image


# ---------------------------------------------------
# FUNCTION: EXTRACT TEXT USING OCR
# ---------------------------------------------------

print("\n=================EXTRACT TEXT USING OCR==============\n")
def extract_text(image):

    # Check whether a valid image was provided
    if image is None:
        print("Error: No image provided for OCR")
        return None

    # Extract text using Tesseract OCR
    extracted_text = pytesseract.image_to_string(image)

    # Return extracted text
    return extracted_text


# ---------------------------------------------------
# CALL THE PREPROCESSING FUNCTION
# ---------------------------------------------------

binary_image = preprocess_image(image_path)


# ---------------------------------------------------
# CHECK IF PREPROCESSING WAS SUCCESSFUL
# ---------------------------------------------------

if binary_image is None:

    print("Preprocessing failed. OCR cannot continue.")

else:

    # ---------------------------------------------------
    # OCR ON PREPROCESSED IMAGE
    # ---------------------------------------------------
    
    processed_text = extract_text(binary_image)
    
    
    # ---------------------------------------------------
    # LOAD ORIGINAL IMAGE
    # ---------------------------------------------------
    
    original_image = cv2.imread(image_path)
    
    
    # ---------------------------------------------------
    # OCR ON ORIGINAL IMAGE
    # ---------------------------------------------------
    
    original_text = extract_text(original_image)
    
    
    # ---------------------------------------------------
    # DISPLAY OCR FROM ORIGINAL IMAGE
    # ---------------------------------------------------
    
    print("\n========== OCR FROM ORIGINAL IMAGE ==========\n")
    print(original_text)
    
    
    # ---------------------------------------------------
    # DISPLAY OCR FROM PREPROCESSED IMAGE
    # ---------------------------------------------------
    
    print("\n========== OCR FROM PREPROCESSED IMAGE ==========\n")
    print(processed_text)
    
    


#=====================================================
# STEP 4 : OCR TEXT EXTRACTION
#====================================================


print("\n============== OCR TEXT EXTRACTION ================")
# Use the preprocessed OCR output for regex extraction
ocr_text = processed_text

print("\n----OCRed Text----")
print(ocr_text)

# Extract Invoice Number from OCRed Text using regex
import re
invoice_match = re.search(r"Invoice no:\s*(\d+)", ocr_text)

if invoice_match:
    invoice_number_ocr = invoice_match.group(1)
    print("\nExtracted Invoice Number from OCRed Text:", invoice_number_ocr)
else:
    invoice_number_ocr = None
    print("\nInvoice Number not found in OCRed Text.")

# Extract Invoice Date from OCRed Text using regex
# Extract invoice date

date_match = re.search(
    r"Date of issue:\s*(\d{2}/\d{2}/\d{4})",
    ocr_text
)

if date_match:

    invoice_date = date_match.group(1)

else:

    # Fallback: find any date in MM/DD/YYYY format
    date_match = re.search(
        r"\b\d{2}/\d{2}/\d{4}\b",
        ocr_text
    )

    if date_match:
        invoice_date = date_match.group(0)
    else:
        invoice_date = None

print("Invoice Date:", invoice_date)

# Extract Vendor Name from OCRed Text using regex
vendor_match = re.search(
    r"Seller:\s*\n+\s*([^\n]+)",
    ocr_text
)

if vendor_match:
    vendor_name = vendor_match.group(1).strip()
else:
    vendor_name = None

print("Vendor Name:", vendor_name)

# ---------------------------------------------------
# Extract Total Amount
# ---------------------------------------------------

# Normalize OCR text for easier searching
normalized_text = re.sub(r"\s+", " ", ocr_text)

# Find monetary values after dollar signs
money_values = re.findall(
    r"\$\s*([\d\s]+,\d{2})",
    normalized_text
)

if money_values:

    # Last dollar value is usually the Gross Total
    total_amount_ocr = money_values[-1]

    # Remove spaces used as thousand separators
    total_amount_ocr = total_amount_ocr.replace(" ", "")

    # Convert European decimal comma to decimal point
    total_amount_ocr = total_amount_ocr.replace(",", ".")

    total_amount_ocr = float(total_amount_ocr)

    print(
        "\nExtracted Total Amount from OCRed Text:",
        total_amount_ocr
    )

else:

    total_amount_ocr = None

    print("\nTotal Amount not found in OCRed Text.")


# ---------------------------------------------------
# EXTRACT ITEM QUANTITIES
# ---------------------------------------------------

print("\n--- Item Numerical Details ---")

# Extract quantities appearing before VAT section
quantity_section = re.search(
    r"Total\s+(.*?)VAT\s*\[%\]",
    ocr_text,
    re.DOTALL
)

quantities = []

if quantity_section:

    quantity_text = quantity_section.group(1)

    quantity_matches = re.findall(
        r"\b\d+[,.]\d{2}\b",
        quantity_text
    )

    quantities = [
        float(q.replace(",", "."))
        for q in quantity_matches
    ]

print("Quantities:", quantities)


# ---------------------------------------------------
# EXTRACT UNIT PRICE, NET WORTH AND VAT
# ---------------------------------------------------

price_section = re.search(
    r"Net price\s+Net worth\s+VAT.*?(?=Net worth\s+VAT)",
    ocr_text,
    re.DOTALL
)

unit_prices = []
net_worths = []
vats = []

if price_section:

    price_text = price_section.group(0)

    rows = re.findall(
    r"(\d+[,.]\d+)\s+(\d+[,.]\d+)\s+(\d+)%",
    price_text
    )

    for row in rows:

        unit_price = float(
            row[0].replace(",", ".")
        )

        net_worth = float(
            row[1].replace(",", ".")
        )

        vat = float(row[2])

        unit_prices.append(unit_price)
        net_worths.append(net_worth)
        vats.append(vat)


print("Unit Prices:", unit_prices)
print("Net Worths:", net_worths)
print("VAT Values:", vats)


# ---------------------------------------------------
# EXTRACT GROSS WORTH VALUES
# ---------------------------------------------------

gross_section = re.search(
    r"Gross\s*\n?worth\s+(.*?)Gross worth",
    ocr_text,
    re.DOTALL
)

gross_worths = []

if gross_section:

    gross_text = gross_section.group(1)

    gross_matches = re.findall(
        r"\b\d+[,.]\d{2}\b",
        gross_text
    )

    gross_worths = [
        float(value.replace(",", "."))
        for value in gross_matches
    ]


print("Gross Worths:", gross_worths)


# ---------------------------------------------------
# EXTRACT ITEM DESCRIPTIONS
# ---------------------------------------------------

description_section = re.search(
    r"Description\s+(.*?)SUMMARY",
    ocr_text,
    re.DOTALL
)

descriptions = []

if description_section:

    description_text = description_section.group(1)

    # Split descriptions using blank lines
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


print("\nDescriptions:")

for i, description in enumerate(
    descriptions,
    start=1
):
    print(f"{i}. {description}")


# ---------------------------------------------------
#  CREATE STRUCTURED INVOICE DATA
# ---------------------------------------------------

invoice_items = []

# Find minimum length to avoid index errors
number_of_items = min(
    len(quantities),
    len(unit_prices),
    len(net_worths),
    len(vats),
    len(gross_worths)
)

print(
    "\nNumber of numerical items extracted:",
    number_of_items
)


for i in range(number_of_items):

    # Use description if available
    if i < len(descriptions):

        description = descriptions[i]

    else:

        description = None


    invoice_items.append({

        "Invoice Number": invoice_number_ocr,

        "Invoice Date": invoice_date,

        "Vendor Name": vendor_name,

        "Total Amount": total_amount_ocr,

        "Description": description,

        "Quantity": quantities[i],

        "Unit Price": unit_prices[i],

        "Net Worth": net_worths[i],

        "VAT": vats[i],

        "Gross Worth": gross_worths[i]

    })


# ---------------------------------------------------
#  CREATE DATAFRAME
# ---------------------------------------------------

invoice_df = pd.DataFrame(invoice_items)

print(
    "\n--- Structured DataFrame for Extracted Invoice Items ---"
)

print(invoice_df)

if invoice_df.empty:
    print("\nNo data extracted. Cannot continue with validation and anomaly preparation.")
    raise SystemExit


# ---------------------------------------------------
#  DATA CLEANING AND VALIDATION
# ---------------------------------------------------

# Check missing values
print(
    "\nMissing values:\n",
    invoice_df.isnull().sum()
)

# Check duplicate records
print(
    "\nDuplicate records:",
    invoice_df.duplicated().sum()
)

# Check data types
print(
    "\nData types before conversion:\n",
    invoice_df.dtypes
)

# Convert Total Amount to float
invoice_df["Total Amount"] = pd.to_numeric(
    invoice_df["Total Amount"],
    errors="coerce"
)

# Convert Invoice Date to datetime
invoice_df["Invoice Date"] = pd.to_datetime(
    invoice_df["Invoice Date"],
    format="%m/%d/%Y",
    errors="coerce"
)

print(
    "\nData types after conversion:\n",
    invoice_df.dtypes
)


# Clean and validate extracted invoice data
print(invoice_df)

# Create a copy of the structured DataFrame for further cleaning and validation
cleaned_df = invoice_df.copy()

# Validate Quantity
invalid_quantity = cleaned_df[cleaned_df["Quantity"] <= 0]
print("\n---Invalid Quantity Records:---\n", invalid_quantity)

# Validate Unit Price
invalid_unit_price = cleaned_df[cleaned_df["Unit Price"] < 0]
print("\n---Invalid Unit Price Records:---\n", invalid_unit_price)

# Validate VAT
invalid_vat = cleaned_df[(cleaned_df["VAT"] < 0) | (cleaned_df["VAT"] > 100)]
print("\n---Invalid VAT Records:---\n", invalid_vat)

# Validate Net Worth
cleaned_df["Expected Net Worth"] = (
    cleaned_df["Quantity"] * cleaned_df["Unit Price"]
)

cleaned_df["Net Worth Difference"] = (
    cleaned_df["Net Worth"] -
    cleaned_df["Expected Net Worth"]
)

invalid_net_worth = cleaned_df[
    cleaned_df["Net Worth Difference"].abs() > 0.01
]

print("\n--- Net Worth Validation ---")
print(invalid_net_worth[
    ["Quantity", "Unit Price", "Net Worth", "Expected Net Worth"]
])

# Validate Gross Worth
cleaned_df["Expected Gross Worth"] = (
    cleaned_df["Net Worth"] *
    (1 + cleaned_df["VAT"] / 100)
)

cleaned_df["Gross Worth Difference"] = (
    cleaned_df["Gross Worth"] -
    cleaned_df["Expected Gross Worth"]
)

invalid_gross_worth = cleaned_df[
    cleaned_df["Gross Worth Difference"].abs() > 0.01
]

print("\n--- Gross Worth Validation ---")
print(invalid_gross_worth[
    ["Net Worth", "VAT", "Gross Worth", "Expected Gross Worth"]
])

# Validate Total Amount
calculated_total = cleaned_df["Gross Worth"].sum()

invoice_total = cleaned_df["Total Amount"].iloc[0]

print("\n--- Invoice Total Validation ---")
print("Extracted Invoice Total:", invoice_total)
print("Calculated Invoice Total:", calculated_total)

if abs(invoice_total - calculated_total) <= 0.01:
    print("Invoice total validation: PASSED")
else:
    print("Invoice total validation: FAILED")

# Final validation summary
print("\n--- Final Cleaned Data ---")
print(cleaned_df)

print("\n--- Final Data Types ---")
print(cleaned_df.dtypes)

# Drop the intermediate validation columns because they are no longer needed for the final cleaned dataset
cleaned_df.drop(
    columns=[
        "Expected Net Worth",
        "Net Worth Difference",
        "Expected Gross Worth",
        "Gross Worth Difference"
    ],
    inplace=True
)

print(cleaned_df.dtypes)

#--------------------------------------------------------
#  Inspect the final cleaned dataset
#--------------------------------------------------------
print("\n--- Final Cleaned Dataset Shape ---")
print(cleaned_df.shape)

print("\n--- Final Cleaned Dataset Columns ---")
print(cleaned_df.columns)

print("\n--- Final Cleaned Dataset ---")
print(cleaned_df)

print("\n--- Numeric columns ---")
print(cleaned_df.select_dtypes(include="number").columns.tolist())

print("\n--- Numeric Data Summary ---")
print(cleaned_df.select_dtypes(include="number").describe())

#----------------------------------------------------------------------
# Prepare Anomaly Detection Features
#----------------------------------------------------------------------

print("\n--- Preparing Anomaly Detection Features ---")

anomaly_features = cleaned_df[["Quantity","Unit Price", "Net Worth", "VAT", "Gross Worth"]].copy()

print("\n---Select Anomaly Detection Features ---")
print(anomaly_features)

print("\n--- Anomaly Detection Features Data Types ---")
print(anomaly_features.dtypes)

print("\n--- Anomaly Detection Features Statistics ---")
print(anomaly_features.describe())

print("\n--- Features Correlation Matrix ---")
print(anomaly_features.corr())

# ==========================================
# STEP 19.1 - Distribution Analysis
# ==========================================

print("\n================== Distribution Analysis ==================\n")

import matplotlib.pyplot as plt

numerical_columns = [
    "Quantity",
    "Unit Price",
    "Net Worth",
    "Gross Worth"
    ]

for column in numerical_columns:

    plt.figure(figsize=(7, 4))

    plt.hist(
        cleaned_df[column],
        bins=5,
        edgecolor='black',
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    # plt.show()

# ==========================================
# Step 19 - Feature Scaling and Normalization
# ==========================================

print("\n=============== Feature Scaling and Normalization ===============")

from sklearn.preprocessing import StandardScaler

print("\n--- Feature Scaling for Anomaly Detection ---")

# Create a StandardScaler object
scaler = StandardScaler()

# Scale the anomaly detection features
scaled_features = scaler.fit_transform(anomaly_features)

# Convert the scaled features back to a DataFrame 
scaled_df = pd.DataFrame(
    scaled_features,
    columns=anomaly_features.columns
)

print("\n--- Scaled Features ---")
print(scaled_df)

print("\n--- Scaled Features Statistics ---")
print(scaled_df.describe())

# ==========================================
# STEP 18 - ANOMALY DETECTION
# USING ISOLATION FOREST
# ==========================================

print("\n=============== Anomaly Detection Using Isolation Forest ===============")

from sklearn.ensemble import IsolationForest

# Create Isolation Forest model
isolation_model = IsolationForest(
    contamination = 0.1,
    random_state = 42
)

# Train the model and predict anomalies
anomamaly_prediction = isolation_model.fit_predict(scaled_df)

print("\n--- Anomaly Predictions ---")
print(anomamaly_prediction)

print(scaled_df)

# ---------------------------------------------------
# ADD ANOMALY PREDICTIONS TO INVOICE DATA
# ---------------------------------------------------

cleaned_df["Anomaly Prediction"] = anomamaly_prediction

print("\n--- Invoice Data with Anomaly Predictions ---")
print(cleaned_df)


cleaned_df["Anomaly Status"] = cleaned_df[
    "Anomaly Prediction"
    ].map({
        1:"Normal",
        -1:"Anomaly"
    })

print(cleaned_df)


# ---------------------------------------------------
# ANOMALY SCORE ANALYSIS
# ---------------------------------------------------

print("\n--- Anomaly Scores ---")

# Calculate anomaly scores using the trained Isolation Forest model

anomaly_score = isolation_model.decision_function(scaled_df)

# Add anomaly scores to the original invoice data

cleaned_df["Anomaly Score"] = anomaly_score

# Display invoice data with anomaly scores

print(
    cleaned_df[
        [
        "Description",
        "Quantity",
        "Unit Price",
        "Net Worth",
        "Gross Worth",
        "Anomaly Prediction",
        "Anomaly Status",
        "Anomaly Score"
        ]
    ]
)

# ---------------------------------------------------
# VISUALIZE ANOMALY SCORES
# ---------------------------------------------------

print("\n--- Anomaly Score Visualization ---")

plt.figure(figsize=(10, 6))

plt.bar(
    range(len(cleaned_df)),
    cleaned_df["Anomaly Score"]
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Isolation Forest Anomaly Scores")
plt.xlabel("Invoice Item Index")
plt.ylabel("Anomaly Score")

plt.xticks(
    range(len(cleaned_df))
)

# plt.show()


# ==========================================
# STEP 19.1 - SELECT MULTIPLE INVOICE IMAGES
# ==========================================

import os

invoice_folder = "data/raw/invoice_dataset/batch1_1"

# Get all JPG invoice images
invoice_files = [
    file for file in os.listdir(invoice_folder)
    if file.lower().endswith(".jpg")
]

# Sort images for consistent selection
image_files = sorted(invoice_files)

# Select first 5 invoice images
selected_images = image_files[:5]

#=============================
# Step 19.3 : DEFINE TEST IMAGES
#=============================
print("\n========== Step 19.3 : DEFINE TEST IMAGES ========== ")

# Use the existing selected_images list for later validation steps.
test_images = selected_images


print("\n--- Selected Invoice Images ---")

for image_name in selected_images:
    print(image_name)

print("Number of selected images:",len(selected_images))


# ==========================================
# STEP 19.2 - BATCH OCR PROCESSING
# ==========================================

print("\n================ BATCH OCR PROCESSING ================\n")

batch_ocr_results = {}

for image_name in selected_images:

    print(f"\nProcessing: {image_name}")

    # Create complete image path
    image_path = os.path.join(
        invoice_folder,
        image_name
    )

    # Preprocess image
    preprocessed_image = preprocess_image(image_path)

    # Check preprocessing
    if preprocessed_image is None:

        print(f"Skipping {image_name} because preprocessing failed.")

        continue

    # Extract text using OCR

    extracted_text = extract_text(preprocessed_image)

    # Store OCR result
    batch_ocr_results[image_name] = extracted_text

    print("OCR completed successfully.")

    print("\n================ BATCH OCR SUMMARY ================\n")

for image_name, text in batch_ocr_results.items():

    print(f"\n--- {image_name} ---")

    print(text[:500])

    print("\n" + "-" * 50)


print(
    "\nTotal invoices successfully processed:",
    len(batch_ocr_results)
)


# =====================================================
# OCR COMPLETENESS VERIFICATION
# =====================================================

print("\n================ OCR COMPLETENESS CHECK ================\n")

for image_name, text in batch_ocr_results.items():

    print("\n" + "=" * 60)
    print(f"INVOICE: {image_name}")
    print("=" * 60)

    # Count total characters extracted by OCR
    print("\nTotal OCR characters:", len(text))

    # Display complete OCR text
    print("\n--- COMPLETE OCR TEXT ---\n")
    print(text)

    print("\n--- END OF OCR TEXT ---\n")


# =====================================================
# STEP 21 : BATCH INVOICE-LEVEL INFORMATION EXTRACTION
# =====================================================

print(
    "\n================ BATCH INVOICE INFORMATION EXTRACTION ================\n"
)


# -----------------------------------------------------
# FUNCTION TO EXTRACT INVOICE-LEVEL INFORMATION
# -----------------------------------------------------

def extract_invoice_information(ocr_text):

    # Create an empty dictionary
    invoice_info = {}


    # -------------------------------------------------
    # Extract Invoice Number
    # -------------------------------------------------

    invoice_match = re.search(
        r"Invoice no:\s*(\d+)",
        ocr_text
    )

    if invoice_match:

        invoice_info["Invoice Number"] = invoice_match.group(1)

    else:

        invoice_info["Invoice Number"] = None


    # -------------------------------------------------
    # Extract Invoice Date
    # -------------------------------------------------

    date_match = re.search(
        r"Date of issue:\s*(\d{2}/\d{2}/\d{4})",
        ocr_text
    )

    if date_match:

        invoice_info["Invoice Date"] = date_match.group(1)

    else:

        # Fallback: search for any date
        date_match = re.search(
            r"\b\d{2}/\d{2}/\d{4}\b",
            ocr_text
        )

        if date_match:

            invoice_info["Invoice Date"] = date_match.group(0)

        else:

            invoice_info["Invoice Date"] = None


    # -------------------------------------------------
    # Extract Vendor Name
    # -------------------------------------------------

    vendor_match = re.search(
        r"Seller:\s*(?:Client:)?\s*\n+\s*([^\n]+)",
        ocr_text
    )


    if vendor_match:

        invoice_info["Vendor Name"] = vendor_match.group(1).strip()

    else:

        invoice_info["Vendor Name"] = None


    # -------------------------------------------------
    # Extract Total Amount
    # -------------------------------------------------

    normalized_text = re.sub(
        r"\s+",
        " ",
        ocr_text
    )

    money_values = re.findall(
        r"\$\s*([\d\s]+[,.]\d{2})",
        normalized_text
    )

    if money_values:

        total_amount = money_values[-1]

        total_amount = total_amount.replace(
            " ",
            ""
        )

        total_amount = total_amount.replace(
            ",",
            "."
        )

        invoice_info["Total Amount"] = float(
            total_amount
        )

    else:

        invoice_info["Total Amount"] = None


    # Return extracted information
    return invoice_info


# -----------------------------------------------------
# APPLY EXTRACTION TO ALL BATCH OCR RESULTS
# -----------------------------------------------------

batch_invoice_information = []


for image_name, text in batch_ocr_results.items():

    # Extract invoice-level information
    extracted_info = extract_invoice_information(
        text
    )

    # Add image name for tracking
    extracted_info["Image Name"] = image_name

    # Store extracted information
    batch_invoice_information.append(
        extracted_info
    )


# -----------------------------------------------------
# CREATE DATAFRAME
# -----------------------------------------------------

batch_invoice_df = pd.DataFrame(
    batch_invoice_information
)


# -----------------------------------------------------
# DISPLAY RESULTS
# -----------------------------------------------------

print(
    "\n--- Batch Invoice-Level Information ---\n"
)

print(
    batch_invoice_df
)


# -----------------------------------------------------
# CHECK MISSING VALUES
# -----------------------------------------------------

print(
    "\n--- Missing Values in Invoice-Level Data ---\n"
)

print(
    batch_invoice_df.isnull().sum()
)



# =====================================================
# HELPER FUNCTION: EXTRACT SEPARATE QUANTITY COLUMN
# =====================================================

def extract_separate_quantities(ocr_text):

    # -------------------------------------------------
    # Find text between "Total" and "VAT [%]"
    # -------------------------------------------------

    quantity_section = re.search(
        r"Total\s+(.*?)VAT\s*\[%\]",
        ocr_text,
        re.DOTALL
    )


    # Create empty list
    quantities = []


    # Continue only if section is found
    if quantity_section:

        # Extract matched text
        quantity_text = quantity_section.group(1)


        # Find values such as 3,00 or 5,00
        quantity_matches = re.findall(
            r"\b\d+[,.]\d{2}\b",
            quantity_text
        )


        # Convert extracted values to float
        quantities = [

            float(
                value.replace(",", ".")
            )

            for value in quantity_matches

        ]


    return quantities


#=============================
# Step 40.12 : IMPROVE UNNUMBERED ITEM EXTRACTION
#=============================
print("\n========== Step 40.12 : IMPROVE UNNUMBERED ITEM EXTRACTION ========== ")


def extract_unnumbered_descriptions(ocr_text):

    descriptions = []

    description_section_match = re.search(
        r"ITEMS\s+Description\s+(.*?)SUMMARY",
        ocr_text,
        re.DOTALL | re.IGNORECASE
    )

    if not description_section_match:
        return descriptions

    description_text = description_section_match.group(1)

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


#=====================================================
# Step 39 — Large-Scale Multi-Invoice Test
#=====================================================

#=====================================================
# Step 39.1 : SELECT ALL INVOICE IMAGES
#=====================================================

print("\n================ 39.1 : SELECT ALL INVOICE IMAGES ================\n")

# Use all invoice images for large-scale testing
all_invoice_images = image_files

print("Total invoice images found:", len(all_invoice_images))

print("\nFirst 5 invoice images:")
print(all_invoice_images[:5])

print("\nLast 5 invoice images:")
print(all_invoice_images[-5:])

#=============================
# Step 39.2 : OCR Cache - Load Existing Results or Run OCR
#=============================
print("\n========== Step 39.2 : OCR Cache - Load Existing Results or Run OCR ========== ")

import os
import pickle
import contextlib
import io

# Get the project folder
project_folder = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Create the processed-data folder path
processed_folder = os.path.join(
    project_folder,
    "data",
    "processed"
)

# Create the folder automatically if it does not exist
os.makedirs(
    processed_folder,
    exist_ok=True
)

# Create the complete path for the OCR cache file
ocr_cache_path = os.path.join(
    processed_folder,
    "all_ocr_results.pkl"
)

print(
    "OCR cache path:",
    ocr_cache_path
)

# Check whether previously generated OCR results already exist
if os.path.exists(ocr_cache_path):

    print(
        "\nExisting OCR cache found."
    )

    print(
        "Skipping image preprocessing and OCR."
    )

    # Load previously saved OCR results
    with open(
        ocr_cache_path,
        "rb"
    ) as file:

        all_ocr_results = pickle.load(
            file
        )

    print(
        "OCR results loaded successfully."
    )

    print(
        "Total OCR results loaded:",
        len(all_ocr_results)
    )

# If cache does not exist, perform OCR
else:

    print(
        "\nNo OCR cache found."
    )

    print(
        "Running preprocessing and OCR for all invoices..."
    )

    # Empty dictionary to store OCR results
    all_ocr_results = {}

    # Process every invoice image
    for count, image_name in enumerate(
        all_invoice_images,
        start=1
    ):

        print(
            f"Processing invoice {count}/{len(all_invoice_images)}: {image_name}"
        )

        image_path = os.path.join(
            invoice_folder,
            image_name
        )

        # Temporarily hide detailed preprocessing messages
        with contextlib.redirect_stdout(
            io.StringIO()
        ):

            preprocessed_image = preprocess_image(
                image_path
            )

        # Check whether image preprocessing was successful
        if preprocessed_image is None:

            print(
                f"Skipping {image_name} because preprocessing failed."
            )

            continue

        # Perform OCR on the preprocessed image
        with contextlib.redirect_stdout(
            io.StringIO()
        ):

            extracted_text = extract_text(
                preprocessed_image
            )

        # Store OCR text using image name as the dictionary key
        all_ocr_results[
            image_name
        ] = extracted_text

    print(
        "\n================ OCR PROCESSING COMPLETED ================\n"
    )

    print(
        "Invoices successfully processed:",
        len(all_ocr_results)
    )

    # Save OCR results so that future runs do not repeat preprocessing/OCR
    with open(
        ocr_cache_path,
        "wb"
    ) as file:

        pickle.dump(
            all_ocr_results,
            file
        )

    print(
        "OCR results saved successfully."
    )

    print(
        "Future runs will load the saved OCR cache instead of processing 499 images."
    )


#=============================
# STEP 22.2 - BATCH ITEM DESCRIPTION AND QUANTITY EXTRACTION
# =====================================================

print("\n================ BATCH ITEM DESCRIPTION AND QUANTITY EXTRACTION ================\n")


def extract_descriptions_and_quantities(ocr_text):

    # -------------------------------------------------
    # Create empty list for extracted items
    # -------------------------------------------------

    extracted_items = []


    # -------------------------------------------------
    # Find ITEMS section
    # -------------------------------------------------

    items_match = re.search(
        r"ITEMS(.*?)(?=SUMMARY)",
        ocr_text,
        re.DOTALL
    )


    # If ITEMS section is not found
    if not items_match:

        return extracted_items


    # Extract only the ITEMS section
    items_text = items_match.group(1)


    # -------------------------------------------------
    # Split text using item numbers
    # -------------------------------------------------

    item_blocks = re.split(
        r"\n(?=\d+\.\s)",
        items_text
    )


    # -------------------------------------------------
    # Process every item block
    # -------------------------------------------------

    for block in item_blocks:


        # Remove unnecessary spaces
        block = block.strip()


        # Skip empty blocks
        if not block:

            continue


        # -------------------------------------------------
        # Extract item number
        # -------------------------------------------------

        item_number_match = re.match(
            r"(\d+)\.\s",
            block
        )


        if not item_number_match:

            continue


        item_number = int(
            item_number_match.group(1)
        )


        # -------------------------------------------------
        # Extract quantity
        # -------------------------------------------------

        quantity_match = re.search(
            r"(\d+[,.]\d{2})\s*(?:each|eac)?",
            block
        )


        if quantity_match:

            quantity = float(
                quantity_match.group(1).replace(",", ".")
            )

        else:

            quantity = None


        # -------------------------------------------------
        # Remove item number from block
        # -------------------------------------------------

        description = re.sub(
            r"^\d+\.\s*",
            "",
            block
        )


        # -------------------------------------------------
        # Remove quantity and everything after it
        # -------------------------------------------------

        description = re.split(
            r"\d+[,.]\d{2}",
            description
        )[0]


        # -------------------------------------------------
        # Clean multiline description
        # -------------------------------------------------

        description = " ".join(
            description.split()
        )


        # -------------------------------------------------
        # Store extracted item
        # -------------------------------------------------

        extracted_items.append({

            "Item Number": item_number,

            "Description": description,

            "Quantity": quantity

        })


    return extracted_items



# =====================================================
# UNIFIED ITEM DESCRIPTION AND QUANTITY EXTRACTION
# =====================================================

def extract_all_items(ocr_text):

    # -------------------------------------------------
    # STEP 1: Try numbered item extraction
    # -------------------------------------------------

    extracted_items = extract_descriptions_and_quantities(
        ocr_text
    )


    # -------------------------------------------------
    # STEP 2: If no numbered items were found,
    # use unnumbered description extraction
    # -------------------------------------------------

    if not extracted_items:

        unnumbered_descriptions = (
            extract_unnumbered_descriptions(
                ocr_text
            )
        )


        # Create item dictionaries
        for i, description in enumerate(
            unnumbered_descriptions,
            start=1
        ):

            extracted_items.append({

                "Item Number": i,

                "Description": description,

                "Quantity": None

            })


    # -------------------------------------------------
    # STEP 3: Extract separate quantity column
    # -------------------------------------------------

    separate_quantities = extract_separate_quantities(
        ocr_text
    )


    # -------------------------------------------------
    # STEP 4: Fill missing quantities
    # -------------------------------------------------

    if separate_quantities:

        for i, item in enumerate(extracted_items):

            if item["Quantity"] is None:

                if i < len(separate_quantities):

                    item["Quantity"] = (
                        separate_quantities[i]
                    )


    # -------------------------------------------------
    # STEP 5: Return final extracted items
    # -------------------------------------------------

    return extracted_items


#=============================
# Step 40.13A : TEST COMPLETE UNNUMBERED ITEM EXTRACTION
#=============================
print("\n========== Step 40.13A : TEST COMPLETE UNNUMBERED ITEM EXTRACTION ========== ")


# -------------------------------------------------
# Test one invoice first
# -------------------------------------------------

test_image = "batch1-0008.jpg"


print("\n==========================================")
print("TEST IMAGE:", test_image)
print("==========================================")


# Call the complete extraction function
extracted_items = extract_all_items(
    all_ocr_results[test_image]
)


# Display number of extracted items
print(
    "Extracted items:",
    len(extracted_items)
)


# Display every extracted item
for item in extracted_items:

    print(
        "Item:",
        item["Item Number"],
        "| Quantity:",
        item["Quantity"],
        "| Description:",
        item["Description"]
    )


#=============================
# Step 40.13B : TEST MULTIPLE FAILED INVOICES
#=============================
print("\n========== Step 40.13B : TEST MULTIPLE FAILED INVOICES ========== ")


# -------------------------------------------------
# Define failed invoices and expected item counts
# -------------------------------------------------

test_cases = {

    "batch1-0008.jpg": 4,

    "batch1-0027.jpg": 4,

    "batch1-0061.jpg": 6,

    "batch1-0074.jpg": 4,

    "batch1-0082.jpg": 7
}


# -------------------------------------------------
# Test every failed invoice
# -------------------------------------------------

for image_name, expected_count in test_cases.items():

    print("\n==========================================")

    print(
        "IMAGE:",
        image_name
    )

    print(
        "EXPECTED ITEMS:",
        expected_count
    )

    print("==========================================")


    # Get OCR text for current invoice
    ocr_text = all_ocr_results[image_name]


    # Extract complete items
    extracted_items = extract_all_items(
        ocr_text
    )


    # Count extracted items
    extracted_count = len(
        extracted_items
    )


    # Compare extracted count with expected count
    if extracted_count == expected_count:

        status = "PASS"

    else:

        status = "FAIL"


    # Display result
    print(
        "EXTRACTED ITEMS:",
        extracted_count
    )

    print(
        "STATUS:",
        status
    )


    # Display extracted items
    for item in extracted_items:

        print(
            "Item:",
            item["Item Number"],
            "| Quantity:",
            item["Quantity"],
            "| Description:",
            item["Description"]
        )


# -------------------------------------------------
# End of testing
# -------------------------------------------------

print(
    "\n========== Step 40.13 TESTING COMPLETED ========== "
)

#=============================
# Step 28 : Extract JSON Item-Level Financial Data
#=====================================================

print("\n================ JSON ITEM-LEVEL DATASET ================\n")

json_item_records = []

for index, row in df.iterrows():

    try:
        json_data = json.loads(row["Json Data"])
    except (json.JSONDecodeError, TypeError):
        continue

    invoice_data = json_data.get("invoice", {})

    invoice_number = str(
        invoice_data.get("invoice_number", "")
    )

    image_name = row["File Name"]

    items = json_data.get("items", [])

    for item_number, item in enumerate(items, start=1):

        json_item_records.append({
            "Invoice Number": invoice_number,
            "Image Name": image_name,
            "Item Number": item_number,
            "JSON Quantity": float(
                item.get("quantity", "0")
                .replace(",", ".")
            ),
            "JSON Total Price": float(
                item.get("total_price", "0")
                .replace(" ", "")
                .replace(",", ".")
            )
        })


json_item_df = pd.DataFrame(json_item_records)

print("--- JSON ITEM DATASET ---\n")
print(json_item_df)

print("\n--- DATASET SHAPE ---")
print("Rows:", json_item_df.shape[0])
print("Columns:", json_item_df.shape[1])

print("\n--- COLUMN NAMES ---")
print(json_item_df.columns.tolist())

#======================================================

# Step 35A : REBUILD JSON ITEM DATASET
#=====================================================

print("\n================ 35A : REBUILD JSON ITEM DATASET ================\n")

json_items = []

for _, row in df.iterrows():

    json_data = json.loads(row["Json Data"])

    invoice = json_data["invoice"]

    invoice_number = invoice["invoice_number"]

    image_name = row["File Name"]

    items = json_data.get("items", [])

    for item_number, item in enumerate(items, start=1):

        quantity = float(
            item["quantity"]
            .replace(" ", "")
            .replace(",", ".")
        )

        total_price = float(
            item["total_price"]
            .replace(" ", "")
            .replace(",", ".")
        )

        json_items.append({
            "Invoice Number": invoice_number,
            "Image Name": image_name,
            "Item Number": item_number,
            "JSON Quantity": quantity,
            "JSON Total Price": total_price
        })


json_item_df = pd.DataFrame(json_items)

print("--- JSON ITEM DATASET ---\n")
print(json_item_df)

print("\n--- DATASET SHAPE ---")
print("Rows:", json_item_df.shape[0])
print("Columns:", json_item_df.shape[1])

# =========================================
# Step 40.14A : TEST REMAINING ZERO-ITEM INVOICES
#=============================
print("\n========== Step 40.14A : TEST REMAINING ZERO-ITEM INVOICES ========== ")


# -------------------------------------------------
# Previously zero-item invoices that still need testing
# -------------------------------------------------

remaining_failed_cases = {

    "batch1-0149.jpg": 4,

    "batch1-0201.jpg": 4,

    "batch1-0217.jpg": 7,

    "batch1-0218.jpg": 4,

    "batch1-0232.jpg": 7,

    "batch1-0341.jpg": 5,

    "batch1-0360.jpg": 4,

    "batch1-0498.jpg": 5
}


# -------------------------------------------------
# Store test results
# -------------------------------------------------

step_40_14_results = []


# -------------------------------------------------
# Test every remaining invoice
# -------------------------------------------------

for image_name, expected_count in remaining_failed_cases.items():

    print("\n==========================================")

    print(
        "IMAGE:",
        image_name
    )

    print(
        "EXPECTED ITEMS:",
        expected_count
    )

    print("==========================================")


    # Get OCR text
    ocr_text = all_ocr_results[image_name]


    # Run complete item extraction
    extracted_items = extract_all_items(
        ocr_text
    )


    # Count extracted items
    extracted_count = len(
        extracted_items
    )


    # Check whether extraction count is correct
    if extracted_count == expected_count:

        status = "PASS"

    else:

        status = "FAIL"


    # Store result
    step_40_14_results.append({

        "Image Name": image_name,

        "Expected Item Count": expected_count,

        "Extracted Item Count": extracted_count,

        "Status": status

    })


    # Display result
    print(
        "EXTRACTED ITEMS:",
        extracted_count
    )

    print(
        "STATUS:",
        status
    )


    # Display extracted items
    for item in extracted_items:

        print(
            "Item:",
            item["Item Number"],
            "| Quantity:",
            item["Quantity"],
            "| Description:",
            item["Description"]
        )


# -------------------------------------------------
# Convert test results into DataFrame
# -------------------------------------------------

step_40_14_results_df = pd.DataFrame(
    step_40_14_results
)


# -------------------------------------------------
# Display summary
# -------------------------------------------------

print(
    "\n========== STEP 40.14A SUMMARY ========== "
)

print(
    step_40_14_results_df
)


# Count passed tests
passed_tests = (
    step_40_14_results_df["Status"] == "PASS"
).sum()


# Count failed tests
failed_tests = (
    step_40_14_results_df["Status"] == "FAIL"
).sum()


print(
    "\nPassed tests:",
    passed_tests
)

print(
    "Failed tests:",
    failed_tests
)

#=============================
# Step 40.14B : VALIDATE EXTRACTED QUANTITIES
#=============================
print("\n========== Step 40.14B : VALIDATE EXTRACTED QUANTITIES ========== ")


quantity_validation_records = []


# -------------------------------------------------
# Process every remaining failed invoice
# -------------------------------------------------

for image_name, expected_count in remaining_failed_cases.items():

    # Get OCR text
    ocr_text = all_ocr_results[image_name]


    # Extract items
    extracted_items = extract_all_items(
        ocr_text
    )


    # Extract invoice number
    invoice_number_match = re.search(
        r"Invoice no:\s*(\d+)",
        ocr_text,
        re.IGNORECASE
    )


    if invoice_number_match:

        invoice_number = (
            invoice_number_match.group(1)
        )

    else:

        invoice_number = None


    # -------------------------------------------------
    # Compare every extracted item with JSON
    # -------------------------------------------------

    for item in extracted_items:

        item_number = item["Item Number"]

        extracted_quantity = item["Quantity"]


        # Find corresponding JSON item
        matching_json_item = json_item_df[
            (
                json_item_df["Invoice Number"]
                == invoice_number
            )
            &
            (
                json_item_df["Image Name"]
                == image_name
            )
            &
            (
                json_item_df["Item Number"]
                == item_number
            )
        ]


        # Continue only if matching JSON item exists
        if not matching_json_item.empty:

            json_quantity = (
                matching_json_item.iloc[0]["JSON Quantity"]
            )


            # Compare OCR quantity with JSON quantity
            quantity_match = (
                extracted_quantity
                == json_quantity
            )


            quantity_validation_records.append({

                "Invoice Number": invoice_number,

                "Image Name": image_name,

                "Item Number": item_number,

                "OCR Quantity": extracted_quantity,

                "JSON Quantity": json_quantity,

                "Quantity Match": quantity_match

            })


# -------------------------------------------------
# Create validation DataFrame
# -------------------------------------------------

step_40_14_quantity_df = pd.DataFrame(
    quantity_validation_records
)


# -------------------------------------------------
# Display validation results
# -------------------------------------------------

print(
    "\n--- QUANTITY VALIDATION RESULTS ---"
)

print(
    step_40_14_quantity_df
)


# -------------------------------------------------
# Calculate quantity accuracy
# -------------------------------------------------

total_quantity_checks = len(
    step_40_14_quantity_df
)


matching_quantities = (
    step_40_14_quantity_df["Quantity Match"]
    .sum()
)


if total_quantity_checks > 0:

    quantity_accuracy = (
        matching_quantities
        / total_quantity_checks
        * 100
    )

else:

    quantity_accuracy = 0


print(
    "\nTotal quantity checks:",
    total_quantity_checks
)

print(
    "Matching quantities:",
    matching_quantities
)

print(
    "Quantity accuracy:",
    round(quantity_accuracy, 2),
    "%"
)

#=====================================================
# step: 24 Extract Financial Information
#=====================================================

print("\n================ FINANCIAL OCR STRUCTURE ================\n")

for image_name, text in batch_ocr_results.items():

    print("\n" + "=" * 70)
    print("IMAGE:", image_name)
    print("=" * 70)

    # Display the OCR text from the ITEMS section to SUMMARY.
    items_section_match = re.search(
        r"ITEMS(.*?)(?=SUMMARY)",
        text,
        re.DOTALL
    )

    if items_section_match:
        items_section = items_section_match.group(1)
        print(items_section)
    else:
        print("Items section not found.")


print("\n================ FINANCIAL FIELD EXTRACTION ================\n")

def extract_financial_fields(ocr_text):

    financial_records = []

    items_match = re.search(
        r"ITEMS(.*?)(?=SUMMARY)",
        ocr_text,
        re.DOTALL
    )

    if not items_match:
        return financial_records

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

        item_number = int(item_number_match.group(1))

        financial_match = re.search(
            r"(\d+[,.]\d{2})\s+each\s+"
            r"([\d\s]+[,.]\d{2})\s+"
            r"([\d\s]+[,.]\d{2})\s+"
            r"(\d+)%\s+"
            r"([\d\s]+[,.]\d{2})",
            block
        )

        if financial_match:

            quantity = float(
                financial_match.group(1).replace(",", ".")
            )

            unit_price = float(
                financial_match.group(2)
                .replace(" ", "")
                .replace(",", ".")
            )

            net_worth = float(
                financial_match.group(3)
                .replace(" ", "")
                .replace(",", ".")
            )

            vat = float(
                financial_match.group(4)
            )

            gross_worth = float(
                financial_match.group(5)
                .replace(" ", "")
                .replace(",", ".")
            )

            financial_records.append({
                "Item Number": item_number,
                "Quantity": quantity,
                "Unit Price": unit_price,
                "Net Worth": net_worth,
                "VAT": vat,
                "Gross Worth": gross_worth
            })

    return financial_records

def calculate_vat_from_values(net_worth, gross_worth):

    if net_worth == 0:
        return None

    vat = ((gross_worth / net_worth) - 1) * 100

    return round(vat, 2)


def extract_separated_financial_fields(ocr_text):

    financial_records = []

    financial_section_match = re.search(
        r"Net\s+price\s+Net\s+worth\s+VAT.*?"
        r"(.*?)(?=Gross\s+worth)",
        ocr_text,
        re.DOTALL | re.IGNORECASE
    )

    if not financial_section_match:
        return financial_records

    financial_text = financial_section_match.group(1)

    financial_values = re.findall(
        r"([\d\s]+[,.]\d{2})\s+"
        r"([\d\s]+[,.]\d{2})",
        financial_text
    )

    gross_section_match = re.search(
        r"Gross\s+worth\s+(.*?)(?=Gross\s+worth|SUMMARY|$)",
        ocr_text,
        re.DOTALL | re.IGNORECASE
    )

    gross_values = []

    if gross_section_match:
        gross_values = re.findall(
            r"\b[\d\s]+[,.]\d{2}\b",
            gross_section_match.group(1)
        )

    extracted_items = extract_all_items(ocr_text)

    item_quantities = [
        item["Quantity"]
        for item in extracted_items
        if item["Quantity"] is not None
    ]

    for i, values in enumerate(financial_values):

        if i >= len(item_quantities):
            break

        net_price = float(
            values[0]
            .replace(" ", "")
            .replace(",", ".")
        )

        net_worth = float(
            values[1]
            .replace(" ", "")
            .replace(",", ".")
        )

        if i < len(gross_values):

            gross_worth = float(
                gross_values[i]
                .replace(" ", "")
                .replace(",", ".")
            )

        else:
            gross_worth = None

        if gross_worth is not None and net_worth != 0:

            vat = round(
                calculate_vat_from_values(
                     net_worth,
                     gross_worth
                     )
)

        else:

            vat = None

        financial_records.append({

            "Item Number": i + 1,

            "Quantity": item_quantities[i],

            "Unit Price": net_price,

            "Net Worth": net_worth,

            "VAT": vat,

            "Gross Worth": gross_worth

        })

    return financial_records

print("\n================ VAT CALCULATION TEST ================\n")

print(
    calculate_vat_from_values(
        1937.31,
        2131.04
    )
)

print(
    calculate_vat_from_values(
        179.96,
        197.96
    )
)

def extract_complete_financial_fields(ocr_text):

    normal_records = extract_financial_fields(ocr_text)

    separated_records = extract_separated_financial_fields(ocr_text)

    financial_records = {}

    for record in normal_records:
        financial_records[record["Item Number"]] = record

    for record in separated_records:

        item_number = record["Item Number"]

        if item_number not in financial_records:

            financial_records[item_number] = record

    financial_records = list(financial_records.values())

    financial_records.sort(
        key=lambda record: record["Item Number"]
    )

    return financial_records


#=============================
# Step 40.15A : TEST QUANTITY RECOVERY FROM FINANCIAL VALUES
#=============================
print("\n========== Step 40.15A : TEST QUANTITY RECOVERY FROM FINANCIAL VALUES ========== ")


# Select the invoice where quantity extraction failed
test_image = "batch1-0218.jpg"


# Get OCR text for this invoice
ocr_text = all_ocr_results[test_image]


# Extract financial information
financial_records = extract_complete_financial_fields(
    ocr_text
)


print("\n==========================================")
print("IMAGE:", test_image)
print("==========================================")


# Process every financial record
for record in financial_records:

    # Read unit price
    unit_price = record["Unit Price"]

    # Read net worth
    net_worth = record["Net Worth"]


    # Calculate quantity from financial values
    if unit_price != 0:

        calculated_quantity = (
            net_worth / unit_price
        )

        # Round because invoice quantities are normally whole numbers
        calculated_quantity = round(
            calculated_quantity
        )

    else:

        calculated_quantity = None


    print(
        "Item:",
        record["Item Number"],
        "| Unit Price:",
        unit_price,
        "| Net Worth:",
        net_worth,
        "| Calculated Quantity:",
        calculated_quantity
    )


#=============================
# Step 40.15B : VALIDATE RECOVERED QUANTITIES
#=============================
print("\n========== Step 40.15B : VALIDATE RECOVERED QUANTITIES ========== ")


# Get the extracted items
extracted_items = extract_all_items(
    ocr_text
)


print("\n==========================================")
print("IMAGE:", test_image)
print("==========================================")


# Compare every extracted item
for index, item in enumerate(extracted_items):

    # Original OCR quantity
    ocr_quantity = item["Quantity"]


    # Get corresponding financial record
    if index < len(financial_records):

        unit_price = financial_records[index]["Unit Price"]

        net_worth = financial_records[index]["Net Worth"]


        # Recover quantity
        if unit_price != 0:

            recovered_quantity = round(
                net_worth / unit_price
            )

        else:

            recovered_quantity = None

    else:

        recovered_quantity = None


    print(
        "Item:",
        item["Item Number"],
        "| OCR Quantity:",
        ocr_quantity,
        "| Recovered Quantity:",
        recovered_quantity
    )


#=============================
# Step 40.15C : CHECK QUANTITY FINANCIAL CONSISTENCY
#=============================
print("\n========== Step 40.15C : CHECK QUANTITY FINANCIAL CONSISTENCY ========== ")


for index, record in enumerate(financial_records):

    # Read financial values
    unit_price = record["Unit Price"]

    net_worth = record["Net Worth"]


    # Calculate quantity from financial values
    if unit_price != 0:

        calculated_quantity = round(
            net_worth / unit_price
        )

    else:

        calculated_quantity = None


    # Recalculate net worth
    if calculated_quantity is not None:

        recalculated_net_worth = (
            calculated_quantity
            * unit_price
        )

    else:

        recalculated_net_worth = None


    # Check whether the values agree
    if recalculated_net_worth is not None:

        quantity_valid = (
            abs(
                net_worth
                - recalculated_net_worth
            ) <= 0.01
        )

    else:

        quantity_valid = False


    print(
        "Item:",
        record["Item Number"],
        "| Recovered Quantity:",
        calculated_quantity,
        "| Original Net Worth:",
        net_worth,
        "| Recalculated Net Worth:",
        round(
            recalculated_net_worth,
            2
        ) if recalculated_net_worth is not None else None,
        "| Valid:",
        quantity_valid
    )



#=============================
# Step 40.12A : TEST UNNUMBERED DESCRIPTION EXTRACTION
#=============================
print("\n========== Step 40.12A : TEST UNNUMBERED DESCRIPTION EXTRACTION ========== ")


test_failed_images = [
    "batch1-0008.jpg",
    "batch1-0027.jpg",
    "batch1-0061.jpg",
    "batch1-0074.jpg",
    "batch1-0082.jpg"
]


for image_name in test_failed_images:

    descriptions = extract_unnumbered_descriptions(
        all_ocr_results[image_name]
    )

    print("\n==========================================")
    print("IMAGE:", image_name)
    print("EXTRACTED ITEMS:", len(descriptions))
    print("==========================================")

    for item_number, description in enumerate(
        descriptions,
        start=1
    ):
        print(
            item_number,
            "->",
            description
        )



# =====================================================
# TEST UNIFIED ITEM EXTRACTION
# =====================================================

print(
    "\n================ UNIFIED ITEM EXTRACTION TEST ================\n"
)


for image_name, text in batch_ocr_results.items():

    # Extract all items
    final_items = extract_all_items(text)


    print("\n" + "=" * 60)

    print(f"INVOICE: {image_name}")

    print("=" * 60)


    print(
        "Number of extracted items:",
        len(final_items)
    )


    for item in final_items:

        print(item)

# =====================================================
# TEST UNNUMBERED DESCRIPTION EXTRACTION
# =====================================================

print(
    "\n================ UNNUMBERED DESCRIPTION EXTRACTION TEST ================\n"
)


ocr_text_0004 = batch_ocr_results[
    "batch1-0004.jpg"
]


unnumbered_descriptions = (
    extract_unnumbered_descriptions(
        ocr_text_0004
    )
)


print(
    "Number of descriptions extracted:",
    len(unnumbered_descriptions)
)


print()


for i, description in enumerate(
    unnumbered_descriptions,
    start=1
):

    print(
        f"Item {i}: {description}"
    )




# =====================================================
# TEST DESCRIPTION AND QUANTITY EXTRACTION
# =====================================================

for image_name, text in batch_ocr_results.items():

    print("\n" + "=" * 60)

    print(f"INVOICE: {image_name}")

    print("=" * 60)


    extracted_items = extract_descriptions_and_quantities(
        text
    )


    print(
        "\nNumber of extracted items:",
        len(extracted_items)
    )


    for item in extracted_items:

        print(item)

        
# =====================================================
# TEST SEPARATE QUANTITY EXTRACTION
# =====================================================

print("\n================ SEPARATE QUANTITY EXTRACTION TEST ================\n")


for image_name, text in batch_ocr_results.items():

    quantities = extract_separate_quantities(text)


    print(image_name)

    print(
        "Separate quantities:",
        quantities
    )

    print()

# =====================================================
# STEP 22.4 - MERGE FALLBACK QUANTITIES
# =====================================================

print("\n================ QUANTITY FALLBACK INTEGRATION TEST ================\n")


for image_name, text in batch_ocr_results.items():

    # Extract items using the main extraction function
    extracted_items = extract_descriptions_and_quantities(
        text
    )


    # Extract quantities from separate column
    separate_quantities = extract_separate_quantities(
        text
    )


    # -------------------------------------------------
    # Apply fallback quantities
    # -------------------------------------------------

    if separate_quantities:

        for i, item in enumerate(extracted_items):

            # Replace missing quantity only
            if item["Quantity"] is None:

                # Make sure index exists
                if i < len(separate_quantities):

                    item["Quantity"] = separate_quantities[i]


    # -------------------------------------------------
    # Display final result
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print(f"INVOICE: {image_name}")
    print("=" * 60)

    print(
        "Number of extracted items:",
        len(extracted_items)
    )

    for item in extracted_items:

        print(item)


# =====================================================
# STEP 22.5 - INSPECT UNNUMBERED DESCRIPTION SECTION
# =====================================================

print(
    "\n================ UNNUMBERED DESCRIPTION SECTION TEST ================\n"
)


# Get OCR text of batch1-0004
ocr_text_0004 = batch_ocr_results["batch1-0004.jpg"]


# Find text between ITEMS header and SUMMARY
description_section_match = re.search(
    r"ITEMS\s+No\.\s*Description\s+(.*?)SUMMARY",
    ocr_text_0004,
    re.DOTALL
)


if description_section_match:

    unnumbered_description_text = (
        description_section_match.group(1)
    )

    print(
        "--- EXTRACTED DESCRIPTION SECTION ---\n"
    )

    print(unnumbered_description_text)

    print(
        "\n--- END OF DESCRIPTION SECTION ---"
    )

else:

    print(
        "Description section could not be found."
    )


# =====================================================
# STEP 23 - CREATE STRUCTURED ITEM-LEVEL DATASET
# =====================================================

print(
    "\n================ STRUCTURED ITEM-LEVEL DATASET ================\n"
)


# Create an empty list to store all invoice items
all_items = []


# Process every invoice in the OCR results
for image_name, text in batch_ocr_results.items():

    # -------------------------------------------------
    # Extract invoice-level information
    # -------------------------------------------------

    invoice_number_match = re.search(
        r"Invoice no:\s*(\d+)",
        text
    )

    if invoice_number_match:

        invoice_number = invoice_number_match.group(1)

    else:

        invoice_number = None


    # -------------------------------------------------
    # Extract all items from the invoice
    # -------------------------------------------------

    extracted_items = extract_all_items(text)


    # -------------------------------------------------
    # Add invoice information to every item
    # -------------------------------------------------

    for item in extracted_items:

        item_record = {

            "Invoice Number": invoice_number,

            "Image Name": image_name,

            "Item Number": item["Item Number"],

            "Description": item["Description"],

            "Quantity": item["Quantity"]

        }


        # Add the item record to the master list
        all_items.append(item_record)


# =====================================================
# CONVERT LIST INTO PANDAS DATAFRAME
# =====================================================

item_level_df = pd.DataFrame(all_items)


# =====================================================
# DISPLAY DATASET
# =====================================================

print("--- ITEM-LEVEL DATASET ---\n")

print(item_level_df)


# =====================================================
# DATASET INFORMATION
# =====================================================

print("\n--- DATASET SHAPE ---")

print(
    "Rows:",
    item_level_df.shape[0]
)

print(
    "Columns:",
    item_level_df.shape[1]
)


print("\n--- COLUMN NAMES ---")

print(
    item_level_df.columns.tolist()
)


#=====================================================
# Step 32F : TEST SEPARATED FINANCIAL EXTRACTION
#=====================================================

print("\n================ 32F : TEST SEPARATED FINANCIAL EXTRACTION ================\n")

for image_name in [
    "batch1-0001.jpg",
    "batch1-0003.jpg",
    "batch1-0004.jpg"
]:

    print("\n" + "=" * 80)
    print("IMAGE:", image_name)
    print("=" * 80)

    ocr_text = batch_ocr_results[image_name]

    separated_financial_data = extract_separated_financial_fields(
        ocr_text
    )

    print("\nExtracted financial records:\n")

    for record in separated_financial_data:
        print(record)


#=====================================================
# Step 25 : Create the Complete Item-Level Dataset
#=====================================================

print("\n================ ALL FINANCIAL RECORDS ================\n")

all_financial_records = []

for image_name, text in batch_ocr_results.items():

    financial_records = extract_financial_fields(text)

    invoice_number_match = re.search(
        r"Invoice no:\s*(\d+)",
        text
    )

    if invoice_number_match:
        invoice_number = invoice_number_match.group(1)
    else:
        invoice_number = None

    for record in financial_records:

        record["Invoice Number"] = invoice_number
        record["Image Name"] = image_name

        all_financial_records.append(record)


financial_df = pd.DataFrame(all_financial_records)

print("--- FINANCIAL DATASET ---\n")
print(financial_df)

print("\n--- DATASET SHAPE ---")
print("Rows:", financial_df.shape[0])
print("Columns:", financial_df.shape[1])

print("\n--- COLUMN NAMES ---")
print(financial_df.columns.tolist())

##=====================================================
# Step 26 : Merge Item-Level + Financial Data
#=====================================================

print("\n================ MERGING ITEM AND FINANCIAL DATA ================\n")

combined_item_df = item_level_df.merge(
    financial_df,
    on=["Invoice Number", "Image Name", "Item Number"],
    how="left",
    suffixes=("", "_financial")
)

# Remove duplicate Quantity column created by the merge.
combined_item_df = combined_item_df.drop(
    columns=["Quantity_financial"]
)

print("--- COMBINED ITEM DATASET ---\n")
print(combined_item_df)

print("\n--- DATASET SHAPE ---")
print("Rows:", combined_item_df.shape[0])
print("Columns:", combined_item_df.shape[1])

print("\n--- MISSING VALUES ---")
print(combined_item_df.isnull().sum())

#======================================================
# Step 27 : Recover the Missing Financial Information
#======================================================

print("\n================ JSON GROUND TRUTH CHECK ================\n")

missing_invoice_numbers = [
    "51109338",
    "19471831",
    "16273983"
]

for invoice_number in missing_invoice_numbers:

    print("\n" + "=" * 70)
    print("INVOICE:", invoice_number)
    print("=" * 70)

    for index, row in df.iterrows():

        try:
            json_data = json.loads(row["Json Data"])
        except (json.JSONDecodeError, TypeError):
            continue

        json_invoice_number = str(
            json_data.get("invoice_number", "")
        )

        if json_invoice_number == invoice_number:

            print("Image:", row["File Name"])
            print("\nJSON DATA:")
            print(json.dumps(json_data, indent=4))

            break

print("\n================ SAMPLE JSON STRUCTURE ================\n")

sample_json = json.loads(df.loc[0, "Json Data"])

print(json.dumps(sample_json, indent=4))

print("\n================ JSON ITEM DATA CHECK ================\n")

missing_invoice_numbers = [
    "51109338",
    "19471831",
    "16273983"
]

for invoice_number in missing_invoice_numbers:

    print("\n" + "=" * 70)
    print("INVOICE:", invoice_number)
    print("=" * 70)

    found = False

    for index, row in df.iterrows():

        try:
            json_data = json.loads(row["Json Data"])
        except (json.JSONDecodeError, TypeError):
            continue

        invoice_data = json_data.get("invoice", {})

        json_invoice_number = str(
            invoice_data.get("invoice_number", "")
        )

        if json_invoice_number == invoice_number:

            found = True

            print("Image:", row["File Name"])

            print("\nInvoice information:")
            print(invoice_data)

            print("\nItems:")

            for item_number, item in enumerate(
                json_data.get("items", []),
                start=1
            ):

                print(
                    item_number,
                    "| Description:",
                    item.get("description"),
                    "| Quantity:",
                    item.get("quantity"),
                    "| Total Price:",
                    item.get("total_price")
                )

            print("\nSubtotal:")
            print(json_data.get("subtotal", {}))

            break

    if not found:
        print("Invoice not found.")

#=====================================================
# Step 29 : Validate OCR Extraction Against JSON
#======================================================

print("\n================ OCR vs JSON VALIDATION ================\n")

validation_df = combined_item_df.merge(
    json_item_df,
    on=["Invoice Number", "Image Name", "Item Number"],
    how="left"
)

validation_df["Quantity Match"] = (
    validation_df["Quantity"] == validation_df["JSON Quantity"]
)

print("--- VALIDATION DATASET ---\n")
print(validation_df[
    [
        "Invoice Number",
        "Image Name",
        "Item Number",
        "Quantity",
        "JSON Quantity",
        "Quantity Match"
    ]
])

print("\n--- QUANTITY VALIDATION SUMMARY ---")

total_items = len(validation_df)

matched_items = validation_df["Quantity Match"].sum()

mismatched_items = total_items - matched_items

print("Total items checked:", total_items)
print("Matching quantities:", matched_items)
print("Mismatching quantities:", mismatched_items)

quantity_accuracy = (
    matched_items / total_items
) * 100

print("Quantity accuracy:", round(quantity_accuracy, 2), "%")

#======================================================
# Step 30 : Validate Item Counts
#======================================================

print("\n================ ITEM COUNT VALIDATION ================\n")

ocr_item_counts = (
    combined_item_df
    .groupby(["Invoice Number", "Image Name"])
    .size()
    .reset_index(name="OCR Item Count")
)

json_item_counts = (
    json_item_df
    .groupby(["Invoice Number", "Image Name"])
    .size()
    .reset_index(name="JSON Item Count")
)

item_count_validation = ocr_item_counts.merge(
    json_item_counts,
    on=["Invoice Number", "Image Name"],
    how="left"
)

item_count_validation["Item Count Match"] = (
    item_count_validation["OCR Item Count"]
    == item_count_validation["JSON Item Count"]
)

print("--- ITEM COUNT COMPARISON ---\n")
print(item_count_validation)

total_invoices = len(item_count_validation)

matching_invoices = (
    item_count_validation["Item Count Match"].sum()
)

mismatching_invoices = (
    total_invoices - matching_invoices
)

print("\n--- ITEM COUNT VALIDATION SUMMARY ---")

print("Total invoices checked:", total_invoices)
print("Matching item counts:", matching_invoices)
print("Mismatching item counts:", mismatching_invoices)

item_count_accuracy = (
    matching_invoices / total_invoices
) * 100

print(
    "Item count accuracy:",
    round(item_count_accuracy, 2),
    "%"
)

#======================================================
# Step 31 : Validate Invoice-Level Fields
#======================================================

print("\n================ JSON INVOICE-LEVEL DATASET ================\n")

json_invoice_records = []

for index, row in df.iterrows():

    try:
        json_data = json.loads(row["Json Data"])
    except (json.JSONDecodeError, TypeError):
        continue

    invoice_data = json_data.get("invoice", {})

    json_invoice_records.append({
        "Invoice Number": str(
            invoice_data.get("invoice_number", "")
        ),
        "Image Name": row["File Name"],
        "JSON Invoice Date": invoice_data.get(
            "invoice_date", ""
        ),
        "JSON Vendor Name": invoice_data.get(
            "seller_name", ""
        ),
        "JSON Total Amount": float(
        json_data.get("subtotal", {})
        .get("total", "0")
        .replace(" ", "")
        .replace(",", ".")
        )
    })

json_invoice_df = pd.DataFrame(json_invoice_records)

print("--- JSON INVOICE DATASET ---\n")
print(json_invoice_df)

print("\n--- DATASET SHAPE ---")
print("Rows:", json_invoice_df.shape[0])
print("Columns:", json_invoice_df.shape[1])

print("\n--- COLUMN NAMES ---")
print(json_invoice_df.columns.tolist())




print("\n================ 31B : INVOICE-LEVEL VALIDATION ================\n")

invoice_validation_df = batch_invoice_df.merge(
    json_invoice_df,
    on=["Invoice Number", "Image Name"],
    how="left"
)

print("--- INVOICE VALIDATION DATASET ---\n")

print(
    invoice_validation_df[
        [
            "Invoice Number",
            "Image Name",
            "Invoice Date",
            "JSON Invoice Date",
            "Vendor Name",
            "JSON Vendor Name",
            "Total Amount",
            "JSON Total Amount"
        ]
    ]
)

print("\n================ 31C : FIELD-LEVEL ACCURACY ================\n")

invoice_validation_df["Invoice Number Match"] = (
    invoice_validation_df["Invoice Number"].astype(str)
    == invoice_validation_df["Invoice Number"].astype(str)
)

invoice_validation_df["Invoice Date Match"] = (
    invoice_validation_df["Invoice Date"].astype(str)
    == invoice_validation_df["JSON Invoice Date"].astype(str)
)

invoice_validation_df["Vendor Name Match"] = (
    invoice_validation_df["Vendor Name"].astype(str).str.strip()
    == invoice_validation_df["JSON Vendor Name"].astype(str).str.strip()
)

invoice_validation_df["Total Amount Match"] = (
    invoice_validation_df["Total Amount"]
    == invoice_validation_df["JSON Total Amount"]
)

print("--- FIELD MATCH RESULTS ---\n")

print(
    invoice_validation_df[
        [
            "Invoice Number",
            "Invoice Number Match",
            "Invoice Date Match",
            "Vendor Name Match",
            "Total Amount Match"
        ]
    ]
)

total_records = len(invoice_validation_df)

invoice_number_accuracy = (
    invoice_validation_df["Invoice Number Match"].sum()
    / total_records
) * 100

invoice_date_accuracy = (
    invoice_validation_df["Invoice Date Match"].sum()
    / total_records
) * 100

vendor_accuracy = (
    invoice_validation_df["Vendor Name Match"].sum()
    / total_records
) * 100

total_amount_accuracy = (
    invoice_validation_df["Total Amount Match"].sum()
    / total_records
) * 100

print("\n--- FIELD-LEVEL ACCURACY ---")

print(
    "Invoice Number Accuracy:",
    round(invoice_number_accuracy, 2),
    "%"
)

print(
    "Invoice Date Accuracy:",
    round(invoice_date_accuracy, 2),
    "%"
)

print(
    "Vendor Name Accuracy:",
    round(vendor_accuracy, 2),
    "%"
)

print(
    "Total Amount Accuracy:",
    round(total_amount_accuracy, 2),
    "%"
)

#======================================================
# Step 32 — Validate Financial Item Data
#=====================================================
print("\n================ 32A : FINANCIAL VALIDATION =================\n")

financial_validation_df = combined_item_df.merge(
    json_item_df,
    on=["Invoice Number", "Image Name", "Item Number"],
    how="left"
)

print("--- FINANCIAL VALIDATION DATASET ---\n")

print(
    financial_validation_df[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Quantity",
            "JSON Quantity",
            "Unit Price",
            "Net Worth",
            "VAT",
            "Gross Worth",
            "JSON Total Price"
        ]
    ]
)

print("\n================ 32B : GROSS WORTH VALIDATION ================\n")

gross_worth_validation_df = financial_validation_df.dropna(
    subset=["Gross Worth", "JSON Total Price"]
).copy()

gross_worth_validation_df["Gross Worth Match"] = (
    gross_worth_validation_df["Gross Worth"]
    == gross_worth_validation_df["JSON Total Price"]
)

print("--- GROSS WORTH MATCH RESULTS ---\n")

print(
    gross_worth_validation_df[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Gross Worth",
            "JSON Total Price",
            "Gross Worth Match"
        ]
    ]
)

total_financial_items = len(gross_worth_validation_df)

matching_gross_worth = (
    gross_worth_validation_df["Gross Worth Match"].sum()
)

mismatching_gross_worth = (
    total_financial_items - matching_gross_worth
)

gross_worth_accuracy = (
    matching_gross_worth / total_financial_items
) * 100

print("\n--- GROSS WORTH VALIDATION SUMMARY ---")

print(
    "Financial items checked:",
    total_financial_items
)

print(
    "Matching Gross Worth values:",
    matching_gross_worth
)

print(
    "Mismatching Gross Worth values:",
    mismatching_gross_worth
)

print(
    "Gross Worth accuracy:",
    round(gross_worth_accuracy, 2),
    "%"
)

print("\n================ 32C : INVOICE TOTAL VALIDATION ================\n")

invoice_total_validation_df = invoice_validation_df.copy()

invoice_total_validation_df["Total Amount Match"] = (
    invoice_total_validation_df["Total Amount"]
    == invoice_total_validation_df["JSON Total Amount"]
)

print("--- INVOICE TOTAL MATCH RESULTS ---\n")

print(
    invoice_total_validation_df[
        [
            "Invoice Number",
            "Image Name",
            "Total Amount",
            "JSON Total Amount",
            "Total Amount Match"
        ]
    ]
)

total_invoices_checked = len(invoice_total_validation_df)

matching_total_amounts = (
    invoice_total_validation_df["Total Amount Match"].sum()
)

mismatching_total_amounts = (
    total_invoices_checked - matching_total_amounts
)

invoice_total_accuracy = (
    matching_total_amounts / total_invoices_checked
) * 100

print("\n--- INVOICE TOTAL VALIDATION SUMMARY ---")

print(
    "Invoices checked:",
    total_invoices_checked
)

print(
    "Matching Total Amount values:",
    matching_total_amounts
)

print(
    "Mismatching Total Amount values:",
    mismatching_total_amounts
)

print(
    "Invoice Total accuracy:",
    round(invoice_total_accuracy, 2),
    "%"
)

print("\n================ 32D : FINANCIAL EXTRACTION COVERAGE ================\n")

financial_fields = [
    "Unit Price",
    "Net Worth",
    "VAT",
    "Gross Worth"
]

total_items = len(combined_item_df)

print("--- FINANCIAL EXTRACTION COVERAGE ---\n")

for field in financial_fields:

    extracted_count = combined_item_df[field].notna().sum()

    missing_count = total_items - extracted_count

    coverage = (
        extracted_count / total_items
    ) * 100

    print(
        field,
        "-> Extracted:",
        extracted_count,
        "| Missing:",
        missing_count,
        "| Coverage:",
        round(coverage, 2),
        "%"
    )

print("\n================ 32E : DIAGNOSE MISSING FINANCIAL CONTENT ================\n")

affected_images = (
    combined_item_df[
        combined_item_df["Gross Worth"].isna()
    ]["Image Name"]
    .unique()
)

print("Affected invoices:\n")

print(affected_images)

for image_name in affected_images:

    print("\n" + "=" * 80)
    print("IMAGE:", image_name)
    print("=" * 80)

    ocr_text = batch_ocr_results[image_name]

    print("\n--- OCR TEXT ---\n")
    print(ocr_text)

    print("\n--- FINANCIAL KEYWORD CHECK ---\n")

    financial_keywords = [
        "Unit Price",
        "Net Worth",
        "VAT",
        "Gross Worth",
        "each"
    ]

    for keyword in financial_keywords:

        if keyword.lower() in ocr_text.lower():
            print(keyword, "-> FOUND")
        else:
            print(keyword, "-> NOT FOUND")


#=============================
# Step 33 : TEST COMPLETE FINANCIAL EXTRACTION
#====================================================

print("\n================ 33 : TEST COMPLETE FINANCIAL EXTRACTION ================\n")

for image_name in [
    "batch1-0001.jpg",
    "batch1-0002.jpg",
    "batch1-0003.jpg",
    "batch1-0004.jpg",
    "batch1-0005.jpg"
]:

    print("\n" + "=" * 80)
    print("IMAGE:", image_name)
    print("=" * 80)

    ocr_text = batch_ocr_results[image_name]

    complete_financial_data = extract_complete_financial_fields(
        ocr_text
    )

    print("Financial records extracted:", len(complete_financial_data))

    for record in complete_financial_data:
        print(record)

#=====================================================
# Step 34 : BUILD COMPLETE FINANCIAL DATASET
#====================================================

print("\n================ 34 : BUILD COMPLETE FINANCIAL DATASET ================\n")

financial_records = []

for image_name in [
    "batch1-0001.jpg",
    "batch1-0002.jpg",
    "batch1-0003.jpg",
    "batch1-0004.jpg",
    "batch1-0005.jpg"
]:

    ocr_text = batch_ocr_results[image_name]

    extracted_records = extract_complete_financial_fields(
        ocr_text
    )

    invoice_number = batch_invoice_df.loc[
        batch_invoice_df["Image Name"] == image_name,
        "Invoice Number"
    ].iloc[0]

    for record in extracted_records:

        record["Invoice Number"] = invoice_number
        record["Image Name"] = image_name

        financial_records.append(record)


financial_df = pd.DataFrame(financial_records)

financial_df = financial_df[
    [
        "Item Number",
        "Quantity",
        "Unit Price",
        "Net Worth",
        "VAT",
        "Gross Worth",
        "Invoice Number",
        "Image Name"
    ]
]


print("--- FINANCIAL DATASET ---\n")
print(financial_df)

print("\n--- DATASET SHAPE ---")
print("Rows:", financial_df.shape[0])
print("Columns:", financial_df.shape[1])

print("\n--- MISSING VALUES ---")
print(financial_df.isnull().sum())



#=====================================================
# Step 35 : RE-VALIDATE COMPLETE FINANCIAL DATASET
#=====================================================

print("\n================ 35 : RE-VALIDATE COMPLETE FINANCIAL DATASET ================\n")

financial_validation_df = financial_df.merge(
    json_item_df,
    on=["Invoice Number", "Image Name", "Item Number"],
    how="left"
)

financial_validation_df["Quantity Match"] = (
    financial_validation_df["Quantity"]
    == financial_validation_df["JSON Quantity"]
)

financial_validation_df["Gross Worth Match"] = (
    abs(
        financial_validation_df["Gross Worth"]
        - financial_validation_df["JSON Total Price"]
    ) < 0.01
)

print("--- FINANCIAL VALIDATION DATASET ---\n")
print(financial_validation_df)

print("\n--- QUANTITY VALIDATION ---")

total_quantity_checks = financial_validation_df["Quantity Match"].count()

matching_quantity = financial_validation_df["Quantity Match"].sum()

print("Items checked:", total_quantity_checks)
print("Matching quantities:", matching_quantity)
print(
    "Quantity accuracy:",
    round((matching_quantity / total_quantity_checks) * 100, 2),
    "%"
)

print("\n--- GROSS WORTH VALIDATION ---")

gross_worth_checks = financial_validation_df["Gross Worth Match"].count()

matching_gross_worth = financial_validation_df["Gross Worth Match"].sum()

print("Items checked:", gross_worth_checks)
print("Matching Gross Worth:", matching_gross_worth)
print(
    "Gross Worth accuracy:",
    round((matching_gross_worth / gross_worth_checks) * 100, 2),
    "%"
)

#=====================================================
# Step 36A : NET WORTH CONSISTENCY CHECK
#=====================================================

print("\n================ 36A : NET WORTH CONSISTENCY ================\n")

financial_validation_df["Calculated Net Worth"] = (
    financial_validation_df["Quantity"]
    * financial_validation_df["Unit Price"]
)

financial_validation_df["Net Worth Match"] = (
    abs(
        financial_validation_df["Net Worth"]
        - financial_validation_df["Calculated Net Worth"]
    ) < 0.01
)

print("--- NET WORTH VALIDATION ---\n")

total_net_worth_checks = (
    financial_validation_df["Net Worth Match"].count()
)

matching_net_worth = (
    financial_validation_df["Net Worth Match"].sum()
)

print("Items checked:", total_net_worth_checks)
print("Matching Net Worth:", matching_net_worth)

print(
    "Net Worth consistency:",
    round(
        (matching_net_worth / total_net_worth_checks) * 100,
        2
    ),
    "%"
)

#=====================================================
# Step 36B : GROSS WORTH CONSISTENCY CHECK
#=====================================================

print("\n================ 36B : GROSS WORTH CONSISTENCY ================\n")

financial_validation_df["Calculated Gross Worth"] = (
    financial_validation_df["Net Worth"]
    * (1 + financial_validation_df["VAT"] / 100)
)

financial_validation_df["Gross Worth Consistency"] = (
    abs(
        financial_validation_df["Gross Worth"]
        - financial_validation_df["Calculated Gross Worth"]
    ) < 0.01
)

print("--- GROSS WORTH VALIDATION ---\n")

total_gross_worth_checks = (
    financial_validation_df["Gross Worth Consistency"].count()
)

matching_gross_worth = (
    financial_validation_df["Gross Worth Consistency"].sum()
)

print("Items checked:", total_gross_worth_checks)
print("Matching Gross Worth:", matching_gross_worth)

print(
    "Gross Worth consistency:",
    round(
        (matching_gross_worth / total_gross_worth_checks) * 100,
        2
    ),
    "%"
)

#=====================================================
# Step 36C : INVOICE-LEVEL FINANCIAL CONSISTENCY CHECK
#=====================================================

print("\n================ 36C : INVOICE-LEVEL FINANCIAL CONSISTENCY ================\n")


invoice_financial_summary = (
    financial_validation_df
    .groupby(["Invoice Number", "Image Name"], as_index=False)
    .agg(
        Calculated_Invoice_Total=("Gross Worth", "sum")
    )
)


invoice_financial_summary = invoice_financial_summary.merge(
    batch_invoice_df[
        ["Invoice Number", "Image Name", "Total Amount"]
    ],
    on=["Invoice Number", "Image Name"],
    how="left"
)


invoice_financial_summary["Invoice Total Difference"] = (
    invoice_financial_summary["Calculated_Invoice_Total"]
    - invoice_financial_summary["Total Amount"]
)


invoice_financial_summary["Invoice Total Match"] = (
    abs(
        invoice_financial_summary["Invoice Total Difference"]
    ).round(2)
    <= 0.01
)


print("--- INVOICE-LEVEL FINANCIAL VALIDATION ---\n")

print(invoice_financial_summary)


total_invoice_checks = (
    invoice_financial_summary["Invoice Total Match"].count()
)

matching_invoice_totals = (
    invoice_financial_summary["Invoice Total Match"].sum()
)


print("\n--- INVOICE TOTAL VALIDATION ---")

print("Invoices checked:", total_invoice_checks)

print("Matching invoice totals:", matching_invoice_totals)

print(
    "Invoice total consistency:",
    round(
        (matching_invoice_totals / total_invoice_checks) * 100,
        2
    ),
    "%"
)

#=====================================================
# Step 37A : FEATURE DATASET CREATION
#=====================================================


print("\n================ 37A : FEATURE DATASET CREATION ================\n")


feature_columns = [
    "Quantity",
    "Unit Price",
    "Net Worth",
    "VAT",
    "Gross Worth"
]


X = financial_validation_df[feature_columns].copy()


print("--- SELECTED FEATURES ---")
print(feature_columns)


print("\n--- FEATURE DATASET ---\n")
print(X)


print("\n--- FEATURE DATASET SHAPE ---")
print("Rows:", X.shape[0])
print("Columns:", X.shape[1])


print("\n--- FEATURE DATA TYPES ---")
print(X.dtypes)


#=====================================================
# Step 37B : FEATURE VARIABILITY ANALYSIS
#=====================================================

print("\n================ 37B : FEATURE VARIABILITY ANALYSIS ================\n")


print("--- UNIQUE VALUES PER FEATURE ---\n")

unique_values = X.nunique()

print(unique_values)


print("\n--- FEATURE STATISTICS ---\n")

print(X.describe())


print("\n--- STANDARD DEVIATION ---\n")

print(X.std())

#=================================================
# Step 37C : FEATURE SCALING
#=================================================

print("\n================ 37C : FEATURE SCALING ================\n")


ml_features = [
    "Quantity",
    "Unit Price",
    "Net Worth",
    "Gross Worth"
]


X_ml = X[ml_features].copy()


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


X_scaled = scaler.fit_transform(X_ml)


X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=ml_features
)


print("--- FEATURES USED FOR MACHINE LEARNING ---")
print(ml_features)


print("\n--- SCALED FEATURE DATASET ---\n")
print(X_scaled_df)


print("\n--- SCALED FEATURE STATISTICS ---\n")
print(X_scaled_df.describe())

#=====================================================
# Step 38A : Build Our First Isolation Forest Model
#=====================================================

print("\n================ 38A : Build Our First Isolation Forest Model ================\n")
from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

isolation_forest.fit(X_scaled)

predictions = isolation_forest.predict(X_scaled)

anomaly_scores = isolation_forest.decision_function(X_scaled)

financial_validation_df["Anomaly Score"] = anomaly_scores
financial_validation_df["Anomaly Prediction"] = predictions

financial_validation_df[    [

        "Invoice Number",
        "Image Name",
        "Item Number",
        "Quantity",
        "Unit Price",
        "Net Worth",
        "VAT",
        "Gross Worth",
        "Anomaly Score",
        "Anomaly Prediction"

    ]
]

#=====================================================
# Step 38B : Analyze Isolation Forest Results
#=====================================================

print("\n================ 38B : Isolation Forest Results ================\n")

print("Total items:", len(financial_validation_df))

print(
    "Normal items:",
    (financial_validation_df["Anomaly Prediction"] == 1).sum()
)

print(
    "Anomalous items:",
    (financial_validation_df["Anomaly Prediction"] == -1).sum()
)

print("\nAnomaly Items:\n")

anomaly_items = financial_validation_df[
    financial_validation_df["Anomaly Prediction"] == -1
]

print(
    anomaly_items[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Quantity",
            "Unit Price",
            "Net Worth",
            "Gross Worth",
            "Anomaly Score",
            "Anomaly Prediction"
        ]
    ]
)

print("\nMost unusual items:\n")

print(
    financial_validation_df[
        [
            "Invoice Number",
            "Item Number",
            "Unit Price",
            "Net Worth",
            "Gross Worth",
            "Anomaly Score"
        ]
    ]
    .sort_values("Anomaly Score")
    .head(10)
)

#=====================================================
# Step 38C — Visualize the Anomaly Results
#=====================================================

print("\n================Step 38C — Visualize the Anomaly Results=====================")
import matplotlib.pyplot as plt

normal_items = financial_validation_df[
    financial_validation_df["Anomaly Prediction"] == 1
]

anomaly_items = financial_validation_df[
    financial_validation_df["Anomaly Prediction"] == -1
]

plt.figure(figsize=(10, 6))

plt.scatter(
    normal_items["Item Number"],
    normal_items["Anomaly Score"],
    label="Normal"
)

plt.scatter(
    anomaly_items["Item Number"],
    anomaly_items["Anomaly Score"],
    label="Anomaly"
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Item Number")
plt.ylabel("Anomaly Score")
plt.title("Isolation Forest Anomaly Detection")
plt.legend()

#plt.show()

#=====================================================
# Step 38D : Validate the Detected Anomalies 
#=====================================================

anomaly_validation_df = financial_validation_df[
    financial_validation_df["Anomaly Prediction"] == -1
].copy()

anomaly_validation_df["Calculated Net Worth"] = (
    anomaly_validation_df["Quantity"]
    * anomaly_validation_df["Unit Price"]
)

anomaly_validation_df["Net Worth Valid"] = (
    abs(
        anomaly_validation_df["Net Worth"]
        - anomaly_validation_df["Calculated Net Worth"]
    ).round(2) <= 0.01
)

anomaly_validation_df["Calculated Gross Worth"] = (
    anomaly_validation_df["Net Worth"]
    * (1 + anomaly_validation_df["VAT"] / 100)
)

anomaly_validation_df["Gross Worth Valid"] = (
    abs(
        anomaly_validation_df["Gross Worth"]
        - anomaly_validation_df["Calculated Gross Worth"]
    ).round(2) <= 0.01
)

print("\n================ Anomaly Validation ================\n")

print(
    anomaly_validation_df[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Quantity",
            "Unit Price",
            "Net Worth",
            "Gross Worth",
            "Anomaly Score",
            "Net Worth Valid",
            "Gross Worth Valid"
        ]
    ]
)

anomaly_validation_df = anomaly_validation_df.drop(
    columns=[
        "JSON Quantity",
        "JSON Total Price",
        "JSON Quantity_x",
        "JSON Quantity_y",
        "JSON Total Price_x",
        "JSON Total Price_y"
    ],
    errors="ignore"
)

anomaly_validation_df = anomaly_validation_df.merge(
    json_item_df[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "JSON Quantity",
            "JSON Total Price"
        ]
    ],
    on=[
        "Invoice Number",
        "Image Name",
        "Item Number"
    ],
    how="left"
)

anomaly_validation_df["Quantity Ground Truth Valid"] = (
    anomaly_validation_df["Quantity"]
    == anomaly_validation_df["JSON Quantity"]
)

anomaly_validation_df["Gross Worth Ground Truth Valid"] = (
    abs(
        anomaly_validation_df["Gross Worth"]
        - anomaly_validation_df["JSON Total Price"]
    ).round(2) <= 0.01
)

print("\n================ Ground Truth Validation ================\n")

print(
    anomaly_validation_df[
        [
            "Invoice Number",
            "Item Number",
            "Quantity",
            "JSON Quantity",
            "Gross Worth",
            "JSON Total Price",
            "Quantity Ground Truth Valid",
            "Gross Worth Ground Truth Valid"
        ]
    ]
)

#=====================================================
# Step 38E : Create the Anomaly Report
#=====================================================

anomaly_report_df = anomaly_validation_df.merge(
    item_level_df[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Description"
        ]
    ],
    on=[
        "Invoice Number",
        "Image Name",
        "Item Number"
    ],
    how="left"
)

anomaly_report_df["Status"] = anomaly_report_df[
    "Anomaly Prediction"
].map({
    1: "Normal",
    -1: "Potential Anomaly"
})
anomaly_report_df["Reason"] = "Unusual financial value"

print("\n================ 38E : ANOMALY REPORT ================\n")

print(
    anomaly_report_df[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Description",
            "Quantity",
            "Unit Price",
            "Net Worth",
            "VAT",
            "Gross Worth",
            "Anomaly Score",
            "Status",
            "Reason"
        ]
    ]
)

#=====================================================
# Step 38G — Anomaly Severity / Ranking
#=====================================================

print("\n================ Step 38F.1 — Rank the anomalies ================\n")

# Sort anomalies from most unusual to least unusual
anomaly_report_df = anomaly_report_df.sort_values(
    "Anomaly Score",
    ascending=True
).reset_index(drop=True)

# Assign a ranking number
anomaly_report_df["Anomaly Rank"] = range(
    1,
    len(anomaly_report_df) + 1
)

# Display the ranked anomalies
print(
    anomaly_report_df[
        [
            "Anomaly Rank",
            "Invoice Number",
            "Item Number",
            "Description",
            "Anomaly Score",
            "Status",
            "Reason"
        ]
    ]
)

#=====================================================
# Step 40.1 : LARGE-SCALE ITEM EXTRACTION
#=============================
print("\n========== Step 40.1 : LARGE-SCALE ITEM EXTRACTION ========== ")

all_item_records = []

for count, (image_name, ocr_text) in enumerate(
    all_ocr_results.items(),
    start=1
):

    print(
        f"Extracting items from invoice {count}/{len(all_ocr_results)}: {image_name}"
    )

    extracted_items = extract_all_items(
        ocr_text
    )

    invoice_number_match = re.search(
        r"Invoice no:\s*(\d+)",
        ocr_text,
        re.IGNORECASE
    )

    if invoice_number_match:
        invoice_number = invoice_number_match.group(1)
    else:
        invoice_number = None

    for item in extracted_items:

        all_item_records.append({
            "Invoice Number": invoice_number,
            "Image Name": image_name,
            "Item Number": item["Item Number"],
            "Description": item["Description"],
            "Quantity": item["Quantity"]
        })


print("\n================ ITEM EXTRACTION COMPLETED ================\n")

print(
    "Total item records extracted:",
    len(all_item_records)
)

#=============================
# Step 40.2 : CREATE LARGE-SCALE ITEM DATASET
#=============================
print("\n========== Step 40.2 : CREATE LARGE-SCALE ITEM DATASET ========== ")

all_item_df = pd.DataFrame(
    all_item_records
)

print("\nDataset shape:")
print(all_item_df.shape)

print("\nDataset columns:")
print(all_item_df.columns.tolist())

print("\nFirst 10 extracted items:")
print(
    all_item_df.head(10)
)

#===============================================
# Step 40.3 : CHECK ITEM EXTRACTION COVERAGE
#===============================================
print("\n========== Step 40.3 : CHECK ITEM EXTRACTION COVERAGE ========== ")

invoices_with_items = (
    all_item_df["Image Name"]
    .nunique()
)

total_invoices = len(
    all_ocr_results
)

item_extraction_coverage = (
    invoices_with_items
    / total_invoices
) * 100

print(
    "Total OCR invoices:",
    total_invoices
)

print(
    "Invoices with extracted items:",
    invoices_with_items
)

print(
    "Item extraction coverage:",
    round(item_extraction_coverage, 2),
    "%"
)   

#==========================================================
 # Step 40.4 : IDENTIFY INVOICES WITH NO EXTRACTED ITEMS
#==========================================================
print("\n========== Step 40.4 : IDENTIFY INVOICES WITH NO EXTRACTED ITEMS ========== ")

extracted_image_names = set(
    all_item_df["Image Name"]
)

all_image_names = set(
    all_ocr_results.keys()
)

missing_item_images = (
    all_image_names
    - extracted_image_names
)

print(
    "Invoices with no extracted items:",
    len(missing_item_images)
)

print("\nFirst 20 invoices with no extracted items:")

print(
    sorted(
        missing_item_images
    )[:20]
)

#=============================
              # Step 40.5 : DIAGNOSE FAILED ITEM EXTRACTION
              #=============================
print("\n========== Step 40.5 : DIAGNOSE FAILED ITEM EXTRACTION ========== ")

for image_name in sorted(missing_item_images):

    ocr_text = all_ocr_results[image_name]

    print("\n---------------------------------------------")
    print("Invoice:", image_name)
    print("---------------------------------------------")

    print(
        "OCR character count:",
        len(ocr_text)
    )

    print(
        "Contains ITEMS:",
        bool(
            re.search(
                r"ITEMS",
                ocr_text,
                re.IGNORECASE
            )
        )
    )

    print(
        "Contains SUMMARY:",
        bool(
            re.search(
                r"SUMMARY",
                ocr_text,
                re.IGNORECASE
            )
        )
    )

    print(
        "Numbered item patterns found:",
        len(
            re.findall(
                r"\b\d+\.\s",
                ocr_text
            )
        )
    )

    print(
        "Contains Description:",
        bool(
            re.search(
                r"Description",
                ocr_text,
                re.IGNORECASE
            )
        )
    )

    print(
        "Contains Quantity:",
        bool(
            re.search(
                r"Quantity",
                ocr_text,
                re.IGNORECASE
            )
        )
    )


#=============================
              # Step 40.6 : LARGE-SCALE ITEM COUNT VALIDATION
              #=============================
print("\n========== Step 40.6 : LARGE-SCALE ITEM COUNT VALIDATION ========== ")

ocr_item_counts = (
    all_item_df
    .groupby(
        ["Invoice Number", "Image Name"]
    )
    .size()
    .reset_index(
        name="OCR Item Count"
    )
)

json_item_counts = (
    json_item_df
    .groupby(
        ["Invoice Number", "Image Name"]
    )
    .size()
    .reset_index(
        name="JSON Item Count"
    )
)

item_count_comparison = json_item_counts.merge(
    ocr_item_counts,
    on=[
        "Invoice Number",
        "Image Name"
    ],
    how="left"
)

item_count_comparison["OCR Item Count"] = (
    item_count_comparison["OCR Item Count"]
    .fillna(0)
    .astype(int)
)

item_count_comparison["Item Count Match"] = (
    item_count_comparison["JSON Item Count"]
    ==
    item_count_comparison["OCR Item Count"]
)

print("\nTotal invoices checked:")
print(
    len(item_count_comparison)
)

print("\nMatching invoices:")
print(
    item_count_comparison["Item Count Match"].sum()
)

print("\nMismatching invoices:")
print(
    (~item_count_comparison["Item Count Match"]).sum()
)

item_count_accuracy = (
    item_count_comparison["Item Count Match"].mean()
    * 100
)

print(
    "\nLarge-scale item count accuracy:",
    round(item_count_accuracy, 2),
    "%"
)

#=============================
              # Step 40.7 : IDENTIFY ITEM COUNT MISMATCHES
              #=============================
print("\n========== Step 40.7 : IDENTIFY ITEM COUNT MISMATCHES ========== ")

item_count_mismatches = item_count_comparison[
    item_count_comparison["Item Count Match"] == False
].copy()

item_count_mismatches["Missing Items"] = (
    item_count_mismatches["JSON Item Count"]
    -
    item_count_mismatches["OCR Item Count"]
)

print(
    "\nNumber of mismatched invoices:",
    len(item_count_mismatches)
)

print("\nItem count mismatches:")

print(
    item_count_mismatches[
        [
            "Invoice Number",
            "Image Name",
            "JSON Item Count",
            "OCR Item Count",
            "Missing Items"
        ]
    ].to_string(index=False)
)

#=============================
              # Step 40.8 : IMAGE-BASED ITEM COUNT VALIDATION
              #=============================
print("\n========== Step 40.8 : IMAGE-BASED ITEM COUNT VALIDATION ========== ")

ocr_image_item_counts = (
    all_item_df
    .groupby(
        "Image Name"
    )
    .size()
    .reset_index(
        name="OCR Item Count"
    )
)

json_image_item_counts = (
    json_item_df
    .groupby(
        "Image Name"
    )
    .size()
    .reset_index(
        name="JSON Item Count"
    )

)

image_item_count_comparison = json_image_item_counts.merge(
    ocr_image_item_counts,
    on="Image Name",
    how="left"
)

image_item_count_comparison["OCR Item Count"] = (
    image_item_count_comparison["OCR Item Count"]
    .fillna(0)
    .astype(int)
)

image_item_count_comparison["Item Count Match"] = (
    image_item_count_comparison["JSON Item Count"]
    ==
    image_item_count_comparison["OCR Item Count"]
)

image_item_count_comparison["Item Count Difference"] = (
    image_item_count_comparison["JSON Item Count"]
    -
    image_item_count_comparison["OCR Item Count"]
)

print(
    "\nTotal invoices checked:",
    len(image_item_count_comparison)
)

print(
    "Matching invoices:",
    image_item_count_comparison[
        "Item Count Match"
    ].sum()
)

print(
    "Mismatching invoices:",
    (
        ~image_item_count_comparison[
            "Item Count Match"
        ]
    ).sum()
)

image_item_count_accuracy = (
    image_item_count_comparison[
        "Item Count Match"
    ].mean()
    * 100
)

print(
    "\nImage-based item count accuracy:",
    round(
        image_item_count_accuracy,
        2
    ),
    "%"
)

print(
    "\nTotal JSON items:",
    image_item_count_comparison[
        "JSON Item Count"
    ].sum()
)

print(
    "Total OCR items:",
    image_item_count_comparison[
        "OCR Item Count"
    ].sum()
)

#=============================
              # Step 40.9 : IDENTIFY IMAGE-BASED ITEM COUNT MISMATCHES
              #=============================
print("\n========== Step 40.9 : IDENTIFY IMAGE-BASED ITEM COUNT MISMATCHES ========== ")

image_item_count_mismatches = (
    image_item_count_comparison[
        image_item_count_comparison[
            "Item Count Match"
        ] == False
    ]
    .copy()
)

print(
    "\nNumber of mismatched invoices:",
    len(image_item_count_mismatches)
)

print(
    "\nItem count mismatches:"
)

print(
    image_item_count_mismatches[
        [
            "Image Name",
            "JSON Item Count",
            "OCR Item Count",
            "Item Count Difference"
        ]
    ]
    .sort_values(
        "Item Count Difference",
        ascending=False
    )
    .to_string(
        index=False
    )
)

#=============================
              # Step 40.10 : CLASSIFY ITEM COUNT ERRORS
              #=============================
print("\n========== Step 40.10 : CLASSIFY ITEM COUNT ERRORS ========== ")

missing_item_invoices = image_item_count_mismatches[
    image_item_count_mismatches[
        "Item Count Difference"
    ] > 0
]

extra_item_invoices = image_item_count_mismatches[
    image_item_count_mismatches[
        "Item Count Difference"
    ] < 0
]

print(
    "\nInvoices with missing extracted items:",
    len(missing_item_invoices)
)

print(
    "Invoices with extra extracted items:",
    len(extra_item_invoices)
)

print(
    "\nTotal missing item records:",
    missing_item_invoices[
        "Item Count Difference"
    ].sum()
)

print(
    "Total extra item records:",
    abs(
        extra_item_invoices[
            "Item Count Difference"
        ].sum()
    )
)

#=============================
              # Step 40.11 : INSPECT OCR TEXT OF FAILED INVOICES
              #=============================
print("\n========== Step 40.11 : INSPECT OCR TEXT OF FAILED INVOICES ========== ")

for image_name in sorted(missing_item_images):

    print("\n\n")
    print("====================================================")
    print("FAILED INVOICE:", image_name)
    print("====================================================")

    print(
        all_ocr_results[image_name]
    )

#==========================================================================================
# Step 40.15 - 40.19 : Quantity Recovery, Financial Validation and Final Test Dataset
#==========================================================================================
print("\n========== Step 40.15 - 40.19 : FINAL EXTRACTION VALIDATION ========== ")

# ============================================================
# STEP 40.15 : QUANTITY RECOVERY
# ============================================================

print("\n========== Step 40.15 : Quantity Recovery ========== ")

# This list will store the final recovered item records
recovered_item_records = []

# Process only the currently selected test invoices
for image_name in all_invoice_images:

    print(
        f"\nProcessing quantity recovery: {image_name}"
    )

    # Get OCR text for the current invoice
    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    # Extract item information using our existing function
    extracted_items = extract_all_items(
        ocr_text
    )

    # Extract financial information using our existing complete extractor
    financial_records = extract_complete_financial_fields(
        ocr_text
    )

    # Create financial lookup using Item Number
    financial_lookup = {
        record["Item Number"]: record
        for record in financial_records
    }

    # Extract invoice number
    invoice_number_match = re.search(
        r"Invoice no:\s*(\d+)",
        ocr_text,
        re.IGNORECASE
    )

    if invoice_number_match:

        invoice_number = invoice_number_match.group(1)

    else:

        invoice_number = None

    # Process every extracted item
    for item in extracted_items:

        item_number = item["Item Number"]

        ocr_quantity = item["Quantity"]

        # Get corresponding financial record
        financial_record = financial_lookup.get(
            item_number
        )

        recovered_quantity = ocr_quantity

        recovery_method = "OCR"

        # If financial information exists, use it when required
        if financial_record is not None:

            unit_price = financial_record["Unit Price"]

            net_worth = financial_record["Net Worth"]

            # Recover quantity if OCR quantity is missing
            if (
                ocr_quantity is None
                and unit_price is not None
                and unit_price != 0
            ):

                recovered_quantity = round(
                    net_worth / unit_price
                )

                recovery_method = "Derived from Net Worth / Unit Price"

            # Also verify an existing OCR quantity
            elif (
                unit_price is not None
                and unit_price != 0
            ):

                calculated_quantity = round(
                    net_worth / unit_price
                )

                # If OCR quantity disagrees with financial values,
                # use the financially derived quantity
                if abs(
                    ocr_quantity - calculated_quantity
                ) > 0.01:

                    recovered_quantity = calculated_quantity

                    recovery_method = "Corrected using Net Worth / Unit Price"

        recovered_item_records.append({

            "Invoice Number": invoice_number,

            "Image Name": image_name,

            "Item Number": item_number,

            "Description": item["Description"],

            "OCR Quantity": ocr_quantity,

            "Recovered Quantity": recovered_quantity,

            "Recovery Method": recovery_method

        })


# Convert recovered records into DataFrame
recovered_quantity_df = pd.DataFrame(
    recovered_item_records
)

print(
    "\nQuantity recovery completed."
)

print(
    "Total items processed:",
    len(recovered_quantity_df)
)

print(
    "\nRecovered quantity results:"
)

print(
    recovered_quantity_df[
        [
            "Image Name",
            "Item Number",
            "OCR Quantity",
            "Recovered Quantity",
            "Recovery Method"
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 40.16 : QUANTITY VALIDATION USING JSON GROUND TRUTH
# ============================================================

print("\n========== Step 40.16 : Quantity Ground Truth Validation ========== ")

# Select JSON quantities for our test invoices
test_json_quantities = json_item_df[
    json_item_df["Image Name"].isin(
        all_invoice_images
    )
].copy()

# Merge recovered quantities with JSON ground truth
quantity_validation_final = recovered_quantity_df.merge(

    test_json_quantities[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "JSON Quantity"
        ]
    ],

    on=[
        "Invoice Number",
        "Image Name",
        "Item Number"
    ],

    how="left"
)

# Compare recovered quantity with ground truth
quantity_validation_final[
    "Quantity Valid"
] = (

    abs(
        quantity_validation_final[
            "Recovered Quantity"
        ]
        -
        quantity_validation_final[
            "JSON Quantity"
        ]
    )

    <= 0.01
)

quantity_matches = quantity_validation_final[
    "Quantity Valid"
].sum()

quantity_total = len(
    quantity_validation_final
)

quantity_accuracy = (
    quantity_matches
    /
    quantity_total
    *
    100
)

print(
    "Items checked:",
    quantity_total
)

print(
    "Matching quantities:",
    quantity_matches
)

print(
    "Final quantity accuracy:",
    round(
        quantity_accuracy,
        2
    ),
    "%"
)

# Display incorrect quantities, if any
quantity_errors = quantity_validation_final[
    ~quantity_validation_final["Quantity Valid"]
]

if len(quantity_errors) > 0:

    print(
        "\nRemaining quantity errors:"
    )

    print(
        quantity_errors[
            [
                "Image Name",
                "Item Number",
                "OCR Quantity",
                "Recovered Quantity",
                "JSON Quantity"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "\nAll recovered quantities match the JSON ground truth."
    )


# ============================================================
# STEP 40.17 : FINAL FINANCIAL DATASET
# ============================================================

print("\n========== Step 40.17 : Build Final Financial Dataset ========== ")

final_financial_records = []

for image_name in all_invoice_images:

    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    financial_records = extract_complete_financial_fields(
        ocr_text
    )

    invoice_number_match = re.search(
        r"Invoice no:\s*(\d+)",
        ocr_text,
        re.IGNORECASE
    )

    if invoice_number_match:

        invoice_number = invoice_number_match.group(1)

    else:

        invoice_number = None

    for record in financial_records:

        final_financial_records.append({

            "Invoice Number": invoice_number,

            "Image Name": image_name,

            "Item Number": record["Item Number"],

            "Quantity": record["Quantity"],

            "Unit Price": record["Unit Price"],

            "Net Worth": record["Net Worth"],

            "VAT": record["VAT"],

            "Gross Worth": record["Gross Worth"]

        })


final_financial_df = pd.DataFrame(
    final_financial_records
)

# Replace quantities with the recovered quantities
final_financial_df = final_financial_df.drop(
    columns=["Quantity"]
)

final_financial_df = final_financial_df.merge(

    recovered_quantity_df[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Recovered Quantity"
        ]
    ],

    on=[
        "Invoice Number",
        "Image Name",
        "Item Number"
    ],

    how="left"
)

# Rename recovered quantity to the final Quantity column
final_financial_df = final_financial_df.rename(
    columns={
        "Recovered Quantity": "Quantity"
    }
)

# Arrange columns in logical order
final_financial_df = final_financial_df[
    [
        "Invoice Number",
        "Image Name",
        "Item Number",
        "Quantity",
        "Unit Price",
        "Net Worth",
        "VAT",
        "Gross Worth"
    ]
]

print(
    "\nFinal financial dataset shape:",
    final_financial_df.shape
)

print(
    "\nFinal financial dataset:"
)

print(
    final_financial_df.to_string(index=False)
)


# ============================================================
# STEP 40.18 : INTERNAL FINANCIAL CONSISTENCY
# ============================================================

print("\n========== Step 40.18 : Financial Consistency Validation ========== ")

# Calculate Net Worth independently
final_financial_df[
    "Calculated Net Worth"
] = (

    final_financial_df[
        "Quantity"
    ]

    *

    final_financial_df[
        "Unit Price"
    ]

)

# Compare extracted Net Worth with calculated Net Worth
final_financial_df[
    "Net Worth Valid"
] = (

    abs(
        final_financial_df[
            "Net Worth"
        ]
        -
        final_financial_df[
            "Calculated Net Worth"
        ]
    ).round(2)

    <= 0.01

)

# Calculate Gross Worth independently
final_financial_df[
    "Calculated Gross Worth"
] = (

    final_financial_df[
        "Net Worth"
    ]

    *

    (
        1
        +
        final_financial_df[
            "VAT"
        ]
        /
        100
    )

)

# Compare extracted Gross Worth with calculated Gross Worth
final_financial_df[
    "Gross Worth Valid"
] = (

    abs(
        final_financial_df[
            "Gross Worth"
        ]
        -
        final_financial_df[
            "Calculated Gross Worth"
        ]
    ).round(2)

    <= 0.01

)

net_worth_matches = final_financial_df[
    "Net Worth Valid"
].sum()

gross_worth_matches = final_financial_df[
    "Gross Worth Valid"
].sum()

total_financial_items = len(
    final_financial_df
)

print(
    "Financial items checked:",
    total_financial_items
)

print(
    "Net Worth matches:",
    net_worth_matches
)

print(
    "Net Worth consistency:",
    round(
        net_worth_matches
        /
        total_financial_items
        *
        100,
        2
    ),
    "%"
)

print(
    "Gross Worth matches:",
    gross_worth_matches
)

print(
    "Gross Worth consistency:",
    round(
        gross_worth_matches
        /
        total_financial_items
        *
        100,
        2
    ),
    "%"
)


# ============================================================
# STEP 40.19 : GROUND TRUTH FINANCIAL VALIDATION
# ============================================================

print("\n========== Step 40.19 : Final Ground Truth Financial Validation ========== ")

# Merge final financial values with JSON ground truth
final_financial_validation = final_financial_df.merge(

    test_json_quantities[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "JSON Quantity",
            "JSON Total Price"
        ]
    ],

    on=[
        "Invoice Number",
        "Image Name",
        "Item Number"
    ],

    how="left"
)

# Validate Quantity
final_financial_validation[
    "Ground Truth Quantity Valid"
] = (

    abs(
        final_financial_validation[
            "Quantity"
        ]
        -
        final_financial_validation[
            "JSON Quantity"
        ]
    )

    <= 0.01

)

# Validate Gross Worth against JSON total_price
final_financial_validation[
    "Ground Truth Gross Worth Valid"
] = (

    abs(
        final_financial_validation[
            "Gross Worth"
        ]
        -
        final_financial_validation[
            "JSON Total Price"
        ]
    ).round(2)

    <= 0.01

)

ground_truth_quantity_matches = final_financial_validation[
    "Ground Truth Quantity Valid"
].sum()

ground_truth_gross_matches = final_financial_validation[
    "Ground Truth Gross Worth Valid"
].sum()

total_ground_truth_items = len(
    final_financial_validation
)

print(
    "Items checked:",
    total_ground_truth_items
)

print(
    "Ground truth quantity matches:",
    ground_truth_quantity_matches
)

print(
    "Ground truth quantity accuracy:",
    round(
        ground_truth_quantity_matches
        /
        total_ground_truth_items
        *
        100,
        2
    ),
    "%"
)

print(
    "Ground truth Gross Worth matches:",
    ground_truth_gross_matches
)

print(
    "Ground truth Gross Worth accuracy:",
    round(
        ground_truth_gross_matches
        /
        total_ground_truth_items
        *
        100,
        2
    ),
    "%"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n============================================================"
)

print(
    "FINAL EXTRACTION VALIDATION SUMMARY"
)

print(
    "============================================================"
)

print(
    "Total invoices processed:",
    len(all_invoice_images)
)

print(
    "Total items validated:",
    total_ground_truth_items
)

print(
    "Quantity accuracy:",
    round(
        ground_truth_quantity_matches
        /
        total_ground_truth_items
        *
        100,
        2
    ),
    "%"
)

print(
    "Gross Worth accuracy:",
    round(
        ground_truth_gross_matches
        /
        total_ground_truth_items
        *
        100,
        2
    ),
    "%"
)

print(
    "Net Worth consistency:",
    round(
        net_worth_matches
        /
        total_financial_items
        *
        100,
        2
    ),
    "%"
)

print(
    "Gross Worth consistency:",
    round(
        gross_worth_matches
        /
        total_financial_items
        *
        100,
        2
    ),
    "%"
)

print(
    "============================================================"
)

print(
    "\nStep 40.15 - 40.19 completed successfully."
)


#=============================
# Step 41 : Anomaly Detection
#=============================
print("\n========== Step 41 : Anomaly Detection ========== ")


#=============================
# Step 41.1 : Prepare Validated Financial Data
#=============================
print("\n========== Step 41.1 : Prepare Validated Financial Data ========== ")

# Select only the records that contain the financial information
# required for anomaly detection.
X = final_financial_df[
    ["Quantity", "Unit Price", "Net Worth", "Gross Worth"]
].copy()

print("\nFinancial data selected for anomaly detection:")
print(X.head())

print("\nShape before numeric conversion:")
print(X.shape)


#=============================
# Step 41.2 : Convert Features to Numeric
#=============================
print("\n========== Step 41.2 : Convert Features to Numeric ========== ")

# Convert all selected financial columns into numeric data types.
# Invalid values are converted into NaN instead of causing an error.
X_ml = X.apply(pd.to_numeric, errors="coerce")

print("\nData types after numeric conversion:")
print(X_ml.dtypes)


#=============================
# Step 41.3 : Remove Incomplete Records
#=============================
print("\n========== Step 41.3 : Remove Incomplete Records ========== ")

# Remove rows where one or more required financial features are missing.
X_ml = X_ml.dropna()

print("\nShape after removing incomplete records:")
print(X_ml.shape)

print("\nMissing values remaining:")
print(X_ml.isnull().sum())


#=============================
# Step 41.4 : Display Feature Statistics
#=============================
print("\n========== Step 41.4 : Feature Statistics ========== ")

# Display descriptive statistics to understand the distribution
# of the financial features before training the model.
print("\nFeature statistics:")
print(X_ml.describe())


#=============================
# Step 41.5 : Feature Scaling
#=============================
print("\n========== Step 41.5 : Feature Scaling ========== ")

# StandardScaler transforms numerical features so that their
# values are represented on a comparable scale.
scaler = StandardScaler()

# Fit the scaler on the financial features and transform them.
X_scaled = scaler.fit_transform(X_ml)

# Convert the scaled NumPy array back into a DataFrame
# so that feature names remain available.
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X_ml.columns,
    index=X_ml.index
)

print("\nScaled feature data:")
print(X_scaled_df.head())


#=============================
# Step 41.6 : Create Isolation Forest Model
#=============================
print("\n========== Step 41.6 : Create Isolation Forest Model ========== ")

# IsolationForest is an unsupervised machine-learning algorithm
# used to identify observations that are statistically unusual.
isolation_forest = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)

print("\nIsolation Forest model created successfully.")


#=============================
# Step 41.7 : Train Isolation Forest
#=============================
print("\n========== Step 41.7 : Train Isolation Forest ========== ")

# Fit the Isolation Forest model using the scaled financial features.
isolation_forest.fit(X_scaled_df)

print("\nIsolation Forest training completed.")


#=============================
# Step 41.8 : Generate Anomaly Predictions
#=============================
print("\n========== Step 41.8 : Generate Anomaly Predictions ========== ")

# Predict whether each financial record is normal or potentially anomalous.
#  1  = normal
# -1  = potential anomaly
predictions = isolation_forest.predict(X_scaled_df)

print("\nPrediction values:")
print(pd.Series(predictions).value_counts().sort_index())


#=============================
# Step 41.9 : Generate Anomaly Scores
#=============================
print("\n========== Step 41.9 : Generate Anomaly Scores ========== ")

# decision_function produces an anomaly score for every record.
# More negative values generally indicate more unusual observations.
anomaly_scores = isolation_forest.decision_function(X_scaled_df)

print("\nFirst 10 anomaly scores:")
print(anomaly_scores[:10])


#=============================
# Step 41.10 : Add Predictions and Scores
#=============================
print("\n========== Step 41.10 : Add Predictions and Scores ========== ")

# Create a copy of the ML feature dataset for storing model results.
anomaly_items = X_ml.copy()

# Add the Isolation Forest prediction to each record.
anomaly_items["Anomaly Prediction"] = predictions

# Add the anomaly score to each record.
anomaly_items["Anomaly Score"] = anomaly_scores

# Convert prediction values into human-readable labels.
anomaly_items["Anomaly Status"] = anomaly_items[
    "Anomaly Prediction"
].map({
    1: "Normal",
    -1: "Potential Anomaly"
})

print("\nAnomaly detection results:")
print(anomaly_items.head())


#=============================
# Step 41.11 : Extract Potential Anomalies
#=============================
print("\n========== Step 41.11 : Extract Potential Anomalies ========== ")

# Select only records that Isolation Forest classified as potential anomalies.
anomaly_items = anomaly_items[
    anomaly_items["Anomaly Prediction"] == -1
].copy()

# Sort anomalies from the most unusual to the least unusual.
anomaly_items = anomaly_items.sort_values(
    by="Anomaly Score",
    ascending=True
)

print("\nPotential anomalies detected:")
print(anomaly_items)


#=============================
# Step 41.12 : Display Anomaly Summary
#=============================
print("\n========== Step 41.12 : Display Anomaly Summary ========== ")

total_records = len(X_ml)

total_anomalies = len(anomaly_items)

total_normal = total_records - total_anomalies

print("\n========== ANOMALY DETECTION SUMMARY ==========")

print("Total validated financial records:", total_records)

print("Normal records:", total_normal)

print("Potential anomalies:", total_anomalies)

if total_records > 0:
    anomaly_percentage = (
        total_anomalies / total_records
    ) * 100

    print(
        "Potential anomaly percentage:",
        round(anomaly_percentage, 2),
        "%"
    )

print("\nMost unusual records:")
print(anomaly_items.head(10))

print("\nStep 41 completed successfully.")


#=============================
# Step 42 : Anomaly Validation
#=============================
print("\n========== Step 42 : Anomaly Validation ========== ")


#=============================
# Step 42.1 : Preserve All Detected Anomalies
#=============================
print("\n========== Step 42.1 : Preserve All Detected Anomalies ========== ")

# Create a copy so that the original anomaly_items dataframe
# remains available for further analysis.
detected_anomalies = anomaly_items.copy()

# Store the original dataframe index.
# This index allows us to reconnect each anomaly with the
# corresponding record in final_financial_df.
detected_anomalies["Original Index"] = detected_anomalies.index

print("\nPotential anomalies available for validation:")
print(len(detected_anomalies))


#=============================
# Step 42.2 : Recover Original Invoice Information
#=============================
print("\n========== Step 42.2 : Recover Original Invoice Information ========== ")

# Select the corresponding rows from the validated financial dataframe.
# The index from anomaly_items identifies the original financial record.
anomaly_source_data = final_financial_df.loc[
    detected_anomalies["Original Index"]
].copy()

# Reset the index so that the dataframe has a clean sequential index.
anomaly_source_data = anomaly_source_data.reset_index(drop=True)

# Reset the anomaly dataframe index as well so both dataframes
# can be combined row by row.
detected_anomalies = detected_anomalies.reset_index(drop=True)

print("\nOriginal invoice information recovered.")


#=============================
# Step 42.3 : Combine Model Results with Invoice Information
#=============================
print("\n========== Step 42.3 : Combine Model Results with Invoice Information ========== ")

# Combine the model's anomaly information with the original
# invoice and item information.
anomaly_validation_df = pd.concat(
    [
        anomaly_source_data,
        detected_anomalies[
            [
                "Anomaly Prediction",
                "Anomaly Score",
                "Anomaly Status"
            ]
        ]
    ],
    axis=1
)

print("\nCombined anomaly validation dataframe:")
print(anomaly_validation_df.head())

print("\nColumns available:")
print(anomaly_validation_df.columns.tolist())


#=============================
# Step 42.4 : Match Anomalies with Ground Truth
#=============================
print("\n========== Step 42.4 : Match Anomalies with Ground Truth ========== ")

# Create a copy of the ground-truth item dataframe.
ground_truth_anomalies = json_item_df.copy()

# Match each detected anomaly with its corresponding invoice image.
# The actual invoice/item matching is performed using invoice number,
# item number, and image name where available.
#
# First inspect the available columns so that the validation
# process is transparent.
print("\nGround truth columns:")
print(ground_truth_anomalies.columns.tolist())


#=============================
# Step 42.5 : Identify Available Matching Columns
#=============================
print("\n========== Step 42.5 : Identify Available Matching Columns ========== ")

# Display a sample of the ground-truth dataframe.
print("\nGround truth sample:")
print(ground_truth_anomalies.head())

# Display a sample of the anomaly dataframe.
print("\nAnomaly sample:")
print(anomaly_validation_df.head())


#=============================
# Step 42.6 : Calculate Financial Consistency
#=============================
print("\n========== Step 42.6 : Calculate Financial Consistency ========== ")

# Calculate the expected net worth from quantity multiplied by unit price.
# This checks whether the extracted financial values are mathematically
# consistent with each other.
anomaly_validation_df["Calculated Net Worth"] = (
    anomaly_validation_df["Quantity"]
    * anomaly_validation_df["Unit Price"]
)

# Calculate the expected gross worth using the 10% VAT rate used
# throughout the validated invoice dataset.
anomaly_validation_df["Calculated Gross Worth"] = (
    anomaly_validation_df["Net Worth"] * 1.10
)

# Calculate the absolute difference between extracted and calculated
# net worth.
anomaly_validation_df["Net Worth Difference"] = (
    anomaly_validation_df["Net Worth"]
    - anomaly_validation_df["Calculated Net Worth"]
).abs()

# Calculate the absolute difference between extracted and calculated
# gross worth.
anomaly_validation_df["Gross Worth Difference"] = (
    anomaly_validation_df["Gross Worth"]
    - anomaly_validation_df["Calculated Gross Worth"]
).abs()

# Consider a financial value consistent when the difference is
# less than or equal to one cent.
anomaly_validation_df["Net Worth Consistent"] = (
    anomaly_validation_df["Net Worth Difference"] <= 0.01
)

anomaly_validation_df["Gross Worth Consistent"] = (
    anomaly_validation_df["Gross Worth Difference"] <= 0.01
)

print("\nFinancial consistency calculated.")


#=============================
# Step 42.7 : Validate Quantity Against Ground Truth
#=============================
print("\n========== Step 42.7 : Validate Quantity Against Ground Truth ========== ")

# Create a column to store the trusted quantity obtained from
# the ground-truth JSON data.
anomaly_validation_df["Ground Truth Quantity"] = np.nan

# Create a column to store whether the extracted quantity
# matches the ground-truth quantity.
anomaly_validation_df["Quantity Ground Truth Match"] = False


# Check that the required matching columns exist in both dataframes.
required_anomaly_columns = [
    "Image Name",
    "Item Number"
]

required_ground_truth_columns = [
    "Image Name",
    "Item Number",
    "JSON Quantity"
]


if all(
    column in anomaly_validation_df.columns
    for column in required_anomaly_columns
) and all(
    column in ground_truth_anomalies.columns
    for column in required_ground_truth_columns
):

    # Check every potential anomaly individually.
    for index, row in anomaly_validation_df.iterrows():

        # Find the corresponding ground-truth record
        # using the invoice image and item number.
        matching_ground_truth = ground_truth_anomalies[
            (
                ground_truth_anomalies["Image Name"]
                == row["Image Name"]
            )
            &
            (
                ground_truth_anomalies["Item Number"]
                == row["Item Number"]
            )
        ]

        # If a matching ground-truth record is found,
        # retrieve its trusted quantity.
        if not matching_ground_truth.empty:

            ground_truth_quantity = matching_ground_truth.iloc[0][
                "JSON Quantity"
            ]

            # Store the ground-truth quantity.
            anomaly_validation_df.loc[
                index,
                "Ground Truth Quantity"
            ] = ground_truth_quantity

            # Compare the extracted quantity with
            # the trusted ground-truth quantity.
            anomaly_validation_df.loc[
                index,
                "Quantity Ground Truth Match"
            ] = np.isclose(
                row["Quantity"],
                ground_truth_quantity,
                atol=0.01
            )

print("\nQuantity ground-truth validation completed.")

print("\nQuantity validation results:")
print(
    anomaly_validation_df[
        [
            "Image Name",
            "Item Number",
            "Quantity",
            "Ground Truth Quantity",
            "Quantity Ground Truth Match"
        ]
    ].head(10)
)


#=============================
# Step 42.8 : Calculate Validation Statistics
#=============================
print("\n========== Step 42.8 : Calculate Validation Statistics ========== ")

total_anomalies_checked = len(anomaly_validation_df)

quantity_matches = anomaly_validation_df[
    "Quantity Ground Truth Match"
].sum()

net_worth_matches = anomaly_validation_df[
    "Net Worth Consistent"
].sum()

gross_worth_matches = anomaly_validation_df[
    "Gross Worth Consistent"
].sum()

print("\n========== ANOMALY VALIDATION SUMMARY ==========")

print(
    "Potential anomalies checked:",
    total_anomalies_checked
)

print(
    "Quantity ground-truth matches:",
    quantity_matches
)

if total_anomalies_checked > 0:
    print(
        "Quantity ground-truth accuracy:",
        round(
            (quantity_matches / total_anomalies_checked) * 100,
            2
        ),
        "%"
    )

print(
    "Net Worth internally consistent:",
    net_worth_matches
)

if total_anomalies_checked > 0:
    print(
        "Net Worth consistency:",
        round(
            (net_worth_matches / total_anomalies_checked) * 100,
            2
        ),
        "%"
    )

print(
    "Gross Worth internally consistent:",
    gross_worth_matches
)

if total_anomalies_checked > 0:
    print(
        "Gross Worth consistency:",
        round(
            (gross_worth_matches / total_anomalies_checked) * 100,
            2
        ),
        "%"
    )


#=============================
# Step 42.9 : Classify Anomalies
#=============================
print("\n========== Step 42.9 : Classify Anomalies ========== ")

# An anomaly that is financially consistent and matches the
# ground truth is considered a genuine statistical anomaly
# rather than an obvious extraction error.
anomaly_validation_df["Validation Result"] = np.where(
    (
        anomaly_validation_df["Net Worth Consistent"]
        &
        anomaly_validation_df["Gross Worth Consistent"]
        &
        anomaly_validation_df["Quantity Ground Truth Match"]
    ),
    "Statistically Unusual - Validated",
    "Requires Further Investigation"
)

print("\nValidation classification:")
print(
    anomaly_validation_df[
        "Validation Result"
    ].value_counts()
)


#=============================
# Step 42.10 : Sort Anomalies by Severity
#=============================
print("\n========== Step 42.10 : Sort Anomalies by Severity ========== ")

# Sort by anomaly score.
# More negative values indicate stronger statistical anomalies.
anomaly_validation_df = anomaly_validation_df.sort_values(
    by="Anomaly Score",
    ascending=True
).reset_index(drop=True)

print("\nTop 20 strongest potential anomalies:")

display_columns = [
    "Quantity",
    "Unit Price",
    "Net Worth",
    "Gross Worth",
    "Anomaly Score",
    "Anomaly Status",
    "Validation Result"
]

# Display only columns that actually exist.
available_display_columns = [
    column
    for column in display_columns
    if column in anomaly_validation_df.columns
]

print(
    anomaly_validation_df[
        available_display_columns
    ].head(20)
)


#=============================
# Step 42.11 : Count Validated Statistical Anomalies
#=============================
print("\n========== Step 42.11 : Count Validated Statistical Anomalies ========== ")

validated_statistical_anomalies = anomaly_validation_df[
    anomaly_validation_df["Validation Result"]
    == "Statistically Unusual - Validated"
].copy()

requires_further_investigation = anomaly_validation_df[
    anomaly_validation_df["Validation Result"]
    == "Requires Further Investigation"
].copy()

print(
    "\nStatistically unusual and validated:",
    len(validated_statistical_anomalies)
)

print(
    "Requires further investigation:",
    len(requires_further_investigation)
)


#=============================
# Step 42.12 : Final Anomaly Validation Output
#=============================
print("\n========== Step 42.12 : Final Anomaly Validation Output ========== ")

print("\n========== FINAL ANOMALY VALIDATION SUMMARY ==========")

print(
    "Total potential anomalies:",
    len(anomaly_validation_df)
)

print(
    "Statistically unusual and validated:",
    len(validated_statistical_anomalies)
)

print(
    "Requires further investigation:",
    len(requires_further_investigation)
)

print("\nTop validated statistical anomalies:")

print(
    validated_statistical_anomalies[
        available_display_columns
    ].head(10)
)

print("\nStep 42 completed successfully.")


#=============================
# Step 43 : Anomaly Analysis / Interpretation
#=============================
print("\n========== Step 43 : Anomaly Analysis / Interpretation ========== ")


#=============================
# Step 43.1 : Prepare Validated Anomaly Data
#=============================
print("\n========== Step 43.1 : Prepare Validated Anomaly Data ========== ")

# Create a separate dataframe containing only the records
# already identified as potential anomalies.
anomaly_analysis_df = anomaly_validation_df[
    anomaly_validation_df["Anomaly Prediction"] == -1
].copy()

print("Potential anomalies available for analysis:")
print(len(anomaly_analysis_df))


#=============================
# Step 43.2 : Rank Anomalies by Severity
#=============================
print("\n========== Step 43.2 : Rank Anomalies by Severity ========== ")

# Isolation Forest produces an anomaly score.
# A more negative score means that the record is more unusual.
anomaly_analysis_df = anomaly_analysis_df.sort_values(
    by="Anomaly Score",
    ascending=True
).reset_index(drop=True)

print("\nTop 20 anomalies by statistical severity:")

print(
    anomaly_analysis_df[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Quantity",
            "Unit Price",
            "Net Worth",
            "Gross Worth",
            "Anomaly Score",
            "Validation Result"
        ]
    ].head(20)
)


#=============================
# Step 43.3 : Analyze Financial Magnitude
#=============================
print("\n========== Step 43.3 : Analyze Financial Magnitude ========== ")

# Calculate the average net worth of all validated financial records.
average_net_worth = X_ml["Net Worth"].mean()

# Calculate the average gross worth of all validated financial records.
average_gross_worth = X_ml["Gross Worth"].mean()

# Calculate how many times larger each anomaly's net worth
# is compared with the average net worth.
anomaly_analysis_df["Net Worth vs Average"] = (
    anomaly_analysis_df["Net Worth"]
    / average_net_worth
)

# Calculate how many times larger each anomaly's gross worth
# is compared with the average gross worth.
anomaly_analysis_df["Gross Worth vs Average"] = (
    anomaly_analysis_df["Gross Worth"]
    / average_gross_worth
)

print("\nAverage Net Worth of validated records:")
print(round(average_net_worth, 2))

print("\nAverage Gross Worth of validated records:")
print(round(average_gross_worth, 2))

print("\nHighest-value anomalous records:")

print(
    anomaly_analysis_df.sort_values(
        by="Net Worth",
        ascending=False
    )[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Quantity",
            "Unit Price",
            "Net Worth",
            "Gross Worth",
            "Net Worth vs Average",
            "Anomaly Score"
        ]
    ].head(10)
)


#=============================
# Step 43.4 : Identify Repeated Financial Patterns
#=============================
print("\n========== Step 43.4 : Identify Repeated Financial Patterns ========== ")

# Group anomalies having the same unit price.
# This helps identify repeated unusual pricing patterns.
repeated_unit_prices = (
    anomaly_analysis_df
    .groupby("Unit Price")
    .size()
    .reset_index(name="Occurrence Count")
    .sort_values(
        by="Occurrence Count",
        ascending=False
    )
)

print("\nRepeated unit prices among anomalies:")

print(
    repeated_unit_prices.head(15)
)


#=============================
# Step 43.5 : Identify Investigation Candidates
#=============================
print("\n========== Step 43.5 : Identify Investigation Candidates ========== ")

# Records requiring further investigation have already been
# identified during Step 42 validation.
investigation_df = anomaly_analysis_df[
    anomaly_analysis_df["Validation Result"]
    == "Requires Further Investigation"
].copy()

print("\nRecords requiring further investigation:")
print(len(investigation_df))

if not investigation_df.empty:

    print("\nInvestigation candidates:")

    print(
        investigation_df[
            [
                "Invoice Number",
                "Image Name",
                "Item Number",
                "Quantity",
                "Unit Price",
                "Net Worth",
                "Gross Worth",
                "Anomaly Score",
                "Validation Result"
            ]
        ]
    )

else:

    print("\nNo additional validation inconsistencies were found.")


#=============================
# Step 43.6 : Create Investigation Priority
#=============================
print("\n========== Step 43.6 : Create Investigation Priority ========== ")

# Create a priority column.
# Higher priority is assigned to records that require
# further investigation.
anomaly_analysis_df["Investigation Priority"] = "Medium"

# Records that are statistically unusual but fully validated
# are given medium priority because they are unusual but
# internally consistent.
anomaly_analysis_df.loc[
    anomaly_analysis_df["Validation Result"]
    == "Statistically Unusual - Validated",
    "Investigation Priority"
] = "Medium"

# Records with validation inconsistencies are given high priority.
anomaly_analysis_df.loc[
    anomaly_analysis_df["Validation Result"]
    == "Requires Further Investigation",
    "Investigation Priority"
] = "High"


# Extremely unusual records are given high priority.
# Here we use an anomaly-score threshold of -0.25.
# This is an analytical ranking threshold, not a fraud threshold.
anomaly_analysis_df.loc[
    anomaly_analysis_df["Anomaly Score"] <= -0.25,
    "Investigation Priority"
] = "High"


print("\nInvestigation priority distribution:")

print(
    anomaly_analysis_df[
        "Investigation Priority"
    ].value_counts()
)


#=============================
# Step 43.7 : Display Highest Priority Records
#=============================
print("\n========== Step 43.7 : Display Highest Priority Records ========== ")

high_priority_anomalies = anomaly_analysis_df[
    anomaly_analysis_df["Investigation Priority"] == "High"
].copy()

high_priority_anomalies = high_priority_anomalies.sort_values(
    by="Anomaly Score",
    ascending=True
)

print("\nHighest-priority anomaly records:")

print(
    high_priority_anomalies[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Quantity",
            "Unit Price",
            "Net Worth",
            "Gross Worth",
            "Anomaly Score",
            "Validation Result",
            "Investigation Priority"
        ]
    ].head(20)
)


#=============================
# Step 43.8 : Generate Anomaly Analysis Summary
#=============================
print("\n========== Step 43.8 : Generate Anomaly Analysis Summary ========== ")

total_anomalies_analyzed = len(anomaly_analysis_df)

high_priority_count = len(
    anomaly_analysis_df[
        anomaly_analysis_df["Investigation Priority"]
        == "High"
    ]
)

medium_priority_count = len(
    anomaly_analysis_df[
        anomaly_analysis_df["Investigation Priority"]
        == "Medium"
    ]
)

validated_anomaly_count = len(
    anomaly_analysis_df[
        anomaly_analysis_df["Validation Result"]
        == "Statistically Unusual - Validated"
    ]
)

investigation_count = len(
    anomaly_analysis_df[
        anomaly_analysis_df["Validation Result"]
        == "Requires Further Investigation"
    ]
)

print("\n========== ANOMALY ANALYSIS SUMMARY ==========")

print(
    "Total anomalies analyzed:",
    total_anomalies_analyzed
)

print(
    "Statistically unusual and validated:",
    validated_anomaly_count
)

print(
    "Requires further investigation:",
    investigation_count
)

print(
    "High-priority anomalies:",
    high_priority_count
)

print(
    "Medium-priority anomalies:",
    medium_priority_count
)

print(
    "Highest anomaly score severity:",
    round(
        anomaly_analysis_df["Anomaly Score"].min(),
        6
    )
)

print(
    "Highest anomalous Net Worth:",
    round(
        anomaly_analysis_df["Net Worth"].max(),
        2
    )
)


#=============================
# Step 43.9 : Store Final Analysis Dataset
#=============================
print("\n========== Step 43.9 : Store Final Analysis Dataset ========== ")

# Save the analyzed anomaly dataframe in the existing
# project variable so later steps can directly use it.
anomaly_report_df = anomaly_analysis_df.copy()

print(
    "\nFinal anomaly analysis dataframe shape:",
    anomaly_report_df.shape
)

print("\nStep 43 completed successfully.")


#=============================
# Step 44 : Final Anomaly Report
#=============================
print("\n========== Step 44 : Final Anomaly Report ========== ")


#=============================
# Step 44.1 : Select Final Report Columns
#=============================
print("\n========== Step 44.1 : Select Final Report Columns ========== ")

# These are the important fields that should appear
# in the final anomaly report.
final_report_columns = [
    "Invoice Number",
    "Image Name",
    "Item Number",
    "Quantity",
    "Unit Price",
    "Net Worth",
    "Gross Worth",
    "Anomaly Score",
    "Validation Result",
    "Investigation Priority"
]

# Check which required columns are actually available.
available_report_columns = [
    column
    for column in final_report_columns
    if column in anomaly_report_df.columns
]

# Create the final report using only available columns.
final_anomaly_report = anomaly_report_df[
    available_report_columns
].copy()

print("\nFinal report columns:")
print(final_anomaly_report.columns.tolist())


#=============================
# Step 44.2 : Create Risk Level
#=============================
print("\n========== Step 44.2 : Create Risk Level ========== ")

# Risk Level is a human-readable interpretation of
# the statistical anomaly and validation information.

final_anomaly_report["Risk Level"] = "Medium"

# Records requiring further investigation receive
# the highest risk level because their validation
# checks contain inconsistencies.
final_anomaly_report.loc[
    final_anomaly_report["Validation Result"]
    == "Requires Further Investigation",
    "Risk Level"
] = "High"

# Extremely unusual statistical records also receive
# high priority for review.
final_anomaly_report.loc[
    final_anomaly_report["Anomaly Score"] <= -0.25,
    "Risk Level"
] = "High"

print("\nRisk level distribution:")
print(
    final_anomaly_report[
        "Risk Level"
    ].value_counts()
)


#=============================
# Step 44.3 : Arrange Report Columns
#=============================
print("\n========== Step 44.3 : Arrange Report Columns ========== ")

# Arrange the columns in a logical order:
# identification -> financial information ->
# ML result -> validation -> investigation.
preferred_column_order = [
    "Invoice Number",
    "Image Name",
    "Item Number",
    "Quantity",
    "Unit Price",
    "Net Worth",
    "Gross Worth",
    "Anomaly Score",
    "Risk Level",
    "Validation Result",
    "Investigation Priority"
]

# Keep only columns that are available.
final_column_order = [
    column
    for column in preferred_column_order
    if column in final_anomaly_report.columns
]

final_anomaly_report = final_anomaly_report[
    final_column_order
].copy()

print("\nFinal report structure:")
print(final_anomaly_report.columns.tolist())


#=============================
# Step 44.4 : Sort Final Report by Severity
#=============================
print("\n========== Step 44.4 : Sort Final Report by Severity ========== ")

# The most negative anomaly score represents the
# strongest statistical anomaly.
final_anomaly_report = final_anomaly_report.sort_values(
    by="Anomaly Score",
    ascending=True
).reset_index(drop=True)

print("\nTop 10 records in final anomaly report:")

print(
    final_anomaly_report.head(10)
)


#=============================
# Step 44.5 : Create High-Priority Report
#=============================
print("\n========== Step 44.5 : Create High-Priority Report ========== ")

# Extract records marked as High risk.
high_priority_report = final_anomaly_report[
    final_anomaly_report["Risk Level"] == "High"
].copy()

print("\nHigh-priority records:")
print(len(high_priority_report))

print("\nTop high-priority records:")

print(
    high_priority_report.head(20)
)


#=============================
# Step 44.6 : Create Investigation Report
#=============================
print("\n========== Step 44.6 : Create Investigation Report ========== ")

# These records specifically failed one or more
# validation conditions and therefore require
# additional investigation.
investigation_report = final_anomaly_report[
    final_anomaly_report["Validation Result"]
    == "Requires Further Investigation"
].copy()

print("\nRecords requiring further investigation:")
print(len(investigation_report))

print("\nInvestigation report:")

print(
    investigation_report
)


#=============================
# Step 44.7 : Create Validated Anomaly Report
#=============================
print("\n========== Step 44.7 : Create Validated Anomaly Report ========== ")

# These records are statistically unusual but passed
# the validation checks performed in Step 42.
validated_anomaly_report = final_anomaly_report[
    final_anomaly_report["Validation Result"]
    == "Statistically Unusual - Validated"
].copy()

print("\nStatistically unusual and validated records:")
print(len(validated_anomaly_report))

print("\nTop validated anomalies:")

print(
    validated_anomaly_report.head(10)
)


#=============================
# Step 44.8 : Calculate Final Report Statistics
#=============================
print("\n========== Step 44.8 : Calculate Final Report Statistics ========== ")

total_report_records = len(
    final_anomaly_report
)

high_risk_records = len(
    final_anomaly_report[
        final_anomaly_report["Risk Level"] == "High"
    ]
)

medium_risk_records = len(
    final_anomaly_report[
        final_anomaly_report["Risk Level"] == "Medium"
    ]
)

validated_records = len(
    validated_anomaly_report
)

investigation_records = len(
    investigation_report
)

# Calculate percentages relative to all potential anomalies.
high_risk_percentage = (
    high_risk_records
    / total_report_records
    * 100
    if total_report_records > 0
    else 0
)

validated_percentage = (
    validated_records
    / total_report_records
    * 100
    if total_report_records > 0
    else 0
)


#=============================
# Step 44.9 : Display Final Report Summary
#=============================
print("\n========== Step 44.9 : Display Final Report Summary ========== ")

print("\n========== FINAL ANOMALY REPORT SUMMARY ==========")

print(
    "Total potential anomalies:",
    total_report_records
)

print(
    "Statistically unusual and validated:",
    validated_records
)

print(
    "Requires further investigation:",
    investigation_records
)

print(
    "High-risk records:",
    high_risk_records
)

print(
    "Medium-risk records:",
    medium_risk_records
)

print(
    "Validated anomaly percentage:",
    round(
        validated_percentage,
        2
    ),
    "%"
)

print(
    "High-risk percentage:",
    round(
        high_risk_percentage,
        2
    ),
    "%"
)

print(
    "Strongest anomaly score:",
    round(
        final_anomaly_report["Anomaly Score"].min(),
        6
    )
)

print(
    "Highest anomalous Net Worth:",
    round(
        final_anomaly_report["Net Worth"].max(),
        2
    )
)


#=============================
# Step 44.10 : Display Top Final Anomalies
#=============================
print("\n========== Step 44.10 : Display Top Final Anomalies ========== ")

print("\nTop 10 final anomaly records:")

print(
    final_anomaly_report[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Quantity",
            "Unit Price",
            "Net Worth",
            "Gross Worth",
            "Anomaly Score",
            "Risk Level",
            "Validation Result",
            "Investigation Priority"
        ]
    ].head(10)
)


#=============================
# Step 44.11 : Store Final Report
#=============================
print("\n========== Step 44.11 : Store Final Report ========== ")

# Keep the final report in a dedicated variable.
# This variable will be used by the documentation
# and final testing stages.
anomaly_report_df = final_anomaly_report.copy()

print(
    "\nFinal anomaly report dataframe shape:",
    anomaly_report_df.shape
)

print(
    "\nFinal anomaly report created successfully."
)


#=============================
# Step 44.12 : Complete Step 44
#=============================
print("\n========== Step 44.12 : Complete Step 44 ========== ")

print("\nStep 44 completed successfully.")


#=============================
# Step 45A : Vendor-wise Z-score Analysis
#=============================
print("\n========== Step 45A : Vendor-wise Z-score Analysis ========== ")


# Import NumPy for numerical calculations
import numpy as np


# -----------------------------------------------------
# CREATE INVOICE-LEVEL DATASET FOR Z-SCORE ANALYSIS
# -----------------------------------------------------

print("\n--- Creating invoice-level dataset ---")


# This list will store extracted invoice information
vendor_invoice_records = []


# Process every OCR result from the complete dataset
for image_name, ocr_text in all_ocr_results.items():

    # Extract invoice-level information using our
    # existing extraction function
    invoice_information = extract_invoice_information(
        ocr_text
    )

    # Add image name 
    invoice_information["Image Name"] = image_name

    # Store the extracted invoice information
    vendor_invoice_records.append(
        invoice_information
    )


# Convert the list of dictionaries into a DataFrame
vendor_invoice_df = pd.DataFrame(
    vendor_invoice_records
)


print("\n--- Invoice-level dataset created ---")

print(
    vendor_invoice_df[
        [
            "Invoice Number",
            "Image Name",
            "Vendor Name",
            "Total Amount"
        ]
    ].head(10)
)


# -----------------------------------------------------
# CLEAN VENDOR AND AMOUNT COLUMNS
# -----------------------------------------------------

print("\n--- Cleaning vendor and amount data ---")


# Remove leading and trailing spaces from vendor names
vendor_invoice_df["Vendor Name"] = (
    vendor_invoice_df["Vendor Name"]
    .astype("string")
    .str.strip()
)


# Convert Total Amount into numerical values
# Invalid values are converted into NaN
vendor_invoice_df["Total Amount"] = pd.to_numeric(
    vendor_invoice_df["Total Amount"],
    errors="coerce"
)


# Remove records where vendor name or invoice amount
# is missing because they cannot participate in
# vendor-wise statistical analysis
zscore_df = vendor_invoice_df.dropna(
    subset=[
        "Vendor Name",
        "Total Amount"
    ]
).copy()


print(
    "\nInvoices available for Z-score analysis:",
    len(zscore_df)
)


# -----------------------------------------------------
# CALCULATE VENDOR STATISTICS
# -----------------------------------------------------

print("\n--- Calculating vendor-wise statistics ---")


# Calculate the average invoice amount for each vendor
zscore_df["Vendor Mean"] = (
    zscore_df
    .groupby("Vendor Name")["Total Amount"]
    .transform("mean")
)


# Calculate the standard deviation of invoice amounts
# for each vendor
zscore_df["Vendor Std"] = (
    zscore_df
    .groupby("Vendor Name")["Total Amount"]
    .transform("std")
)


# Count how many invoices belong to each vendor
zscore_df["Vendor Invoice Count"] = (
    zscore_df
    .groupby("Vendor Name")["Total Amount"]
    .transform("count")
)


# -----------------------------------------------------
# CALCULATE Z-SCORE
# -----------------------------------------------------

print("\n--- Calculating Z-scores ---")


zscore_df["Vendor Z-score"] = (
    (
        zscore_df["Total Amount"]
        - zscore_df["Vendor Mean"]
    )
    / zscore_df["Vendor Std"]
)


# -----------------------------------------------------
# IDENTIFY STATISTICAL OUTLIERS
# -----------------------------------------------------

print("\n--- Identifying vendor-wise statistical outliers ---")


zscore_df["Absolute Z-score"] = (
    zscore_df["Vendor Z-score"].abs()
)


zscore_df["Z-score Outlier"] = (
    zscore_df["Absolute Z-score"] >= 3
)


# -----------------------------------------------------
# ADD INTERPRETATION
# -----------------------------------------------------

print("\n--- Creating Z-score interpretation ---")


# Default classification
zscore_df["Z-score Status"] = "Normal"


# Mark statistically unusual invoices
zscore_df.loc[
    zscore_df["Z-score Outlier"] == True,
    "Z-score Status"
] = "Statistical Outlier"


# Records without a valid Z-score occur mainly when there are insufficient invoices for a vendor.
zscore_df.loc[
    zscore_df["Vendor Z-score"].isna(),
    "Z-score Status"
] = "Insufficient Vendor History"


#=============================
# Step 45A.1 : Display Vendor-wise Z-score Results
#=============================
print("\n========== Step 45A.1 : Display Vendor-wise Z-score Results ========== ")


print("\nAvailable Z-score columns:")

print(
    zscore_df.columns.tolist()
)


# -----------------------------------------------------
# RECREATE ABSOLUTE Z-SCORE
# -----------------------------------------------------


zscore_df["Absolute Z-score"] = (
    zscore_df["Vendor Z-score"].abs()
)


# -----------------------------------------------------
# DISPLAY TOP STATISTICAL OUTLIERS
# -----------------------------------------------------

print(
    "\n================ VENDOR-WISE Z-SCORE RESULTS ================\n"
)


# Sort invoices according to the magnitude of their
# Z-score.

zscore_display_df = (
    zscore_df[
        [
            "Invoice Number",
            "Image Name",
            "Vendor Name",
            "Total Amount",
            "Vendor Invoice Count",
            "Vendor Mean",
            "Vendor Std",
            "Vendor Z-score",
            "Absolute Z-score",
            "Z-score Outlier",
            "Z-score Status"
        ]
    ]
    .sort_values(
        "Absolute Z-score",
        ascending=False,
        na_position="last"
    )
    .head(20)
)


print(
    zscore_display_df
)


# -----------------------------------------------------
# Z-SCORE SUMMARY
# -----------------------------------------------------

print("\n================ Z-SCORE SUMMARY ================\n")


# Count invoices having a valid Z-score
valid_zscore_count = (
    zscore_df["Vendor Z-score"]
    .notna()
    .sum()
)


# Count statistical outliers
zscore_outlier_count = (
    zscore_df["Z-score Outlier"]
    .sum()
)


# Count records without sufficient vendor history
insufficient_history_count = (
    zscore_df["Z-score Status"]
    == "Insufficient Vendor History"
).sum()


print(
    "Invoices available for Z-score analysis:",
    len(zscore_df)
)


print(
    "Invoices with valid Z-scores:",
    valid_zscore_count
)


print(
    "Statistical outliers (|Z| >= 3):",
    zscore_outlier_count
)


print(
    "Invoices with insufficient vendor history:",
    insufficient_history_count
)


# -----------------------------------------------------
# DISPLAY STRONGEST STATISTICAL OUTLIERS
# -----------------------------------------------------

print(
    "\n--- Strongest Vendor-wise Statistical Outliers ---\n"
)


strongest_zscore_outliers = zscore_df[
    zscore_df["Z-score Outlier"] == True
].sort_values(
    "Absolute Z-score",
    ascending=False
)


if strongest_zscore_outliers.empty:

    print(
        "No vendor-wise statistical outliers were detected."
    )

else:

    print(
        strongest_zscore_outliers[
            [
                "Invoice Number",
                "Image Name",
                "Vendor Name",
                "Total Amount",
                "Vendor Mean",
                "Vendor Z-score",
                "Z-score Status"
            ]
        ]
    )


# -----------------------------------------------------
# STORE FINAL Z-SCORE DATASET
# -----------------------------------------------------

vendor_zscore_df = zscore_df.copy()


print(
    "\nVendor-wise Z-score analysis completed successfully."
)


#=============================
# Step 45B : Date Logic Validation
#=============================
print("\n========== Step 45B : Date Logic Validation ========== ")

# -----------------------------------------------------
# CREATE DATE VALIDATION DATASET
# -----------------------------------------------------

print("\n--- Preparing invoice dates ---")


# Make a copy so that the original invoice dataframe is not modified.
date_validation_df = vendor_invoice_df.copy()


# Convert the Invoice Date column into Pandas datetime.

date_validation_df["Invoice Date"] = pd.to_datetime(
    date_validation_df["Invoice Date"],
    errors="coerce"
)


# -----------------------------------------------------
# CHECK AVAILABLE DATES
# -----------------------------------------------------

valid_invoice_dates = (
    date_validation_df["Invoice Date"]
    .dropna()
)


print(
    "\nTotal invoices:",
    len(date_validation_df)
)


print(
    "Invoices with valid dates:",
    len(valid_invoice_dates)
)


print(
    "Invoices with missing/invalid dates:",
    date_validation_df["Invoice Date"].isna().sum()
)


# -----------------------------------------------------
# DEFINE REFERENCE DATE
# -----------------------------------------------------

print("\n--- Defining reference date ---")


if len(valid_invoice_dates) > 0:

    reference_date = valid_invoice_dates.max()

else:

    reference_date = pd.Timestamp.today().normalize()


print(
    "Reference date:",
    reference_date.date()
)


# -----------------------------------------------------
# DEFINE AGE THRESHOLD
# -----------------------------------------------------

DATE_THRESHOLD_DAYS = 90


print(
    "Invoice age threshold:",
    DATE_THRESHOLD_DAYS,
    "days"
)


# -----------------------------------------------------
# CALCULATE INVOICE AGE
# -----------------------------------------------------

print("\n--- Calculating invoice age ---")


# Calculate how many days have passed between the invoice date and the reference date.

date_validation_df["Invoice Age Days"] = (
    reference_date
    - date_validation_df["Invoice Date"]
).dt.days


# -----------------------------------------------------
# FUTURE DATE VALIDATION
# -----------------------------------------------------

print("\n--- Checking future invoice dates ---")


# An invoice is considered a future invoice if its date occurs after the reference date.

date_validation_df["Future Date Flag"] = (
    date_validation_df["Invoice Date"]
    > reference_date
)


# -----------------------------------------------------
# OLD INVOICE VALIDATION
# -----------------------------------------------------

print("\n--- Checking invoices older than threshold ---")


# An invoice is considered old when its age exceeds the configured threshold.

date_validation_df["Older Than Threshold"] = (
    date_validation_df["Invoice Age Days"]
    > DATE_THRESHOLD_DAYS
)


# -----------------------------------------------------
# CREATE DATE VALIDATION STATUS
# -----------------------------------------------------

print("\n--- Creating date validation status ---")


date_validation_df["Date Validation Status"] = (
    "Valid"
)


date_validation_df.loc[
    date_validation_df["Invoice Date"].isna(),
    "Date Validation Status"
] = "Invalid/Missing Date"


# Mark future dates.
date_validation_df.loc[
    date_validation_df["Future Date Flag"] == True,
    "Date Validation Status"
] = "Future Invoice Date"


# Mark invoices older than the configured threshold.
date_validation_df.loc[
    date_validation_df["Older Than Threshold"] == True,
    "Date Validation Status"
] = "Older Than 90 Days"


# -----------------------------------------------------
# CREATE DATE REASON
# -----------------------------------------------------

print("\n--- Creating date validation reasons ---")


# Default reason for valid dates.
date_validation_df["Date Validation Reason"] = (
    "Date within acceptable range"
)


# Missing/invalid date reason.
date_validation_df.loc[
    date_validation_df["Invoice Date"].isna(),
    "Date Validation Reason"
] = "Invoice date is missing or invalid"


# Future date reason.
date_validation_df.loc[
    date_validation_df["Future Date Flag"] == True,
    "Date Validation Reason"
] = "Invoice date occurs after the reference date"


# Old invoice reason.
date_validation_df.loc[
    date_validation_df["Older Than Threshold"] == True,
    "Date Validation Reason"
] = (
    "Invoice is older than "
    + str(DATE_THRESHOLD_DAYS)
    + " days"
)


# -----------------------------------------------------
# DISPLAY DATE VALIDATION RESULTS
# -----------------------------------------------------

print(
    "\n================ DATE VALIDATION RESULTS ================\n"
)


print(
    date_validation_df[
        [
            "Invoice Number",
            "Image Name",
            "Invoice Date",
            "Invoice Age Days",
            "Future Date Flag",
            "Older Than Threshold",
            "Date Validation Status",
            "Date Validation Reason"
        ]
    ]
    .head(20)
)


# -----------------------------------------------------
# DISPLAY FLAGGED INVOICES
# -----------------------------------------------------

print(
    "\n--- Invoices flagged by date validation ---\n"
)


flagged_date_records = date_validation_df[
    (
        date_validation_df["Future Date Flag"] == True
    )
    |
    (
        date_validation_df["Older Than Threshold"] == True
    )
    |
    (
        date_validation_df["Invoice Date"].isna()
    )
].copy()


if flagged_date_records.empty:

    print(
        "No invoices were flagged by date validation."
    )

else:

    print(
        flagged_date_records[
            [
                "Invoice Number",
                "Image Name",
                "Invoice Date",
                "Invoice Age Days",
                "Date Validation Status",
                "Date Validation Reason"
            ]
        ]
    )


# -----------------------------------------------------
# DATE VALIDATION SUMMARY
# -----------------------------------------------------

print(
    "\n================ DATE VALIDATION SUMMARY ================\n"
)


future_date_count = (
    date_validation_df["Future Date Flag"]
    .sum()
)


old_invoice_count = (
    date_validation_df["Older Than Threshold"]
    .sum()
)


missing_date_count = (
    date_validation_df["Invoice Date"]
    .isna()
    .sum()
)


valid_date_count = (
    (
        date_validation_df["Date Validation Status"]
        == "Valid"
    )
    .sum()
)


print(
    "Total invoices checked:",
    len(date_validation_df)
)


print(
    "Valid invoice dates:",
    valid_date_count
)


print(
    "Future invoice dates:",
    future_date_count
)


print(
    "Invoices older than",
    DATE_THRESHOLD_DAYS,
    "days:",
    old_invoice_count
)


print(
    "Missing/invalid invoice dates:",
    missing_date_count
)


print(
    "\nDate logic validation completed successfully."
)


# -----------------------------------------------------
# STORE FINAL DATE VALIDATION DATASET
# -----------------------------------------------------

date_logic_report_df = date_validation_df.copy()


#=============================
# Step 45C : Duplicate Detection with Fuzzy Matching
#=============================
print("\n========== Step 45C : Duplicate Detection with Fuzzy Matching ========== ")

from difflib import SequenceMatcher


# Create full invoice-level dataset
full_invoice_records = []

for image_name, text in all_ocr_results.items():

    invoice_info = extract_invoice_information(text)

    invoice_info["Image Name"] = image_name

    full_invoice_records.append(invoice_info)


full_invoice_df = pd.DataFrame(
    full_invoice_records
)


# Clean invoice number and amount
full_invoice_df["Invoice Number"] = (
    full_invoice_df["Invoice Number"]
    .astype(str)
    .str.strip()
)

full_invoice_df["Total Amount"] = pd.to_numeric(
    full_invoice_df["Total Amount"],
    errors="coerce"
)


# Create exact duplicate flags
full_invoice_df["Duplicate Invoice Number"] = (
    full_invoice_df
    .duplicated(
        subset=["Invoice Number"],
        keep=False
    )
)

full_invoice_df["Duplicate Amount"] = (
    full_invoice_df
    .duplicated(
        subset=["Total Amount"],
        keep=False
    )
)


# Create fuzzy duplicate columns
full_invoice_df["Fuzzy Duplicate Invoice"] = False
full_invoice_df["Fuzzy Duplicate Amount"] = False


# Compare invoice numbers and amounts
for i in range(len(full_invoice_df)):

    for j in range(i + 1, len(full_invoice_df)):

        invoice_1 = full_invoice_df.iloc[i]["Invoice Number"]
        invoice_2 = full_invoice_df.iloc[j]["Invoice Number"]

        amount_1 = full_invoice_df.iloc[i]["Total Amount"]
        amount_2 = full_invoice_df.iloc[j]["Total Amount"]


        # Fuzzy invoice number comparison
        if (
            invoice_1 != "nan"
            and invoice_2 != "nan"
        ):

            invoice_similarity = SequenceMatcher(
                None,
                invoice_1,
                invoice_2
            ).ratio()

            if (
                invoice_similarity >= 0.90
                and invoice_1 != invoice_2
            ):

                full_invoice_df.loc[
                    [i, j],
                    "Fuzzy Duplicate Invoice"
                ] = True


        # Fuzzy amount comparison
        if (
            pd.notna(amount_1)
            and pd.notna(amount_2)
            and amount_1 != amount_2
        ):

            amount_difference = abs(
                amount_1 - amount_2
            )

            if amount_difference <= 0.01:

                full_invoice_df.loc[
                    [i, j],
                    "Fuzzy Duplicate Amount"
                ] = True


# Create duplicate status
full_invoice_df["Duplicate Status"] = "Unique"

full_invoice_df.loc[
    full_invoice_df["Duplicate Invoice Number"],
    "Duplicate Status"
] = "Exact Duplicate Invoice Number"

full_invoice_df.loc[
    (
        ~full_invoice_df["Duplicate Invoice Number"]
        &
        full_invoice_df["Duplicate Amount"]
    ),
    "Duplicate Status"
] = "Exact Duplicate Amount"

full_invoice_df.loc[
    (
        ~full_invoice_df["Duplicate Invoice Number"]
        &
        ~full_invoice_df["Duplicate Amount"]
        &
        (
            full_invoice_df["Fuzzy Duplicate Invoice"]
            |
            full_invoice_df["Fuzzy Duplicate Amount"]
        )
    ),
    "Duplicate Status"
] = "Fuzzy Potential Duplicate"


# Create duplicate reason
full_invoice_df["Duplicate Reason"] = "No duplicate detected"

full_invoice_df.loc[
    full_invoice_df["Duplicate Invoice Number"],
    "Duplicate Reason"
] = "Exact invoice number appears more than once"

full_invoice_df.loc[
    (
        ~full_invoice_df["Duplicate Invoice Number"]
        &
        full_invoice_df["Duplicate Amount"]
    ),
    "Duplicate Reason"
] = "Exact invoice amount appears more than once"

full_invoice_df.loc[
    (
        ~full_invoice_df["Duplicate Invoice Number"]
        &
        ~full_invoice_df["Duplicate Amount"]
        &
        full_invoice_df["Fuzzy Duplicate Invoice"]
    ),
    "Duplicate Reason"
] = "Invoice number is highly similar to another invoice number"

full_invoice_df.loc[
    (
        ~full_invoice_df["Duplicate Invoice Number"]
        &
        ~full_invoice_df["Duplicate Amount"]
        &
        ~full_invoice_df["Fuzzy Duplicate Invoice"]
        &
        full_invoice_df["Fuzzy Duplicate Amount"]
    ),
    "Duplicate Reason"
] = "Invoice amount is within fuzzy matching tolerance"


# Store duplicate report
duplicate_report_df = full_invoice_df.copy()


print("\n================ DUPLICATE DETECTION RESULTS ================\n")

print(
    duplicate_report_df[
        [
            "Invoice Number",
            "Image Name",
            "Total Amount",
            "Duplicate Status",
            "Duplicate Reason"
        ]
    ].head(20)
)


# Create duplicate-only dataset
duplicate_records_df = duplicate_report_df[
    duplicate_report_df["Duplicate Status"] != "Unique"
].copy()


print("\n================ DUPLICATE SUMMARY ================\n")

print(
    "Total invoices checked:",
    len(duplicate_report_df)
)

print(
    "Exact duplicate invoice numbers:",
    duplicate_report_df[
        "Duplicate Invoice Number"
    ].sum()
)

print(
    "Exact duplicate amounts:",
    duplicate_report_df[
        "Duplicate Amount"
    ].sum()
)

print(
    "Fuzzy duplicate invoice numbers:",
    duplicate_report_df[
        "Fuzzy Duplicate Invoice"
    ].sum()
)

print(
    "Fuzzy duplicate amounts:",
    duplicate_report_df[
        "Fuzzy Duplicate Amount"
    ].sum()
)

print(
    "Total potential duplicate records:",
    len(duplicate_records_df)
)


print("\n--- Potential Duplicate Records ---")

if len(duplicate_records_df) > 0:

    print(
        duplicate_records_df[
            [
                "Invoice Number",
                "Image Name",
                "Total Amount",
                "Duplicate Status",
                "Duplicate Reason"
            ]
        ]
    )

else:

    print(
        "No duplicate invoice records detected."
    )


print(
    "\nDuplicate detection with fuzzy matching completed successfully."
)

#=============================
# Step 45D : Final Item-level Output CSV
#=============================
print("\n========== Step 45D : Final Item-level Output CSV ========== ")


# Create complete item-level base dataset
base_item_df = all_item_df[
    [
        "Invoice Number",
        "Image Name",
        "Item Number",
        "Description",
        "Quantity"
    ]
].drop_duplicates(
    subset=[
        "Invoice Number",
        "Image Name",
        "Item Number"
    ]
)


# Add financial information
financial_information_df = final_financial_df[
    [
        "Invoice Number",
        "Image Name",
        "Item Number",
        "Quantity",
        "Unit Price",
        "Net Worth",
        "VAT",
        "Gross Worth"
    ]
].drop_duplicates(
    subset=[
        "Invoice Number",
        "Image Name",
        "Item Number"
    ]
)


final_csv_df = base_item_df.merge(
    financial_information_df,
    on=[
        "Invoice Number",
        "Image Name",
        "Item Number"
    ],
    how="left",
    suffixes=("", "_Financial")
)


# Use validated financial quantity when available
final_csv_df["Quantity"] = (
    final_csv_df["Quantity_Financial"]
    .combine_first(final_csv_df["Quantity"])
)


final_csv_df = final_csv_df.drop(
    columns=["Quantity_Financial"]
)


# Add invoice-level information
invoice_information_df = full_invoice_df[
    [
        "Invoice Number",
        "Invoice Date",
        "Vendor Name",
        "Total Amount"
    ]
].drop_duplicates(
    subset=["Invoice Number"]
)


final_csv_df = final_csv_df.merge(
    invoice_information_df,
    on="Invoice Number",
    how="left"
)


# Add anomaly information
anomaly_information_df = anomaly_report_df[
    [
        "Invoice Number",
        "Image Name",
        "Item Number",
        "Anomaly Score",
        "Validation Result",
        "Investigation Priority"
    ]
].drop_duplicates(
    subset=[
        "Invoice Number",
        "Image Name",
        "Item Number"
    ]
)


final_csv_df = final_csv_df.merge(
    anomaly_information_df,
    on=[
        "Invoice Number",
        "Image Name",
        "Item Number"
    ],
    how="left"
)


# Create anomaly flag
final_csv_df["IsAnomaly"] = (
    final_csv_df["Anomaly Score"].notna()
)


# Create anomaly reason
final_csv_df["AnomalyReason"] = "Normal"

final_csv_df.loc[
    final_csv_df["IsAnomaly"],
    "AnomalyReason"
] = final_csv_df.loc[
    final_csv_df["IsAnomaly"],
    "Validation Result"
]


# Add duplicate information
duplicate_information_df = duplicate_report_df[
    [
        "Invoice Number",
        "Duplicate Status",
        "Duplicate Reason"
    ]
].drop_duplicates(
    subset=["Invoice Number"]
)


final_csv_df = final_csv_df.merge(
    duplicate_information_df,
    on="Invoice Number",
    how="left"
)


# Fill missing anomaly information
final_csv_df["Anomaly Score"] = (
    final_csv_df["Anomaly Score"]
    .fillna(0)
)

final_csv_df["Validation Result"] = (
    final_csv_df["Validation Result"]
    .fillna("Normal")
)

final_csv_df["Investigation Priority"] = (
    final_csv_df["Investigation Priority"]
    .fillna("None")
)


# Fill missing duplicate information
final_csv_df["Duplicate Status"] = (
    final_csv_df["Duplicate Status"]
    .fillna("Unique")
)

final_csv_df["Duplicate Reason"] = (
    final_csv_df["Duplicate Reason"]
    .fillna("No duplicate detected")
)


# Arrange final columns
final_csv_df = final_csv_df[
    [
        "Invoice Number",
        "Image Name",
        "Item Number",
        "Invoice Date",
        "Vendor Name",
        "Description",
        "Quantity",
        "Unit Price",
        "Net Worth",
        "VAT",
        "Gross Worth",
        "Total Amount",
        "IsAnomaly",
        "AnomalyReason",
        "Anomaly Score",
        "Validation Result",
        "Investigation Priority",
        "Duplicate Status",
        "Duplicate Reason"
    ]
]


# Save final CSV
final_csv_path = "reports/final_invoice_output.csv"

final_csv_df.to_csv(
    final_csv_path,
    index=False
)


# Display final output
print("\n================ FINAL CSV PREVIEW ================\n")

print(
    final_csv_df.head(20).to_string(
        index=False
    )
)


print("\n================ FINAL CSV SUMMARY ================\n")

print(
    "Total invoice items in final CSV:",
    len(final_csv_df)
)

print(
    "Unique invoices:",
    final_csv_df["Invoice Number"].nunique()
)

print(
    "Items with descriptions:",
    final_csv_df["Description"].notna().sum()
)

print(
    "Anomalous items:",
    final_csv_df["IsAnomaly"].sum()
)

print(
    "Potential duplicate invoice items:",
    (
        final_csv_df["Duplicate Status"] != "Unique"
    ).sum()
)

print(
    "Final CSV columns:",
    list(final_csv_df.columns)
)

print(
    "\nFinal CSV saved to:",
    final_csv_path
)

print(
    "\nFinal item-level output CSV created successfully."
)

#=============================
# Step 45D.1 : Missing Value Recovery
#=============================
print("\n========== Step 45D.1 : Missing Value Recovery ========== ")


# Recover financial values from validated financial data
recovery_financial_df = final_financial_df[
    [
        "Invoice Number",
        "Image Name",
        "Item Number",
        "Quantity",
        "Unit Price",
        "Net Worth",
        "VAT",
        "Gross Worth"
    ]
].drop_duplicates(
    subset=[
        "Invoice Number",
        "Image Name",
        "Item Number"
    ]
)


# Merge validated financial values
final_csv_df = final_csv_df.merge(
    recovery_financial_df,
    on=[
        "Invoice Number",
        "Image Name",
        "Item Number"
    ],
    how="left",
    suffixes=("", "_Recovery")
)


# Recover Unit Price
unit_price_missing = final_csv_df["Unit Price"].isna()

unit_price_recoverable = (
    unit_price_missing
    & final_csv_df["Net Worth_Recovery"].notna()
    & final_csv_df["Quantity_Recovery"].notna()
    & (final_csv_df["Quantity_Recovery"] != 0)
)

final_csv_df.loc[
    unit_price_recoverable,
    "Unit Price"
] = (
    final_csv_df.loc[
        unit_price_recoverable,
        "Net Worth_Recovery"
    ]
    /
    final_csv_df.loc[
        unit_price_recoverable,
        "Quantity_Recovery"
    ]
)


# Recover Net Worth
net_worth_missing = final_csv_df["Net Worth"].isna()

net_worth_recoverable = (
    net_worth_missing
    & final_csv_df["Quantity_Recovery"].notna()
    & final_csv_df["Unit Price"].notna()
)

final_csv_df.loc[
    net_worth_recoverable,
    "Net Worth"
] = (
    final_csv_df.loc[
        net_worth_recoverable,
        "Quantity_Recovery"
    ]
    *
    final_csv_df.loc[
        net_worth_recoverable,
        "Unit Price"
    ]
)


# Recover VAT
vat_missing = final_csv_df["VAT"].isna()

vat_recoverable = (
    vat_missing
    & final_csv_df["VAT_Recovery"].notna()
)

final_csv_df.loc[
    vat_recoverable,
    "VAT"
] = final_csv_df.loc[
    vat_recoverable,
    "VAT_Recovery"
]


# Recover Gross Worth
gross_worth_missing = final_csv_df["Gross Worth"].isna()

gross_worth_recoverable = (
    gross_worth_missing
    & final_csv_df["Net Worth"].notna()
    & final_csv_df["VAT"].notna()
)

final_csv_df.loc[
    gross_worth_recoverable,
    "Gross Worth"
] = (
    final_csv_df.loc[
        gross_worth_recoverable,
        "Net Worth"
    ]
    *
    (
        1
        +
        final_csv_df.loc[
            gross_worth_recoverable,
            "VAT"
        ]
        / 100
    )
)


# Recover invoice dates from original dataset
date_recovery_df = df[
    [
        "File Name",
        "OCRed Text"
    ]
].copy()


date_recovery_df["Recovered Invoice Date"] = (
    date_recovery_df["OCRed Text"]
    .str.extract(
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        expand=False
    )
)


date_recovery_df = date_recovery_df.rename(
    columns={
        "File Name": "Image Name"
    }
)


final_csv_df = final_csv_df.merge(
    date_recovery_df[
        [
            "Image Name",
            "Recovered Invoice Date"
        ]
    ],
    on="Image Name",
    how="left"
)


# Recover missing invoice dates
date_missing = final_csv_df["Invoice Date"].isna()

date_recoverable = (
    date_missing
    & final_csv_df["Recovered Invoice Date"].notna()
)

final_csv_df.loc[
    date_recoverable,
    "Invoice Date"
] = final_csv_df.loc[
    date_recoverable,
    "Recovered Invoice Date"
]


# Remove temporary recovery columns
final_csv_df = final_csv_df.drop(
    columns=[
        "Quantity_Recovery",
        "Unit Price_Recovery",
        "Net Worth_Recovery",
        "VAT_Recovery",
        "Gross Worth_Recovery",
        "Recovered Invoice Date"
    ],
    errors="ignore"
)


# Round recovered financial values
final_csv_df["Unit Price"] = (
    final_csv_df["Unit Price"].round(2)
)

final_csv_df["Net Worth"] = (
    final_csv_df["Net Worth"].round(2)
)

final_csv_df["VAT"] = (
    final_csv_df["VAT"].round(2)
)

final_csv_df["Gross Worth"] = (
    final_csv_df["Gross Worth"].round(2)
)


# Save recovered final CSV
final_csv_df.to_csv(
    final_csv_path,
    index=False
)


# Display recovery summary
print("\n================ RECOVERY SUMMARY ================\n")

print(
    "Unit Price recovered:",
    unit_price_recoverable.sum()
)

print(
    "Net Worth recovered:",
    net_worth_recoverable.sum()
)

print(
    "VAT recovered:",
    vat_recoverable.sum()
)

print(
    "Gross Worth recovered:",
    gross_worth_recoverable.sum()
)

print(
    "Invoice dates recovered:",
    date_recoverable.sum()
)

print(
    "Remaining Unit Price missing:",
    final_csv_df["Unit Price"].isna().sum()
)

print(
    "Remaining Net Worth missing:",
    final_csv_df["Net Worth"].isna().sum()
)

print(
    "Remaining VAT missing:",
    final_csv_df["VAT"].isna().sum()
)

print(
    "Remaining Gross Worth missing:",
    final_csv_df["Gross Worth"].isna().sum()
)

print(
    "Remaining Invoice Date missing:",
    final_csv_df["Invoice Date"].isna().sum()
)

print(
    "\nRecovered final CSV saved to:",
    final_csv_path
)

#=============================
# Step 45D.2 : Image-Based Financial Recovery
#=============================
print("\n========== Step 45D.2 : Image-Based Financial Recovery ========== ")


financial_recovery_records = []


for image_name in final_csv_df[
    final_csv_df["Unit Price"].isna()
]["Image Name"].unique():

    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    if not ocr_text:
        continue

    recovered_financial = extract_complete_financial_fields(
        ocr_text
    )

    for record in recovered_financial:

        financial_recovery_records.append(
            {
                "Image Name": image_name,
                "Item Number": record.get("Item Number"),
                "Unit Price Recovery": record.get("Unit Price"),
                "Net Worth Recovery": record.get("Net Worth"),
                "VAT Recovery": record.get("VAT"),
                "Gross Worth Recovery": record.get("Gross Worth")
            }
        )


financial_recovery_df = pd.DataFrame(
    financial_recovery_records
)


print(
    "\nFinancial records extracted during second OCR pass:",
    len(financial_recovery_df)
)


if not financial_recovery_df.empty:

    final_csv_df = final_csv_df.merge(
        financial_recovery_df,
        on=[
            "Image Name",
            "Item Number"
        ],
        how="left"
    )


    unit_price_mask = (
        final_csv_df["Unit Price"].isna()
        &
        final_csv_df["Unit Price Recovery"].notna()
    )

    final_csv_df.loc[
        unit_price_mask,
        "Unit Price"
    ] = final_csv_df.loc[
        unit_price_mask,
        "Unit Price Recovery"
    ]


    net_worth_mask = (
        final_csv_df["Net Worth"].isna()
        &
        final_csv_df["Net Worth Recovery"].notna()
    )

    final_csv_df.loc[
        net_worth_mask,
        "Net Worth"
    ] = final_csv_df.loc[
        net_worth_mask,
        "Net Worth Recovery"
    ]


    vat_mask = (
        final_csv_df["VAT"].isna()
        &
        final_csv_df["VAT Recovery"].notna()
    )

    final_csv_df.loc[
        vat_mask,
        "VAT"
    ] = final_csv_df.loc[
        vat_mask,
        "VAT Recovery"
    ]


    gross_worth_mask = (
        final_csv_df["Gross Worth"].isna()
        &
        final_csv_df["Gross Worth Recovery"].notna()
    )

    final_csv_df.loc[
        gross_worth_mask,
        "Gross Worth"
    ] = final_csv_df.loc[
        gross_worth_mask,
        "Gross Worth Recovery"
    ]


    final_csv_df = final_csv_df.drop(
        columns=[
            "Unit Price Recovery",
            "Net Worth Recovery",
            "VAT Recovery",
            "Gross Worth Recovery"
        ],
        errors="ignore"
    )


final_csv_df["Unit Price"] = (
    final_csv_df["Unit Price"].round(2)
)

final_csv_df["Net Worth"] = (
    final_csv_df["Net Worth"].round(2)
)

final_csv_df["VAT"] = (
    final_csv_df["VAT"].round(2)
)

final_csv_df["Gross Worth"] = (
    final_csv_df["Gross Worth"].round(2)
)


final_csv_df.to_csv(
    final_csv_path,
    index=False
)


print("\n================ IMAGE-BASED RECOVERY SUMMARY ================\n")

print(
    "Unit Price recovered:",
    unit_price_mask.sum()
    if not financial_recovery_df.empty
    else 0
)

print(
    "Net Worth recovered:",
    net_worth_mask.sum()
    if not financial_recovery_df.empty
    else 0
)

print(
    "VAT recovered:",
    vat_mask.sum()
    if not financial_recovery_df.empty
    else 0
)

print(
    "Gross Worth recovered:",
    gross_worth_mask.sum()
    if not financial_recovery_df.empty
    else 0
)

print(
    "\nRemaining Unit Price missing:",
    final_csv_df["Unit Price"].isna().sum()
)

print(
    "Remaining Net Worth missing:",
    final_csv_df["Net Worth"].isna().sum()
)

print(
    "Remaining VAT missing:",
    final_csv_df["VAT"].isna().sum()
)

print(
    "Remaining Gross Worth missing:",
    final_csv_df["Gross Worth"].isna().sum()
)

print(
    "\nImage-based financial recovery completed."
)


#=============================
# Step 45D.3 : Diagnose Financial Recovery Matching
#=============================
print("\n========== Step 45D.3 : Diagnose Financial Recovery Matching ========== ")


print("\n========== RECOVERY DATA TYPES ==========\n")

if not financial_recovery_df.empty:

    print(
        "Financial recovery columns:",
        financial_recovery_df.columns.tolist()
    )

    print(
        "\nFinancial recovery data types:"
    )

    print(
        financial_recovery_df[
            ["Image Name", "Item Number"]
        ].dtypes
    )

    print(
        "\nFirst 10 recovered records:"
    )

    print(
        financial_recovery_df[
            [
                "Image Name",
                "Item Number",
                "Unit Price Recovery",
                "Net Worth Recovery",
                "VAT Recovery",
                "Gross Worth Recovery"
            ]
        ].head(10).to_string(index=False)
    )


print("\n========== FINAL CSV KEY DATA TYPES ==========\n")

print(
    final_csv_df[
        ["Image Name", "Item Number"]
    ].dtypes
)

print(
    "\nFirst 10 final CSV records:"
)

print(
    final_csv_df[
        [
            "Image Name",
            "Item Number",
            "Unit Price",
            "Net Worth",
            "VAT",
            "Gross Worth"
        ]
    ].head(10).to_string(index=False)
)


if not financial_recovery_df.empty:

    recovery_keys = financial_recovery_df[
        [
            "Image Name",
            "Item Number"
        ]
    ].drop_duplicates()

    final_keys = final_csv_df[
        [
            "Image Name",
            "Item Number"
        ]
    ].drop_duplicates()


    matched_keys = recovery_keys.merge(
        final_keys,
        on=[
            "Image Name",
            "Item Number"
        ],
        how="inner"
    )


    print(
        "\nRecovery records with matching image + item:",
        len(matched_keys)
    )

    print(
        "Recovery records without matching image + item:",
        len(recovery_keys) - len(matched_keys)
    )


    print(
        "\n========== SAMPLE UNMATCHED RECOVERY RECORDS ==========\n"
    )


    unmatched_recovery = recovery_keys.merge(
        final_keys,
        on=[
            "Image Name",
            "Item Number"
        ],
        how="left",
        indicator=True
    )

    unmatched_recovery = unmatched_recovery[
        unmatched_recovery["_merge"] == "left_only"
    ].drop(
        columns=["_merge"]
    )


    print(
        unmatched_recovery.head(20).to_string(
            index=False
        )
    )


print(
    "\n========== STEP 45D.3 COMPLETED ==========\n"
)

#=============================
# Step 45D.4 : Apply Confirmed Image-Based Financial Recovery
#=============================
print("\n========== Step 45D.4 : Apply Confirmed Image-Based Financial Recovery ========== ")


recovery_merge_df = final_csv_df[
    [
        "Image Name",
        "Item Number"
    ]
].merge(
    financial_recovery_df,
    on=[
        "Image Name",
        "Item Number"
    ],
    how="left"
)


recovery_merge_df = recovery_merge_df.drop_duplicates(
    subset=[
        "Image Name",
        "Item Number"
    ]
)


final_csv_df = final_csv_df.merge(
    recovery_merge_df,
    on=[
        "Image Name",
        "Item Number"
    ],
    how="left",
    suffixes=(
        "",
        "_Recovered"
    )
)


unit_price_recovery_mask = (
    final_csv_df["Unit Price"].isna()
    &
    final_csv_df["Unit Price Recovery"].notna()
)

final_csv_df.loc[
    unit_price_recovery_mask,
    "Unit Price"
] = final_csv_df.loc[
    unit_price_recovery_mask,
    "Unit Price Recovery"
]


net_worth_recovery_mask = (
    final_csv_df["Net Worth"].isna()
    &
    final_csv_df["Net Worth Recovery"].notna()
)

final_csv_df.loc[
    net_worth_recovery_mask,
    "Net Worth"
] = final_csv_df.loc[
    net_worth_recovery_mask,
    "Net Worth Recovery"
]


vat_recovery_mask = (
    final_csv_df["VAT"].isna()
    &
    final_csv_df["VAT Recovery"].notna()
)

final_csv_df.loc[
    vat_recovery_mask,
    "VAT"
] = final_csv_df.loc[
    vat_recovery_mask,
    "VAT Recovery"
]


gross_worth_recovery_mask = (
    final_csv_df["Gross Worth"].isna()
    &
    final_csv_df["Gross Worth Recovery"].notna()
)

final_csv_df.loc[
    gross_worth_recovery_mask,
    "Gross Worth"
] = final_csv_df.loc[
    gross_worth_recovery_mask,
    "Gross Worth Recovery"
]


final_csv_df = final_csv_df.drop(
    columns=[
        "Unit Price Recovery",
        "Net Worth Recovery",
        "VAT Recovery",
        "Gross Worth Recovery"
    ],
    errors="ignore"
)


final_csv_df["Unit Price"] = (
    final_csv_df["Unit Price"].round(2)
)

final_csv_df["Net Worth"] = (
    final_csv_df["Net Worth"].round(2)
)

final_csv_df["VAT"] = (
    final_csv_df["VAT"].round(2)
)

final_csv_df["Gross Worth"] = (
    final_csv_df["Gross Worth"].round(2)
)


final_csv_df.to_csv(
    final_csv_path,
    index=False
)


print("\n========== RECOVERY RESULTS ==========\n")

print(
    "Confirmed matching recovery records:",
    len(matched_keys)
)

print(
    "Unit Price recovered:",
    unit_price_recovery_mask.sum()
)

print(
    "Net Worth recovered:",
    net_worth_recovery_mask.sum()
)

print(
    "VAT recovered:",
    vat_recovery_mask.sum()
)

print(
    "Gross Worth recovered:",
    gross_worth_recovery_mask.sum()
)

print(
    "\nRemaining Unit Price missing:",
    final_csv_df["Unit Price"].isna().sum()
)

print(
    "Remaining Net Worth missing:",
    final_csv_df["Net Worth"].isna().sum()
)

print(
    "Remaining VAT missing:",
    final_csv_df["VAT"].isna().sum()
)

print(
    "Remaining Gross Worth missing:",
    final_csv_df["Gross Worth"].isna().sum()
)

print(
    "\nConfirmed image-based financial recovery applied successfully."
)


#=============================
# Step 45D.5 : Identify Missing Financial Item Keys
#=============================
print("\n========== Step 45D.5 : Identify Missing Financial Item Keys ========== ")


missing_financial_items = final_csv_df[
    final_csv_df["Unit Price"].isna()
][
    [
        "Image Name",
        "Item Number",
        "Description",
        "Quantity"
    ]
].copy()


recovered_financial_keys = financial_recovery_df[
    [
        "Image Name",
        "Item Number"
    ]
].drop_duplicates()


missing_financial_keys = missing_financial_items[
    [
        "Image Name",
        "Item Number"
    ]
].drop_duplicates()


matched_missing_keys = missing_financial_keys.merge(
    recovered_financial_keys,
    on=[
        "Image Name",
        "Item Number"
    ],
    how="inner"
)


unmatched_missing_keys = missing_financial_keys.merge(
    recovered_financial_keys,
    on=[
        "Image Name",
        "Item Number"
    ],
    how="left",
    indicator=True
)


unmatched_missing_keys = unmatched_missing_keys[
    unmatched_missing_keys["_merge"] == "left_only"
].drop(
    columns=["_merge"]
)


print("\n========== MISSING FINANCIAL ITEMS ==========\n")

print(
    "Total missing financial item rows:",
    len(missing_financial_items)
)

print(
    "Unique missing invoice-item keys:",
    len(missing_financial_keys)
)

print(
    "Missing items found by OCR financial parser:",
    len(matched_missing_keys)
)

print(
    "Missing items NOT found by OCR financial parser:",
    len(unmatched_missing_keys)
)


print(
    "\n========== SAMPLE MISSING ITEMS ==========\n"
)

print(
    missing_financial_items.head(20).to_string(
        index=False
    )
)


print(
    "\n========== SAMPLE OCR-MATCHED MISSING ITEMS ==========\n"
)

if not matched_missing_keys.empty:

    print(
        matched_missing_keys.head(20).to_string(
            index=False
        )
    )

else:

    print(
        "No missing financial item was recovered by the current OCR parser."
    )


print(
    "\n========== SAMPLE UNMATCHED MISSING ITEMS ==========\n"
)

print(
    unmatched_missing_keys.head(30).to_string(
        index=False
    )
)


print(
    "\n========== STEP 45D.5 COMPLETED ==========\n"
)

#=============================
# Step 45D.6 : Inspect Missing Financial OCR Layout
#=============================
print("\n========== Step 45D.6 : Inspect Missing Financial OCR Layout ========== ")


missing_images = (
    missing_financial_items[
        "Image Name"
    ]
    .drop_duplicates()
    .head(5)
    .tolist()
)


for image_name in missing_images:

    print(
        "\n\n============================================================"
    )

    print(
        "IMAGE:",
        image_name
    )

    print(
        "============================================================\n"
    )


    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )


    if not ocr_text:

        print(
            "OCR text not available."
        )

        continue


    print(
        ocr_text
    )


print(
    "\n========== STEP 45D.6 COMPLETED ==========\n"
)

#=============================
# Step 45D.7 : Analyze Financial Column Alignment
#=============================
print("\n========== Step 45D.7 : Analyze Financial Column Alignment ========== ")


analysis_records = []


for image_name in missing_financial_items[
    "Image Name"
].drop_duplicates().head(20):


    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )


    if not ocr_text:
        continue


    invoice_items = all_item_df[
        all_item_df["Image Name"] == image_name
    ]


    financial_records = extract_complete_financial_fields(
        ocr_text
    )


    analysis_records.append(
        {
            "Image Name": image_name,
            "Invoice Item Count": len(invoice_items),
            "Financial Record Count": len(financial_records)
        }
    )


financial_alignment_df = pd.DataFrame(
    analysis_records
)


print(
    "\n========== FINANCIAL ALIGNMENT RESULTS ==========\n"
)


if not financial_alignment_df.empty:

    print(
        financial_alignment_df.to_string(
            index=False
        )
    )


    financial_alignment_df["Counts Match"] = (
        financial_alignment_df["Invoice Item Count"]
        ==
        financial_alignment_df["Financial Record Count"]
    )


    print(
        "\nNumber of invoices where item count matches financial record count:",
        financial_alignment_df["Counts Match"].sum()
    )

    print(
        "Number of invoices where counts do not match:",
        (~financial_alignment_df["Counts Match"]).sum()
    )


print(
    "\n========== STEP 45D.7 COMPLETED ==========\n"
)

#=============================
# Step 45D.8 : Test Financial Column Extraction
#=============================
print("\n========== Step 45D.8 : Test Financial Column Extraction ========== ")


def extract_financial_columns_from_ocr(ocr_text):

    net_price_values = []
    net_worth_values = []
    vat_values = []
    gross_worth_values = []

    lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if line.strip()
    ]


    net_price_start = None
    net_worth_start = None
    gross_worth_start = None


    for index, line in enumerate(lines):

        line_lower = line.lower()


        if (
            "net price" in line_lower
            and "net worth" in line_lower
        ):

            net_price_start = index


        elif (
            line_lower == "net worth"
            and net_worth_start is None
        ):

            net_worth_start = index


        elif (
            line_lower == "gross worth"
            and gross_worth_start is None
        ):

            gross_worth_start = index


    if net_price_start is None:

        return {
            "Net Price": [],
            "Net Worth": [],
            "VAT": [],
            "Gross Worth": []
        }


    financial_section = lines[
        net_price_start:
    ]


    number_pattern = re.compile(
        r"^\$?\s*\d[\d\s,]*\.\d{2}$"
    )


    financial_numbers = []


    for line in financial_section:

        cleaned_line = (
            line
            .replace("$", "")
            .strip()
        )


        if number_pattern.match(cleaned_line):

            financial_numbers.append(
                cleaned_line
            )


    if net_worth_start is not None:

        net_worth_section = lines[
            net_worth_start:
        ]

        for line in net_worth_section:

            cleaned_line = (
                line
                .replace("$", "")
                .strip()
            )


            if number_pattern.match(cleaned_line):

                value = (
                    cleaned_line
                    .replace(" ", "")
                    .replace(",", ".")
                )


                try:

                    net_worth_values.append(
                        float(value)
                    )

                except ValueError:

                    pass


    print(
        "\nDetected financial section:",
        len(financial_section),
        "OCR lines"
    )

    print(
        "Detected financial numbers:",
        len(financial_numbers)
    )

    print(
        "Detected Net Worth values:",
        len(net_worth_values)
    )


    return {
        "Net Price": net_price_values,
        "Net Worth": net_worth_values,
        "VAT": vat_values,
        "Gross Worth": gross_worth_values
    }


test_financial_images = [
    "batch1-0006.jpg",
    "batch1-0012.jpg",
    "batch1-0015.jpg",
    "batch1-0018.jpg",
    "batch1-0038.jpg"
]


for image_name in test_financial_images:

    print(
        "\n============================================================"
    )

    print(
        "IMAGE:",
        image_name
    )

    print(
        "============================================================"
    )


    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )


    financial_test_result = (
        extract_financial_columns_from_ocr(
            ocr_text
        )
    )


    print(
        "Net Worth values:",
        financial_test_result["Net Worth"]
    )


print(
    "\n========== STEP 45D.8 COMPLETED ==========\n"
)

#=============================
# Step 45D.8 : Corrected Financial Number Detection
#=============================
print("\n========== Step 45D.8 : Corrected Financial Number Detection ========== ")


def extract_financial_numbers_from_ocr(ocr_text):

    lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if line.strip()
    ]


    number_pattern = re.compile(
        r"^\$?\s*\d[\d\s]*[,.]\d{2}$"
    )


    detected_numbers = []


    for line in lines:

        cleaned_line = (
            line
            .replace("$", "")
            .strip()
        )


        if number_pattern.match(cleaned_line):

            detected_numbers.append(
                cleaned_line
            )


    return detected_numbers


test_financial_images = [
    "batch1-0006.jpg",
    "batch1-0012.jpg",
    "batch1-0015.jpg",
    "batch1-0018.jpg",
    "batch1-0038.jpg"
]


for image_name in test_financial_images:

    print(
        "\n============================================================"
    )

    print(
        "IMAGE:",
        image_name
    )

    print(
        "============================================================"
    )


    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )


    detected_numbers = (
        extract_financial_numbers_from_ocr(
            ocr_text
        )
    )


    print(
        "\nTotal decimal financial numbers detected:",
        len(detected_numbers)
    )


    print(
        "\nDetected values:"
    )


    print(
        detected_numbers
    )


print(
    "\n========== STEP 45D.8 COMPLETED ==========\n"
)

#=============================
# Step 45D.9 : Detect Financial Table Sections
#=============================
print("\n========== Step 45D.9 : Detect Financial Table Sections ========== ")


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

            section_positions["net_price"] = index


        elif (
            line_lower == "net worth"
            and section_positions["net_worth"] is None
        ):

            section_positions["net_worth"] = index


        elif (
            line_lower == "gross worth"
            and section_positions["gross_worth"] is None
        ):

            section_positions["gross_worth"] = index


    sections = {}


    if section_positions["net_price"] is not None:

        start = section_positions["net_price"]


        if section_positions["net_worth"] is not None:

            end = section_positions["net_worth"]

        else:

            end = len(lines)


        sections["net_price"] = lines[
            start:end
        ]


    if section_positions["net_worth"] is not None:

        start = section_positions["net_worth"]


        if section_positions["gross_worth"] is not None:

            end = section_positions["gross_worth"]

        else:

            end = len(lines)


        sections["net_worth"] = lines[
            start:end
        ]


    if section_positions["gross_worth"] is not None:

        start = section_positions["gross_worth"]

        sections["gross_worth"] = lines[
            start:
        ]


    return section_positions, sections


test_financial_images = [
    "batch1-0006.jpg",
    "batch1-0012.jpg",
    "batch1-0015.jpg",
    "batch1-0018.jpg",
    "batch1-0038.jpg"
]


for image_name in test_financial_images:

    print(
        "\n============================================================"
    )

    print(
        "IMAGE:",
        image_name
    )

    print(
        "============================================================"
    )


    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )


    section_positions, sections = (
        extract_financial_sections(
            ocr_text
        )
    )


    print(
        "\nSection positions:"
    )

    print(
        section_positions
    )


    for section_name, section_lines in sections.items():

        print(
            "\n---",
            section_name.upper(),
            "---"
        )


        print(
            section_lines
        )


print(
    "\n========== STEP 45D.9 COMPLETED ==========\n"
)

#=============================
# Step 45D.10 : Extract and Align Financial Table Values
#=============================
print("\n========== Step 45D.10 : EXTRACT AND ALIGN FINANCIAL TABLE VALUES ========== ")

def extract_financial_table_values(ocr_text):
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    net_price_values = []
    net_worth_values = []
    gross_worth_values = []
    vat_values = []

    financial_start = False
    gross_start = False

    for line in lines:

        lower_line = line.lower()

        if "net price" in lower_line and "net worth" in lower_line:
            financial_start = True
            continue

        if "net worth" in lower_line and "vat" in lower_line:
            financial_start = True
            continue

        if lower_line == "gross worth":
            gross_start = True
            financial_start = False
            continue

        if financial_start and not gross_start:

            parts = line.replace("$", "").split()

            if len(parts) >= 2 and "%" in line:
                numeric_parts = []

                for part in parts:
                    if "%" in part:
                        vat_text = part.replace("%", "").replace(",", ".")
                        try:
                            vat_values.append(float(vat_text))
                        except ValueError:
                            pass
                    else:
                        value_text = part.replace(",", ".").replace(" ", "")
                        try:
                            numeric_parts.append(float(value_text))
                        except ValueError:
                            pass

                if len(numeric_parts) >= 2:
                    net_price_values.append(numeric_parts[0])
                    net_worth_values.append(numeric_parts[1])

                elif len(numeric_parts) == 1:
                    net_worth_values.append(numeric_parts[0])

        elif gross_start:

            if lower_line == "vat":
                continue

            if "gross worth" in lower_line:
                continue

            value_text = line.replace("$", "").replace(",", ".").replace(" ", "")

            try:
                gross_value = float(value_text)

                if gross_value > 0:
                    gross_worth_values.append(gross_value)

            except ValueError:
                continue

    return {
        "Net Price": net_price_values,
        "Net Worth": net_worth_values,
        "VAT": vat_values,
        "Gross Worth": gross_worth_values
    }


financial_alignment_records = []

for image_name in all_invoice_images:

    ocr_text = all_ocr_results.get(image_name, "")

    if not ocr_text:
        continue

    financial_values = extract_financial_table_values(ocr_text)

    image_items = all_item_df[
        all_item_df["Image Name"] == image_name
    ].copy()

    if image_items.empty:
        continue

    image_items = image_items.sort_values("Item Number")

    item_count = len(image_items)

    net_price_values = financial_values["Net Price"]
    net_worth_values = financial_values["Net Worth"]
    vat_values = financial_values["VAT"]
    gross_worth_values = financial_values["Gross Worth"]

    for position in range(item_count):

        record = {
            "Image Name": image_name,
            "Item Number": image_items.iloc[position]["Item Number"],
            "Unit Price": None,
            "Net Worth": None,
            "VAT": None,
            "Gross Worth": None
        }

        if position < len(net_price_values):
            record["Unit Price"] = net_price_values[position]

        if position < len(net_worth_values):
            record["Net Worth"] = net_worth_values[position]

        if position < len(vat_values):
            record["VAT"] = vat_values[position]

        if position < len(gross_worth_values):
            record["Gross Worth"] = gross_worth_values[position]

        financial_alignment_records.append(record)


financial_alignment_df = pd.DataFrame(financial_alignment_records)

print("\nFinancial alignment records:", len(financial_alignment_df))

print("\nFinancial alignment sample:")

if not financial_alignment_df.empty:
    print(
        financial_alignment_df.head(20).to_string(index=False)
    )

print("\n========== FINANCIAL ALIGNMENT SUMMARY ==========")

if not financial_alignment_df.empty:

    print(
        "Images processed:",
        financial_alignment_df["Image Name"].nunique()
    )

    print(
        "Financial item records:",
        len(financial_alignment_df)
    )

    print(
        "Unit Price extracted:",
        financial_alignment_df["Unit Price"].notna().sum()
    )

    print(
        "Net Worth extracted:",
        financial_alignment_df["Net Worth"].notna().sum()
    )

    print(
        "VAT extracted:",
        financial_alignment_df["VAT"].notna().sum()
    )

    print(
        "Gross Worth extracted:",
        financial_alignment_df["Gross Worth"].notna().sum()
    )

print("\n========== STEP 45D.10 COMPLETED ========== ")

#=============================
# Step 45D.11 : Build Safer Financial Table Parser
#=============================
print("\n========== Step 45D.11 : BUILD SAFER FINANCIAL TABLE PARSER ========== ")

def extract_financial_table_rows(ocr_text):

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    combined_rows = []
    separated_net_rows = []
    gross_rows = []

    mode = None

    for line in lines:

        lower_line = line.lower()

        if "net price" in lower_line and "net worth" in lower_line:
            mode = "combined"
            continue

        if "net worth" in lower_line and "vat" in lower_line:
            mode = "separated_net"
            continue

        if lower_line == "gross worth":
            mode = "gross"
            continue

        if lower_line == "vat":
            continue

        if lower_line == "gross":
            continue

        if mode == "combined":

            match = re.match(
                r"^\$?\s*([\d\s]+[,.]\d{2})\s+([\d\s]+[,.]\d{2})\s+(\d+(?:[,.]\d+)?)%$",
                line
            )

            if match:

                unit_price = convert_ocr_number(match.group(1))
                net_worth = convert_ocr_number(match.group(2))
                vat = convert_ocr_number(match.group(3))

                combined_rows.append(
                    {
                        "Unit Price": unit_price,
                        "Net Worth": net_worth,
                        "VAT": vat
                    }
                )

        elif mode == "separated_net":

            match = re.match(
                r"^\$?\s*([\d\s]+[,.]\d{2})\s+(\d+(?:[,.]\d+)?)%$",
                line
            )

            if match:

                net_worth = convert_ocr_number(match.group(1))
                vat = convert_ocr_number(match.group(2))

                separated_net_rows.append(
                    {
                        "Net Worth": net_worth,
                        "VAT": vat
                    }
                )

        elif mode == "gross":

            match = re.match(
                r"^\$?\s*([\d\s]+[,.]\d{2})$",
                line
            )

            if match:

                gross_value = convert_ocr_number(match.group(1))

                gross_rows.append(gross_value)


    return {
        "combined_rows": combined_rows,
        "separated_net_rows": separated_net_rows,
        "gross_rows": gross_rows
    }


print("\nTesting safer financial parser...")

test_financial_images = [
    "batch1-0006.jpg",
    "batch1-0012.jpg",
    "batch1-0015.jpg",
    "batch1-0018.jpg",
    "batch1-0038.jpg"
]

for image_name in test_financial_images:

    ocr_text = all_ocr_results.get(image_name, "")

    financial_rows = extract_financial_table_rows(ocr_text)

    print("\n" + "=" * 60)
    print("IMAGE:", image_name)
    print("=" * 60)

    print("\nCombined rows:")
    print(financial_rows["combined_rows"])

    print("\nSeparated Net Worth rows:")
    print(financial_rows["separated_net_rows"])

    print("\nGross Worth rows:")
    print(financial_rows["gross_rows"])


print("\n========== STEP 45D.11 COMPLETED ========== ")

#=============================
# Step 45D.12 : Extract Item-Level Gross Worth Values
#=============================
print("\n========== Step 45D.12 : EXTRACT ITEM-LEVEL GROSS WORTH VALUES ========== ")

def extract_item_gross_worth_values(ocr_text):

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    gross_start_positions = []

    for index, line in enumerate(lines):

        if line.lower() == "gross worth":
            gross_start_positions.append(index)

    gross_values = []

    if not gross_start_positions:
        return gross_values

    first_gross_position = gross_start_positions[0]

    second_gross_position = None

    if len(gross_start_positions) > 1:
        second_gross_position = gross_start_positions[1]

    if second_gross_position is not None:

        start_position = first_gross_position + 1
        end_position = second_gross_position

        candidate_lines = lines[start_position:end_position]

    else:

        start_position = first_gross_position + 1
        candidate_lines = lines[start_position:]

    for line in candidate_lines:

        value_text = line.replace("$", "").strip()

        value_text = value_text.replace(" ", "")

        if not re.match(
            r"^\d+[,.]\d{2}$",
            value_text
        ):
            continue

        try:

            value = convert_ocr_number(value_text)

            if value > 0:
                gross_values.append(value)

        except ValueError:
            continue

    return gross_values


test_gross_images = [
    "batch1-0006.jpg",
    "batch1-0012.jpg",
    "batch1-0015.jpg",
    "batch1-0018.jpg",
    "batch1-0038.jpg"
]

for image_name in test_gross_images:

    ocr_text = all_ocr_results.get(image_name, "")

    gross_values = extract_item_gross_worth_values(ocr_text)

    print("\n" + "=" * 60)
    print("IMAGE:", image_name)
    print("=" * 60)

    print("Item-level Gross Worth values:")
    print(gross_values)

    print("Number of Gross Worth values:", len(gross_values))


print("\n========== STEP 45D.12 COMPLETED ========== ")

#=============================
# Step 45D.13 : Detect Split Gross Worth Section
#=============================
print("\n========== Step 45D.13 : DETECT SPLIT GROSS WORTH SECTION ========== ")

def extract_item_gross_worth_values_v2(ocr_text):

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    gross_section_positions = []

    index = 0

    while index < len(lines):

        current_line = lines[index].lower()

        if current_line == "gross worth":
            gross_section_positions.append(index)

        elif (
            current_line == "gross"
            and index + 1 < len(lines)
            and lines[index + 1].lower() == "worth"
        ):
            gross_section_positions.append(index)

        index += 1

    if not gross_section_positions:
        return []

    if len(gross_section_positions) >= 2:

        first_position = gross_section_positions[0]
        second_position = gross_section_positions[1]

        start_position = first_position + 1

        if lines[first_position].lower() == "gross":
            start_position += 1

        end_position = second_position

    else:

        first_position = gross_section_positions[0]

        start_position = first_position + 1

        if lines[first_position].lower() == "gross":
            start_position += 1

        end_position = len(lines)

    gross_values = []

    for line in lines[start_position:end_position]:

        value_text = line.replace("$", "").replace(" ", "")

        if not re.match(
            r"^\d+[,.]\d{2}$",
            value_text
        ):
            continue

        try:

            value = convert_ocr_number(value_text)

            if value > 0:
                gross_values.append(value)

        except ValueError:
            continue

    return gross_values


test_gross_images_v2 = [
    "batch1-0006.jpg",
    "batch1-0012.jpg",
    "batch1-0015.jpg",
    "batch1-0018.jpg",
    "batch1-0038.jpg"
]

for image_name in test_gross_images_v2:

    ocr_text = all_ocr_results.get(image_name, "")

    gross_values = extract_item_gross_worth_values_v2(ocr_text)

    print("\n" + "=" * 60)
    print("IMAGE:", image_name)
    print("=" * 60)

    print("Item-level Gross Worth values:")
    print(gross_values)

    print("Number of Gross Worth values:", len(gross_values))


print("\n========== STEP 45D.13 COMPLETED ========== ")


#=============================
# Step 45D.14 : Combine All Financial Values
#=============================
print("\n========== Step 45D.14 : COMBINE ALL FINANCIAL VALUES ========== ")

financial_combined_records = []

for image_name in all_invoice_images:

    ocr_text = all_ocr_results.get(image_name, "")

    if not ocr_text:
        continue

    image_items = all_item_df[
        all_item_df["Image Name"] == image_name
    ].copy()

    if image_items.empty:
        continue

    image_items = image_items.sort_values("Item Number")

    financial_rows = extract_financial_table_rows(ocr_text)

    combined_rows = financial_rows["combined_rows"]

    separated_net_rows = financial_rows["separated_net_rows"]

    gross_values = extract_item_gross_worth_values_v2(ocr_text)

    item_count = len(image_items)

    for position in range(item_count):

        item_number = image_items.iloc[position]["Item Number"]

        unit_price = None
        net_worth = None
        vat = None

        if position < len(combined_rows):

            unit_price = combined_rows[position]["Unit Price"]
            net_worth = combined_rows[position]["Net Worth"]
            vat = combined_rows[position]["VAT"]

        elif position < len(separated_net_rows):

            net_worth = separated_net_rows[position]["Net Worth"]
            vat = separated_net_rows[position]["VAT"]

        gross_worth = None

        if position < len(gross_values):
            gross_worth = gross_values[position]

        financial_combined_records.append(
            {
                "Image Name": image_name,
                "Item Number": item_number,
                "Unit Price": unit_price,
                "Net Worth": net_worth,
                "VAT": vat,
                "Gross Worth": gross_worth
            }
        )


financial_combined_df = pd.DataFrame(
    financial_combined_records
)

print("\nFinancial combined records:", len(financial_combined_df))

print("\nFinancial combined sample:")

if not financial_combined_df.empty:
    print(
        financial_combined_df.head(30).to_string(index=False)
    )

print("\n========== FINANCIAL EXTRACTION SUMMARY ==========")

if not financial_combined_df.empty:

    total_records = len(financial_combined_df)

    unit_price_count = financial_combined_df["Unit Price"].notna().sum()
    net_worth_count = financial_combined_df["Net Worth"].notna().sum()
    vat_count = financial_combined_df["VAT"].notna().sum()
    gross_worth_count = financial_combined_df["Gross Worth"].notna().sum()

    print(
        "Total item records:",
        total_records
    )

    print(
        "Unit Price extracted:",
        unit_price_count,
        f"({unit_price_count / total_records * 100:.2f}%)"
    )

    print(
        "Net Worth extracted:",
        net_worth_count,
        f"({net_worth_count / total_records * 100:.2f}%)"
    )

    print(
        "VAT extracted:",
        vat_count,
        f"({vat_count / total_records * 100:.2f}%)"
    )

    print(
        "Gross Worth extracted:",
        gross_worth_count,
        f"({gross_worth_count / total_records * 100:.2f}%)"
    )


print("\n========== STEP 45D.14 COMPLETED ========== ")

#=============================
# Step 45D.15 : Diagnose Remaining Financial Extraction
#=============================
print("\n========== Step 45D.15 : DIAGNOSE REMAINING FINANCIAL EXTRACTION ========== ")

diagnostic_df = financial_combined_df.copy()

diagnostic_df["Available Fields"] = diagnostic_df[
    ["Unit Price", "Net Worth", "VAT", "Gross Worth"]
].notna().sum(axis=1)


def classify_financial_record(row):

    available_fields = row["Available Fields"]

    if available_fields == 4:
        return "Complete"

    elif available_fields == 3:
        return "Three Fields"

    elif available_fields == 2:
        return "Two Fields"

    elif available_fields == 1:
        return "One Field"

    else:
        return "No Financial Fields"


diagnostic_df["Financial Status"] = diagnostic_df.apply(
    classify_financial_record,
    axis=1
)


print("\n========== FINANCIAL EXTRACTION STATUS ==========")

status_counts = diagnostic_df[
    "Financial Status"
].value_counts()

print(status_counts.to_string())


print("\n========== FIELD AVAILABILITY ==========")

print(
    diagnostic_df[
        ["Unit Price", "Net Worth", "VAT", "Gross Worth"]
    ].notna().sum()
)


print("\n========== SAMPLE PARTIAL RECORDS ==========")

partial_records = diagnostic_df[
    diagnostic_df["Financial Status"] != "Complete"
].copy()

print(
    partial_records.head(20).to_string(index=False)
)


print("\n========== REPRESENTATIVE IMAGES ==========")

representative_images = []

for status in [
    "No Financial Fields",
    "One Field",
    "Two Fields",
    "Three Fields"
]:

    status_images = diagnostic_df[
        diagnostic_df["Financial Status"] == status
    ]["Image Name"].drop_duplicates().tolist()

    if status_images:
        representative_images.append(
            (status, status_images[0])
        )


for status, image_name in representative_images:

    print("\n" + "=" * 60)
    print("STATUS:", status)
    print("IMAGE:", image_name)
    print("=" * 60)

    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    print(ocr_text[:5000])


print("\n========== STEP 45D.15 COMPLETED ========== ")

#=============================
# Step 45D.16 : Parse Complete Single-Line Invoice Items
#=============================
print("\n========== Step 45D.16 : PARSE COMPLETE SINGLE-LINE INVOICE ITEMS ========== ")

def extract_single_line_financial_rows(ocr_text):

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    records = []

    for line in lines:

        if "each" not in line.lower():
            continue

        match = re.search(
            r"([\d\s]+[,.]\d{2})\s+each\s+"
            r"([\d\s]+[,.]\d{2})\s+"
            r"([\d\s]+[,.]\d{2})\s+"
            r"(\d+(?:[,.]\d+)?)%\s+"
            r"([\d\s]+[,.]\d{2})",
            line,
            re.IGNORECASE
        )

        if not match:
            continue

        quantity = convert_ocr_number(match.group(1))
        unit_price = convert_ocr_number(match.group(2))
        net_worth = convert_ocr_number(match.group(3))
        vat = convert_ocr_number(match.group(4))
        gross_worth = convert_ocr_number(match.group(5))

        records.append(
            {
                "Quantity": quantity,
                "Unit Price": unit_price,
                "Net Worth": net_worth,
                "VAT": vat,
                "Gross Worth": gross_worth
            }
        )

    return records


test_single_line_images = [
    "batch1-0002.jpg",
    "batch1-0027.jpg",
    "batch1-0061.jpg",
    "batch1-0074.jpg",
    "batch1-0082.jpg"
]

for image_name in test_single_line_images:

    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    single_line_rows = extract_single_line_financial_rows(
        ocr_text
    )

    print("\n" + "=" * 60)
    print("IMAGE:", image_name)
    print("=" * 60)

    print("Extracted financial rows:")

    for row in single_line_rows:
        print(row)

    print(
        "Number of extracted rows:",
        len(single_line_rows)
    )


print("\n========== STEP 45D.16 COMPLETED ========== ")

#=============================
# Step 45D.17 : Diagnose Single-Line Financial Layouts
#=============================
print("\n========== Step 45D.17 : DIAGNOSE SINGLE-LINE FINANCIAL LAYOUTS ========== ")

diagnostic_images = [
    "batch1-0002.jpg",
    "batch1-0027.jpg",
    "batch1-0061.jpg",
    "batch1-0074.jpg",
    "batch1-0082.jpg"
]

for image_name in diagnostic_images:

    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if line.strip()
    ]

    print("\n" + "=" * 80)
    print("IMAGE:", image_name)
    print("=" * 80)

    print("\n========== LINES CONTAINING 'EACH' ==========")

    each_lines_found = False

    for line_number, line in enumerate(lines, start=1):

        if "each" in line.lower():

            each_lines_found = True

            print(
                f"Line {line_number}: {line}"
            )

    if not each_lines_found:
        print("No lines containing 'each' found.")

    print("\n========== FINANCIAL-LOOKING LINES ==========")

    financial_lines_found = False

    for line_number, line in enumerate(lines, start=1):

        decimal_count = len(
            re.findall(
                r"\d[\d\s]*[,.]\d{2}",
                line
            )
        )

        percentage_found = re.search(
            r"\d+(?:[,.]\d+)?%",
            line
        )

        if decimal_count >= 2 or percentage_found:

            financial_lines_found = True

            print(
                f"Line {line_number}: {line}"
            )

    if not financial_lines_found:
        print("No financial-looking lines found.")


print("\n========== STEP 45D.17 COMPLETED ========== ")


#=============================
# Step 45D.18 : Build Layout-Aware Financial Parser
#=============================
print("\n========== Step 45D.18 : BUILD LAYOUT-AWARE FINANCIAL PARSER ========== ")

def extract_layout_aware_financial_rows(ocr_text):

    lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if line.strip()
    ]

    records = []

    # Detect complete single-line financial rows
    single_line_pattern = re.compile(
        r"^\s*\d+\.\s+.*?"
        r"([\d\s]+[,.]\d{2})\s+each\s+"
        r"([\d\s]+[,.]\d{2})\s+"
        r"([\d\s]+[,.]\d{2})\s+"
        r"(\d+(?:[,.]\d+)?)%\s+"
        r"([\d\s]+[,.]\d{2})\s*$",
        re.IGNORECASE
    )

    for line in lines:

        match = single_line_pattern.search(line)

        if match:

            quantity = convert_ocr_number(
                match.group(1)
            )

            unit_price = convert_ocr_number(
                match.group(2)
            )

            net_worth = convert_ocr_number(
                match.group(3)
            )

            vat = convert_ocr_number(
                match.group(4)
            )

            gross_worth = convert_ocr_number(
                match.group(5)
            )

            records.append(
                {
                    "Quantity": quantity,
                    "Unit Price": unit_price,
                    "Net Worth": net_worth,
                    "VAT": vat,
                    "Gross Worth": gross_worth,
                    "Extraction Layout": "Single-Line"
                }
            )

    return records


layout_test_images = [
    "batch1-0002.jpg",
    "batch1-0027.jpg",
    "batch1-0061.jpg",
    "batch1-0074.jpg",
    "batch1-0082.jpg"
]

for image_name in layout_test_images:

    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    layout_rows = extract_layout_aware_financial_rows(
        ocr_text
    )

    print("\n" + "=" * 70)
    print("IMAGE:", image_name)
    print("=" * 70)

    for row in layout_rows:
        print(row)

    print(
        "Extracted rows:",
        len(layout_rows)
    )


print("\n========== STEP 45D.18 COMPLETED ========== ")

#=============================
# Step 45D.19 : Extract Single-Line Financial Values
#=============================
print("\n========== Step 45D.19 : EXTRACT SINGLE-LINE FINANCIAL VALUES ========== ")

def extract_single_line_financial_values(ocr_text):

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

        match = financial_pattern.search(line)

        if not match:
            continue

        unit_price = convert_ocr_number(
            match.group(1)
        )

        net_worth = convert_ocr_number(
            match.group(2)
        )

        vat = convert_ocr_number(
            match.group(3)
        )

        gross_worth = convert_ocr_number(
            match.group(4)
        )

        records.append(
            {
                "Unit Price": unit_price,
                "Net Worth": net_worth,
                "VAT": vat,
                "Gross Worth": gross_worth,
                "Extraction Layout": "Single-Line"
            }
        )

    return records


for image_name in layout_test_images:

    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    financial_rows = extract_single_line_financial_values(
        ocr_text
    )

    print("\n" + "=" * 70)
    print("IMAGE:", image_name)
    print("=" * 70)

    for row in financial_rows:
        print(row)

    print(
        "Extracted financial rows:",
        len(financial_rows)
    )


print("\n========== STEP 45D.19 COMPLETED ========== ")


#=============================
# Step 45D.20 : Extract Separated Net Price and Net Worth Tables
#=============================
print("\n========== Step 45D.20 : EXTRACT SEPARATED NET PRICE AND NET WORTH TABLES ========== ")

def extract_separated_net_financial_rows(ocr_text):

    lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if line.strip()
    ]

    records = []

    combined_header_found = False

    for line in lines:

        lower_line = line.lower()

        if "net price" in lower_line and "net worth" in lower_line:
            combined_header_found = True
            continue

        if lower_line == "net worth":
            continue

        if lower_line == "net price":
            continue

        if lower_line == "vat":
            continue

        if lower_line == "gross worth":
            continue

        match = re.match(
            r"^\$?\s*"
            r"([\d\s]+[,.]\d{2})\s+"
            r"([\d\s]+[,.]\d{2})\s+"
            r"(\d+(?:[,.]\d+)?)%"
            r"\s*$",
            line,
            re.IGNORECASE
        )

        if match:

            unit_price = convert_ocr_number(
                match.group(1)
            )

            net_worth = convert_ocr_number(
                match.group(2)
            )

            vat = convert_ocr_number(
                match.group(3)
            )

            records.append(
                {
                    "Unit Price": unit_price,
                    "Net Worth": net_worth,
                    "VAT": vat,
                    "Extraction Layout": "Separated-Net"
                }
            )

    return records


separated_net_test_images = [
    "batch1-0027.jpg",
    "batch1-0061.jpg",
    "batch1-0074.jpg",
    "batch1-0082.jpg"
]

for image_name in separated_net_test_images:

    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    separated_rows = extract_separated_net_financial_rows(
        ocr_text
    )

    print("\n" + "=" * 70)
    print("IMAGE:", image_name)
    print("=" * 70)

    for row in separated_rows:
        print(row)

    print(
        "Extracted separated financial rows:",
        len(separated_rows)
    )


print("\n========== STEP 45D.20 COMPLETED ========== ")


#=============================
# Step 45D.21 : Diagnose Special Financial Layout
#=============================
print("\n========== Step 45D.21 : DIAGNOSE SPECIAL FINANCIAL LAYOUT ========== ")

image_name = "batch1-0074.jpg"

ocr_text = all_ocr_results.get(
    image_name,
    ""
)

lines = [
    line.strip()
    for line in ocr_text.splitlines()
    if line.strip()
]

print("\n" + "=" * 80)
print("IMAGE:", image_name)
print("=" * 80)

print("\n========== COMPLETE OCR STRUCTURE ==========")

for line_number, line in enumerate(lines, start=1):

    print(
        f"{line_number:03d}: {line}"
    )

print("\n========== FINANCIAL SECTION MARKERS ==========")

for line_number, line in enumerate(lines, start=1):

    lower_line = line.lower()

    if (
        "net price" in lower_line
        or "net worth" in lower_line
        or "gross worth" in lower_line
        or lower_line == "gross"
        or "vat" in lower_line
        or "%" in line
    ):

        print(
            f"{line_number:03d}: {line}"
        )


print("\n========== STEP 45D.21 COMPLETED ========== ")


#=============================
# Step 45D.22 : Parse Special Separated Financial Layout
#=============================
print("\n========== Step 45D.22 : PARSE SPECIAL SEPARATED FINANCIAL LAYOUT ========== ")

def extract_special_separated_financial_rows(ocr_text):

    lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if line.strip()
    ]

    net_price_values = []
    net_worth_values = []
    vat_values = []
    gross_worth_values = []

    net_price_start = None
    net_worth_start = None
    gross_worth_start = None

    for index, line in enumerate(lines):

        lower_line = line.lower()

        if lower_line == "net price":
            net_price_start = index

        elif lower_line == "net worth":
            if net_worth_start is None:
                net_worth_start = index

        elif lower_line == "gross worth":
            if gross_worth_start is None:
                gross_worth_start = index

        elif lower_line == "gross":
            if (
                index + 1 < len(lines)
                and lines[index + 1].lower() == "worth"
            ):
                if gross_worth_start is None:
                    gross_worth_start = index

    if net_price_start is not None:

        end_position = (
            net_worth_start
            if net_worth_start is not None
            else len(lines)
        )

        for line in lines[
            net_price_start + 1 : end_position
        ]:

            if re.fullmatch(
                r"\$?\s*[\d\s]+[,.]\d{2}",
                line
            ):

                net_price_values.append(
                    convert_ocr_number(line)
                )

    if net_worth_start is not None:

        end_position = (
            gross_worth_start
            if gross_worth_start is not None
            else len(lines)
        )

        for line in lines[
            net_worth_start + 1 : end_position
        ]:

            match = re.fullmatch(
                r"\$?\s*([\d\s]+[,.]\d{2})\s+"
                r"(\d+(?:[,.]\d+)?)%",
                line
            )

            if match:

                net_worth_values.append(
                    convert_ocr_number(
                        match.group(1)
                    )
                )

                vat_values.append(
                    convert_ocr_number(
                        match.group(2)
                    )
                )

    if gross_worth_start is not None:

        end_position = len(lines)

        for index in range(
            gross_worth_start + 1,
            len(lines)
        ):

            if lines[index].lower() == "gross worth":

                end_position = index
                break

        for line in lines[
            gross_worth_start + 1 : end_position
        ]:

            if re.fullmatch(
                r"\$?\s*[\d\s]+[,.]\d{2}",
                line
            ):

                gross_worth_values.append(
                    convert_ocr_number(line)
                )

    item_count = min(
        len(net_price_values),
        len(net_worth_values),
        len(vat_values),
        len(gross_worth_values)
    )

    records = []

    for index in range(item_count):

        records.append(
            {
                "Unit Price": net_price_values[index],
                "Net Worth": net_worth_values[index],
                "VAT": vat_values[index],
                "Gross Worth": gross_worth_values[index],
                "Extraction Layout": "Special-Separated"
            }
        )

    return records


special_test_images = [
    "batch1-0074.jpg"
]

for image_name in special_test_images:

    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    special_rows = extract_special_separated_financial_rows(
        ocr_text
    )

    print("\n" + "=" * 70)
    print("IMAGE:", image_name)
    print("=" * 70)

    for row in special_rows:
        print(row)

    print(
        "Extracted financial rows:",
        len(special_rows)
    )


print("\n========== STEP 45D.22 COMPLETED ========== ")


#=============================
# Step 45D.23 : Combine OCR Financial Parsers
#=============================
print("\n========== Step 45D.23 : COMBINE OCR FINANCIAL PARSERS ========== ")

def extract_all_ocr_financial_rows(ocr_text):

    single_line_rows = extract_single_line_financial_values(
        ocr_text
    )

    if single_line_rows:
        return single_line_rows

    separated_rows = extract_separated_net_financial_rows(
        ocr_text
    )

    special_rows = extract_special_separated_financial_rows(
        ocr_text
    )

    if special_rows:
        return special_rows

    if separated_rows:
        return separated_rows

    return []


parser_test_images = [
    "batch1-0002.jpg",
    "batch1-0027.jpg",
    "batch1-0061.jpg",
    "batch1-0074.jpg",
    "batch1-0082.jpg"
]

for image_name in parser_test_images:

    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    financial_rows = extract_all_ocr_financial_rows(
        ocr_text
    )

    print("\n" + "=" * 70)
    print("IMAGE:", image_name)
    print("=" * 70)

    for row in financial_rows:
        print(row)

    print(
        "Total extracted financial rows:",
        len(financial_rows)
    )


print("\n========== STEP 45D.23 COMPLETED ========== ")


#=============================
# Step 45D.24 : Run Combined Financial Parser on All Invoices
#=============================
print("\n========== Step 45D.24 : RUN COMBINED FINANCIAL PARSER ON ALL INVOICES ========== ")

financial_recovery_records = []

for image_name in all_invoice_images:

    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    financial_rows = extract_all_ocr_financial_rows(
        ocr_text
    )

    for item_index, row in enumerate(
        financial_rows,
        start=1
    ):

        financial_recovery_records.append(
            {
                "Image Name": image_name,
                "Item Number": item_index,
                "Unit Price": row.get("Unit Price"),
                "Net Worth": row.get("Net Worth"),
                "VAT": row.get("VAT"),
                "Gross Worth": row.get("Gross Worth"),
                "Extraction Layout": row.get("Extraction Layout")
            }
        )


financial_recovery_df = pd.DataFrame(
    financial_recovery_records
)

print(
    "\nTotal OCR financial records extracted:",
    len(financial_recovery_df)
)

print(
    "Unique invoices with financial extraction:",
    financial_recovery_df["Image Name"].nunique()
    if not financial_recovery_df.empty
    else 0
)

print(
    "Invoices without financial extraction:",
    len(all_invoice_images)
    - (
        financial_recovery_df["Image Name"].nunique()
        if not financial_recovery_df.empty
        else 0
    )
)

if not financial_recovery_df.empty:

    print("\n========== EXTRACTION BY LAYOUT ==========")

    print(
        financial_recovery_df[
            "Extraction Layout"
        ].value_counts()
    )

    print("\n========== FINANCIAL FIELD COVERAGE ==========")

    print(
        "Unit Price:",
        financial_recovery_df["Unit Price"].notna().sum()
    )

    print(
        "Net Worth:",
        financial_recovery_df["Net Worth"].notna().sum()
    )

    print(
        "VAT:",
        financial_recovery_df["VAT"].notna().sum()
    )

    print(
        "Gross Worth:",
        financial_recovery_df["Gross Worth"].notna().sum()
    )

    print("\n========== SAMPLE RECOVERY RECORDS ==========")

    print(
        financial_recovery_df.head(20).to_string(
            index=False
        )
    )


print("\n========== STEP 45D.24 COMPLETED ========== ")


#=============================
# Step 45D.25 : Add Item-Level Gross Worth to Financial Records
#=============================
print("\n========== Step 45D.25 : ADD ITEM-LEVEL GROSS WORTH TO FINANCIAL RECORDS ========== ")

gross_recovery_records = []

for image_name in all_invoice_images:

    ocr_text = all_ocr_results.get(
        image_name,
        ""
    )

    gross_values = extract_item_gross_worth_values_v2(
        ocr_text
    )

    for item_index, gross_value in enumerate(
        gross_values,
        start=1
    ):

        gross_recovery_records.append(
            {
                "Image Name": image_name,
                "Item Number": item_index,
                "Gross Worth Recovery": gross_value
            }
        )


gross_recovery_df = pd.DataFrame(
    gross_recovery_records
)

print(
    "\nTotal item-level Gross Worth records:",
    len(gross_recovery_df)
)

print(
    "Invoices with item-level Gross Worth:",
    gross_recovery_df["Image Name"].nunique()
    if not gross_recovery_df.empty
    else 0
)

if not gross_recovery_df.empty:

    print(
        "\nGross Worth values extracted:",
        gross_recovery_df["Gross Worth Recovery"].notna().sum()
    )

    print("\n========== SAMPLE GROSS WORTH RECOVERY ==========")

    print(
        gross_recovery_df.head(20).to_string(
            index=False
        )
    )


print("\n========== STEP 45D.25 COMPLETED ========== ")


#=============================
# Step 45D.26 : Build Complete OCR Financial Recovery Dataset
#=============================
print("\n========== Step 45D.26 : BUILD COMPLETE OCR FINANCIAL RECOVERY DATASET ========== ")

# Merge item-level Gross Worth recovered from OCR
financial_recovery_complete_df = financial_recovery_df.merge(
    gross_recovery_df,
    on=["Image Name", "Item Number"],
    how="outer"
)

# Fill missing Gross Worth values using the item-level OCR recovery
financial_recovery_complete_df["Gross Worth"] = (
    financial_recovery_complete_df["Gross Worth"]
    .fillna(
        financial_recovery_complete_df["Gross Worth Recovery"]
    )
)

# Remove the temporary recovery column
financial_recovery_complete_df = (
    financial_recovery_complete_df.drop(
        columns=["Gross Worth Recovery"]
    )
)

print(
    "\nTotal OCR financial recovery records:",
    len(financial_recovery_complete_df)
)

print(
    "Invoices with OCR financial recovery:",
    financial_recovery_complete_df["Image Name"].nunique()
)

print("\n========== FINANCIAL FIELD COVERAGE ==========")

print(
    "Unit Price:",
    financial_recovery_complete_df["Unit Price"].notna().sum()
)

print(
    "Net Worth:",
    financial_recovery_complete_df["Net Worth"].notna().sum()
)

print(
    "VAT:",
    financial_recovery_complete_df["VAT"].notna().sum()
)

print(
    "Gross Worth:",
    financial_recovery_complete_df["Gross Worth"].notna().sum()
)

print("\n========== SAMPLE COMPLETE OCR RECOVERY ==========")

print(
    financial_recovery_complete_df.head(20).to_string(
        index=False
    )
)

print("\n========== STEP 45D.26 COMPLETED ========== ")


#=============================
# Step 45D.27 : Recover Missing Financial Values
#=============================
print("\n========== Step 45D.27 : RECOVER MISSING FINANCIAL VALUES ========== ")

# Create a copy of the current final CSV dataframe
financial_recovery_output_df = final_csv_df.copy()

# Merge OCR financial recovery using invoice image and item number
financial_recovery_output_df = financial_recovery_output_df.merge(
    financial_recovery_complete_df,
    on=["Image Name", "Item Number"],
    how="left",
    suffixes=("", "_OCR")
)

# Count missing values before recovery
missing_unit_before = (
    financial_recovery_output_df["Unit Price"].isna().sum()
)

missing_net_before = (
    financial_recovery_output_df["Net Worth"].isna().sum()
)

missing_vat_before = (
    financial_recovery_output_df["VAT"].isna().sum()
)

missing_gross_before = (
    financial_recovery_output_df["Gross Worth"].isna().sum()
)

# Fill only missing values using OCR-recovered financial data
financial_recovery_output_df["Unit Price"] = (
    financial_recovery_output_df["Unit Price"]
    .fillna(
        financial_recovery_output_df["Unit Price_OCR"]
    )
)

financial_recovery_output_df["Net Worth"] = (
    financial_recovery_output_df["Net Worth"]
    .fillna(
        financial_recovery_output_df["Net Worth_OCR"]
    )
)

financial_recovery_output_df["VAT"] = (
    financial_recovery_output_df["VAT"]
    .fillna(
        financial_recovery_output_df["VAT_OCR"]
    )
)

financial_recovery_output_df["Gross Worth"] = (
    financial_recovery_output_df["Gross Worth"]
    .fillna(
        financial_recovery_output_df["Gross Worth_OCR"]
    )
)

# Count values recovered
unit_recovered = (
    missing_unit_before
    - financial_recovery_output_df["Unit Price"].isna().sum()
)

net_recovered = (
    missing_net_before
    - financial_recovery_output_df["Net Worth"].isna().sum()
)

vat_recovered = (
    missing_vat_before
    - financial_recovery_output_df["VAT"].isna().sum()
)

gross_recovered = (
    missing_gross_before
    - financial_recovery_output_df["Gross Worth"].isna().sum()
)

# Remove temporary OCR columns
financial_recovery_output_df = (
    financial_recovery_output_df.drop(
        columns=[
            "Unit Price_OCR",
            "Net Worth_OCR",
            "VAT_OCR",
            "Gross Worth_OCR",
            "Extraction Layout"
        ],
        errors="ignore"
    )
)

print("\n========== RECOVERY RESULTS ==========")

print(
    "Unit Price recovered:",
    unit_recovered
)

print(
    "Net Worth recovered:",
    net_recovered
)

print(
    "VAT recovered:",
    vat_recovered
)

print(
    "Gross Worth recovered:",
    gross_recovered
)

print("\n========== REMAINING MISSING VALUES ==========")

print(
    "Unit Price remaining:",
    financial_recovery_output_df["Unit Price"].isna().sum()
)

print(
    "Net Worth remaining:",
    financial_recovery_output_df["Net Worth"].isna().sum()
)

print(
    "VAT remaining:",
    financial_recovery_output_df["VAT"].isna().sum()
)

print(
    "Gross Worth remaining:",
    financial_recovery_output_df["Gross Worth"].isna().sum()
)

print(
    "\nFinal recovery dataframe shape:",
    financial_recovery_output_df.shape
)

print("\n========== STEP 45D.27 COMPLETED ========== ")

#=============================
# Step 45D.28 : Analyze Remaining Financial Records
#=============================
print("\n========== Step 45D.28 : ANALYZE REMAINING FINANCIAL RECORDS ========== ")

# Identify records still missing the core financial fields
remaining_financial_df = financial_recovery_output_df[
    financial_recovery_output_df[
        ["Unit Price", "Net Worth", "VAT"]
    ].isna().all(axis=1)
].copy()

print(
    "\nRemaining records missing Unit Price, Net Worth and VAT:",
    len(remaining_financial_df)
)

print(
    "Unique invoices with remaining financial gaps:",
    remaining_financial_df["Image Name"].nunique()
)

print("\n========== SAMPLE REMAINING RECORDS ==========")

print(
    remaining_financial_df[
        [
            "Image Name",
            "Item Number",
            "Description",
            "Quantity"
        ]
    ]
    .head(30)
    .to_string(index=False)
)

# Get the unique invoice images that still contain financial gaps
remaining_images = (
    remaining_financial_df["Image Name"]
    .dropna()
    .unique()
    .tolist()
)

print(
    "\nTotal invoice images requiring further analysis:",
    len(remaining_images)
)

print("\n========== STEP 45D.28 COMPLETED ========== ")


#=============================
# Step 45D.29 : Build Final OCR-Only Production Dataset
#=============================
print("\n========== Step 45D.29 : BUILD FINAL OCR-ONLY PRODUCTION DATASET ========== ")

# Start from all OCR-extracted item records
production_df = all_item_df.copy()

# Keep the required invoice and item columns
production_df = production_df[
    [
        "Image Name",
        "Item Number",
        "Description",
        "Quantity"
    ]
].copy()

# Merge the best OCR-derived financial values
production_df = production_df.merge(
    financial_recovery_output_df[
        [
            "Image Name",
            "Item Number",
            "Unit Price",
            "Net Worth",
            "VAT",
            "Gross Worth"
        ]
    ],
    on=[
        "Image Name",
        "Item Number"
    ],
    how="left"
)

# Add invoice-level information extracted from OCR
production_df = production_df.merge(
    full_invoice_df[
        [
            "Image Name",
            "Invoice Number",
            "Invoice Date",
            "Vendor Name",
            "Total Amount"
        ]
    ],
    on="Image Name",
    how="left"
)

# Add anomaly information
production_df = production_df.merge(
    anomaly_report_df[
        [
            "Image Name",
            "Item Number",
            "Anomaly Score",
            "Validation Result",
            "Investigation Priority"
        ]
    ],
    on=[
        "Image Name",
        "Item Number"
    ],
    how="left"
)

# Add duplicate detection information
production_df = production_df.merge(
    duplicate_report_df[
        [
            "Image Name",
            "Duplicate Status",
            "Duplicate Reason"
        ]
    ],
    on="Image Name",
    how="left"
)

# Create the required anomaly flag
production_df["IsAnomaly"] = (
    production_df["Anomaly Score"].notna()
)

# Create the required anomaly reason
production_df["AnomalyReason"] = ""

production_df.loc[
    production_df["IsAnomaly"],
    "AnomalyReason"
] = (
    production_df.loc[
        production_df["IsAnomaly"],
        "Validation Result"
    ]
)

# Arrange final production columns
production_df = production_df[
    [
        "Invoice Number",
        "Image Name",
        "Item Number",
        "Invoice Date",
        "Vendor Name",
        "Description",
        "Quantity",
        "Unit Price",
        "Net Worth",
        "VAT",
        "Gross Worth",
        "Total Amount",
        "IsAnomaly",
        "AnomalyReason",
        "Anomaly Score",
        "Validation Result",
        "Investigation Priority",
        "Duplicate Status",
        "Duplicate Reason"
    ]
].copy()

# Sort by invoice and item number
production_df = production_df.sort_values(
    by=[
        "Invoice Number",
        "Item Number"
    ],
    na_position="last"
).reset_index(drop=True)

print("\n========== FINAL PRODUCTION DATASET ==========")

print(
    "Total item records:",
    len(production_df)
)

print(
    "Unique invoices:",
    production_df["Invoice Number"].nunique()
)

print(
    "Descriptions available:",
    production_df["Description"].notna().sum()
)

print(
    "Quantity available:",
    production_df["Quantity"].notna().sum()
)

print(
    "Unit Price available:",
    production_df["Unit Price"].notna().sum()
)

print(
    "Net Worth available:",
    production_df["Net Worth"].notna().sum()
)

print(
    "VAT available:",
    production_df["VAT"].notna().sum()
)

print(
    "Gross Worth available:",
    production_df["Gross Worth"].notna().sum()
)

print(
    "Anomalous records:",
    production_df["IsAnomaly"].sum()
)

print("\n========== SAMPLE PRODUCTION DATA ==========")

print(
    production_df.head(20).to_string(
        index=False
    )
)

print(
    "\nFinal production dataframe shape:",
    production_df.shape
)

print("\n========== STEP 45D.29 COMPLETED ========== ")


#=============================
# Step 45D.30 : Final Production Dataset Sanity Check
#=============================
print("\n========== Step 45D.30 : FINAL PRODUCTION DATASET SANITY CHECK ========== ")

# Check required columns
required_columns = [
    "Invoice Number",
    "Image Name",
    "Item Number",
    "Invoice Date",
    "Vendor Name",
    "Description",
    "Quantity",
    "Unit Price",
    "Net Worth",
    "VAT",
    "Gross Worth",
    "Total Amount",
    "IsAnomaly",
    "AnomalyReason"
]

missing_columns = [
    column
    for column in required_columns
    if column not in production_df.columns
]

print(
    "\nMissing required columns:",
    missing_columns
)

# Check duplicate invoice-item combinations
duplicate_item_keys = production_df.duplicated(
    subset=[
        "Image Name",
        "Item Number"
    ]
).sum()

print(
    "Duplicate invoice-item records:",
    duplicate_item_keys
)

# Check numeric fields for negative values
numeric_columns = [
    "Quantity",
    "Unit Price",
    "Net Worth",
    "VAT",
    "Gross Worth",
    "Total Amount"
]

print("\n========== NEGATIVE VALUE CHECK ==========")

for column in numeric_columns:

    negative_count = (
        production_df[column]
        .dropna()
        .lt(0)
        .sum()
    )

    print(
        f"{column}:",
        negative_count
    )

# Check anomaly flag consistency
anomaly_flag_count = (
    production_df["IsAnomaly"]
    .eq(True)
    .sum()
)

anomaly_score_count = (
    production_df["Anomaly Score"]
    .notna()
    .sum()
)

print("\n========== ANOMALY CONSISTENCY CHECK ==========")

print(
    "IsAnomaly = True:",
    anomaly_flag_count
)

print(
    "Records with Anomaly Score:",
    anomaly_score_count
)

# Check required text fields
print("\n========== REQUIRED FIELD CHECK ==========")

for column in [
    "Image Name",
    "Description"
]:

    print(
        f"{column} missing:",
        production_df[column].isna().sum()
    )

# Final structural summary
print("\n========== FINAL SANITY SUMMARY ==========")

print(
    "Rows:",
    len(production_df)
)

print(
    "Columns:",
    len(production_df.columns)
)

print(
    "Unique invoice images:",
    production_df["Image Name"].nunique()
)

print(
    "Unique invoice numbers:",
    production_df["Invoice Number"].nunique()
)

print(
    "Anomalies:",
    production_df["IsAnomaly"].sum()
)

if (
    len(missing_columns) == 0
    and duplicate_item_keys == 0
):
    print(
        "\nSANITY CHECK STATUS: PASSED"
    )
else:
    print(
        "\nSANITY CHECK STATUS: REVIEW REQUIRED"
    )

print("\n========== STEP 45D.30 COMPLETED ========== ")


#=============================
# Step 45D.31 : Export Final Production CSV
#=============================
print("\n========== Step 45D.31 : EXPORT FINAL PRODUCTION CSV ========== ")

# Define the final CSV output path
final_csv_path = os.path.join(
    "reports",
    "final_invoice_output.csv"
)

# Export the production dataset
production_df.to_csv(
    final_csv_path,
    index=False
)

# Confirm the exported file
print(
    "\nFinal CSV saved to:",
    final_csv_path
)

print(
    "Rows exported:",
    len(production_df)
)

print(
    "Columns exported:",
    len(production_df.columns)
)

# Verify the exported CSV can be loaded
exported_csv_df = pd.read_csv(
    final_csv_path
)

print(
    "\nExport verification rows:",
    len(exported_csv_df)
)

print(
    "Export verification columns:",
    len(exported_csv_df.columns)
)

if (
    len(exported_csv_df) == len(production_df)
    and len(exported_csv_df.columns) == len(production_df.columns)
):
    print(
        "\nCSV EXPORT STATUS: PASSED"
    )
else:
    print(
        "\nCSV EXPORT STATUS: REVIEW REQUIRED"
    )

print("\n========== STEP 45D.31 COMPLETED ========== ")


#=============================
# Step 46.1 : Financial Outlier Box Plots
#=============================
print("\n========== Step 46.1 : FINANCIAL OUTLIER BOX PLOTS ========== ")

# Load the final production CSV
visualization_df = pd.read_csv(
    final_csv_path
)

# Create the reports visualization directory
visualization_folder = os.path.join(
    "reports",
    "visualizations"
)

os.makedirs(
    visualization_folder,
    exist_ok=True
)

# Define financial fields for visualization
financial_plot_columns = [
    "Unit Price",
    "Net Worth",
    "Gross Worth"
]

# Create one box plot for each financial field
for column in financial_plot_columns:

    plot_data = visualization_df[column].dropna()

    plt.figure(
        figsize=(8, 6)
    )

    plt.boxplot(
        plot_data
    )

    plt.title(
        f"Box Plot of {column}"
    )

    plt.ylabel(
        column
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plot_filename = os.path.join(
        visualization_folder,
        f"{column.lower().replace(' ', '_')}_boxplot.png"
    )

    plt.savefig(
        plot_filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"{column} box plot saved to:",
        plot_filename
    )

print(
    "\nVisualization files created:",
    len(financial_plot_columns)
)

print("\n========== STEP 46.1 COMPLETED ========== ")

#=============================
# Step 46.2 : Net Worth vs Gross Worth Anomaly Scatter Plot
#=============================
print("\n========== Step 46.2 : NET WORTH VS GROSS WORTH SCATTER PLOT ========== ")

# Prepare financial data for visualization
scatter_df = visualization_df[
    [
        "Net Worth",
        "Gross Worth",
        "IsAnomaly"
    ]
].dropna()

# Separate normal and anomalous records
normal_records = scatter_df[
    scatter_df["IsAnomaly"] == False
]

anomaly_records = scatter_df[
    scatter_df["IsAnomaly"] == True
]

# Create scatter plot
plt.figure(
    figsize=(10, 7)
)

plt.scatter(
    normal_records["Net Worth"],
    normal_records["Gross Worth"],
    label="Normal"
)

plt.scatter(
    anomaly_records["Net Worth"],
    anomaly_records["Gross Worth"],
    label="Potential Anomaly"
)

plt.title(
    "Net Worth vs Gross Worth - Anomaly Detection"
)

plt.xlabel(
    "Net Worth"
)

plt.ylabel(
    "Gross Worth"
)

plt.legend()

plt.grid(
    alpha=0.3
)

# Save scatter plot
scatter_plot_path = os.path.join(
    visualization_folder,
    "net_worth_vs_gross_worth_anomalies.png"
)

plt.savefig(
    scatter_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print(
    "Normal records plotted:",
    len(normal_records)
)

print(
    "Potential anomaly records plotted:",
    len(anomaly_records)
)

print(
    "Scatter plot saved to:",
    scatter_plot_path
)

print("\n========== STEP 46.2 COMPLETED ========== ")

#=============================
# Step 46.3 : Anomaly Score vs Net Worth Visualization
#=============================
print("\n========== Step 46.3 : ANOMALY SCORE VS NET WORTH ========== ")

# Prepare the exact records used by Isolation Forest
score_df = X_ml[
    [
        "Net Worth"
    ]
].reset_index(
    drop=True
).copy()

# Add Isolation Forest scores and predictions
score_df["Anomaly Score"] = anomaly_scores
score_df["Prediction"] = predictions

# Separate normal and anomalous records
normal_score_records = score_df[
    score_df["Prediction"] == 1
]

anomaly_score_records = score_df[
    score_df["Prediction"] == -1
]

# Create scatter plot
plt.figure(
    figsize=(10, 7)
)

plt.scatter(
    normal_score_records["Net Worth"],
    normal_score_records["Anomaly Score"],
    label="Normal"
)

plt.scatter(
    anomaly_score_records["Net Worth"],
    anomaly_score_records["Anomaly Score"],
    label="Potential Anomaly"
)

# Add reference line at zero
plt.axhline(
    y=0,
    linestyle="--",
    label="Reference Line"
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

plt.legend()

plt.grid(
    alpha=0.3
)

# Save anomaly score visualization
anomaly_score_plot_path = os.path.join(
    visualization_folder,
    "anomaly_score_vs_net_worth.png"
)

plt.savefig(
    anomaly_score_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print(
    "Total records plotted:",
    len(score_df)
)

print(
    "Normal records plotted:",
    len(normal_score_records)
)

print(
    "Potential anomaly records plotted:",
    len(anomaly_score_records)
)

print(
    "Anomaly score visualization saved to:",
    anomaly_score_plot_path
)

print("\n========== STEP 46.3 COMPLETED ========== ")


#=============================
# Step 46.4 : Anomaly Distribution Visualization
#=============================
print("\n========== Step 46.4 : ANOMALY DISTRIBUTION ========== ")

# Calculate normal and anomaly counts
normal_count = int(
    np.sum(predictions == 1)
)

anomaly_count = int(
    np.sum(predictions == -1)
)

total_model_records = (
    normal_count + anomaly_count
)

# Calculate percentages
normal_percentage = (
    normal_count / total_model_records
) * 100

anomaly_percentage = (
    anomaly_count / total_model_records
) * 100

# Create anomaly distribution bar chart
plt.figure(
    figsize=(8, 6)
)

plt.bar(
    ["Normal", "Potential Anomaly"],
    [normal_count, anomaly_count]
)

plt.title(
    "Isolation Forest Anomaly Distribution"
)

plt.xlabel(
    "Classification"
)

plt.ylabel(
    "Number of Records"
)

# Display record counts above bars
plt.text(
    0,
    normal_count,
    f"{normal_count}",
    ha="center",
    va="bottom"
)

plt.text(
    1,
    anomaly_count,
    f"{anomaly_count}",
    ha="center",
    va="bottom"
)

plt.grid(
    axis="y",
    alpha=0.3
)

# Save anomaly distribution chart
anomaly_distribution_path = os.path.join(
    visualization_folder,
    "anomaly_distribution.png"
)

plt.savefig(
    anomaly_distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print(
    "Normal records:",
    normal_count,
    f"({normal_percentage:.2f}%)"
)

print(
    "Potential anomaly records:",
    anomaly_count,
    f"({anomaly_percentage:.2f}%)"
)

print(
    "Total records:",
    total_model_records
)

print(
    "Anomaly distribution chart saved to:",
    anomaly_distribution_path
)

print("\n========== STEP 46.4 COMPLETED ========== ")

#=============================
# Step 46.5 : Anomaly Validation Status Visualization
#=============================
print("\n========== Step 46.5 : ANOMALY VALIDATION STATUS ========== ")

# Calculate validation status counts
validated_count = len(
    validated_anomaly_report
)

investigation_count = len(
    investigation_report
)

total_anomaly_records = (
    validated_count + investigation_count
)

# Calculate percentages
validated_percentage = (
    validated_count / total_anomaly_records
) * 100

investigation_percentage = (
    investigation_count / total_anomaly_records
) * 100

# Create validation status bar chart
plt.figure(
    figsize=(8, 6)
)

plt.bar(
    [
        "Statistically Unusual\nand Validated",
        "Requires Further\nInvestigation"
    ],
    [
        validated_count,
        investigation_count
    ]
)

plt.title(
    "Anomaly Validation Status"
)

plt.xlabel(
    "Validation Status"
)

plt.ylabel(
    "Number of Records"
)

# Display record counts above bars
plt.text(
    0,
    validated_count,
    f"{validated_count}",
    ha="center",
    va="bottom"
)

plt.text(
    1,
    investigation_count,
    f"{investigation_count}",
    ha="center",
    va="bottom"
)

plt.grid(
    axis="y",
    alpha=0.3
)

# Save validation status chart
validation_status_path = os.path.join(
    visualization_folder,
    "anomaly_validation_status.png"
)

plt.savefig(
    validation_status_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print(
    "Statistically unusual and validated:",
    validated_count,
    f"({validated_percentage:.2f}%)"
)

print(
    "Requires further investigation:",
    investigation_count,
    f"({investigation_percentage:.2f}%)"
)

print(
    "Total potential anomalies:",
    total_anomaly_records
)

print(
    "Validation status chart saved to:",
    validation_status_path
)

print("\n========== STEP 46.5 COMPLETED ========== ")

#=============================
# Step 47.1 : Final Project Performance Summary
#=============================
print("\n========== Step 47.1 : FINAL PROJECT PERFORMANCE SUMMARY ========== ")

# Calculate dataset metrics
total_images = len(all_invoice_images)

ocr_success_count = len(all_ocr_results)

total_item_records = len(all_item_df)

financial_complete_count = len(final_financial_df)

# Calculate extraction coverage
ocr_coverage = (
    ocr_success_count / total_images
) * 100

item_extraction_coverage = (
    len(all_item_df["Image Name"].unique()) / total_images
) * 100

financial_extraction_coverage = (
    financial_complete_count / total_item_records
) * 100

# Calculate anomaly metrics
total_model_records = len(predictions)

normal_count = int(
    np.sum(predictions == 1)
)

potential_anomaly_count = int(
    np.sum(predictions == -1)
)

anomaly_percentage = (
    potential_anomaly_count / total_model_records
) * 100

# Calculate anomaly validation metrics
validated_anomaly_count = len(
    validated_anomaly_report
)

investigation_count = len(
    investigation_report
)

validated_anomaly_percentage = (
    validated_anomaly_count / potential_anomaly_count
) * 100

# Calculate risk metrics
high_risk_count = len(
    high_priority_report
)

high_risk_percentage = (
    high_risk_count / potential_anomaly_count
) * 100

# Calculate duplicate metrics
duplicate_record_count = len(
    duplicate_report_df
)

# Identify the existing date-status column
date_status_columns = [
    column
    for column in date_validation_df.columns
    if "status" in column.lower()
]

if len(date_status_columns) > 0:

    date_status_column = date_status_columns[0]

    invalid_date_count = int(
        date_validation_df[
            date_status_column
        ].astype(str).str.contains(
            "missing|invalid",
            case=False,
            na=False
        ).sum()
    )

    old_date_count = int(
        date_validation_df[
            date_status_column
        ].astype(str).str.contains(
            "older",
            case=False,
            na=False
        ).sum()
    )

    future_date_count = int(
        date_validation_df[
            date_status_column
        ].astype(str).str.contains(
            "future",
            case=False,
            na=False
        ).sum()
    )

else:

    invalid_date_count = 0
    old_date_count = 0
    future_date_count = 0

    print(
        "Warning: No date-status column found."
    )

# Display final performance summary
print("\n----- DATASET -----")

print(
    "Total invoice images:",
    total_images
)

print(
    "OCR records processed:",
    ocr_success_count
)

print(
    f"OCR coverage: {ocr_coverage:.2f}%"
)

print(
    "Total extracted item records:",
    total_item_records
)

print(
    f"Invoice item extraction coverage: {item_extraction_coverage:.2f}%"
)

print(
    "Complete financial records:",
    financial_complete_count
)

print(
    f"Complete financial extraction coverage: {financial_extraction_coverage:.2f}%"
)

print("\n----- ANOMALY DETECTION -----")

print(
    "Records analyzed by Isolation Forest:",
    total_model_records
)

print(
    "Normal records:",
    normal_count
)

print(
    "Potential anomalies:",
    potential_anomaly_count
)

print(
    f"Potential anomaly percentage: {anomaly_percentage:.2f}%"
)

print("\n----- ANOMALY VALIDATION -----")

print(
    "Statistically unusual and validated:",
    validated_anomaly_count
)

print(
    f"Validated anomaly percentage: {validated_anomaly_percentage:.2f}%"
)

print(
    "Requires further investigation:",
    investigation_count
)

print(
    "High-risk records:",
    high_risk_count
)

print(
    f"High-risk percentage: {high_risk_percentage:.2f}%"
)

print("\n----- DATE VALIDATION -----")

print(
    "Missing/invalid dates:",
    invalid_date_count
)

print(
    "Dates older than threshold:",
    old_date_count
)

print(
    "Future dates:",
    future_date_count
)

print("\n----- DUPLICATE DETECTION -----")

print(
    "Potential duplicate records:",
    duplicate_record_count
)

print("\n========== STEP 47.1 COMPLETED ========== ")

#=============================
# Step 47.1 : Final Project Performance Summary
#=============================
print("\n========== Step 47.1 : FINAL PROJECT PERFORMANCE SUMMARY ========== ")

# Dataset metrics
total_images = len(all_invoice_images)

ocr_success_count = len(all_ocr_results)

total_item_records = len(all_item_df)

# Item extraction coverage
item_invoice_count = 486

item_extraction_coverage = (
    item_invoice_count / total_images
) * 100

# Financial field coverage
quantity_count = production_df["Quantity"].notna().sum()

unit_price_count = production_df["Unit Price"].notna().sum()

net_worth_count = production_df["Net Worth"].notna().sum()

vat_count = production_df["VAT"].notna().sum()

gross_worth_count = production_df["Gross Worth"].notna().sum()

quantity_coverage = (
    quantity_count / total_item_records
) * 100

unit_price_coverage = (
    unit_price_count / total_item_records
) * 100

net_worth_coverage = (
    net_worth_count / total_item_records
) * 100

vat_coverage = (
    vat_count / total_item_records
) * 100

gross_worth_coverage = (
    gross_worth_count / total_item_records
) * 100

# Anomaly detection metrics
total_model_records = len(predictions)

normal_count = int(
    np.sum(predictions == 1)
)

potential_anomaly_count = int(
    np.sum(predictions == -1)
)

anomaly_percentage = (
    potential_anomaly_count / total_model_records
) * 100

# Anomaly validation metrics
validated_anomaly_count = len(
    validated_anomaly_report
)

investigation_count = len(
    investigation_report
)

validated_anomaly_percentage = (
    validated_anomaly_count / potential_anomaly_count
) * 100

# Risk metrics
high_risk_count = len(
    high_priority_report
)

high_risk_percentage = (
    high_risk_count / potential_anomaly_count
) * 100

# Duplicate detection result from Step 45C
potential_duplicate_count = 21

# Date validation results from Step 45B
missing_invalid_dates = 20

old_dates = 468

future_dates = 0

# Display dataset results
print("\n----- DATASET -----")

print(
    "Total invoice images:",
    total_images
)

print(
    "OCR records processed:",
    ocr_success_count
)

print(
    f"OCR coverage: {ocr_success_count / total_images * 100:.2f}%"
)

print(
    "Total extracted item records:",
    total_item_records
)

print(
    "Invoices with extracted items:",
    item_invoice_count
)

print(
    f"Invoice item extraction coverage: {item_extraction_coverage:.2f}%"
)

# Display financial extraction coverage
print("\n----- FINANCIAL FIELD EXTRACTION -----")

print(
    f"Quantity: {quantity_count}/{total_item_records} "
    f"({quantity_coverage:.2f}%)"
)

print(
    f"Unit Price: {unit_price_count}/{total_item_records} "
    f"({unit_price_coverage:.2f}%)"
)

print(
    f"Net Worth: {net_worth_count}/{total_item_records} "
    f"({net_worth_coverage:.2f}%)"
)

print(
    f"VAT: {vat_count}/{total_item_records} "
    f"({vat_coverage:.2f}%)"
)

print(
    f"Gross Worth: {gross_worth_count}/{total_item_records} "
    f"({gross_worth_coverage:.2f}%)"
)

# Display anomaly detection results
print("\n----- ANOMALY DETECTION -----")

print(
    "Records analyzed by Isolation Forest:",
    total_model_records
)

print(
    "Normal records:",
    normal_count
)

print(
    "Potential anomalies:",
    potential_anomaly_count
)

print(
    f"Potential anomaly percentage: {anomaly_percentage:.2f}%"
)

# Display anomaly validation results
print("\n----- ANOMALY VALIDATION -----")

print(
    "Statistically unusual and validated:",
    validated_anomaly_count
)

print(
    f"Validated anomaly percentage: "
    f"{validated_anomaly_percentage:.2f}%"
)

print(
    "Requires further investigation:",
    investigation_count
)

print(
    "High-risk records:",
    high_risk_count
)

print(
    f"High-risk percentage: {high_risk_percentage:.2f}%"
)

# Display date validation results
print("\n----- DATE VALIDATION -----")

print(
    "Missing/invalid dates:",
    missing_invalid_dates
)

print(
    "Dates older than 90 days:",
    old_dates
)

print(
    "Future dates:",
    future_dates
)

# Display duplicate detection results
print("\n----- DUPLICATE DETECTION -----")

print(
    "Potential duplicate records:",
    potential_duplicate_count
)

print("\n========== STEP 47.1 COMPLETED ========== ")

#=============================
# Step 47.2 : Strongest Anomaly Cases
#=============================
print("\n========== Step 47.2 : STRONGEST ANOMALY CASES ========== ")

# Create a copy of the final anomaly report
strongest_anomaly_cases = anomaly_report_df.copy()

# Sort anomalies from strongest to weakest
strongest_anomaly_cases = strongest_anomaly_cases.sort_values(
    by="Anomaly Score",
    ascending=True
).reset_index(
    drop=True
)

# Select the top 10 strongest anomalies
top_10_anomalies = strongest_anomaly_cases.head(
    10
).copy()

# Identify the highest Net Worth anomaly
highest_net_worth_anomaly = strongest_anomaly_cases.loc[
    strongest_anomaly_cases["Net Worth"].idxmax()
]

# Identify the highest Gross Worth anomaly
highest_gross_worth_anomaly = strongest_anomaly_cases.loc[
    strongest_anomaly_cases["Gross Worth"].idxmax()
]

# Display top 10 strongest anomalies
print("\n----- TOP 10 STRONGEST ANOMALIES -----")

print(
    top_10_anomalies[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Quantity",
            "Unit Price",
            "Net Worth",
            "Gross Worth",
            "Anomaly Score",
            "Validation Result",
            "Investigation Priority"
        ]
    ].to_string(
        index=False
    )
)

# Display highest Net Worth anomaly
print("\n----- HIGHEST NET WORTH ANOMALY -----")

print(
    "Invoice Number:",
    highest_net_worth_anomaly["Invoice Number"]
)

print(
    "Image Name:",
    highest_net_worth_anomaly["Image Name"]
)

print(
    "Item Number:",
    highest_net_worth_anomaly["Item Number"]
)

print(
    "Net Worth:",
    highest_net_worth_anomaly["Net Worth"]
)

print(
    "Anomaly Score:",
    highest_net_worth_anomaly["Anomaly Score"]
)

# Display highest Gross Worth anomaly
print("\n----- HIGHEST GROSS WORTH ANOMALY -----")

print(
    "Invoice Number:",
    highest_gross_worth_anomaly["Invoice Number"]
)

print(
    "Image Name:",
    highest_gross_worth_anomaly["Image Name"]
)

print(
    "Item Number:",
    highest_gross_worth_anomaly["Item Number"]
)

print(
    "Gross Worth:",
    highest_gross_worth_anomaly["Gross Worth"]
)

print(
    "Anomaly Score:",
    highest_gross_worth_anomaly["Anomaly Score"]
)

# Display investigation cases
print("\n----- CASES REQUIRING FURTHER INVESTIGATION -----")

print(
    investigation_report[
        [
            "Invoice Number",
            "Image Name",
            "Item Number",
            "Quantity",
            "Unit Price",
            "Net Worth",
            "Gross Worth",
            "Anomaly Score",
            "Validation Result",
            "Investigation Priority"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nTotal strongest anomaly cases available:",
    len(strongest_anomaly_cases)
)

print(
    "Top cases selected for reporting:",
    len(top_10_anomalies)
)

print(
    "\n========== STEP 47.2 COMPLETED ========== "
)

#=============================
# Step 47.3 : Financial Anomaly Analysis
#=============================
print("\n========== Step 47.3 : FINANCIAL ANOMALY ANALYSIS ========== ")

# Prepare financial anomaly analysis dataframe
financial_anomaly_analysis = anomaly_report_df[
    [
        "Invoice Number",
        "Image Name",
        "Item Number",
        "Quantity",
        "Unit Price",
        "Net Worth",
        "Gross Worth",
        "Anomaly Score"
    ]
].copy()

# Calculate overall financial averages
average_net_worth = X_ml["Net Worth"].mean()

average_gross_worth = X_ml["Gross Worth"].mean()

average_unit_price = X_ml["Unit Price"].mean()

# Calculate anomaly financial extremes
highest_anomaly_net_worth = financial_anomaly_analysis[
    "Net Worth"
].max()

highest_anomaly_gross_worth = financial_anomaly_analysis[
    "Gross Worth"
].max()

highest_anomaly_unit_price = financial_anomaly_analysis[
    "Unit Price"
].max()

highest_anomaly_quantity = financial_anomaly_analysis[
    "Quantity"
].max()

lowest_anomaly_quantity = financial_anomaly_analysis[
    "Quantity"
].min()

strongest_anomaly_score = financial_anomaly_analysis[
    "Anomaly Score"
].min()

weakest_anomaly_score = financial_anomaly_analysis[
    "Anomaly Score"
].max()

# Calculate how much the highest Net Worth exceeds the overall average
net_worth_multiple = (
    highest_anomaly_net_worth /
    average_net_worth
)

gross_worth_multiple = (
    highest_anomaly_gross_worth /
    average_gross_worth
)

# Display financial analysis
print("\n----- OVERALL FINANCIAL DISTRIBUTION -----")

print(
    f"Average Net Worth: {average_net_worth:.2f}"
)

print(
    f"Average Gross Worth: {average_gross_worth:.2f}"
)

print(
    f"Average Unit Price: {average_unit_price:.2f}"
)

print("\n----- ANOMALOUS FINANCIAL EXTREMES -----")

print(
    f"Highest anomalous Net Worth: "
    f"{highest_anomaly_net_worth:.2f}"
)

print(
    f"Highest anomalous Gross Worth: "
    f"{highest_anomaly_gross_worth:.2f}"
)

print(
    f"Highest anomalous Unit Price: "
    f"{highest_anomaly_unit_price:.2f}"
)

print(
    f"Highest anomalous Quantity: "
    f"{highest_anomaly_quantity:.2f}"
)

print(
    f"Lowest anomalous Quantity: "
    f"{lowest_anomaly_quantity:.2f}"
)

print("\n----- COMPARISON WITH OVERALL DATA -----")

print(
    f"Highest anomalous Net Worth is "
    f"{net_worth_multiple:.2f}x the overall average."
)

print(
    f"Highest anomalous Gross Worth is "
    f"{gross_worth_multiple:.2f}x the overall average."
)

print("\n----- ANOMALY SCORE RANGE -----")

print(
    f"Strongest anomaly score: "
    f"{strongest_anomaly_score:.6f}"
)

print(
    f"Weakest potential anomaly score: "
    f"{weakest_anomaly_score:.6f}"
)

# Display the three financially largest anomalous records
largest_financial_anomalies = financial_anomaly_analysis.sort_values(
    by="Net Worth",
    ascending=False
).head(
    3
)

print(
    "\n----- TOP 3 ANOMALIES BY NET WORTH -----"
)

print(
    largest_financial_anomalies.to_string(
        index=False
    )
)

print(
    "\n========== STEP 47.3 COMPLETED ========== "
)

#=============================
# Step 47.4 : Final Validation Findings
#=============================
print("\n========== Step 47.4 : FINAL VALIDATION FINDINGS ========== ")

# Rule-based financial validation results
print("\n----- FINANCIAL CONSISTENCY -----")

print(
    "Net Worth internally consistent:",
    141
)

print(
    "Gross Worth internally consistent:",
    142
)

print(
    "Quantity ground-truth matches:",
    141
)

print(
    "Quantity validation accuracy: 97.92%"
)

print(
    "Net Worth consistency: 97.92%"
)

print(
    "Gross Worth consistency: 98.61%"
)

# Vendor-wise Z-score results
print("\n----- VENDOR-WISE Z-SCORE -----")

print(
    "Total invoices checked:",
    499
)

print(
    "Valid vendor Z-scores:",
    6
)

print(
    "Insufficient vendor history:",
    493
)

print(
    "Strong vendor-level outliers (|Z| >= 3):",
    0
)

# Date validation results
print("\n----- DATE LOGIC -----")

print(
    "Total invoices checked:",
    499
)

print(
    "Missing/invalid dates:",
    20
)

print(
    "Dates older than 90 days:",
    468
)

print(
    "Future dates:",
    0
)

# Duplicate detection results
print("\n----- DUPLICATE DETECTION -----")

print(
    "Exact duplicate invoice numbers:",
    0
)

print(
    "Exact duplicate amounts:",
    10
)

print(
    "Fuzzy duplicate invoice numbers:",
    0
)

print(
    "Fuzzy duplicate amounts:",
    13
)

print(
    "Total potential duplicate records:",
    21
)

# Final anomaly validation
print("\n----- ANOMALY VALIDATION -----")

print(
    "Potential anomalies detected:",
    144
)

print(
    "Statistically unusual and validated:",
    140
)

print(
    "Requires further investigation:",
    4
)

print(
    "Validation rate of potential anomalies: 97.22%"
)

# Final interpretation
print("\n----- FINAL VALIDATION INTERPRETATION -----")

print(
    "The anomaly detection pipeline identified statistically unusual "
    "records and subsequently validated them using financial consistency "
    "and available reference information."
)

print(
    "Vendor-wise Z-score analysis was limited by insufficient repeated "
    "vendor observations in the dataset."
)

print(
    "Date and duplicate checks identified additional records requiring "
    "attention without modifying or fabricating the original data."
)

print(
    "\n========== STEP 47.4 COMPLETED ========== "
)

#=============================
# Step 47.5 : Final Project Conclusions
#=============================
print("\n========== Step 47.5 : FINAL PROJECT CONCLUSIONS ========== ")

# Final project conclusion
print("\n----- PROJECT ACHIEVEMENTS -----")

print(
    "The system successfully processed 499 invoice images "
    "through the OCR and automated invoice-processing pipeline."
)

print(
    "OCR processing achieved 100% coverage across the available "
    "invoice images."
)

print(
    "The system extracted 1805 invoice line-item records with "
    "97.39% invoice-level item extraction coverage."
)

print(
    "Financial field extraction achieved varying coverage depending "
    "on invoice layout and OCR quality."
)

print("\n----- ANOMALY DETECTION CONCLUSION -----")

print(
    "Isolation Forest analyzed 1205 complete financial records."
)

print(
    "The model identified 144 statistically unusual records, "
    "representing 11.95% of the analyzed records."
)

print(
    "After validation, 140 records were classified as statistically "
    "unusual and validated."
)

print(
    "Four records were classified as requiring further investigation."
)

print(
    "Seventeen records were assigned high investigation priority."
)

print("\n----- VALIDATION CONCLUSION -----")

print(
    "Financial consistency checks demonstrated high validation "
    "consistency for Net Worth and Gross Worth."
)

print(
    "Date validation identified missing or invalid dates and "
    "historically old dates without modifying the original data."
)

print(
    "Duplicate detection identified 21 potential duplicate records "
    "using exact and fuzzy matching techniques."
)

print(
    "Vendor-wise Z-score analysis was limited by insufficient "
    "repeated vendor observations."
)

print("\n----- IMPORTANT ANOMALY FINDING -----")

print(
    "The strongest detected anomaly was Invoice 28488969, "
    "Item 1, with Net Worth 84438.10 and Gross Worth 92881.91."
)

print(
    "Its Isolation Forest anomaly score was -0.329661."
)

print(
    "The anomalous Net Worth was approximately 41.57 times the "
    "overall average Net Worth."
)

print("\n----- PROJECT LIMITATIONS -----")

print(
    "Some invoice layouts produced incomplete financial extraction "
    "because of OCR and document-layout variability."
)

print(
    "Unrecoverable values were left blank instead of being fabricated."
)

print(
    "Vendor-wise statistical analysis was limited by insufficient "
    "historical observations for most vendors."
)

print(
    "Isolation Forest identifies statistical unusualness and does "
    "not independently prove fraud."
)

print("\n----- FINAL PROJECT ASSESSMENT -----")

print(
    "The project successfully implements an end-to-end automated "
    "invoice processing and anomaly detection workflow combining "
    "OCR, information extraction, rule-based validation, statistical "
    "analysis, duplicate detection, and unsupervised machine learning."
)

print(
    "\n========== STEP 47.5 COMPLETED ========== "
)

