import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pytesseract

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




#======================================================================================
# Step 40.15 - 40.19 : Quantity Recovery, Financial Validation and Final Test Dataset
#======================================================================================
print("\n========== Step 40.15 - 40.19 : FINAL EXTRACTION VALIDATION ========== ")

# ============================================================
# STEP 40.15 : QUANTITY RECOVERY
# ============================================================

print("\n========== Step 40.15 : Quantity Recovery ========== ")

# This list will store the final recovered item records
recovered_item_records = []

# Process only the currently selected test invoices
for image_name in test_images:

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
    "Total test items:",
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
        test_images
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

for image_name in test_images:

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
    "Test invoices:",
    len(test_images)
)

print(
    "Test items:",
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


# =====================================================
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


#=========================================================
# Step 40.13A : TEST COMPLETE UNNUMBERED ITEM EXTRACTION
#=========================================================
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


#=============================================================
# Step 40.15A : TEST QUANTITY RECOVERY FROM FINANCIAL VALUES
#=============================================================
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


#=====================================================
# Step 40.15C : CHECK QUANTITY FINANCIAL CONSISTENCY
#=====================================================
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



#=======================================================
# Step 40.12A : TEST UNNUMBERED DESCRIPTION EXTRACTION
#=======================================================
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