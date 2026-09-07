# Automated Invoice Processing and Anomaly Detection

## 1. Project Overview

This project is an end-to-end Machine Learning system for automated invoice processing and anomaly detection.

The system takes invoice images as input and performs:

**Invoice Image → Image Preprocessing → OCR → Information Extraction → Validation → Anomaly Detection → Final CSV**

The objective is to reduce manual invoice processing and identify potentially unusual or inconsistent invoice records.

---

## 2. Problem Statement

Manual invoice processing requires significant time and can introduce human errors during data entry, calculation, and verification.

This project automates the extraction of important invoice information and performs automated checks to identify potentially anomalous records.

---

## 3. Objectives

* Extract text from invoice images using OCR.
* Extract invoice and line-item information automatically.
* Convert unstructured invoice information into structured tabular data.
* Validate financial relationships between invoice fields.
* Detect unusual invoice records using unsupervised Machine Learning.
* Identify potential duplicate invoices and amounts through statistical analysis.
* Generate a final CSV containing extracted information, validation results, and anomaly analysis.
* Generate visualizations for financial distributions and anomaly analysis.

---

## 4. System Architecture

```text
                Invoice Image
                     │
                     ▼
          ┌─────────────────────┐
          │ Image Preprocessing │
          │       OpenCV        │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │        OCR          │
          │     Tesseract       │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Information Parsing │
          │       Regex         │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │     Validation      │
          │   Financial Rules   │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Anomaly Detection   │
          │  Isolation Forest   │
          └──────────┬──────────┘
                     │
                     ▼
                Final CSV
```

---

## 5. Project Structure

```text
automated_invoice_processing/
│
├── data/
│   └── raw/
│       └── invoice_dataset/
│
├── reports/
│   ├── final_invoice_output.csv
│   ├── technical_report.pdf
│   └── visualizations/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── ocr_engine.py
│   ├── parser.py
│   ├── anomaly_detector.py
│   ├── validation.py
│   └── main.py
│
├── tests/
│
├── generate_visualizations.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 6. Core Modules

### preprocessing.py

Responsible for image preprocessing before OCR.

Operations include:

* Grayscale conversion
* Gaussian noise reduction
* Image binarization using Otsu thresholding

The purpose is to improve the quality of the image supplied to the OCR engine.

### ocr_engine.py

Responsible for extracting machine-readable text from invoice images.

The project uses **Tesseract OCR** to convert the processed invoice image into text.

OCR timeout handling is also included so that an individual problematic invoice does not unnecessarily stop the complete processing workflow.

### parser.py

Responsible for extracting structured information from OCR text.

Regex-based parsing is used to identify:

* Invoice Number
* Invoice Date
* Vendor Name
* Total Amount
* Description
* Quantity
* Unit Price
* Net Worth
* VAT
* Gross Worth

The parser also handles different financial layouts encountered in the invoice dataset, including single-line and separated financial layouts.

### validation.py

Performs rule-based financial validation.

Examples include:

```text
Quantity × Unit Price ≈ Net Worth

Net Worth × (1 + VAT / 100) ≈ Gross Worth
```

A small tolerance is used to account for decimal rounding.

The validation module produces calculated values and validation status for the extracted financial records.

### anomaly_detector.py

Contains the Machine Learning anomaly detection implementation.

The project uses **Isolation Forest**, an unsupervised learning algorithm that identifies observations that are easier to isolate from the rest of the data.

The model uses numerical invoice features including:

* Quantity
* Unit Price
* Net Worth
* Gross Worth

The detector produces:

* Prediction
* Anomaly Score
* IsAnomaly
* AnomalyReason

### main.py

Connects the different modules and provides the main invoice-processing workflow.

The pipeline processes invoice images, performs OCR, extracts invoice and item information, applies financial validation, and performs batch-level anomaly detection.

### generate_visualizations.py

Generates final visualizations from the production CSV.

The script creates:

* Unit Price Box Plot
* Net Worth Box Plot
* Gross Worth Box Plot
* Net Worth vs Gross Worth Anomalies
* Anomaly Score vs Net Worth
* Anomaly Distribution
* Anomaly Validation Status

---

## 7. Technologies Used

| Technology          | Purpose                   |
| ------------------- | ------------------------- |
| Python              | Main programming language |
| OpenCV              | Image preprocessing       |
| Tesseract OCR       | Text extraction           |
| Pandas              | Data processing           |
| NumPy               | Numerical operations      |
| Regular Expressions | Information extraction    |
| Scikit-learn        | Machine Learning          |
| Isolation Forest    | Anomaly detection         |
| Matplotlib          | Data visualization        |

---

## 8. Dataset

The project uses a dataset containing **499 invoice images**.

The accompanying JSON information contains reference invoice and item information.

The JSON data was used only as **ground truth/reference for evaluation** and was not used to fill missing production fields.

The production pipeline obtains its extracted information from the invoice image through:

**Image → Preprocessing → OCR → Parsing**

---

## 9. Verified Production Results

The current modular production workflow processed:

* **499 invoice images**
* **100% OCR processing coverage**
* **486 invoices with extracted items**
* **1,805 extracted item records**

### Field Extraction

| Field       |      Extraction Result |
| ----------- | ---------------------: |
| Quantity    | 1,788 / 1,805 — 99.06% |
| Unit Price  | 1,574 / 1,805 — 87.20% |
| Net Worth   | 1,574 / 1,805 — 87.20% |
| VAT         | 1,574 / 1,805 — 87.20% |
| Gross Worth | 1,699 / 1,805 — 94.13% |

The difference between fields is mainly caused by variations in invoice layouts and OCR quality.

---

## 10. Current Validation Results

The final production CSV contains **1,805 item-level records**.

The current validation workflow produced:

* **474 valid financial records**
* **1,331 invalid or incomplete financial records**

An invalid validation result does **not automatically indicate fraud**.

A record may fail validation because one or more financial fields were missing or incorrectly extracted by OCR. Therefore, validation results must be interpreted together with the extracted data and anomaly results.

---

## 11. Current Anomaly Detection Results

Isolation Forest was applied to the production dataset after financial feature extraction.

The current production output contains:

* **1,805 total records**
* **1,543 normal records**
* **262 potential anomalies**
* **14.52% potential anomaly rate**

The calculation is:

```text
262 / 1805 × 100 ≈ 14.52%
```

These records represent potentially unusual numerical patterns and are **not automatically considered fraudulent**.

An anomaly score indicates how unusual a record is according to the Isolation Forest model. Records identified as anomalies should be reviewed together with their validation results and extracted financial values.

---

## 12. Statistical and Additional Analysis

Additional statistical analysis was performed during project evaluation.

### Vendor Z-Score

Z-score analysis was used to identify invoice values that are unusually far from a vendor's historical mean.

Because many vendors had insufficient repeated invoice history:

* 499 invoices were checked.
* 6 had sufficient repeated-vendor history.
* 493 had insufficient history.
* No record reached the `|Z| >= 3` threshold.

### Date Validation

Invoice dates were checked for unusual date conditions.

Results:

* 20 missing or invalid dates
* 468 invoices older than 90 days
* 0 future-dated invoices

### Duplicate Detection

Exact and fuzzy matching were used during additional analysis to identify potentially duplicated invoice information.

Results:

* Exact duplicate invoice numbers: 0
* Exact duplicate amounts: 10
* Fuzzy duplicate invoice numbers: 0
* Fuzzy duplicate amounts: 13
* Total potential duplicate records: 21

These results represent potential duplicate patterns and require further review before being treated as confirmed duplicates.

---

## 13. Investigation Cases

Four records were identified during the earlier detailed anomaly investigation because their extracted financial values showed stronger inconsistencies or unusual patterns.

| Invoice  | Item | Quantity | Unit Price | Net Worth | Gross Worth |
| -------- | ---: | -------: | ---------: | --------: | ----------: |
| 66053252 |    7 |        0 |   45150.30 |   4515.03 |     1315.51 |
| 60665075 |    5 |       25 |     174.25 |   4348.50 |     4783.35 |
| 97781335 |    2 |        2 |    3600.00 |   7200.00 |      920.00 |
| 52230192 |    1 |       30 |      17.50 |    517.50 |      569.25 |

These cases should be manually reviewed rather than automatically classified as fraud.

---

## 14. Visualizations

The final production workflow includes seven visualizations:

* Unit Price Box Plot
* Net Worth Box Plot
* Gross Worth Box Plot
* Net Worth vs Gross Worth Anomalies
* Anomaly Score vs Net Worth
* Anomaly Distribution
* Anomaly Validation Status

All visualizations are available in:

```text
reports/visualizations/
```

They are generated using:

```text
generate_visualizations.py
```

---

## 15. Final Production CSV

The final production output is:

```text
reports/final_invoice_output.csv
```

The current CSV contains **1,805 records and 21 columns**.

The output includes:

* `Image Name`
* `Invoice Number`
* `Invoice Date`
* `Vendor Name`
* `Total Amount`
* `Item Number`
* `Description`
* `Quantity`
* `Unit Price`
* `Net Worth`
* `VAT`
* `Gross Worth`
* `Calculated Net Worth`
* `Net Worth Valid`
* `Calculated Gross Worth`
* `Gross Worth Valid`
* `Validation Result`
* `Prediction`
* `Anomaly Score`
* `IsAnomaly`
* `AnomalyReason`

---

## 16. Limitations

* OCR quality depends on invoice image quality and layout.
* Different invoice templates can produce different extraction results.
* Some financial fields may remain unavailable because of OCR or layout differences.
* Vendor-level Z-score analysis is limited when vendors have insufficient historical records.
* Isolation Forest requires complete numerical features for reliable model input.
* Fuzzy duplicate detection identifies potential similarity and requires human confirmation.
* Anomaly detection indicates unusual behavior and does not by itself prove fraud.
* Validation failures can result from incomplete or incorrect OCR extraction.
* Two negative VAT values remain in the production data and should be manually reviewed rather than being silently corrected.

---

## 17. Conclusion

The project demonstrates a complete invoice-processing workflow combining Computer Vision, OCR, rule-based validation, data processing, statistical analysis, and unsupervised Machine Learning.

The current modular production pipeline successfully processed **499 invoice images** and produced **1,805 item-level records**.

The system combines:

**OpenCV → Tesseract OCR → Regex Parsing → Financial Validation → Isolation Forest → Final CSV**

Isolation Forest is used to identify potentially unusual invoice patterns without requiring pre-existing fraud labels.

The modular Python architecture separates image preprocessing, OCR, parsing, validation, anomaly detection, and pipeline execution, making the system easier to understand, maintain, and extend.

---

## 18. Future Scope

Possible improvements include:

* More advanced OCR models
* Named Entity Recognition for invoice fields
* Improved handling of new invoice layouts
* Larger vendor histories for statistical analysis
* Human-in-the-loop anomaly review
* Real-time invoice processing
* Web-based invoice processing interface
* Model monitoring and MLOps integration
* Improved anomaly explanations
* Larger and more diverse invoice datasets
