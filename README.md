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
* Identify potential duplicate invoices and amounts.
* Generate a final CSV containing extracted information and anomaly analysis.

---

## 4. System Architecture

```text
                Invoice Image
                     │
                     ▼
          ┌─────────────────────┐
          │ Image Preprocessing │
          │      OpenCV         │
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
          │ Financial Rules     │
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

The parser also handles different financial layouts encountered in the invoice dataset.

### anomaly_detector.py

Contains the Machine Learning anomaly detection implementation.

The project uses **Isolation Forest**, an unsupervised learning algorithm that identifies observations that are easier to isolate from the rest of the data.

The model uses numerical invoice features including:

* Quantity
* Unit Price
* Net Worth
* Gross Worth

### validation.py

Performs rule-based financial validation.

Examples include:

```text
Quantity × Unit Price ≈ Net Worth

Net Worth × (1 + VAT / 100) ≈ Gross Worth
```

A small tolerance is used to account for decimal rounding.

### main.py

Connects the different modules and provides the main invoice-processing workflow.

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

The dataset contains invoice information and OCR-related data that can be used as reference during evaluation.

The accompanying JSON information was used only as **ground truth/reference for evaluation** and was not used to fill missing production fields.

---

## 9. Verified Production Results

The verified production workflow processed:

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

## 10. Anomaly Detection Results

Isolation Forest was applied to **1,205 complete financial records**.

The model identified:

* **1,061 normal records**
* **144 potential anomalies**
* **11.95% potential anomaly rate**

These records represent potentially unusual patterns and are not automatically considered fraudulent.

Further validation identified records requiring investigation.

---

## 11. Statistical Validation

### Vendor Z-Score

Z-score analysis was used to identify invoice values that are unusually far from the vendor's historical mean.

Because many vendors had insufficient repeated invoice history:

* 499 invoices were checked.
* 6 had sufficient repeated-vendor history.
* 493 had insufficient history.
* No record reached the `|Z| >= 3` threshold.

### Date Validation

The invoice dates were checked for unusual date conditions.

Results:

* 20 missing or invalid dates
* 468 invoices older than 90 days
* 0 future-dated invoices

### Duplicate Detection

Both exact and fuzzy matching were used to identify potentially duplicated records.

Results:

* Exact duplicate invoice numbers: 0
* Exact duplicate amounts: 10
* Fuzzy duplicate invoice numbers: 0
* Fuzzy duplicate amounts: 13
* Total potential duplicate records: 21

---

## 12. Investigation Cases

Four records require further investigation because their extracted financial values show stronger inconsistencies or unusual patterns.

| Invoice  | Item | Quantity | Unit Price | Net Worth | Gross Worth |
| -------- | ---: | -------: | ---------: | --------: | ----------: |
| 66053252 |    7 |        0 |   45150.30 |   4515.03 |     1315.51 |
| 60665075 |    5 |       25 |     174.25 |   4348.50 |     4783.35 |
| 97781335 |    2 |        2 |    3600.00 |   7200.00 |      920.00 |
| 52230192 |    1 |       30 |      17.50 |    517.50 |      569.25 |

These cases should be manually reviewed rather than automatically classified as fraud.

---

## 13. Visualizations

The project includes visual analysis of invoice financial data and anomaly results.

Available visualizations include:

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

---

## 14. Limitations

* OCR quality depends on invoice image quality and layout.
* Different invoice templates can produce different extraction results.
* Some financial fields may remain unavailable because of OCR or layout differences.
* Vendor-level Z-score analysis is limited when vendors have insufficient historical records.
* Isolation Forest requires complete numerical features for reliable model input.
* Fuzzy duplicate detection identifies potential similarity and requires human confirmation.
* Anomaly detection indicates unusual behavior and does not by itself prove fraud.
* Two negative VAT values remain in the production data and should be manually reviewed rather than being silently corrected.

---

## 15. Final Output

The verified production output is:

```text
reports/final_invoice_output.csv
```

The output contains extracted invoice and item information together with:

* `IsAnomaly`
* `AnomalyReason`
* `Anomaly Score`
* `Validation Result`
* `Investigation Priority`
* `Duplicate Status`
* `Duplicate Reason`

---

## 16. Conclusion

The project demonstrates a complete invoice-processing workflow combining Computer Vision, OCR, rule-based validation, data processing, statistical analysis, and unsupervised Machine Learning.

The system successfully processed 499 invoice images and produced 1,805 verified item-level records. Isolation Forest was used to identify potentially unusual invoice patterns without requiring pre-existing fraud labels.

The modular Python architecture separates image preprocessing, OCR, parsing, validation, anomaly detection, and pipeline execution, making the system easier to understand, maintain, and extend.

---

## 17. Future Scope

Possible improvements include:

* More advanced OCR models
* Named Entity Recognition for invoice fields
* Improved handling of new invoice layouts
* Larger vendor histories for statistical analysis
* Human-in-the-loop anomaly review
* Real-time invoice processing
* Web-based invoice processing interface
* Model monitoring and MLOps integration

```

### Now do exactly this

**Open `README.md` → select all → replace with the README above → save.**

Then tell me only:

**`README saved`**

and we immediately move to the **final GitHub/repository check**.
```
