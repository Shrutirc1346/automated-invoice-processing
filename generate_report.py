#=============================
# Step 48.29 : Generate Technical Report
#=============================
print("\n========== Step 48.29 : GENERATE TECHNICAL REPORT ========== ")

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)


#=============================
# Report Configuration
#=============================
OUTPUT_FILE = os.path.join(
    "reports",
    "technical_report.pdf"
)

VISUALIZATION_FOLDER = os.path.join(
    "reports",
    "visualizations"
)

os.makedirs(
    "reports",
    exist_ok=True
)


#=============================
# Report Styles
#=============================
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "ReportTitle",
    parent=styles["Title"],
    alignment=TA_CENTER,
    fontSize=18,
    leading=22,
    spaceAfter=12
)

subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=styles["Heading2"],
    alignment=TA_CENTER,
    fontSize=12,
    leading=15,
    spaceAfter=15
)

heading_style = ParagraphStyle(
    "Heading",
    parent=styles["Heading2"],
    fontSize=13,
    leading=16,
    spaceBefore=8,
    spaceAfter=7
)

body_style = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontSize=9.5,
    leading=13,
    spaceAfter=7
)

small_style = ParagraphStyle(
    "Small",
    parent=styles["BodyText"],
    fontSize=8,
    leading=10
)


#=============================
# Create PDF Document
#=============================
document = SimpleDocTemplate(
    OUTPUT_FILE,
    pagesize=A4,
    rightMargin=42,
    leftMargin=42,
    topMargin=42,
    bottomMargin=42
)

story = []


#=============================
# Title
#=============================
story.append(
    Paragraph(
        "Automated Invoice Processing and Anomaly Detection",
        title_style
    )
)

story.append(
    Paragraph(
        "Technical Project Report",
        subtitle_style
    )
)

story.append(
    Spacer(
        1,
        10
    )
)


#=============================
# Section 1 : Project Overview
#=============================
story.append(
    Paragraph(
        "1. Project Overview",
        heading_style
    )
)

story.append(
    Paragraph(
        "This project implements an end-to-end Machine Learning system "
        "for automated invoice processing and anomaly detection. Invoice "
        "images are processed using OpenCV, converted into text using "
        "Tesseract OCR, parsed into structured invoice and line-item "
        "information, validated using financial consistency rules, and "
        "analyzed for unusual records using statistical and unsupervised "
        "Machine Learning techniques.",
        body_style
    )
)


#=============================
# Section 2 : Problem Statement
#=============================
story.append(
    Paragraph(
        "2. Problem Statement",
        heading_style
    )
)

story.append(
    Paragraph(
        "Manual invoice processing requires significant time and may "
        "introduce errors during data entry, calculation, verification, "
        "and duplicate identification. The proposed system automates "
        "invoice information extraction and performs validation and "
        "anomaly screening to assist invoice-processing workflows.",
        body_style
    )
)


#=============================
# Section 3 : System Architecture
#=============================
story.append(
    Paragraph(
        "3. System Architecture",
        heading_style
    )
)

architecture_data = [
    [
        "Stage",
        "Technology / Module",
        "Purpose"
    ],
    [
        "Image preprocessing",
        "OpenCV / preprocessing.py",
        "Grayscale conversion, noise reduction and binarization"
    ],
    [
        "OCR",
        "Tesseract / ocr_engine.py",
        "Convert invoice image into machine-readable text"
    ],
    [
        "Information extraction",
        "Regex / parser.py",
        "Extract invoice, item and financial fields"
    ],
    [
        "Validation",
        "validation.py",
        "Check financial consistency rules"
    ],
    [
        "Anomaly detection",
        "Isolation Forest / anomaly_detector.py",
        "Identify unusual multivariate invoice records"
    ],
    [
        "Pipeline",
        "main.py",
        "Connect processing components"
    ]
]

architecture_table = Table(
    architecture_data,
    colWidths=[
        1.25 * inch,
        1.75 * inch,
        3.25 * inch
    ]
)

architecture_table.setStyle(
    TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold"
        ),
        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            8
        ),
        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "TOP"
        ),
        (
            "PADDING",
            (0, 0),
            (-1, -1),
            5
        )
    ])
)

story.append(
    architecture_table
)


#=============================
# Section 4 : Modular Codebase
#=============================
story.append(
    Paragraph(
        "4. Modular Codebase",
        heading_style
    )
)

story.append(
    Paragraph(
        "The project follows a modular Python architecture. "
        "<b>preprocessing.py</b> performs image cleanup, "
        "<b>ocr_engine.py</b> performs OCR, <b>parser.py</b> extracts "
        "structured invoice fields, <b>anomaly_detector.py</b> contains "
        "the Isolation Forest implementation, <b>validation.py</b> "
        "performs financial consistency checks, and <b>main.py</b> "
        "connects the processing workflow.",
        body_style
    )
)


#=============================
# Section 5 : Information Extraction
#=============================
story.append(
    Paragraph(
        "5. Information Extraction",
        heading_style
    )
)

story.append(
    Paragraph(
        "OCR converts the visual content of an invoice into machine-readable "
        "text. Regular Expressions (Regex) are then used to locate structured "
        "patterns within the OCR output. The parser extracts Invoice Number, "
        "Invoice Date, Vendor Name, Total Amount, Description, Quantity, "
        "Unit Price, Net Worth, VAT and Gross Worth.",
        body_style
    )
)

story.append(
    Paragraph(
        "Different invoice layouts can arrange financial information "
        "differently. The parser therefore includes handling for several "
        "financial layouts encountered in the dataset. The JSON information "
        "provided with the dataset was used only as reference ground truth "
        "for evaluation and was not used to fill production output fields.",
        body_style
    )
)


#=============================
# Section 6 : Verified Production Results
#=============================
story.append(
    Paragraph(
        "6. Verified Production Results",
        heading_style
    )
)

results_data = [
    [
        "Metric",
        "Result"
    ],
    [
        "Invoice images processed",
        "499"
    ],
    [
        "OCR processing coverage",
        "100%"
    ],
    [
        "Invoices with extracted items",
        "486 / 499 (97.39%)"
    ],
    [
        "Extracted item records",
        "1,805"
    ],
    [
        "Quantity extraction",
        "1,788 / 1,805 (99.06%)"
    ],
    [
        "Unit Price extraction",
        "1,574 / 1,805 (87.20%)"
    ],
    [
        "Net Worth extraction",
        "1,574 / 1,805 (87.20%)"
    ],
    [
        "VAT extraction",
        "1,574 / 1,805 (87.20%)"
    ],
    [
        "Gross Worth extraction",
        "1,699 / 1,805 (94.13%)"
    ],
    [
        "Records processed by production pipeline",
        "1,805"
    ],
    [
        "Valid financial records",
        "474"
    ],
    [
        "Invalid or incomplete financial records",
        "1,331"
    ]
]

results_table = Table(
    results_data,
    colWidths=[
        3.7 * inch,
        2.55 * inch
    ]
)

results_table.setStyle(
    TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold"
        ),
        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            8.5
        ),
        (
            "PADDING",
            (0, 0),
            (-1, -1),
            5
        )
    ])
)

story.append(
    results_table
)


#=============================
# Section 7 : Rule-Based Validation
#=============================
story.append(
    Paragraph(
        "7. Rule-Based Validation",
        heading_style
    )
)

story.append(
    Paragraph(
        "Rule-based validation checks whether extracted financial values "
        "are internally consistent. For each line item, Quantity multiplied "
        "by Unit Price is compared with Net Worth. Net Worth and VAT are then "
        "used to calculate the expected Gross Worth. A tolerance of 0.01 is "
        "used to account for normal decimal rounding.",
        body_style
    )
)

story.append(
    Paragraph(
        "Additional checks include invoice-date logic and duplicate detection. "
        "These rules help distinguish records that are mathematically "
        "consistent from records that require further investigation.",
        body_style
    )
)

story.append(
    Paragraph(
        "The current production output contains 474 valid financial records "
        "and 1,331 invalid or incomplete records. An invalid validation result "
        "does not automatically indicate fraud because missing or incorrectly "
        "extracted OCR fields can cause financial consistency checks to fail.",
        body_style
    )
)


#=============================
# Section 8 : Isolation Forest
#=============================
story.append(
    Paragraph(
        "8. Anomaly Detection Using Isolation Forest",
        heading_style
    )
)

story.append(
    Paragraph(
        "Isolation Forest is an unsupervised Machine Learning algorithm "
        "designed to identify unusual observations. It works by repeatedly "
        "partitioning observations; unusual observations generally require "
        "fewer partitions to isolate.",
        body_style
    )
)

story.append(
    Paragraph(
        "Isolation Forest was selected because the project does not have "
        "reliable pre-labelled fraud data. It can identify unusual "
        "multivariate numerical behavior and is suitable for screening "
        "potentially anomalous invoice records.",
        body_style
    )
)

story.append(
    Paragraph(
        "The current production pipeline generated anomaly predictions for "
        "1,805 item-level records using Quantity, Unit Price, Net Worth and "
        "Gross Worth. The output identified 1,543 records as normal and "
        "262 records as potential anomalies, corresponding to an anomaly "
        "rate of 14.52%. These records represent potential anomalies and "
        "should not automatically be interpreted as confirmed fraud.",
        body_style
    )
)


#=============================
# Section 9 : Statistical Checks
#=============================
story.append(
    Paragraph(
        "9. Statistical and Duplicate Checks",
        heading_style
    )
)

checks_data = [
    [
        "Check",
        "Result"
    ],
    [
        "Vendor Z-score",
        "499 checked; 6 had sufficient repeated-vendor history; "
        "0 reached |Z| >= 3"
    ],
    [
        "Date logic",
        "20 missing/invalid; 468 older than 90 days; "
        "0 future dates"
    ],
    [
        "Exact duplicate invoice numbers",
        "0"
    ],
    [
        "Exact duplicate amounts",
        "10"
    ],
    [
        "Fuzzy duplicate invoice numbers",
        "0"
    ],
    [
        "Fuzzy duplicate amounts",
        "13"
    ],
    [
        "Total potential duplicate records",
        "21"
    ]
]

checks_table = Table(
    checks_data,
    colWidths=[
        2.55 * inch,
        3.7 * inch
    ]
)

checks_table.setStyle(
    TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold"
        ),
        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            8
        ),
        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "TOP"
        ),
        (
            "PADDING",
            (0, 0),
            (-1, -1),
            5
        )
    ])
)

story.append(
    checks_table
)


#=============================
# Section 10 : Investigation Cases
#=============================
story.append(
    Paragraph(
        "10. Investigation Cases",
        heading_style
    )
)

story.append(
    Paragraph(
        "Four records require further investigation because their extracted "
        "financial values show stronger inconsistencies or unusual patterns.",
        body_style
    )
)

investigation_data = [
    [
        "Invoice",
        "Item",
        "Quantity",
        "Unit Price",
        "Net Worth",
        "Gross Worth"
    ],
    [
        "66053252",
        "7",
        "0",
        "45150.30",
        "4515.03",
        "1315.51"
    ],
    [
        "60665075",
        "5",
        "25",
        "174.25",
        "4348.50",
        "4783.35"
    ],
    [
        "97781335",
        "2",
        "2",
        "3600.00",
        "7200.00",
        "920.00"
    ],
    [
        "52230192",
        "1",
        "30",
        "17.50",
        "517.50",
        "569.25"
    ]
]

investigation_table = Table(
    investigation_data,
    colWidths=[
        0.95 * inch,
        0.5 * inch,
        0.75 * inch,
        1.05 * inch,
        1.05 * inch,
        1.05 * inch
    ]
)

investigation_table.setStyle(
    TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold"
        ),
        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            7.5
        ),
        (
            "PADDING",
            (0, 0),
            (-1, -1),
            4
        )
    ])
)

story.append(
    investigation_table
)

story.append(
    Spacer(
        1,
        8
    )
)

story.append(
    Paragraph(
        "Two negative VAT values remain in the production data. They were "
        "not silently corrected because the available OCR evidence is "
        "insufficient to determine the intended values. They should remain "
        "flagged for manual review.",
        body_style
    )
)


#=============================
# Section 11 : Visual Analysis
#=============================
story.append(
    Paragraph(
        "11. Visual Analysis",
        heading_style
    )
)

visualizations = [
    "unit_price_boxplot.png",
    "net_worth_boxplot.png",
    "gross_worth_boxplot.png",
    "net_worth_vs_gross_worth_anomalies.png",
    "anomaly_score_vs_net_worth.png",
    "anomaly_distribution.png",
    "anomaly_validation_status.png"
]

for filename in visualizations:

    image_path = os.path.join(
        VISUALIZATION_FOLDER,
        filename
    )

    if os.path.exists(image_path):

        story.append(
            Paragraph(
                filename.replace(
                    ".png",
                    ""
                ).replace(
                    "_",
                    " "
                ).title(),
                small_style
            )
        )

        image = Image(
            image_path
        )

        image.drawWidth = 5.7 * inch
        image.drawHeight = 1.55 * inch

        story.append(
            image
        )

        story.append(
            Spacer(
                1,
                5
            )
        )


#=============================
# Section 12 : Limitations
#=============================
story.append(
    PageBreak()
)

story.append(
    Paragraph(
        "12. Limitations",
        heading_style
    )
)

story.append(
    Paragraph(
        "The production validation results include invalid or incomplete "
        "records caused by missing or inaccurate OCR extraction. Such records "
        "require review and should not be treated as confirmed fraudulent "
        "invoices.",
        body_style
    )
)

story.append(
    Paragraph(
        "OCR accuracy depends on invoice image quality and document layout. "
        "Different invoice templates can produce different extraction results. "
        "Some financial fields may remain unavailable because of OCR errors or "
        "layout differences.",
        body_style
    )
)

story.append(
    Paragraph(
        "Vendor-level Z-score analysis is limited when vendors have insufficient "
        "historical records. Isolation Forest requires complete numerical "
        "features for model analysis. Fuzzy duplicate detection identifies "
        "potential similarity and therefore requires human confirmation.",
        body_style
    )
)

story.append(
    Paragraph(
        "Anomaly detection identifies unusual behavior and does not by itself "
        "prove fraud.",
        body_style
    )
)


#=============================
# Section 13 : Conclusion
#=============================
story.append(
    Paragraph(
        "13. Conclusion",
        heading_style
    )
)

story.append(
    Paragraph(
        "The project demonstrates an automated invoice-processing workflow "
        "combining Computer Vision, OCR, Regex-based information extraction, "
        "rule-based validation, statistical analysis, duplicate detection and "
        "unsupervised Machine Learning.",
        body_style
    )
)

story.append(
    Paragraph(
        "The verified production workflow processed 499 invoice images and "
        "produced 1,805 item-level records. The current production anomaly "
        "detection output identified 1,543 normal records and 262 potential "
        "anomalies, corresponding to an anomaly rate of 14.52%.",
        body_style
    )
)

story.append(
    Paragraph(
        "The modular Python architecture separates image preprocessing, OCR, "
        "parsing, validation, anomaly detection and pipeline execution, "
        "satisfying the required modular codebase structure.",
        body_style
    )
)


#=============================
# Section 14 : Final Deliverables
#=============================
story.append(
    Paragraph(
        "14. Final Deliverables",
        heading_style
    )
)

story.append(
    Paragraph(
        "The repository contains preprocessing.py for image preprocessing, "
        "ocr_engine.py for OCR, parser.py for information extraction, "
        "anomaly_detector.py for the Machine Learning implementation, "
        "validation.py for financial validation, and main.py for pipeline "
        "integration.",
        body_style
    )
)

story.append(
    Paragraph(
        "The verified production CSV is stored at "
        "<b>reports/final_invoice_output.csv</b>. Project visualizations "
        "are stored in <b>reports/visualizations/</b>.",
        body_style
    )
)


#=============================
# Build PDF
#=============================
document.build(
    story
)

print(
    "\nTechnical report created successfully:"
)

print(
    OUTPUT_FILE
)

print(
    "\n========== STEP 48.29 COMPLETED =========="
)