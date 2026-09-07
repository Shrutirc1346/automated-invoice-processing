#=============================
# Step 48.12A : Add OCR Timeout Protection
#=============================
print("\n========== Step 48.12A : ADD OCR TIMEOUT PROTECTION ========== ")

import os
import shutil
import pytesseract

from src.preprocessing import preprocess_image


TESSERACT_PATH = shutil.which(
    "tesseract"
)


if TESSERACT_PATH:

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

else:

    WINDOWS_TESSERACT_PATH = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe\tesseract.exe"
    )

    if os.path.exists(
        WINDOWS_TESSERACT_PATH
    ):

        pytesseract.pytesseract.tesseract_cmd = (
            WINDOWS_TESSERACT_PATH
        )

    else:

        raise FileNotFoundError(
            "Tesseract OCR was not found."
        )


#=============================
# Step 48.12C : Handle OCR Timeout Safely
#=============================
print("\n========== Step 48.12C : HANDLE OCR TIMEOUT SAFELY ========== ")

import pytesseract

from src.preprocessing import preprocess_image


TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def extract_text(image_path):

    processed_image = preprocess_image(image_path)

    try:

        text = pytesseract.image_to_string(
            processed_image,
            timeout=30
        )

        return text

    except RuntimeError as error:

        print(
            f"\nOCR timeout/error for {image_path}: {error}"
        )

        return ""


print("\nOCR timeout handling added successfully.")
print("Function available: extract_text()")

print("\n========== STEP 48.12C COMPLETED ========== ")