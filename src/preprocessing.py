#=============================
# Step 48.4 : Build Image Preprocessing Module
#=============================
print("\n========== Step 48.4 : BUILD IMAGE PREPROCESSING MODULE ========== ")

import cv2


def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Invoice image could not be loaded: {image_path}"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, binary = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


print("\nPreprocessing module loaded successfully.")
print("Function available: preprocess_image()")

print("\n========== STEP 48.4 COMPLETED ========== ")