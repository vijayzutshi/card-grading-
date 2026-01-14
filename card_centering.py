import cv2
import numpy as np


# ---------------------------------------------------------
# 1. Detect outer border of the card
# ---------------------------------------------------------
def detect_card_border(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found — cannot detect card border.")

    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(approx)
    return x, y, w, h


# ---------------------------------------------------------
# 2. Detect inner artwork rectangle
# ---------------------------------------------------------
def detect_artwork_rectangle(image, outer_box):
    ox, oy, ow, oh = outer_box
    card_region = image[oy:oy+oh, ox:ox+ow]

    gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        raise ValueError("Not enough contours to detect artwork.")

    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    artwork_contour = contours_sorted[1]

    peri = cv2.arcLength(artwork_contour, True)
    approx = cv2.approxPolyDP(artwork_contour, 0.02 * peri, True)
    ax, ay, aw, ah = cv2.boundingRect(approx)

    return ox + ax, oy + ay, aw, ah


# ---------------------------------------------------------
# 3. Compute border thickness
# ---------------------------------------------------------
def compute_border_thickness(outer, inner):
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner

    left = ix - ox
    right = (ox + ow) - (ix + iw)
    top = iy - oy
    bottom = (oy + oh) - (iy + ih)

    return {
        "left": float(left),
        "right": float(right),
        "top": float(top),
        "bottom": float(bottom)
    }


# ---------------------------------------------------------
# 4. Compute PSA-style axis ratio
# ---------------------------------------------------------
def compute_psa_axis_ratio(border_a, border_b):
    total = border_a + border_b
    if total <= 0:
        return 0.0

    worse = min(border_a, border_b)
    ratio = worse / total
    return ratio * 100.0


def compute_psa_centering_ratio_from_borders(borders):
    left = borders["left"]
    right = borders["right"]
    top = borders["top"]
    bottom = borders["bottom"]

    horizontal_ratio = compute_psa_axis_ratio(left, right)
    vertical_ratio = compute_psa_axis_ratio(top, bottom)

    limiting_ratio = min(horizontal_ratio, vertical_ratio)

    details = {
        "horizontal_ratio_percent": horizontal_ratio,
        "vertical_ratio_percent": vertical_ratio,
        "limiting_ratio_percent": limiting_ratio
    }

    return limiting_ratio, details


# ---------------------------------------------------------
# 5. Map PSA ratios to centering grade
# ---------------------------------------------------------
def map_psa_centering_grade(front_ratio, back_ratio):
    if front_ratio >= 45.0 and back_ratio >= 25.0:
        return 10
    if front_ratio >= 40.0 and back_ratio >= 20.0:
        return 9
    if front_ratio >= 35.0 and back_ratio >= 15.0:
        return 8
    if front_ratio >= 30.0 and back_ratio >= 10.0:
        return 7
    if front_ratio >= 25.0 and back_ratio >= 10.0:
        return 6
    if front_ratio >= 20.0:
        return 5
    return 4


# ---------------------------------------------------------
# 6. FINAL FUNCTION (with new ratio fields added)
# ---------------------------------------------------------
def analyze_centering(front_path: str, back_path: str) -> dict:

    # ---------------- FRONT ----------------
    front_img = cv2.imread(front_path)
    if front_img is None:
        raise FileNotFoundError(f"Could not read front image: {front_path}")

    front_outer = detect_card_border(front_img)
    front_inner = detect_artwork_rectangle(front_img, front_outer)
    front_borders = compute_border_thickness(front_outer, front_inner)
    front_center_ratio, front_ratio_details = compute_psa_centering_ratio_from_borders(front_borders)

    # ---------------- BACK ----------------
    back_img = cv2.imread(back_path)
    if back_img is None:
        raise FileNotFoundError(f"Could not read back image: {back_path}")

    back_outer = detect_card_border(back_img)
    back_inner = detect_artwork_rectangle(back_img, back_outer)
    back_borders = compute_border_thickness(back_outer, back_inner)
    back_center_ratio, back_ratio_details = compute_psa_centering_ratio_from_borders(back_borders)

    # ---------------- PSA GRADE ----------------
    centering_grade = map_psa_centering_grade(front_center_ratio, back_center_ratio)

    # ---------------- NEW FIELDS ----------------
    # Front
    front_left = 100 - front_center_ratio
    front_right = front_center_ratio
    frontRatioString = f"{int(front_left)}/{int(front_right)}"
    frontRatioValue = front_right / 100.0

    # Back
    back_left = 100 - back_center_ratio
    back_right = back_center_ratio
    backRatioString = f"{int(back_left)}/{int(back_right)}"
    backRatioValue = back_right / 100.0

    # ---------------- RESULT ----------------
    result = {
        "front": {
            "outer_border": front_outer,
            "artwork_rectangle": front_inner,
            "borders": front_borders,
            "ratios": front_ratio_details,
            "psa_centering_ratio": front_center_ratio
        },
        "back": {
            "outer_border": back_outer,
            "artwork_rectangle": back_inner,
            "borders": back_borders,
            "ratios": back_ratio_details,
            "psa_centering_ratio": back_center_ratio
        },
        "summary": {
            "frontCenteringRatio": front_center_ratio,
            "backCenteringRatio": back_center_ratio,
            "centeringGrade": centering_grade,

            # NEW FIELDS FOR FLUTTERFLOW
            "frontRatioString": frontRatioString,
            "backRatioString": backRatioString,
            "frontRatioValue": frontRatioValue,
            "backRatioValue": backRatioValue
        }
    }

    return result
