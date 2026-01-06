import cv2
import numpy as np


# ---------------------------------------------------------
# 1. Detect outer border of the card
# ---------------------------------------------------------
def detect_card_border(image):
    """
    Detect the outer border of the card using edge detection and contour analysis.
    Returns the bounding rectangle (x, y, w, h).
    """

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
    """
    Detect the inner artwork rectangle inside the card.
    Returns bounding rectangle (x, y, w, h).
    """

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
    """
    Compute left, right, top, bottom border thickness.
    """

    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner

    left = ix - ox
    right = (ox + ow) - (ix + iw)
    top = iy - oy
    bottom = (oy + oh) - (iy + ih)

    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom
    }


# ---------------------------------------------------------
# 4. Compute centering ratios
# ---------------------------------------------------------
def compute_centering_ratios(borders):
    """
    Compute horizontal and vertical centering ratios.
    """

    left = borders["left"]
    right = borders["right"]
    top = borders["top"]
    bottom = borders["bottom"]

    horizontal_ratio = min(left, right) / max(left, right) if max(left, right) > 0 else 0
    vertical_ratio = min(top, bottom) / max(top, bottom) if max(top, bottom) > 0 else 0

    return {
        "horizontal_ratio": horizontal_ratio,
        "vertical_ratio": vertical_ratio
    }


# ---------------------------------------------------------
# 5. Map ratios to PSA grade
# ---------------------------------------------------------
def map_to_psa_grade(horizontal_ratio, vertical_ratio, is_back=False):
    """
    Map centering ratios to PSA grade based on front/back rules.
    """

    if not is_back:
        thresholds = [
            (10, 0.818),
            (9, 0.666),
            (8, 0.538),
            (7, 0.428),
        ]
    else:
        thresholds = [
            (10, 0.333),
            (9, 0.111),
            (8, 0.052),
            (7, 0.0),
        ]

    limiting_ratio = min(horizontal_ratio, vertical_ratio)

    for grade, min_ratio in thresholds:
        if limiting_ratio >= min_ratio:
            return grade

    return 1


# ---------------------------------------------------------
# 6. FINAL INTEGRATED FUNCTION FOR YOUR APP
# ---------------------------------------------------------
def analyze_centering(front_path: str, back_path: str) -> dict:
    """
    Full centering analysis for front and back images.
    Returns a dictionary ready for FlutterFlow or API use.
    """

    # ---------------- FRONT ----------------
    front_img = cv2.imread(front_path)
    if front_img is None:
        raise FileNotFoundError(f"Could not read front image: {front_path}")

    front_outer = detect_card_border(front_img)
    front_inner = detect_artwork_rectangle(front_img, front_outer)
    front_borders = compute_border_thickness(front_outer, front_inner)
    front_ratios = compute_centering_ratios(front_borders)
    front_grade = map_to_psa_grade(
        front_ratios["horizontal_ratio"],
        front_ratios["vertical_ratio"],
        is_back=False
    )

    # ---------------- BACK ----------------
    back_img = cv2.imread(back_path)
    if back_img is None:
        raise FileNotFoundError(f"Could not read back image: {back_path}")

    back_outer = detect_card_border(back_img)
    back_inner = detect_artwork_rectangle(back_img, back_outer)
    back_borders = compute_border_thickness(back_outer, back_inner)
    back_ratios = compute_centering_ratios(back_borders)
    back_grade = map_to_psa_grade(
        back_ratios["horizontal_ratio"],
        back_ratios["vertical_ratio"],
        is_back=True
    )

    # ---------------- RESULT ----------------
    return {
        "front": {
            "outer_border": front_outer,
            "artwork_rectangle": front_inner,
            "borders": front_borders,
            "ratios": front_ratios,
            "psa_grade": front_grade
        },
        "back": {
            "outer_border": back_outer,
            "artwork_rectangle": back_inner,
            "borders": back_borders,
            "ratios": back_ratios,
            "psa_grade": back_grade
        }
    }
