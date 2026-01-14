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

    # Convert back to full-image coordinates
    return ox + ax, oy + ay, aw, ah


# ---------------------------------------------------------
# 3. Compute border thickness
# ---------------------------------------------------------
def compute_border_thickness(outer, inner):
    """
    Compute left, right, top, bottom border thickness in pixels.
    """

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
# 4. Compute PSA-style axis ratio from borders
# ---------------------------------------------------------
def compute_psa_axis_ratio(border_a, border_b):
    """
    Compute PSA-style centering percentage for one axis (horizontal or vertical).

    PSA cares about the worse (smaller) side compared to the total of both sides.
    Example:
        left = 55, right = 45 → worse side = 45 → ratio = 45 / (55 + 45) = 0.45 → 45%
        This corresponds to 55/45 front centering.
    """

    total = border_a + border_b
    if total <= 0:
        return 0.0

    worse = min(border_a, border_b)
    ratio = worse / total  # 0.0–0.5
    return ratio * 100.0   # percentage, e.g. 45.0 for 55/45


def compute_psa_centering_ratio_from_borders(borders):
    """
    Compute a single PSA-style centering ratio for the card side (front or back),
    based on the worst axis (horizontal vs vertical).

    Returns:
        ratio_percent: float  # e.g. 45.0 meaning 55/45
        details: dict with horizontal/vertical ratios for debugging/analysis
    """

    left = borders["left"]
    right = borders["right"]
    top = borders["top"]
    bottom = borders["bottom"]

    horizontal_ratio = compute_psa_axis_ratio(left, right)
    vertical_ratio = compute_psa_axis_ratio(top, bottom)

    # PSA effectively uses the worse axis as the limiting factor
    limiting_ratio = min(horizontal_ratio, vertical_ratio)

    details = {
        "horizontal_ratio_percent": horizontal_ratio,
        "vertical_ratio_percent": vertical_ratio,
        "limiting_ratio_percent": limiting_ratio
    }

    return limiting_ratio, details


# ---------------------------------------------------------
# 5. Map PSA ratios to overall centering grade (front + back)
# ---------------------------------------------------------
def map_psa_centering_grade(front_ratio, back_ratio):
    """
    Map PSA-style centering ratios (front/back) to a PSA centering grade 1–10.

    front_ratio, back_ratio:
        - These are the worse-side percentages in PSA format.
        - Example: 45.0 → 55/45, 40.0 → 60/40, etc.

    PSA Front (worse side %):
        10: 45 or better (55/45 or better)
        9 : 40 or better (60/40 or better)
        8 : 35 or better (65/35 or better)
        7 : 30 or better (70/30 or better)
        6 : 25 or better (75/25 or better)
        5 : 20 or better (80/20 or better)

    PSA Back (worse side %):
        10: 25 or better (75/25 or better)
        9 : 20 or better (80/20 or better)
        8 : 15 or better (85/15 or better)
        7 : 10 or better (90/10 or better)
        6 : 10 or better (90/10 or better)
        5 : 10 or better (90/10 or better, but front already limits heavily)
    """

    # PSA 10
    if front_ratio >= 45.0 and back_ratio >= 25.0:
        return 10

    # PSA 9
    if front_ratio >= 40.0 and back_ratio >= 20.0:
        return 9

    # PSA 8
    if front_ratio >= 35.0 and back_ratio >= 15.0:
        return 8

    # PSA 7
    if front_ratio >= 30.0 and back_ratio >= 10.0:
        return 7

    # PSA 6
    if front_ratio >= 25.0 and back_ratio >= 10.0:
        return 6

    # PSA 5
    if front_ratio >= 20.0:
        return 5

    # Anything worse is 4 or below (you can refine further if you want)
    return 4


# ---------------------------------------------------------
# 6. FINAL INTEGRATED FUNCTION FOR YOUR APP (PSA-STYLE)
# ---------------------------------------------------------
def analyze_centering(front_path: str, back_path: str) -> dict:
    """
    Full centering analysis for front and back images using PSA-style logic.

    Returns a dictionary designed to be API/FlutterFlow friendly, including:
        - Raw borders per side
        - PSA-style centering ratios for front and back (percent, worse side)
        - Overall PSA centering grade (1–10)
    """

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

    # ---------------- PSA CENTERING GRADE (COMBINED) ----------------
    centering_grade = map_psa_centering_grade(
        front_ratio=front_center_ratio,
        back_ratio=back_center_ratio
    )

    # ---------------- RESULT STRUCTURE ----------------
    # This is what you’ll return via your API and map in FlutterFlow.
    result = {
        "front": {
            "outer_border": front_outer,                  # [x, y, w, h]
            "artwork_rectangle": front_inner,             # [x, y, w, h]
            "borders": front_borders,                     # {left, right, top, bottom}
            "ratios": front_ratio_details,                # detailed axis ratios
            "psa_centering_ratio": front_center_ratio     # e.g. 45.0 → 55/45
        },
        "back": {
            "outer_border": back_outer,
            "artwork_rectangle": back_inner,
            "borders": back_borders,
            "ratios": back_ratio_details,
            "psa_centering_ratio": back_center_ratio      # e.g. 25.0 → 75/25
        },
        "summary": {
            # These three are the ones you’ll store in Firestore:
            # frontCenteringRatio, backCenteringRatio, centeringGrade
            "frontCenteringRatio": front_center_ratio,    # Number (e.g. 45.0)
            "backCenteringRatio": back_center_ratio,      # Number (e.g. 25.0)
            "centeringGrade": centering_grade             # PSA 1–10
        }
    }

    return result
