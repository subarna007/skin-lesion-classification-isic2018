"""Image quality checks and simple lesion shape/colour descriptors.

These descriptors are computed with classical image processing (Otsu
segmentation). They are shown next to the model output for context only and
have not been clinically validated. Diameter and evolution (the D and E of the
ABCDE rule) are not estimated: a single photo has no physical scale and no
history.
"""
import cv2
import numpy as np

# Approximate RGB values of the six colours used in dermoscopy colour counts.
REFERENCE_COLOURS = {
    "white": (225, 225, 225),
    "red": (190, 60, 60),
    "light brown": (175, 120, 80),
    "dark brown": (95, 60, 40),
    "blue-grey": (100, 115, 135),
    "black": (30, 30, 30),
}


def _resize_max(img, side=512):
    h, w = img.shape[:2]
    s = side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def quality_checks(rgb):
    """Return (ok, warnings). `ok` is False only when the image cannot be used."""
    h, w = rgb.shape[:2]
    if min(h, w) < 100:
        return False, ["The image is smaller than 100 pixels on one side."]
    gray = cv2.cvtColor(_resize_max(rgb), cv2.COLOR_RGB2GRAY)
    warnings = []
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 20:
        warnings.append("The image looks blurry; the result may be unreliable.")
    brightness = float(gray.mean())
    if brightness < 40:
        warnings.append("The image is very dark.")
    elif brightness > 235:
        warnings.append("The image is very bright or overexposed.")
    return True, warnings


def segment_lesion(rgb):
    """Binary mask of the lesion: the dark region (Otsu) whose centre is closest
    to the image centre. Returns None when no plausible lesion is found."""
    img = _resize_max(rgb)
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = mask.shape
    centre = np.array([w / 2, h / 2])
    best, best_dist = None, None
    for c in contours:
        area = cv2.contourArea(c)
        if not 0.01 * h * w < area < 0.9 * h * w:
            continue
        m = cv2.moments(c)
        dist = np.linalg.norm(np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) - centre)
        if best is None or dist < best_dist:
            best, best_dist = c, dist
    if best is None:
        return None, img
    out = np.zeros_like(mask)
    cv2.drawContours(out, [best], -1, 255, -1)
    return out, img


def _asymmetry(mask):
    """Mean non-overlap after flipping the lesion about its two principal axes
    (0 = perfectly symmetric)."""
    m = cv2.moments(mask, binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    angle = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    h, w = mask.shape
    side = int(2 * np.hypot(h, w))
    shift = np.float32([[1, 0, side / 2 - cx], [0, 1, side / 2 - cy]])
    centred = cv2.warpAffine(mask, shift, (side, side))
    rot = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
    aligned = cv2.warpAffine(centred, rot, (side, side)) > 127
    area = aligned.sum()
    scores = [np.logical_xor(aligned, np.flip(aligned, axis)).sum() / (2 * area) for axis in (0, 1)]
    return float(np.mean(scores))


def lesion_descriptors(rgb):
    mask, img = segment_lesion(rgb)
    if mask is None:
        return {"segmentation_ok": False}
    contour = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
                  key=cv2.contourArea)
    area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
    pixels = img[mask > 0].astype(np.float32)
    refs = np.array(list(REFERENCE_COLOURS.values()), dtype=np.float32)
    nearest = np.argmin(((pixels[:, None, :] - refs[None]) ** 2).sum(-1), axis=1)
    share = np.bincount(nearest, minlength=len(refs)) / len(nearest)
    present = [name for name, s in zip(REFERENCE_COLOURS, share) if s >= 0.05]
    return {
        "segmentation_ok": True,
        "asymmetry": round(_asymmetry(mask), 3),
        "border_irregularity": round(float(max(0.0, 1 - 4 * np.pi * area / perimeter ** 2)), 3),
        "colours": present,
        "lesion_area_fraction": round(float(area / mask.size), 3),
    }


def mask_outline(rgb):
    """The image with the segmented lesion outline drawn, for display."""
    mask, img = segment_lesion(rgb)
    out = img.copy()
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    return out
