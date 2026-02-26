import cv2
import numpy as np
from pathlib import Path
import csv
import shutil

# ============================================================
#                CONFIGURATION 
# ============================================================

REFERENCE_IMAGE = r"...\References\Coated Brake Rotor.png"
INPUT_FOLDER = r"...\Input\Coated Brake Rotor"
OUTPUT_FOLDER = r"...\Output\Coated Brake Rotor"

# Output subfolder for REVIEW images
REVIEW_FOLDER = str(Path(OUTPUT_FOLDER) / "Image_to_REVIEW")

# --- Parameter controls (PLAY WITH THESE) ---

# Single value, not a range. 0.0 = no luminance adjustment, 1.0 = full match
LUMINANCE_STRENGTH = 0.85

# 0.0 = preserve original a/b, 1.0 = match reference, >1.0 = force beyond reference
COLOR_STRENGTH = 1.0

# Manual color shifts on a and b
A_SHIFT = 0      # negative = greener, positive = redder
B_SHIFT = 0      # negative = bluer,  positive = yellower

# Force final L mean to match reference
FORCE_L_MEAN_TO_REF = True
L_MEAN_TOLERANCE = 8.0     # if |L_mean - L_ref| > this, apply final L shift
MAX_L_MEAN_DELTA = 10.0    # maximum allowed global L shift in the second stage

# Print stats before/after to console
PRINT_STATS = True

# Name of the CSV file created inside OUTPUT_FOLDER
CSV_NAME = "stats.csv"

# Thresholds for automatic quality flag (OK / ACCEPTABLE / REVIEW)
DELTA_L_OK = 10.0
DELTA_L_ACCEPTABLE = 20.0

DELTA_AB_OK = 10.0
DELTA_AB_ACCEPTABLE = 20.0

DELTA_LSTD_OK = 8.0
DELTA_LSTD_ACCEPTABLE = 15.0

# ============================================================
#                       FUNCTIONS
# ============================================================

def load_image_with_alpha(path):
    """
    Load image as BGRA (with alpha if available).
    Returns BGR image and alpha mask.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not read image: {path.resolve()}")

    # Handle grayscale, BGR, BGRA
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    elif img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)

    return bgr, alpha


def compute_lab_stats(bgr, alpha, min_alpha=10):
    """
    Compute mean and std of L, a, b channels using only product pixels
    (where alpha > min_alpha).
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    mask = alpha > min_alpha

    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    L_vals = L[mask]
    A_vals = A[mask]
    B_vals = B[mask]

    if L_vals.size == 0:
        # No product pixels: avoid crash, return neutral stats
        mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return mean, std

    mean = np.array(
        [L_vals.mean(), A_vals.mean(), B_vals.mean()],
        dtype=np.float32,
    )
    std = np.array(
        [L_vals.std() + 1e-6, A_vals.std() + 1e-6, B_vals.std() + 1e-6],
        dtype=np.float32,
    )

    return mean, std


def match_color_to_reference(
    bgr,
    alpha,
    ref_mean,
    ref_std,
    luminance_strength,
    color_strength,
    a_shift,
    b_shift,
):
    """
    Adjust the color of 'bgr' image so that its Lab statistics move towards
    the reference statistics (Reinhard-style transfer + blending).
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    cur_mean, cur_std = compute_lab_stats(bgr, alpha)
    cur_mean_L, cur_mean_A, cur_mean_B = cur_mean
    cur_std_L, cur_std_A, cur_std_B = cur_std

    ref_mean_L, ref_mean_A, ref_mean_B = ref_mean
    ref_std_L, ref_std_A, ref_std_B = ref_std

    mask = alpha > 10

    L_out = L.copy()
    A_out = A.copy()
    B_out = B.copy()

    # --- Match color channels (a and b) with strength factor ---
    if cur_std_A > 0 and cur_std_B > 0:
        A_target = (A[mask] - cur_mean_A) * (ref_std_A / cur_std_A) + ref_mean_A
        B_target = (B[mask] - cur_mean_B) * (ref_std_B / cur_std_B) + ref_mean_B

        A_out[mask] = (1.0 - color_strength) * A[mask] + color_strength * A_target
        B_out[mask] = (1.0 - color_strength) * B[mask] + color_strength * B_target

    # Manual shifts in a and b
    A_out[mask] += a_shift
    B_out[mask] += b_shift

    # --- Match luminance channel (L) with strength factor (blend) ---
    if luminance_strength > 0.0 and cur_std_L > 0:
        L_target = (L[mask] - cur_mean_L) * (ref_std_L / cur_std_L) + ref_mean_L
        L_out[mask] = (
            (1.0 - luminance_strength) * L[mask] +
            luminance_strength * L_target
        )

    lab_out = np.stack([L_out, A_out, B_out], axis=2)
    lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)

    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return bgr_out


def force_l_mean_to_ref(bgr, alpha, L_ref, tolerance, max_delta, min_alpha=10):
    """
    After the main correction, optionally force the mean L of the product region
    closer to the reference L_ref by applying a uniform shift on L.
    The shift is limited by 'max_delta' to avoid blowing out highlights.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    mask = alpha > min_alpha

    L_vals = L[mask]
    if L_vals.size == 0:
        return bgr  # no product pixels; nothing to adjust

    L_mean = float(L_vals.mean())
    delta = L_ref - L_mean

    if abs(delta) > tolerance:
        delta = float(np.clip(delta, -max_delta, max_delta))
        L[mask] = L[mask] + delta
        lab_out = np.stack([L, A, B], axis=2)
        lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)
        bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
        return bgr_out
    else:
        return bgr


def debug_stats(label, bgr, alpha):
    """
    Print Lab statistics to console.
    """
    mean, std = compute_lab_stats(bgr, alpha)
    print(
        f"{label} - L mean: {mean[0]:6.2f}, a mean: {mean[1]:6.2f}, "
        f"b mean: {mean[2]:6.2f} | L std: {std[0]:6.2f}"
    )
    return mean, std


def classify_ok_flag(delta_L_abs_after, delta_a_abs_after, delta_b_abs_after, delta_L_std_abs_after):
    """
    Classify image quality based on deviations in L, a, b and L_std.
    """
    if (
        delta_L_abs_after     <= DELTA_L_OK and
        delta_a_abs_after     <= DELTA_AB_OK and
        delta_b_abs_after     <= DELTA_AB_OK and
        delta_L_std_abs_after <= DELTA_LSTD_OK
    ):
        return "OK"
    elif (
        delta_L_abs_after     <= DELTA_L_ACCEPTABLE and
        delta_a_abs_after     <= DELTA_AB_ACCEPTABLE and
        delta_b_abs_after     <= DELTA_AB_ACCEPTABLE and
        delta_L_std_abs_after <= DELTA_LSTD_ACCEPTABLE
    ):
        return "ACCEPTABLE"
    else:
        return "REVIEW"


def process_folder():
    """
    Process all PNG files in INPUT_FOLDER, save corrected images in OUTPUT_FOLDER,
    and write a CSV file with statistics before and after, including deltas and
    an automatic OK_flag.
    """
    reference_path = Path(REFERENCE_IMAGE)
    input_folder = Path(INPUT_FOLDER)
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    review_folder = Path(REVIEW_FOLDER)
    review_folder.mkdir(parents=True, exist_ok=True)

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_path.resolve()}")

    # --- Load reference and compute stats ---
    ref_bgr, ref_alpha = load_image_with_alpha(reference_path)
    ref_mean, ref_std = compute_lab_stats(ref_bgr, ref_alpha)

    if PRINT_STATS:
        print("REFERENCE stats:")
        debug_stats("REF", ref_bgr, ref_alpha)
        print()

    # CSV setup (open once, write as we go)
    csv_path = output_folder / CSV_NAME
    fieldnames = [
        "filename",
        "L_mean_before", "a_mean_before", "b_mean_before", "L_std_before",
        "L_mean_after",  "a_mean_after",  "b_mean_after",  "L_std_after",
        "L_mean_ref",    "a_mean_ref",    "b_mean_ref",    "L_std_ref",
        "delta_L_before",
        "delta_L_after",
        "delta_L_abs_after",
        "delta_a_after",
        "delta_a_abs_after",
        "delta_b_after",
        "delta_b_abs_after",
        "delta_L_std_after",
        "OK_flag",
    ]

    png_files = list(input_folder.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in: {input_folder.resolve()}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for img_path in png_files:
            try:
                print(f"Processing {img_path.name}...")
                bgr, alpha = load_image_with_alpha(img_path)

                # Stats BEFORE
                mean_before, std_before = compute_lab_stats(bgr, alpha)
                if PRINT_STATS:
                    debug_stats("  BEFORE", bgr, alpha)

                # Main Lab-based color/luminance matching
                bgr_corr = match_color_to_reference(
                    bgr,
                    alpha,
                    ref_mean,
                    ref_std,
                    luminance_strength=LUMINANCE_STRENGTH,
                    color_strength=COLOR_STRENGTH,
                    a_shift=A_SHIFT,
                    b_shift=B_SHIFT,
                )

                # Optional second-stage L mean correction
                if FORCE_L_MEAN_TO_REF:
                    bgr_out = force_l_mean_to_ref(
                        bgr_corr,
                        alpha,
                        L_ref=ref_mean[0],
                        tolerance=L_MEAN_TOLERANCE,
                        max_delta=MAX_L_MEAN_DELTA,
                    )
                else:
                    bgr_out = bgr_corr

                # Stats AFTER
                mean_after, std_after = compute_lab_stats(bgr_out, alpha)
                if PRINT_STATS:
                    debug_stats("  AFTER ", bgr_out, alpha)
                    print()

                # Save corrected image with original alpha
                bgra_out = cv2.merge(
                    [bgr_out[:, :, 0], bgr_out[:, :, 1], bgr_out[:, :, 2], alpha]
                )
                out_path = output_folder / img_path.name
                cv2.imwrite(str(out_path), bgra_out)

                # --- Compute deltas ---
                L_before, a_before, b_before = mean_before
                L_after,  a_after,  b_after  = mean_after
                L_ref,    a_ref,    b_ref    = ref_mean

                L_std_b = std_before[0]
                L_std_a = std_after[0]
                L_std_r = ref_std[0]

                delta_L_before        = float(L_before - L_ref)
                delta_L_after         = float(L_after  - L_ref)
                delta_L_abs_after     = float(abs(delta_L_after))

                delta_a_after         = float(a_after - a_ref)
                delta_a_abs_after     = float(abs(delta_a_after))

                delta_b_after         = float(b_after - b_ref)
                delta_b_abs_after     = float(abs(delta_b_after))

                delta_L_std_after     = float(L_std_a - L_std_r)
                delta_L_std_abs_after = float(abs(delta_L_std_after))

                ok_flag = classify_ok_flag(
                    delta_L_abs_after,
                    delta_a_abs_after,
                    delta_b_abs_after,
                    delta_L_std_abs_after,
                )

                # If flag is REVIEW, move corrected image to REVIEW subfolder
                if ok_flag == "REVIEW":
                    review_out_path = review_folder / img_path.name
                    shutil.move(out_path, review_out_path)

                row = {
                    "filename": img_path.name,
                    "L_mean_before": float(L_before),
                    "a_mean_before": float(a_before),
                    "b_mean_before": float(b_before),
                    "L_std_before":  float(L_std_b),
                    "L_mean_after":  float(L_after),
                    "a_mean_after":  float(a_after),
                    "b_mean_after":  float(b_after),
                    "L_std_after":   float(L_std_a),
                    "L_mean_ref":    float(L_ref),
                    "a_mean_ref":    float(a_ref),
                    "b_mean_ref":    float(b_ref),
                    "L_std_ref":     float(L_std_r),
                    "delta_L_before":    delta_L_before,
                    "delta_L_after":     delta_L_after,
                    "delta_L_abs_after": delta_L_abs_after,
                    "delta_a_after":     delta_a_after,
                    "delta_a_abs_after": delta_a_abs_after,
                    "delta_b_after":     delta_b_after,
                    "delta_b_abs_after": delta_b_abs_after,
                    "delta_L_std_after": delta_L_std_after,
                    "OK_flag":           ok_flag,
                }

                writer.writerow(row)

            except Exception as e:
                print(f"ERROR processing {img_path.name}: {e}")

    print(f"\nColor correction finished.")
    print(f"Stats saved to: {csv_path.resolve()}\n")


# ============================================================
#                        MAIN
# ============================================================

if __name__ == "__main__":
    process_folder()
