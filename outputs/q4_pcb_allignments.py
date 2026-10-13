"""EN3160 Assignment 02 - Q4: PCB image alignment and difference analysis.

Expected project structure:
    EN3160-Assignment-02/
        images/AECC1.jpg   # reference/template
        images/AECC2.jpg
        images/AECC3.jpg
        images/AECC4.jpg   # other PCB images (or any AECC*.jpg)
        outputs/q4_pcb_alignment.py

Run from any working directory:
    python outputs/q4_pcb_alignment.py

Outputs are written to the project's outputs/ directory.
"""
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_DIR / "images"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_PATH = IMAGE_DIR / "AECC1.jpg"
RATIO_TEST = 0.75
RANSAC_REPROJ_THRESHOLD = 5.0
MIN_GOOD_MATCHES = 8


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)    
    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {path}\n"
            "Check that the file exists and that its name matches exactly."
        )
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save output image: {path}")


def align_to_template(template: np.ndarray, source: np.ndarray, source_path: Path) -> None:
    """Detect SIFT features, match, estimate homography, warp, and compare."""
    name = source_path.stem
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints_template, descriptors_template = sift.detectAndCompute(template_gray, None)
    keypoints_source, descriptors_source = sift.detectAndCompute(source_gray, None)

    if descriptors_template is None or descriptors_source is None:
        print(f"[SKIP] {source_path.name}: SIFT found too few usable features.")
        return

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(descriptors_source, descriptors_template, k=2)
    good_matches = [m for pair in knn_matches if len(pair) == 2
                    for m, n in [pair] if m.distance < RATIO_TEST * n.distance]

    if len(good_matches) < MIN_GOOD_MATCHES:
        print(f"[SKIP] {source_path.name}: only {len(good_matches)} good matches; "
              f"need at least {MIN_GOOD_MATCHES}.")
        return

    # Source points map to corresponding template points.
    source_points = np.float32(
        [keypoints_source[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    template_points = np.float32(
        [keypoints_template[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(
        source_points, template_points, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD
    )
    if homography is None or inlier_mask is None:
        print(f"[SKIP] {source_path.name}: homography estimation failed.")
        return

    height, width = template.shape[:2]
    aligned = cv2.warpPerspective(source, homography, (width, height))

    # Difference is computed after converting both images to the same size.
    difference = cv2.absdiff(template, aligned)
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    # Contrast-enhanced version makes small differences easier to inspect.
    difference_enhanced = cv2.normalize(
        difference_gray, None, 0, 255, cv2.NORM_MINMAX
    )
    overlay = cv2.addWeighted(template, 0.5, aligned, 0.5, 0)

    inlier_matches = [m for m, keep in zip(good_matches, inlier_mask.ravel()) if keep]
    matches_image = cv2.drawMatches(
        source, keypoints_source,
        template, keypoints_template,
        inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    save_image(OUTPUT_DIR / f"q4_{name}_matches.png", matches_image)
    save_image(OUTPUT_DIR / f"q4_{name}_aligned.png", aligned)
    save_image(OUTPUT_DIR / f"q4_{name}_difference.png", difference)
    save_image(OUTPUT_DIR / f"q4_{name}_difference_gray.png", difference_enhanced)
    save_image(OUTPUT_DIR / f"q4_{name}_overlay.png", overlay)

    inliers = int(inlier_mask.sum())
    print(f"\n{name} -> AECC1.jpg")
    print(f"  SIFT keypoints: source={len(keypoints_source)}, template={len(keypoints_template)}")
    print(f"  Good matches: {len(good_matches)}; RANSAC inliers: {inliers}")
    print("  Homography (source -> template):")
    print(homography)
    print(f"  Saved matches, aligned image, difference, enhanced difference, and overlay to {OUTPUT_DIR}")


def main() -> None:
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Reference image not found: {TEMPLATE_PATH}\n"
            "Place AECC1.jpg in the project's images/ folder."
        )

    template = read_image(TEMPLATE_PATH)
    candidates = sorted(
        p for p in IMAGE_DIR.glob("AECC*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        and p.resolve() != TEMPLATE_PATH.resolve()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No PCB images found in {IMAGE_DIR}. Expected files such as AECC2.jpg, AECC3.jpg, AECC4.jpg."
        )

    print(f"Reference template: {TEMPLATE_PATH}")
    print(f"Found {len(candidates)} PCB image(s) to align. Output folder: {OUTPUT_DIR}")
    for source_path in candidates:
        try:
            source = read_image(source_path)
            align_to_template(template, source, source_path)
        except (FileNotFoundError, OSError, cv2.error) as exc:
            print(f"[ERROR] {source_path.name}: {exc}")

    print("\nQ4 processing finished.")
    print("Review q4_*_matches.png, q4_*_aligned.png, and q4_*_difference.png for each image.")


if __name__ == "__main__":
    main()