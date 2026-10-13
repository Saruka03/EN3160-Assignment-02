"""
Q3: Homography and image compositing.
Run from the repository root: python outputs/q3_homography.py
Click four destination points on the architectural image in this order:
top-left, top-right, bottom-right, bottom-left.
"""
from pathlib import Path
import cv2
import numpy as np

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_DIR / "images"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

flag = cv2.imread(str(IMAGE_DIR / "flag.png"), cv2.IMREAD_COLOR)
scene = cv2.imread(str(IMAGE_DIR / "wadham.jpg"), cv2.IMREAD_COLOR)
if flag is None or scene is None:
    raise FileNotFoundError("Expected flag.png and wadham.jpg in a2images/a2images")

fh, fw = flag.shape[:2]
# The four corners of the source flag.
src = np.float32([[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]])

clicked = []
display = scene.copy()
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
        clicked.append([x, y])
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(display, str(len(clicked)), (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Select target quadrilateral: TL, TR, BR, BL", display)

cv2.namedWindow("Select target quadrilateral: TL, TR, BR, BL")
cv2.setMouseCallback("Select target quadrilateral: TL, TR, BR, BL", on_mouse)
cv2.imshow("Select target quadrilateral: TL, TR, BR, BL", display)
print("Click four corners in order: top-left, top-right, bottom-right, bottom-left. Press Esc to cancel.")
while len(clicked) < 4:
    if cv2.waitKey(20) == 27:
        cv2.destroyAllWindows()
        raise SystemExit("Cancelled before four points were selected.")
cv2.destroyAllWindows()

dst = np.float32(clicked)
H = cv2.getPerspectiveTransform(src, dst)
h, w = scene.shape[:2]
warped = cv2.warpPerspective(flag, H, (w, h))
mask_src = np.full(flag.shape[:2], 255, np.uint8)
mask = cv2.warpPerspective(mask_src, H, (w, h))
# Use the warped flag only inside its projected quadrilateral.
result = scene.copy()
result[mask > 0] = warped[mask > 0]
cv2.imwrite(str(OUTPUT_DIR / "q3_flag_composite.png"), result)
cv2.imwrite(str(OUTPUT_DIR / "q3_warped_flag.png"), warped)
np.savetxt(OUTPUT_DIR / "q3_homography.txt", H, fmt="%.8f")
print("Saved q3_flag_composite.png, q3_warped_flag.png, q3_homography.txt")
print("Homography matrix:\n", H)
