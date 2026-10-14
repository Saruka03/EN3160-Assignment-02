import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.ndimage import gaussian_laplace, maximum_filter

# Run from the EN3160-Assignment-02 project root.
IMAGE_PATH = Path("images/the_berry_farms_sunflower_field.jpeg")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters to tune after inspecting the output.
SIGMA_MIN = 2.2
SIGMA_MAX = 30.0          # Wider search to check larger flower blobs.
NUMBER_OF_SCALES = 32       # More scales improve sampling across the wider range.
FIELD_START_ROW = 170
RESPONSE_THRESHOLD_FRACTION = 0.12
MIN_SIGMA = 2.9              # Discard very small texture responses.
MIN_CENTER_DISTANCE = 10.0   # Suppress detections near stronger detections.

im = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_REDUCED_COLOR_4)
if im is None:
    raise FileNotFoundError(
        f"Cannot read image: {IMAGE_PATH.resolve()}\n"
        "Check that the sunflower image is in the images folder."
    )

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
print("Image shape:", gray.shape)

# Create scale-normalized absolute LoG responses.
sigmas = np.linspace(SIGMA_MIN, SIGMA_MAX, NUMBER_OF_SCALES)
responses = []
for sigma in sigmas:
    log_response = gaussian_laplace(gray, sigma=float(sigma))
    responses.append(np.abs((float(sigma) ** 2) * log_response))

scale_space = np.stack(responses, axis=0)
print(f"Sigma range tested: {sigmas[0]:.2f} to {sigmas[-1]:.2f}")
print("Number of scales:", len(sigmas))
print("Scale step:", f"{sigmas[1] - sigmas[0]:.3f}" if len(sigmas) > 1 else "N/A")

# Restrict detections to the sunflower field, excluding most of the sky.
field_mask = np.zeros_like(gray, dtype=bool)
field_mask[FIELD_START_ROW:, :] = True

# Find local maxima across scale, row, and column.
local_maxima = maximum_filter(
    scale_space, size=(3, 7, 7), mode="nearest"
)
threshold = RESPONSE_THRESHOLD_FRACTION * float(scale_space.max())
extrema_mask = (scale_space == local_maxima) & (scale_space >= threshold)

scale_idx, ys, xs = np.where(extrema_mask)
keep = field_mask[ys, xs] & (sigmas[scale_idx] >= MIN_SIGMA)
scale_idx, ys, xs = scale_idx[keep], ys[keep], xs[keep]

strengths = scale_space[scale_idx, ys, xs]
detected_sigmas = sigmas[scale_idx]
radii = np.sqrt(2.0) * detected_sigmas
candidates = np.column_stack((xs, ys, radii, detected_sigmas, strengths))

# Greedy non-maximum suppression: retain stronger candidates first and
# suppress nearby centers. This reduces duplicate circles but may remove
# closely spaced real flowers, so inspect the output and tune the distance.
if len(candidates):
    candidates = candidates[np.argsort(candidates[:, 4])[::-1]]

selected = []
for candidate in candidates:
    x, y = candidate[:2]
    if all(
        np.hypot(x - previous[0], y - previous[1]) >= MIN_CENTER_DISTANCE
        for previous in selected
    ):
        selected.append(candidate)

detections = np.asarray(selected, dtype=float).reshape(-1, 5)

# Draw retained circles.
result = im.copy()
for x, y, radius, sigma, strength in detections:
    cv2.circle(
        result,
        (int(round(x)), int(round(y))),
        max(1, int(round(radius))),
        (0, 255, 0),
        1
    )

print("Response threshold:", threshold)
print("Minimum sigma retained:", MIN_SIGMA)
print("Minimum center distance:", MIN_CENTER_DISTANCE)
print("Detections before spatial suppression:", len(candidates))
print("Detections after spatial suppression:", len(detections))

# Report how detections are distributed across scales. A large number at the
# largest tested scale can indicate that SIGMA_MAX should be increased again.
if len(detections):
    print("Detections by sigma:")
    unique_sigmas, sigma_counts = np.unique(detections[:, 3], return_counts=True)
    for sigma_value, count in zip(unique_sigmas, sigma_counts):
        print(f"  sigma={sigma_value:.2f}: {count}")

    largest_scale_count = int(np.count_nonzero(
        np.isclose(detections[:, 3], sigmas[-1])
    ))
    largest_scale_fraction = largest_scale_count / len(detections)
    print(
        f"Detections at largest tested sigma ({sigmas[-1]:.2f}): "
        f"{largest_scale_count}/{len(detections)} "
        f"({largest_scale_fraction:.1%})"
    )
    if largest_scale_fraction >= 0.10:
        print(
            "NOTE: At least 10% of retained detections are at the upper scale "
            "limit. Consider increasing SIGMA_MAX and rerunning to check for "
            "scale-boundary truncation."
        )
    else:
        print(
            "Upper-scale diagnostic: fewer than 10% of retained detections "
            "are at the largest tested scale; inspect the scale distribution "
            "and image before concluding the range is sufficient."
        )
else:
    print("No detections retained; consider reviewing the threshold and masks.")
print("Top detections:")
for x, y, radius, sigma, strength in detections[:20]:
    print(
        f"x={x:.0f}, y={y:.0f}, radius={radius:.2f}, "
        f"sigma={sigma:.2f}, response={strength:.5f}"
    )

result_path = OUTPUT_DIR / "q1_sunflowers.png"
plot_path = OUTPUT_DIR / "q1_sunflowers_plot.png"
csv_path = OUTPUT_DIR / "q1_detections.csv"

if not cv2.imwrite(str(result_path), result):
    raise OSError(f"Could not save image to {result_path}")

np.savetxt(
    csv_path,
    detections,
    delimiter=",",
    header="x,y,radius,sigma,response",
    comments="",
    fmt="%.5f"
)

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Sunflower Blob Detection using Scale-Normalized LoG")
plt.axis("off")
plt.tight_layout()
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.show()

print("Saved image:", result_path)
print("Saved plot:", plot_path)
print("Saved detections:", csv_path)