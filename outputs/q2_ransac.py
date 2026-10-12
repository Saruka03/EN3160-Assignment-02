import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

# EN3160 Assignment 2 - Question 2
# RANSAC line fitting followed by RANSAC circle fitting.
# The line and circle RANSAC loops below are implemented directly.

# ----------------------------- Configuration -----------------------------
RANDOM_SEED = 0
N = 100
HALF_N = N // 2

LINE_THRESHOLD = 1.4       # Perpendicular/normal distance threshold
LINE_MIN_INLIERS = 35
LINE_ITERATIONS = 2000

CIRCLE_THRESHOLD = 1.2     # Absolute radial error threshold
CIRCLE_MIN_INLIERS = 35
CIRCLE_ITERATIONS = 4000

np.random.seed(RANDOM_SEED)

# ------------------------- Generate assignment data ----------------------
# Circle points: centre (2, 3), radius 10, Gaussian radial noise r/16.
r_gt = 10.0
x0_gt, y0_gt = 2.0, 3.0
s_circle = r_gt / 16.0

t = np.random.uniform(0, 2 * np.pi, HALF_N)
n = s_circle * np.random.randn(HALF_N)
x_circle = x0_gt + (r_gt + n) * np.cos(t)
y_circle = y0_gt + (r_gt + n) * np.sin(t)
X_circle_gt = np.column_stack((x_circle, y_circle))

# Line points: y = -x + 2, with Gaussian noise sigma = 1.
s_line = 1.0
m_gt, b_gt = -1.0, 2.0
x_line = np.linspace(-12, 12, HALF_N)
y_line = m_gt * x_line + b_gt + s_line * np.random.randn(HALF_N)
X_line_gt = np.column_stack((x_line, y_line))

X = np.vstack((X_circle_gt, X_line_gt))
rng = np.random.default_rng(RANDOM_SEED + 1)


# ------------------------------ Line model -------------------------------
def line_from_two_points(p1, p2):
    """Return unit-normal line [a,b,d] where a*x + b*y = d."""
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return None

    # A normal is perpendicular to the direction.
    normal = np.array([-direction[1], direction[0]], dtype=float)
    normal /= np.linalg.norm(normal)
    d = float(normal @ p1)

    # Canonical sign makes plots and printed parameters consistent.
    if normal[0] < 0 or (abs(normal[0]) < 1e-12 and normal[1] < 0):
        normal = -normal
        d = -d

    return np.array([normal[0], normal[1], d])


def line_residuals(points, line):
    """Absolute perpendicular distances; normal is constrained to unit length."""
    a, b, d = line
    return np.abs(points @ np.array([a, b]) - d)


def fit_line_tls(points):
    """Total least-squares fit; returns unit normal [a,b] and distance d."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1].astype(float)
    normal /= np.linalg.norm(normal)
    d = float(normal @ centroid)

    if normal[0] < 0 or (abs(normal[0]) < 1e-12 and normal[1] < 0):
        normal = -normal
        d = -d
    return np.array([normal[0], normal[1], d])


best_line = None
best_line_inliers = np.zeros(len(X), dtype=bool)
best_line_sample = None
best_line_count = -1

for _ in range(LINE_ITERATIONS):
    sample_indices = rng.choice(len(X), 2, replace=False)
    sample = X[sample_indices]
    candidate = line_from_two_points(sample[0], sample[1])
    if candidate is None:
        continue

    inliers = line_residuals(X, candidate) < LINE_THRESHOLD
    count = int(inliers.sum())

    if count > best_line_count:
        best_line = candidate
        best_line_inliers = inliers
        best_line_sample = sample.copy()
        best_line_count = count

if best_line is None or best_line_count < LINE_MIN_INLIERS:
    raise RuntimeError("RANSAC did not find a convincing line. Adjust thresholds/iterations.")

# Refit the best line using all of its consensus inliers.
best_line = fit_line_tls(X[best_line_inliers])
line_inliers = line_residuals(X, best_line) < LINE_THRESHOLD

# Remove the line consensus before circle estimation.
remaining = X[~line_inliers]
remaining_original_indices = np.flatnonzero(~line_inliers)


# ----------------------------- Circle model ------------------------------
def circle_from_three_points(points):
    """Calculate a circle from three non-collinear 2D points."""
    p1, p2, p3 = points
    matrix = 2.0 * np.array([
        [p2[0] - p1[0], p2[1] - p1[1]],
        [p3[0] - p1[0], p3[1] - p1[1]]
    ])
    rhs = np.array([
        p2 @ p2 - p1 @ p1,
        p3 @ p3 - p1 @ p1
    ])
    if abs(np.linalg.det(matrix)) < 1e-8:
        return None

    center = np.linalg.solve(matrix, rhs)
    radius = float(np.linalg.norm(p1 - center))
    if not np.isfinite(radius) or radius <= 0:
        return None
    return np.array([center[0], center[1], radius])


def circle_radial_residuals(points, circle):
    cx, cy, radius = circle
    return np.abs(np.linalg.norm(points - np.array([cx, cy]), axis=1) - radius)


def fit_circle_optimizer(points, initial_circle):
    """Refine (cx,cy,r) by minimizing squared radial residuals."""
    def objective(params):
        cx, cy, radius = params
        if radius <= 0:
            return 1e12 + abs(radius) * 1e9
        radial_errors = (
            np.linalg.norm(points - np.array([cx, cy]), axis=1) - radius
        )
        return float(np.sum(radial_errors ** 2))

    result = minimize(
        objective,
        x0=np.asarray(initial_circle, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-9, "fatol": 1e-9}
    )
    cx, cy, radius = result.x
    if radius < 0:
        radius = abs(radius)
    return np.array([cx, cy, radius])


best_circle_sample_model = None
best_circle_sample = None
best_circle_inliers_local = np.zeros(len(remaining), dtype=bool)
best_circle_count = -1

for _ in range(CIRCLE_ITERATIONS):
    if len(remaining) < 3:
        raise RuntimeError("Not enough points remain for circle fitting.")

    sample_indices = rng.choice(len(remaining), 3, replace=False)
    sample = remaining[sample_indices]
    candidate = circle_from_three_points(sample)
    if candidate is None:
        continue

    inliers = circle_radial_residuals(remaining, candidate) < CIRCLE_THRESHOLD
    count = int(inliers.sum())

    if count > best_circle_count:
        best_circle_sample_model = candidate
        best_circle_sample = sample.copy()
        best_circle_inliers_local = inliers
        best_circle_count = count

if best_circle_sample_model is None or best_circle_count < CIRCLE_MIN_INLIERS:
    raise RuntimeError("RANSAC did not find a convincing circle. Adjust thresholds/iterations.")

# Optimize the circle using the consensus set from the best RANSAC sample.
circle_consensus = remaining[best_circle_inliers_local]
best_circle = fit_circle_optimizer(circle_consensus, best_circle_sample_model)

# Recalculate consensus using the refined model.
circle_inliers_local = (
    circle_radial_residuals(remaining, best_circle) < CIRCLE_THRESHOLD
)
circle_inliers = np.zeros(len(X), dtype=bool)
circle_inliers[remaining_original_indices[circle_inliers_local]] = True


# -------------------------------- Results --------------------------------
a, b, d = best_line
cx, cy, radius = best_circle

print("\n--- RANSAC line result ---")
print(f"Line model: a*x + b*y = d")
print(f"a = {a:.6f}, b = {b:.6f}, d = {d:.6f}")
print(f"Normal length sqrt(a^2+b^2) = {np.hypot(a, b):.6f}")
print(f"Line inliers = {int(line_inliers.sum())} / {len(X)}")
print(f"Line threshold = {LINE_THRESHOLD:.2f}")

print("\n--- RANSAC circle result ---")
print(f"Centre = ({cx:.6f}, {cy:.6f})")
print(f"Radius = {radius:.6f}")
print(f"Circle inliers = {int(circle_inliers.sum())}")
print(f"Circle radial-error threshold = {CIRCLE_THRESHOLD:.2f}")
print(f"Best sample circle (cx, cy, r) = {best_circle_sample_model}")
print("Three points in the best circle sample:")
print(best_circle_sample)

# ------------------------------- Plotting --------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# All points, with detected consensus sets emphasized.
ax.scatter(X[:, 0], X[:, 1], s=24, c="lightgray",
           label="All points", zorder=1)
ax.scatter(X[line_inliers, 0], X[line_inliers, 1], s=28,
           label="Line inliers", zorder=2)
ax.scatter(X[circle_inliers, 0], X[circle_inliers, 1], s=28,
           label="Circle inliers", zorder=3)

# Best line sample and the line from that two-point sample.
sample_line = line_from_two_points(best_line_sample[0], best_line_sample[1])
x_plot = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 400)

def line_y_values(line, x_values):
    aa, bb, dd = line
    if abs(bb) < 1e-8:
        return None
    return (dd - aa * x_values) / bb

sample_y = line_y_values(sample_line, x_plot)
if sample_y is not None:
    ax.plot(x_plot, sample_y, "--", linewidth=1.5,
            label="Line from best RANSAC sample")

best_y = line_y_values(best_line, x_plot)
if best_y is not None:
    ax.plot(x_plot, best_y, linewidth=2.2, label="Best-fit line")

# Best circle model from the three-point RANSAC sample.
theta = np.linspace(0, 2 * np.pi, 500)
scx, scy, sr = best_circle_sample_model
ax.plot(scx + sr * np.cos(theta), scy + sr * np.sin(theta), "--",
        linewidth=1.5, label="Circle from best RANSAC sample")
ax.scatter(best_circle_sample[:, 0], best_circle_sample[:, 1],
           marker="x", s=100, linewidths=2.5,
           label="Three-point circle sample", zorder=5)

# Optimized best-fit circle.
ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta),
        linewidth=2.2, label="Best-fit circle")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Q2: RANSAC Line and Circle Fitting")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.25)
ax.legend(loc="best", fontsize=8)
fig.tight_layout()

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
plot_path = output_dir / "q2_ransac_line_circle.png"
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"\nSaved plot: {plot_path}")
