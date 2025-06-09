import numpy as np

def compute_sat_distance(A: np.ndarray, B: np.ndarray) -> float:
    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        norms = np.linalg.norm(edges, axis=1)
        valid = norms > 1e-8  # 忽略长度为 0 的边
        edges = edges[valid]
        if edges.shape[0] == 0:
            raise ValueError("Degenerate polygon with no valid edges.")
        normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        return normals / np.linalg.norm(normals, axis=1, keepdims=True)

    if A.shape[0] < 3 or B.shape[0] < 3:
        raise ValueError("Polygons must have at least 3 vertices.")

    axes = np.vstack([get_normals(A), get_normals(B)])
    max_penetration = -np.inf
    found_valid_axis = False

    for axis in axes:
        if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-8:
            continue

        proj_A = A @ axis
        proj_B = B @ axis

        if not np.isfinite(proj_A).all() or not np.isfinite(proj_B).all():
            continue

        A_min, A_max = proj_A.min(), proj_A.max()
        B_min, B_max = proj_B.min(), proj_B.max()
        sep = max(B_min - A_max, A_min - B_max)

        found_valid_axis = True

        if sep > 0:
            return sep
        else:
            max_penetration = max(max_penetration, sep)

    if not found_valid_axis:
        raise ValueError("No valid axis found: check for NaNs or degenerate edges.")

    return max_penetration


# ======================
# ✅ Test: overlapping rectangles
A = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

B = np.array([
    [0.5, 0.5],
    [1.5, 0.5],
    [1.5, 1.5],
    [0.5, 1.5]
])

distance = compute_sat_distance(A, B)
print("SAT distance:", distance)

assert distance < 0, "Expected penetration (negative value) but got non-negative!"