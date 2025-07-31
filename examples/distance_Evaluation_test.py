import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from functools import partial
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# ——— Distance function implementations ———

@jax.jit
def get_normals(v: jnp.ndarray) -> jnp.ndarray:
    """Compute normalized edge normals (outward) for polygon vertices v."""
    edges = jnp.roll(v, -1, axis=0) - v
    normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
    eps = 1e-9
    return normals / (jnp.linalg.norm(normals, axis=1, keepdims=True) + eps)
@jax.jit
def smooth_abs(x: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """Smooth absolute function via x * tanh(alpha * x)."""
    return x * jnp.tanh(alpha * x)

@jax.jit
def compute_distance_smoothabs(
    robot: jnp.ndarray,
    poly: jnp.ndarray,
    alpha_axes: float = 20.,
    alpha_sabs: float = 20.
) -> float:
    """
    Fused, JAX-optimized version of compute_distance_smoothabs.
    """
    # Get separating axes (K, 2)
    axes = jnp.concatenate([get_normals(robot), get_normals(poly)], axis=0)  # (K, 2)
    K = axes.shape[0]

    # Center difference
    center_robot = jnp.mean(robot, axis=0)  # (2,)
    center_poly = jnp.mean(poly, axis=0)
    delta_p = center_poly - center_robot  # (2,)
    proj_center = smooth_abs(axes @ delta_p, alpha_sabs)  # shape (K,)

    # Project all points to all axes in one batch
    robot_centered = robot - center_robot  # (N, 2)
    poly_centered  = poly  - center_poly   # (M, 2)

    proj_robot = robot_centered @ axes.T  # (N, K)
    proj_poly  = poly_centered  @ axes.T  # (M, K)

    # Compute smooth_abs, then max per axis (K,)
    rA = jnp.max(smooth_abs(proj_robot, alpha_sabs), axis=0)  # (K,)
    rB = jnp.max(smooth_abs(proj_poly,  alpha_sabs), axis=0)  # (K,)

    axis_gaps = proj_center - (rA + rB)  # shape (K,)

    return (1.0 / alpha_axes) * logsumexp(alpha_axes * axis_gaps)


# === Main distance function: One-LogSAT with shift ===
@jax.jit
def compute_distance_onelogminus(robot: jnp.ndarray, poly: jnp.ndarray, alpha: float = 20.0) -> float:
    """
    Smoothed one-log version of SAT-based polygon separation distance.

    Args:
        robot: shape (N, 2) polygon
        poly: shape (M, 2) polygon
        alpha: float, smoothing parameter (higher = sharper max)

    Returns:
        scalar smooth separation distance
    """
    Rn = get_normals(robot)
    Pn = get_normals(poly)
    axes = jnp.concatenate([Rn, Pn], axis=0)  # shape (K, 2)

    # Project both polygons
    proj_R = robot @ axes.T  # (N, K)
    proj_P = poly  @ axes.T  # (M, K)

    # Min-max bounds
    R_min, R_max = jnp.min(proj_R, axis=0), jnp.max(proj_R, axis=0)
    P_min, P_max = jnp.min(proj_P, axis=0), jnp.max(proj_P, axis=0)

    # Signed separations
    gaps = jnp.concatenate([P_min - R_max, R_min - P_max], axis=0)  # shape (2K,)
    h_raw = (1.0 / alpha) * logsumexp(alpha * gaps)
    error_bound = jnp.log(gaps.shape[0]) / alpha
    return h_raw - error_bound


def compute_distance_sat(robot, poly):
    """
    Basic SAT distance: maximum of axis-wise penetration/separation.
    Non-smooth, JIT-compiled separately.
    """
    def get_normals(v):
        e = jnp.roll(v, -1, axis=0) - v
        n = jnp.stack([-e[:,1], e[:,0]], axis=1)
        return n / jnp.linalg.norm(n, axis=1, keepdims=True)
    Rn = get_normals(robot)
    Pn = get_normals(poly)
    axes = jnp.concatenate([Rn, Pn], axis=0)
    prj_R = robot @ axes.T
    prj_P = poly  @ axes.T
    d1 = prj_P.min(0) - prj_R.max(0)
    d2 = prj_R.min(0) - prj_P.max(0)
    return jnp.max(jnp.maximum(d1, d2))

# JIT-compile the two-step and SAT versions
xtanh_jit = compute_distance_smoothabs
onelog_jit = jax.jit(compute_distance_onelogminus)
sat_jit    = jax.jit(compute_distance_sat)

# ——— Helper functions ———

def regular_ngon(N, radius=1.0):
    """
    Generate vertices of a regular N-gon centered at origin.
    """
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.stack([radius * np.cos(angles),
                    radius * np.sin(angles)], axis=1)

def irregular_ngon(N, radius=1.0, r_jitter=0.2, angle_jitter=0.1, seed=None):
    """
    Generate a strictly convex N-gon by generating random points and taking convex hull.

    Args:
        N (int): number of vertices in final polygon (exact)
        radius (float): nominal radius of bounding circle
        r_jitter (float): noise added to generate initial points
        seed (int or None): random seed

    Returns:
        (N, 2) np.ndarray of convex polygon vertices
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate more random points on a circle
    M = int(2.5 * N)  # oversample
    angles = np.sort(np.random.uniform(0, 2 * np.pi, M))
    radii = radius * (1 + np.random.uniform(-r_jitter, r_jitter, size=M))

    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.stack([x, y], axis=1)

    # Take convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Resample hull to exactly N vertices (optional: uniform arc-length sampling)
    if len(hull_points) > N:
        idx = np.round(np.linspace(0, len(hull_points) - 1, N)).astype(int)
        hull_points = hull_points[idx]

    return hull_points

def benchmark(fn, *args, n=1000):
    """
    Measure average latency (ms) of fn(*args) over n runs.
    Uses .block_until_ready() to ensure synchronous timing.
    """
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
        try:
            out.block_until_ready()
        except AttributeError:
            pass
    t1 = time.perf_counter()
    return (t1 - t0) / n * 1e3

def benchmark_all_pairs(fn, robots, obstacles, n=1000):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        for r in robots:
            for o in obstacles:
                out = fn(r, o)
                try:
                    out.block_until_ready()
                except AttributeError:
                    pass
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return np.mean(times) / (len(robots) * len(obstacles))

def align_polygons_for_contact(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Move polygon B so it just contacts polygon A along center-to-center axis.

    Returns:
        B_aligned: np.ndarray, same shape as B
    """
    center_A = A.mean(axis=0)
    center_B = B.mean(axis=0)
    dir_vec = center_B - center_A
    dir_unit = dir_vec / np.linalg.norm(dir_vec)

    proj_A = A @ dir_unit
    proj_B = B @ dir_unit
    A_max = np.max(proj_A)
    B_min = np.min(proj_B)
    gap = A_max - B_min
    return B + gap * dir_unit

def average_polygon_radius(poly: np.ndarray) -> float:
    center = poly.mean(axis=0)
    return np.mean(np.linalg.norm(poly - center, axis=1))

# ——— Main experiment: repeat trials and average ———


def pad_polygon(poly: np.ndarray, target_N: int) -> np.ndarray:
    pad_len = target_N - poly.shape[0]
    return np.pad(poly, ((0, pad_len), (0, 0)), mode='constant')

if __name__ == "__main__":
    batched_xtanh = jax.jit(jax.vmap(compute_distance_smoothabs, in_axes=(0, 0, None, None)))
    batched_onelog = jax.jit(jax.vmap(compute_distance_onelogminus, in_axes=(0, 0, None)))
    batched_sat = jax.jit(jax.vmap(compute_distance_sat, in_axes=(0, 0)))  # 你需要定义好 sat 函数

    def pad_polygon(poly: np.ndarray, target_N: int) -> np.ndarray:
        pad_len = target_N - poly.shape[0]
        return np.pad(poly, ((0, pad_len), (0, 0)), mode='constant')

    polygon_sizes = [4, 8, 16, 32, 64]
    outer_repeats = 1000
    batch_n = 32

    print("    N   | xtanh(ms) | onelog(ms) | SAT(ms) | speedup | xtanh_err (%) [min,max] | onelog_err (%) [min,max]")
    print("--------+-----------+------------+---------+---------+--------------------------+---------------------------")


    for N in polygon_sizes:
        # === Generate polygon batches ===
        As = [regular_ngon(4, radius=1.0) for _ in range(batch_n)]
        Bs = [irregular_ngon(N, radius=5.0, seed=100 + i) for i in range(batch_n)]
        B_aligneds = [align_polygons_for_contact(A, B) for A, B in zip(As, Bs)]
        max_N = max(p.shape[0] for p in B_aligneds)

        A_batch = jnp.array(As)  # (batch, 4, 2)
        B_batch = jnp.array([pad_polygon(B, max_N) for B in B_aligneds])  # (batch, max_N, 2)

        # === Warmup ===
        _ = batched_xtanh(A_batch, B_batch, 30., 30.).block_until_ready()
        _ = batched_onelog(A_batch, B_batch, 30.).block_until_ready()
        _ = batched_sat(A_batch, B_batch).block_until_ready()

        # === Timed batch evaluation ===
        def time_fn(fn, *args):
            times = []
            for _ in range(outer_repeats):
                t0 = time.perf_counter()
                out = fn(*args)
                _ = out.block_until_ready()
                t1 = time.perf_counter()
                times.append(t1 - t0)
            return np.mean(times) * 1e3  # ms per pair

        t_xtanh = time_fn(batched_xtanh, A_batch, B_batch, 30., 30.)
        t_onelog = time_fn(batched_onelog, A_batch, B_batch, 30.)
        t_sat = time_fn(batched_sat, A_batch, B_batch)
        speedup = t_xtanh / t_onelog if t_onelog > 0 else float('nan')

        # === 误差评估（逐对） ===
        sum_xtanh_d = sum_onelog_d = 0.0
        xtanh_err_list = []
        onelog_err_list = []
        radius_sum = 0.0

        for A, B in zip(As, B_aligneds):
            d_sat = float(sat_jit(jnp.array(A), jnp.array(B)))
            d_xtanh = float(xtanh_jit(jnp.array(A), jnp.array(B)))
            d_onelog = float(onelog_jit(jnp.array(A), jnp.array(B)))
            r = average_polygon_radius(np.array(B))
            radius_sum += r

            xtanh_err_list.append((d_xtanh - d_sat) / r)
            onelog_err_list.append((d_onelog - d_sat) / r)

        xtanh_err_array = np.array(xtanh_err_list)
        onelog_err_array = np.array(onelog_err_list)

        rel_err_xtanh_mean = np.mean(xtanh_err_array)
        rel_err_onelog_mean = np.mean(onelog_err_array)

        rel_err_xtanh_min = np.min(xtanh_err_array)
        rel_err_xtanh_max = np.max(xtanh_err_array)

        rel_err_onelog_min = np.min(onelog_err_array)
        rel_err_onelog_max = np.max(onelog_err_array)

        # print(f"{N:7d} | {t_xtanh:9.3f} | {t_onelog:10.3f} | {t_sat:7.3f} | {speedup:7.2f} |"
        #     f"     {rel_err_xtanh:+.6f}   |   {rel_err_onelog:+.6f}")

        print(f"{N:7d} | {t_xtanh:9.3f} | {t_onelog:10.3f} | {t_sat:7.3f} | {speedup:7.2f} |"
      f" {rel_err_xtanh_mean:+.6f} [{rel_err_xtanh_min:+.3f}, {rel_err_xtanh_max:+.3f}] |"
      f" {rel_err_onelog_mean:+.6f} [{rel_err_onelog_min:+.3f}, {rel_err_onelog_max:+.3f}]")


        
    def pad_polygon(poly: np.ndarray, target_N: int) -> np.ndarray:
        pad_len = target_N - poly.shape[0]
        return np.pad(poly, ((0, pad_len), (0, 0)), mode='constant')

    batch_sizes = [32, 64, 128, 256]
    polygon_N = 8
    repeats = 10

    print("\nBatch size | xtanh (ms/pair) | onelog (ms/pair) | speedup")
    print("-----------+----------------+-------------------+---------")

    for batch_size in batch_sizes:
        # Generate batched polygon pairs
        As = [regular_ngon(4, radius=0.5) for _ in range(batch_size)]
        Bs = [irregular_ngon(polygon_N, radius=1.0, seed=100 + i) for i in range(batch_size)]
        B_aligneds = [align_polygons_for_contact(A, B) for A, B in zip(As, Bs)]

        A_batch = jnp.array(As)
        max_N = max(p.shape[0] for p in B_aligneds)
        B_padded = np.stack([pad_polygon(p, max_N) for p in B_aligneds])
        B_batch = jnp.array(B_padded)

        # Batched vmap-wrapped functions
        batched_xtanh = jax.jit(jax.vmap(compute_distance_smoothabs, in_axes=(0, 0, None, None)))
        batched_onelog = jax.jit(jax.vmap(compute_distance_onelogminus, in_axes=(0, 0, None)))

        # Warmup
        _ = batched_xtanh(A_batch, B_batch, 30., 30.).block_until_ready()
        _ = batched_onelog(A_batch, B_batch, 30.).block_until_ready()

        def time_fn(fn, *args):
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = fn(*args)
                _ = out.block_until_ready()
                t1 = time.perf_counter()
                times.append(t1 - t0)
            return np.mean(times) * 1e3  # ms per pair

        t_xtanh = time_fn(batched_xtanh, A_batch, B_batch, 30., 30.)
        t_onelog = time_fn(batched_onelog, A_batch, B_batch, 30.)

        print(f"{batch_size:10d} |     {t_xtanh:10.4f}     |     {t_onelog:10.4f}     |  {t_xtanh / t_onelog:6.2f}")