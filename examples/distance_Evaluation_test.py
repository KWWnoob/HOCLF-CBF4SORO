import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# ——— Distance function implementations ———

def smooth_abs(x, alpha):
    return x * jnp.tanh(alpha * x)
@jax.jit
def compute_distance_smoothabs(robot, poly, alpha_axes=9000., alpha_sabs=1000.):
    """
    Distance using smooth absolute value replacement.
    """
    def get_normals(v):
        e = jnp.roll(v, -1, axis=0) - v
        n = jnp.stack([-e[:,1], e[:,0]], axis=1)
        return n / jnp.linalg.norm(n, axis=1, keepdims=True)
    
    # Centroids
    p_robot = robot.mean(axis=0)
    p_poly  = poly.mean(axis=0)
    delta_p = p_poly - p_robot

    # Normals
    axes = jnp.concatenate([get_normals(robot), get_normals(poly)], axis=0)

    def half_extent(v, axis):
        projections = (v - v.mean(axis=0)) @ axis
        return smooth_abs(projections, alpha_sabs).max()

    axis_gaps = []
    for ni in axes:
        proj_dist = smooth_abs(ni @ delta_p, alpha_sabs)
        rA = half_extent(robot, ni)
        rB = half_extent(poly, ni)
        gap_i = proj_dist - (rA + rB)
        axis_gaps.append(gap_i)

    axis_gaps = jnp.stack(axis_gaps)
    h = (1 / alpha_axes) * logsumexp(alpha_axes * axis_gaps)
    return h

def compute_distance_twolog(robot, poly, alpha_pair=2080., alpha_axes=9000.):
    """
    Two-step smooth distance using nested LogSumExp:
    1) Smooth max between penetration and separation on each axis.
    2) Smooth max across all axes.
    """
    def get_normals(v):
        # Compute outward normals of each edge
        e = jnp.roll(v, -1, axis=0) - v
        n = jnp.stack([-e[:,1], e[:,0]], axis=1)
        return n / jnp.linalg.norm(n, axis=1, keepdims=True)
    Rn = get_normals(robot)
    Pn = get_normals(poly)
    axes = jnp.concatenate([Rn, Pn], axis=0)
    prj_R = robot @ axes.T
    prj_P = poly  @ axes.T
    Rmin, Rmax = prj_R.min(0), prj_R.max(0)
    Pmin, Pmax = prj_P.min(0), prj_P.max(0)
    d1 = Pmin - Rmax  # separation in forward direction
    d2 = Rmin - Pmax  # separation in reverse direction
    # Smooth max over d1, d2 for each axis
    axis_gaps = (1/alpha_pair) * logsumexp(alpha_pair * jnp.stack([d1, d2]), axis=0)
    # Smooth max over all axes
    h = (1/alpha_axes) * logsumexp(alpha_axes * axis_gaps)
    return h

@jax.jit
def compute_distance_onelog(robot, poly, alpha_axes=9000.):
    """
    Single-step smooth distance using one LogSumExp over all axis gaps.
    JIT-compiled for efficiency.
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
    flat = jnp.stack([d1, d2]).reshape(-1)
    # Smooth max across all gaps
    return (1.0/alpha_axes) * logsumexp(alpha_axes * flat)

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
twolog_jit = jax.jit(compute_distance_twolog)
sat_jit    = jax.jit(compute_distance_sat)

# ——— Helper functions ———

def regular_ngon(N, radius=1.0):
    """
    Generate vertices of a regular N-gon centered at origin.
    """
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.stack([radius * np.cos(angles),
                     radius * np.sin(angles)], axis=1)

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

# ——— Main experiment: repeat trials and average ———

if __name__ == "__main__":
    polygon_sizes = [4, 8, 16, 32]
    outer_repeats = 100  # number of outer average loops

    print("    N   | lse-xtanh(ms) | one-log(ms) | SAT(ms)  | speedup")
    print("--------+-------------+-------------+----------+---------")
    for N in polygon_sizes:
        # Create robot (square) and obstacle (N-gon)
        A = regular_ngon(4, radius=0.5)
        B = regular_ngon(N, radius=1.0)
        A_j = jnp.array(A)
        B_j = jnp.array(B)

        # Warm up JIT
        # _ = twolog_jit(A_j, B_j).block_until_ready()
        _ = compute_distance_onelog(A_j, B_j).block_until_ready()
        _ = sat_jit(A_j, B_j).block_until_ready()
        _ = xtanh_jit(A_j, B_j).block_until_ready()

        # Accumulate timings
        sum_two = sum_one = sum_sat = sum_xtanh = 0.0
        for _ in range(outer_repeats):
            # sum_two += benchmark(twolog_jit, A_j, B_j, n=1000)
            sum_one += benchmark(compute_distance_onelog, A_j, B_j, n=1000)
            sum_sat += benchmark(sat_jit, A_j, B_j, n=1000)
            sum_xtanh += benchmark(xtanh_jit, A_j, B_j, n=1000)

        # Compute average
        avg_two = sum_two / outer_repeats
        avg_one = sum_one / outer_repeats
        avg_sat = sum_sat / outer_repeats
        avg_xtanh = sum_xtanh / outer_repeats
        speedup = avg_xtanh / avg_one if avg_one > 0 else float('nan')

        print(f"{N:7d} | {avg_xtanh:11.3f} | {avg_one:11.3f} | {avg_sat:8.3f} | {speedup:7.2f}")
