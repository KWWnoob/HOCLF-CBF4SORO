import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

# ——— Two-step LogSumExp distance ———
def compute_distance_twolog(robot: jnp.ndarray,
                            poly:  jnp.ndarray,
                            alpha_pair: float = 2080.0,
                            alpha_axes: float = 9000.0) -> jnp.ndarray:
    def get_normals(v):
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

    d1 = Pmin - Rmax
    d2 = Rmin - Pmax

    axis_gaps = (1.0/alpha_pair) * logsumexp(alpha_pair * jnp.stack([d1, d2]), axis=0)
    h = (1.0/alpha_axes) * logsumexp(alpha_axes * axis_gaps)
    return h

twolog_jit = jax.jit(compute_distance_twolog)


# ——— One-step LogSumExp distance ———
@jax.jit
def compute_distance_onelog(pt:   jnp.ndarray,
                            robot:jnp.ndarray,
                            poly: jnp.ndarray,
                            alpha_axes: float = 9000.0) -> jnp.ndarray:
    r = robot + pt

    def get_normals(v):
        e = jnp.roll(v, -1, axis=0) - v
        n = jnp.stack([-e[:,1], e[:,0]], axis=1)
        return n / jnp.linalg.norm(n, axis=1, keepdims=True)

    Rn = get_normals(r)
    Pn = get_normals(poly)
    axes = jnp.concatenate([Rn, Pn], axis=0)

    prj_R = r    @ axes.T
    prj_P = poly @ axes.T

    d1 = prj_P.min(0) - prj_R.max(0)
    d2 = prj_R.min(0) - prj_P.max(0)

    flat = jnp.stack([d1, d2]).reshape(-1)
    h = (1.0/alpha_axes) * logsumexp(alpha_axes * flat)
    return h


# ——— Basic SAT distance ———
def compute_distance_sat(robot: jnp.ndarray,
                         poly:  jnp.ndarray) -> jnp.ndarray:
    def get_normals(v):
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

    d1 = Pmin - Rmax
    d2 = Rmin - Pmax

    gaps = jnp.maximum(d1, d2)
    return jnp.max(gaps)

sat_jit = jax.jit(compute_distance_sat)


# ——— Top-K One-step LSE distance ———
@partial(jax.jit, static_argnames=('K',))
def compute_distance_topk(pt:   jnp.ndarray,
                          robot:jnp.ndarray,
                          poly: jnp.ndarray,
                          alpha_axes: float = 9000.0,
                          K: int = 16) -> jnp.ndarray:
    r = robot + pt

    def get_normals(v):
        e = jnp.roll(v, -1, axis=0) - v
        n = jnp.stack([-e[:,1], e[:,0]], axis=1)
        return n / jnp.linalg.norm(n, axis=1, keepdims=True)

    Rn = get_normals(r)
    Pn = get_normals(poly)
    axes = jnp.concatenate([Rn, Pn], axis=0)

    prj_R = r    @ axes.T
    prj_P = poly @ axes.T

    d1 = prj_P.min(0) - prj_R.max(0)
    d2 = prj_R.min(0) - prj_P.max(0)

    flat = jnp.concatenate([d1, d2])
    topk_vals, _ = lax.top_k(flat, K)

    return (1.0/alpha_axes) * logsumexp(alpha_axes * topk_vals)


# ——— Benchmark helper ———
def benchmark(fn, *args, n=5000):
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
        out.block_until_ready()
    t1 = time.perf_counter()
    return (t1 - t0) / n * 1e3  # ms


# ——— Main: ———
if __name__ == "__main__":
    A = np.array([[-0.5, -0.5],
                  [ 0.5, -0.5],
                  [ 0.5,  0.5],
                  [-0.5,  0.5]])
    B = np.array([[0.2, 0.1],
                  [1.0, 0.0],
                  [1.8, 0.3],
                  [2.0, 1.0],
                  [1.5, 1.8],
                  [0.7, 2.0],
                  [0.1, 1.5],
                  [0.0, 0.5]])
    A_j  = jnp.array(A)
    B_j  = jnp.array(B)
    pt0 = jnp.array([0.0, 0.0])

    # JIT warmup
    _ = twolog_jit(A_j, B_j).block_until_ready()
    _ = compute_distance_onelog(pt0, A_j, B_j).block_until_ready()
    _ = sat_jit(A_j, B_j).block_until_ready()
    _ = compute_distance_topk(pt0, A_j, B_j, 9000.0, 16).block_until_ready()

    print("Method           | avg latency (ms)")
    print("-----------------+-----------------")
    print(f"two-step LSE     | {benchmark(twolog_jit,          A_j, B_j):9.3f}")
    print(f"one-step LSE     | {benchmark(compute_distance_onelog, pt0, A_j, B_j):9.3f}")
    print(f"basic SAT        | {benchmark(sat_jit,             A_j, B_j):9.3f}")
    for K in [4, 8, 16, 32]:
        t = benchmark(compute_distance_topk, pt0, A_j, B_j, 9000.0, K)
        print(f"one-step LSE Top-{K:<2d} | {t:9.3f}")
