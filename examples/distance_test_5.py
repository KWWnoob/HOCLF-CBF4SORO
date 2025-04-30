import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

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

A_j = jnp.array(A)
B_j = jnp.array(B)
pt0 = jnp.array([0.0, 0.0])

def compute_distance_twolog(robot, poly, alpha_pair=2080., alpha_axes=9000.):
    def get_normals(v):
        e = jnp.roll(v, -1, axis=0) - v
        n = jnp.stack([-e[:,1], e[:,0]],axis=1)
        return n / jnp.linalg.norm(n,axis=1,keepdims=True)
    Rn = get_normals(robot); Pn = get_normals(poly)
    axes = jnp.concatenate([Rn, Pn], axis=0)
    prj_R = robot @ axes.T; prj_P = poly @ axes.T
    Rmin,Rmax = prj_R.min(0), prj_R.max(0)
    Pmin,Pmax = prj_P.min(0), prj_P.max(0)
    d1 = Pmin - Rmax; d2 = Rmin - Pmax
    axis_gaps = (1/alpha_pair) * logsumexp(alpha_pair * jnp.stack([d1,d2]), axis=0)
    h = (1/alpha_axes) * logsumexp(alpha_axes * axis_gaps)
    return h

@jax.jit
def compute_distance_onelog(robot, poly, alpha_axes=9000.):
    def get_normals(v):
        e = jnp.roll(v, -1, axis=0) - v
        n = jnp.stack([-e[:,1], e[:,0]],axis=1)
        return n / jnp.linalg.norm(n,axis=1,keepdims=True)
    Rn = get_normals(robot); Pn = get_normals(poly)
    axes = jnp.concatenate([Rn, Pn], axis=0)
    prj_R = robot @ axes.T; prj_P = poly @ axes.T
    d1 = prj_P.min(0) - prj_R.max(0)
    d2 = prj_R.min(0) - prj_P.max(0)
    flat = jnp.stack([d1, d2]).reshape(-1)
    h = (1/alpha_axes) * logsumexp(alpha_axes * flat)
    return h


def compute_distance_sat(robot, poly):
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


# JIT    
twolog_jit = jax.jit(compute_distance_twolog)
sat_jit = jax.jit(compute_distance_sat)
_ = sat_jit(A_j, B_j).block_until_ready()
_ = twolog_jit(A_j, B_j).block_until_ready()
_ = compute_distance_onelog(A_j, B_j).block_until_ready()

def benchmark(fn, *args, n=20000):
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
        try:
            out.block_until_ready()
        except:
            pass
    t1 = time.perf_counter()
    return (t1 - t0) / n * 1e3  

t_twolog = benchmark(twolog_jit,       A_j, B_j)
t_onelog = benchmark(compute_distance_onelog, A_j, B_j)
t_sat    = benchmark(sat_jit,          A_j, B_j)

print(f"Two-step LSE average latency: {t_twolog:.3f} ms")
print(f"One-step LSE average latency: {t_onelog:.3f} ms")
print(f"Basic SAT average latency:     {t_sat:.3f} ms")
