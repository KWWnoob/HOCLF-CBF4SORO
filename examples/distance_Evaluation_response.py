import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from dpax.endpoints import proximity  # dpax: pip install git+https://github.com/kevin-tracy/dpax

# ----- DCOL via dpax: for 2D capsules -----
@jax.jit
def dcol_dpax_capsule():
    R1 = 0.2
    a1 = jnp.array([0.0, 0.0, 0.0])
    b1 = jnp.array([1.0, 0.0, 0.0])
    R2 = 0.2
    a2 = jnp.array([0.5, 0.3, 0.0])
    b2 = jnp.array([1.5, 0.3, 0.0])
    return proximity(R1, a1, b1, R2, a2, b2)

# ----- OneLogMinus: for 2D polygons -----
@jax.jit
def onelogminus(robot, poly, alpha_outer=500, alpha_inner=500):
    def get_normals(verts):
        edges = jnp.roll(verts, -1, axis=0) - verts
        normals = jnp.stack([-edges[:,1], edges[:,0]], axis=1)
        return normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
    
    Rn = get_normals(robot)
    Pn = get_normals(poly)
    axes = jnp.vstack((Rn, Pn))

    proj_R = robot @ axes.T
    proj_P = poly  @ axes.T

    R_max = (1./alpha_inner)*logsumexp(alpha_inner * proj_R, axis=0)
    R_min = -(1./alpha_inner)*logsumexp(-alpha_inner * proj_R, axis=0)
    P_max = (1./alpha_inner)*logsumexp(alpha_inner * proj_P, axis=0)
    P_min = -(1./alpha_inner)*logsumexp(-alpha_inner * proj_P, axis=0)

    d1 = P_min - R_max
    d2 = R_min - P_max
    gap_soft = (1./alpha_inner)*logsumexp(alpha_inner*jnp.stack([d1,d2]), axis=0)

    h = (1./alpha_outer)*logsumexp(alpha_outer * gap_soft)
    err = jnp.log(gap_soft.shape[0]) / alpha_outer
    return h - err

# ----- Geometry Generator -----
def regular_ngon(N, radius=1.0):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

# ----- Benchmark Helper -----
def benchmark(fn, *args, n=1000):
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args) if args else fn()
        try: out.block_until_ready()
        except: pass
    t1 = time.perf_counter()
    return (t1 - t0) / n * 1e3  # ms

# ----- Run Benchmarks -----
if __name__ == "__main__":
    print("Comparing dpax (capsule) vs OneLogMinus (polygon):\n")

    # Prepare capsule + warmup
    _ = dcol_dpax_capsule().block_until_ready()
    dpax_time = benchmark(dcol_dpax_capsule, n=1000)

    # Prepare polygon + warmup
    robot = regular_ngon(4, 0.5)
    poly = regular_ngon(8, 1.0) + np.array([0.2, 0.3])
    robot_j = jnp.array(robot)
    poly_j = jnp.array(poly)
    _ = onelogminus(robot_j, poly_j).block_until_ready()
    onelog_time = benchmark(onelogminus, robot_j, poly_j, n=1000)

    # Print results
    print(f"DCOL dpax (capsule):      {dpax_time:.3f} ms")
    print(f"OneLogMinus (polygon):    {onelog_time:.3f} ms")
    print(f"Speedup (OLSAT faster):   {dpax_time / onelog_time:.2f}x\n")
