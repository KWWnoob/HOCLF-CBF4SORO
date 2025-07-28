import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

@jax.jit
def segment_segment_closest_points(a0, a1, b0, b1):
    """Compute closest points on two segments a0–a1 and b0–b1."""
    A = a1 - a0
    B = b1 - b0
    T = a0 - b0

    A_dot_A = jnp.dot(A, A)
    B_dot_B = jnp.dot(B, B)
    A_dot_B = jnp.dot(A, B)
    A_dot_T = jnp.dot(A, T)
    B_dot_T = jnp.dot(B, T)

    denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B

    def compute_st():
        s = (A_dot_B * B_dot_T - B_dot_B * A_dot_T) / denom
        t = (A_dot_A * B_dot_T - A_dot_B * A_dot_T) / denom
        return s, t

    def fallback_st():
        return 0.0, jnp.clip(B_dot_T / B_dot_B, 0.0, 1.0)

    s, t = jax.lax.cond(denom > 1e-8, compute_st, fallback_st)

    s = jnp.clip(s, 0.0, 1.0)
    t = jnp.clip(t, 0.0, 1.0)

    p_closest = a0 + s * A
    q_closest = b0 + t * B
    dist = jnp.linalg.norm(p_closest - q_closest)

    return p_closest, q_closest, dist

@jax.jit
def find_closest_segment_pair(poly1: jnp.ndarray, poly2: jnp.ndarray):
    N1 = poly1.shape[0]
    N2 = poly2.shape[0]

    seg1_start = poly1
    seg1_end = jnp.roll(poly1, -1, axis=0)
    seg2_start = poly2
    seg2_end = jnp.roll(poly2, -1, axis=0)

    def one_edge_pair(a0, a1):
        def inner(b0, b1):
            return segment_segment_closest_points(a0, a1, b0, b1)
        return jax.vmap(inner)(seg2_start, seg2_end)

    # (N1, N2, 3)
    p_closest, q_closest, dists = jax.vmap(one_edge_pair)(seg1_start, seg1_end)

    dists_flat = dists.reshape(-1)
    p_flat = p_closest.reshape(-1, 2)
    q_flat = q_closest.reshape(-1, 2)

    idx = jnp.argmin(dists_flat)
    return p_flat[idx], q_flat[idx], dists_flat[idx]

poly1 = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [2.0, 1.0],
    [0.0, 1.0]
])

poly2 = np.array([
    [3.0, 0.5],
    [5.0, 0.5],
    [5.0, 2.0],
    [3.0, 2.0]
]) - np.array([
    [0.0, 0.5]
])

# Assume closest points were computed via JAX
p_closest = np.array([2.0, 0.5])
q_closest = np.array([3.0, 0.5])
dist = np.linalg.norm(p_closest - q_closest)

# Plot polygons
plt.figure(figsize=(6, 6))
plt.plot(*np.vstack([poly1, poly1[0]]).T, 'b-o', label='Polygon 1')
plt.plot(*np.vstack([poly2, poly2[0]]).T, 'r-o', label='Polygon 2')

# Draw closest points and connecting segment
plt.plot([p_closest[0], q_closest[0]], [p_closest[1], q_closest[1]], 'k--', lw=2, label=f'Distance: {dist:.2f}')
plt.scatter(*p_closest, color='blue', s=60, label='Closest on Poly1')
plt.scatter(*q_closest, color='red', s=60, label='Closest on Poly2')

plt.legend()
plt.grid(True)
plt.axis('equal')
plt.title("Closest Points Between Two Polygons (Edge-to-Edge)")
plt.show()
