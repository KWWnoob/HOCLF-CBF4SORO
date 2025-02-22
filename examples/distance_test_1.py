import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# GJK–EPA Helper Functions
# ===============================
def support(shape, direction):
    dots = jnp.dot(shape, direction)
    idx = jnp.argmax(dots)
    return shape[idx]

def minkowski_support(shape1, shape2, direction):
    return support(shape1, direction) - support(shape2, -direction)

def update_simplex(simplex, direction):
    # For 2D, handle line and triangle cases
    if simplex.shape[0] == 2:
        A = simplex[1]
        B = simplex[0]
        AB = B - A
        AO = -A
        # Perpendicular to AB that points towards the origin
        perp = jnp.array([-AB[1], AB[0]])
        perp = jax.lax.select(jnp.dot(perp, AO) < 0, -perp, perp)
        direction = perp
        return False, simplex, direction
    elif simplex.shape[0] == 3:
        # In this simplified 2D example, we assume the origin is inside
        return True, simplex, direction
    else:
        return False, simplex, direction

def gjk(shape1, shape2, max_iters=20):
    direction = jnp.array([1.0, 0.0])
    A = minkowski_support(shape1, shape2, direction)
    simplex = jnp.expand_dims(A, axis=0)
    direction = -A
    collision = False
    for i in range(max_iters):
        A = minkowski_support(shape1, shape2, direction)
        if jnp.dot(A, direction) <= 0:
            collision = False
            break
        simplex = jnp.concatenate([simplex, jnp.expand_dims(A, axis=0)], axis=0)
        collision, simplex, direction = update_simplex(simplex, direction)
        if collision:
            break
    return collision, simplex

def find_closest_edge(polytope):
    min_distance = 1e9
    closest_edge_index = -1
    closest_normal = jnp.array([0.0, 0.0])
    N = polytope.shape[0]
    for i in range(N):
        j = (i + 1) % N
        A = polytope[i]
        B = polytope[j]
        edge = B - A
        # Outward normal
        normal = jnp.array([edge[1], -edge[0]])
        normal_norm = jnp.linalg.norm(normal) + 1e-8
        normal = normal / normal_norm
        distance = jnp.dot(normal, A)
        if distance < min_distance:
            min_distance = distance
            closest_edge_index = j
            closest_normal = normal
    return closest_edge_index, min_distance, closest_normal

def epa(shape1, shape2, simplex, max_iters=30, tolerance=1e-6):
    polytope = simplex
    for i in range(max_iters):
        edge_index, distance, normal = find_closest_edge(polytope)
        support_point = minkowski_support(shape1, shape2, normal)
        d = jnp.dot(support_point, normal)
        if d - distance < tolerance:
            penetration_depth = d
            penetration_vector = normal * penetration_depth
            return penetration_vector
        polytope = jnp.insert(polytope, edge_index, support_point, axis=0)
    return normal * d

def distance_to_simplex(simplex):
    if simplex.shape[0] == 1:
        return jnp.linalg.norm(simplex[0])
    elif simplex.shape[0] == 2:
        A = simplex[0]
        B = simplex[1]
        AB = B - A
        t = -jnp.dot(A, AB) / (jnp.dot(AB, AB) + 1e-8)
        t = jnp.clip(t, 0.0, 1.0)
        closest = A + t * AB
        return jnp.linalg.norm(closest)
    elif simplex.shape[0] == 3:
        def point_to_segment_distance(A, B):
            AB = B - A
            t = -jnp.dot(A, AB) / (jnp.dot(AB, AB) + 1e-8)
            t = jnp.clip(t, 0.0, 1.0)
            closest = A + t * AB
            return jnp.linalg.norm(closest)
        d0 = point_to_segment_distance(simplex[0], simplex[1])
        d1 = point_to_segment_distance(simplex[1], simplex[2])
        d2 = point_to_segment_distance(simplex[2], simplex[0])
        return jnp.minimum(jnp.minimum(d0, d1), d2)
    else:
        return 1e9

def compute_signed_distance_gjk(shape1, shape2):
    collision, simplex = gjk(shape1, shape2)
    if collision:
        penetration_vector = epa(shape1, shape2, simplex)
        penetration_depth = jnp.linalg.norm(penetration_vector)
        signed_distance = -penetration_depth
        return signed_distance
    else:
        separation_distance = distance_to_simplex(simplex)
        return separation_distance

# ===============================
# Kiwan’s SAT Helper Functions
# ===============================
def get_normals(vertices):
    edges = jnp.roll(vertices, -1, axis=0) - vertices
    normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
    norms = jnp.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals / norms

def compute_polygon_centroid(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_next = jnp.roll(x, -1)
    y_next = jnp.roll(y, -1)
    cross = x * y_next - x_next * y
    area = jnp.sum(cross) / 2.0
    Cx = jnp.sum((x + x_next) * cross) / (6.0 * area)
    Cy = jnp.sum((y + y_next) * cross) / (6.0 * area)
    return jnp.array([Cx, Cy])

def cross2D(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def ray_polygon_intersection(O, d, vertices, eps=1e-8):
    A = vertices
    B = jnp.concatenate([vertices[1:], vertices[:1]], axis=0)
    BA = B - A
    denom = cross2D(d, BA)
    A_minus_O = A - O
    t = cross2D(A_minus_O, BA) / denom
    u = cross2D(A_minus_O, d) / denom
    valid = (jnp.abs(denom) > eps) & (t >= 0) & (u >= 0) & (u <= 1)
    t_valid = jnp.where(valid, t, jnp.inf)
    return jnp.min(t_valid)

def compute_gap_along_centers(vertices1, vertices2, eps=1e-8):
    C1 = compute_polygon_centroid(vertices1)
    C2 = compute_polygon_centroid(vertices2)
    d_vec = C2 - C1
    d_norm = jnp.linalg.norm(d_vec) + 1e-8
    d = d_vec / d_norm
    r1 = ray_polygon_intersection(C1, d, vertices1, eps)
    r2 = ray_polygon_intersection(C2, -d, vertices2, eps)
    gap = d_norm - (r1 + r2)
    return gap

def compute_distance_sat(vertices1, vertices2):
    robot_normals = get_normals(vertices1)
    poly_normals  = get_normals(vertices2)
    candidate_axes = jnp.concatenate([robot_normals, poly_normals], axis=0)
    
    proj_robot = vertices1 @ candidate_axes.T
    proj_poly  = vertices2 @ candidate_axes.T
    
    min_R = jnp.min(proj_robot, axis=0)
    max_R = jnp.max(proj_robot, axis=0)
    min_P = jnp.min(proj_poly, axis=0)
    max_P = jnp.max(proj_poly, axis=0)
    
    separated_mask = (max_R < min_P) | (max_P < min_R)
    penetration = jnp.minimum(max_R, max_P) - jnp.maximum(min_R, min_P)
    
    def separated_case(_):
        gap = compute_gap_along_centers(vertices1, vertices2)
        return gap
    
    def overlapping_case(_):
        pen = -jnp.min(penetration)
        return pen

    overall_distance = jax.lax.cond(
        jnp.any(separated_mask),
        separated_case,
        overlapping_case,
        operand=None
    )
    return overall_distance

# ===============================
# Comparison over Circular Motion
# ===============================
def compare_algorithms():
    # Define the two polygons (using the same shapes for both methods)
    fixed_polygon = jnp.array([
        [0.0, 0.2],
        [0.2, 0.0],
        [0.8, 0.0],
        [1.0, 0.2],
        [1.0, 0.8],
        [0.8, 1.0],
        [0.2, 1.0],
        [0.0, 0.8]
    ])
    
    base_moving_polygon = jnp.array([
        [1.5, 0.5],
        [2.5, 0.5],
        [2.5, 1.5],
        [1.5, 1.5]
    ])
    
    num_steps = 400
    angles = np.linspace(0, 2 * np.pi, num_steps)
    radius = 1.0
    
    signed_distances_gjk = []
    signed_distances_sat = []
    
    for angle in angles:
        translation = jnp.array([radius * np.cos(angle), radius * np.sin(angle)])
        moved_polygon = base_moving_polygon + translation
        
        # GJK–EPA signed distance
        sd_gjk = compute_signed_distance_gjk(fixed_polygon, moved_polygon)
        signed_distances_gjk.append(float(sd_gjk))
        
        # Kiwan's SAT signed distance
        sd_sat = compute_distance_sat(fixed_polygon, moved_polygon)
        signed_distances_sat.append(float(sd_sat))
    
# Plot the two results with transparency
    plt.figure(figsize=(10, 5))
    plt.plot(angles, signed_distances_gjk, label="GJK–EPA", marker='o', markersize=3, linestyle='-', alpha=1.0)
    plt.plot(angles, signed_distances_sat, label="Kiwan’s SAT", marker='s', markersize=3, linestyle='--', alpha=0.3)
    plt.xlabel("Angle (radians)")
    plt.ylabel("Signed Distance")
    plt.title("Comparison of Signed Distances (GJK–EPA vs. Kiwan’s SAT)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    compare_algorithms()
