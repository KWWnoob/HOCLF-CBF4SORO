'''
GJK - EPA
'''

import jax
import jax.numpy as jnp

# -------------------------------
# Support function: Compute the furthest point in a shape along a given direction
def support(shape, direction):
    # shape: [N, 2] array
    dots = jnp.dot(shape, direction)
    idx = jnp.argmax(dots)
    return shape[idx]

# Minkowski difference support function
def minkowski_support(shape1, shape2, direction):
    return support(shape1, direction) - support(shape2, -direction)

# -------------------------------
# Update simplex (simplified version for 2D only)
def update_simplex(simplex, direction):
    # Assume the last point added in simplex is A
    if simplex.shape[0] == 2:
        # Line segment case: let A be the newest point, B be the previous point
        A = simplex[1]
        B = simplex[0]
        AB = B - A
        AO = -A
        # Compute the perpendicular direction of AB
        perp = jnp.array([-AB[1], AB[0]])
        # Ensure the perpendicular direction points towards the origin
        perp = jax.lax.select(jnp.dot(perp, AO) < 0, -perp, perp)
        direction = perp
        return False, simplex, direction
    elif simplex.shape[0] == 3:
        # Triangle case: for the example, assume the origin is inside (collision)
        return True, simplex, direction
    else:
        return False, simplex, direction

# -------------------------------
# GJK algorithm (2D example)
def gjk(shape1, shape2, max_iters=20):
    # Initial search direction
    direction = jnp.array([1.0, 0.0])
    A = minkowski_support(shape1, shape2, direction)
    simplex = jnp.expand_dims(A, axis=0)  # Initial simplex is a single point
    direction = -A

    collision = False
    for i in range(max_iters):
        A = minkowski_support(shape1, shape2, direction)
        # If the projection of the new point along the search direction is not greater than 0,
        # then the origin is not within the Minkowski difference
        if jnp.dot(A, direction) <= 0:
            collision = False
            break
        # Add the new point to the simplex
        simplex = jnp.concatenate([simplex, jnp.expand_dims(A, axis=0)], axis=0)
        collision, simplex, direction = update_simplex(simplex, direction)
        if collision:
            break
    return collision, simplex

# -------------------------------
# EPA algorithm: Expand the simplex to obtain the penetration vector during collision
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
        # Compute the outward normal of the edge (normalized)
        normal = jnp.array([edge[1], -edge[0]])
        normal_norm = jnp.linalg.norm(normal) + 1e-8
        normal = normal / normal_norm
        # Distance from the origin to edge A (projection)
        distance = jnp.dot(normal, A)
        if distance < min_distance:
            min_distance = distance
            closest_edge_index = j
            closest_normal = normal
    return closest_edge_index, min_distance, closest_normal

def epa(shape1, shape2, simplex, max_iters=30, tolerance=1e-6):
    polytope = simplex  # Use the initial polytope from GJK
    for i in range(max_iters):
        edge_index, distance, normal = find_closest_edge(polytope)
        support_point = minkowski_support(shape1, shape2, normal)
        d = jnp.dot(support_point, normal)
        # When the difference between the new support point's distance and the edge distance is less than tolerance, consider converged
        if d - distance < tolerance:
            penetration_depth = d
            penetration_vector = normal * penetration_depth
            return penetration_vector
        # Insert the new point into the polytope
        polytope = jnp.insert(polytope, edge_index, support_point, axis=0)
    return normal * d

# -------------------------------
# Compute the shortest distance from the origin to the simplex (used when there is no collision)
def distance_to_simplex(simplex):
    if simplex.shape[0] == 1:
        # Single point
        return jnp.linalg.norm(simplex[0])
    elif simplex.shape[0] == 2:
        # Line segment: compute the shortest distance from the origin to the line segment
        A = simplex[0]
        B = simplex[1]
        AB = B - A
        # Projection coefficient
        t = -jnp.dot(A, AB) / (jnp.dot(AB, AB) + 1e-8)
        t = jnp.clip(t, 0.0, 1.0)
        closest = A + t * AB
        return jnp.linalg.norm(closest)
    elif simplex.shape[0] == 3:
        # Triangle: compute the distance from the origin to each edge and take the minimum
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

# -------------------------------
# Comprehensive example: Compute the "signed distance"
#
# If there is no collision: return a positive value indicating the separation distance between the shapes;
# If there is a collision: return a negative value, whose absolute value represents the penetration depth.
def compute_signed_distance(shape1, shape2):
    collision, simplex = gjk(shape1, shape2)
    if collision:
        # In collision, use EPA to obtain the penetration vector and depth
        penetration_vector = epa(shape1, shape2, simplex)
        penetration_depth = jnp.linalg.norm(penetration_vector)
        # Define signed distance as negative
        signed_distance = -penetration_depth
        return signed_distance, penetration_vector
    else:
        # When no collision, use the current simplex to estimate the separation distance
        separation_distance = distance_to_simplex(simplex)
        return separation_distance, None

# -------------------------------
# Example: Define two 2D shapes (polygons)
#
# Example 1: Non-collision case
shape1 = jnp.array([
    [0.0, 0.2],
    [0.2, 0.0],
    [0.8, 0.0],
    [1.0, 0.2],
    [1.0, 0.8],
    [0.8, 1.0],
    [0.2, 1.0],
    [0.0, 0.8]
])


import numpy as np
import matplotlib.pyplot as plt

# Fixed polygon: use shape1 as an example (square)
fixed_polygon = shape1

# Moving polygon: based on shape2 from the example (square)
base_moving_polygon = jnp.array([[1.5, 0.5],
                                 [2.5, 0.5],
                                 [2.5, 1.5],
                                 [1.5, 1.5]])

# Set parameters for circular motion
num_steps = 400
angles = np.linspace(0, 2 * np.pi, num_steps)
radius = 1.0  # Radius of the circle

signed_distances = []  # Save the signed distance for each angle

for angle in angles:
    # Compute the translation vector corresponding to the current angle (rotation around the origin)
    translation = jnp.array([radius * np.cos(angle), radius * np.sin(angle)])
    # Translate the moving polygon
    moved_polygon = base_moving_polygon + translation

    # Compute the signed distance and penetration vector
    # Note: if there is a collision, signed_distance is negative; if not, it is positive
    signed_distance, penetration_vector = compute_signed_distance(fixed_polygon, moved_polygon)
    signed_distances.append(signed_distance)

# Convert JAX array to numpy array for plotting
signed_distances = np.array([float(sd) for sd in signed_distances])

plt.figure(figsize=(8, 4))
plt.plot(angles, signed_distances, marker='o')
plt.xlabel("Angle (radians)")
plt.ylabel("Signed Distance")
plt.title("Signed Distance during Circular Motion of a Polygon")
plt.grid(True)
plt.show()

import matplotlib.animation as animation

# Fixed polygon: shape1 (square), converted to numpy array (since matplotlib accepts numpy arrays)
fixed_polygon_np = np.array(shape1)

# Moving polygon: based on base_moving_polygon (example shape2), converted to numpy array
base_moving_polygon_np = np.array([[1.5, 0.5],
                                   [2.5, 0.5],
                                   [2.5, 1.5],
                                   [1.5, 1.5]])

# Set circular motion parameters
num_steps = 100
angles = np.linspace(0, 2 * np.pi, num_steps)
radius = 1.0

# Create figure and axes
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Circular Motion Animation of Polygons\nBlue: Fixed Polygon, Red: Moving Polygon")

# Draw the fixed polygon
fixed_patch = plt.Polygon(fixed_polygon_np, closed=True, fill=False, edgecolor='blue', linewidth=2)
ax.add_patch(fixed_patch)

# Draw the moving polygon in its initial state
moving_patch = plt.Polygon(base_moving_polygon_np, closed=True, fill=False, edgecolor='red', linewidth=2)
ax.add_patch(moving_patch)

# Text to display the current angle and signed distance
distance_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=10,
                          verticalalignment='top')

def init():
    """Initialize animation"""
    moving_patch.set_xy(base_moving_polygon_np)
    distance_text.set_text("")
    return moving_patch, distance_text

def animate(i):
    """Update function for each frame of the animation"""
    angle = angles[i]
    # Compute translation vector
    translation = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    # Translate the moving polygon
    moved_polygon = base_moving_polygon_np + translation
    moving_patch.set_xy(moved_polygon)

    # Use GJK/EPA to compute the signed distance for the current state
    # Note: compute_signed_distance expects a JAX array, so conversion is necessary
    moved_polygon_jax = jnp.array(moved_polygon)
    signed_distance, penetration_vector = compute_signed_distance(shape1, moved_polygon_jax)

    # Update text information
    distance_text.set_text(f"Angle: {angle:.2f} rad\nSigned Distance: {signed_distance:.2f}")
    return moving_patch, distance_text

# Create animation, interval indicates the time between frames (in milliseconds)
ani = animation.FuncAnimation(fig, animate, frames=num_steps, init_func=init,
                              interval=100, blit=True)

plt.show()
