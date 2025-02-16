'''
SAT_Kiwan Version: Two AXIS Translational
'''

import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Define Kiwan SAT ---
def Kiwan_SAT():

    def get_normals(vertices):
        edges = jnp.roll(vertices, -1, axis=0) - vertices
        normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
        norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
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
        """
        Compute the 2D cross product (scalar) for vectors a and b.
        Supports vectorized inputs; a and b can have shape (..., 2).
        """
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def ray_polygon_intersection(O, d, vertices, eps=1e-8):
        """
        Compute the intersection of a ray starting at point O in direction d with a polygon's edges.
        The polygon is defined by its vertices (assumed to be ordered).
        
        Parameters:
        O: Origin of the ray, shape (2,).
        d: Ray direction (unit vector), shape (2,).
        vertices: Array of polygon vertices, shape (N, 2).
        eps: Tolerance to check for near-zero denominators (parallelism).
        
        Returns:
        The smallest positive t value (distance along the ray) for which the ray
        intersects any of the polygon's edges. If no valid intersection is found, returns jnp.inf.
        """
        # Create edge endpoints: A is each vertex, and B is the next vertex (with wrapping)
        A = vertices
        B = jnp.concatenate([vertices[1:], vertices[:1]], axis=0)
        BA = B - A  # Direction vectors for each edge

        # Compute the denominator for the intersection formula for each edge
        denom = cross2D(d, BA)

        # Vector from ray origin O to each vertex A
        A_minus_O = A - O

        # Calculate ray parameter t and segment parameter u for each edge:
        # The intersection is given by: O + t*d = A + u*(B - A)
        t = cross2D(A_minus_O, BA) / denom
        u = cross2D(A_minus_O, d) / denom

        # Determine valid intersections:
        # - Denom must be significantly non-zero.
        # - t must be non-negative (intersection is along the ray).
        # - u must be between 0 and 1 (intersection lies on the segment).
        valid = (jnp.abs(denom) > eps) & (t >= 0) & (u >= 0) & (u <= 1)

        # Replace invalid intersection t values with infinity so they are ignored when taking the minimum
        t_valid = jnp.where(valid, t, jnp.inf)

        # Return the smallest t value among all valid intersections
        return jnp.min(t_valid)

    def compute_gap_along_centers(vertices1, vertices2, eps=1e-8):
        """
        Compute the gap between two polygons along the line connecting their centroids:
        
        gap = (distance between centroids) - (radius of polygon1 in the given direction +
                                                radius of polygon2 in the opposite direction)
        
        The "radius" of a polygon is determined by finding the intersection between a ray
        emanating from the polygon's centroid and the polygon's boundary.
        
        Note: This function assumes that the centroid (computed by compute_polygon_centroid)
        is located inside the polygon.
        """
        # Assume compute_polygon_centroid is defined elsewhere to compute the centroid (shape (2,))
        C1 = compute_polygon_centroid(vertices1)
        C2 = compute_polygon_centroid(vertices2)
        
        # Compute the vector between centroids, its magnitude, and the unit direction vector d
        d_vec = C2 - C1
        d_norm = jnp.linalg.norm(d_vec)
        d = d_vec / d_norm

        # Compute the "radius" for each polygon along the specified directions
        # For polygon1, along the direction d; for polygon2, along the opposite direction -d.
        r1 = ray_polygon_intersection(C1, d, vertices1, eps)
        r2 = ray_polygon_intersection(C2, -d, vertices2, eps)
        
        # The gap is the center-to-center distance minus the sum of the two radii
        gap = d_norm - (r1 + r2)
        return gap

    def compute_distance(robot_vertices, polygon_vertices):
        robot_normals = get_normals(robot_vertices)
        poly_normals  = get_normals(polygon_vertices)
        candidate_axes = jnp.concatenate([robot_normals, poly_normals], axis=0)
        
        proj_robot = robot_vertices @ candidate_axes.T
        proj_poly  = polygon_vertices @ candidate_axes.T
        
        min_R = jnp.min(proj_robot, axis=0)
        max_R = jnp.max(proj_robot, axis=0)
        min_P = jnp.min(proj_poly, axis=0)
        max_P = jnp.max(proj_poly, axis=0)
        
        separated_mask = (max_R < min_P) | (max_P < min_R)
        
        penetration = jnp.minimum(max_R, max_P) - jnp.maximum(min_R, min_P)
        
        def separated_case(_):
            gap = compute_gap_along_centers(robot_vertices, polygon_vertices)
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

    # -- Plot for translational motion --
    num_sides = 8
    angles = jnp.linspace(0, 2 * jnp.pi, num_sides + 1)[:-1] 
    center = jnp.array([1.5, 1.5])
    radius = 1.0
    obstacle_poly = jnp.stack([center[0] + radius * jnp.cos(angles),
                               center[1] + radius * jnp.sin(angles)], axis=1)

    base_robot_poly = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    num_frames = 100
    angles = np.linspace(0, 2 * np.pi, num_frames)
    radius = 1.0  

    signed_distances = []  
    translations = []

    for angle in angles:
        # Calculate the translation vector corresponding to the current angle (rotation around the origin)
        translation = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        translations.append(translation)
        # Translate the moving polygon
        moved_polygon = base_robot_poly + translation

        # Compute the signed distance and penetration vector
        # Note: If the two shapes collide, signed_distance is negative; if not, it is positive.
        signed_distance = compute_distance(obstacle_poly, moved_polygon)
        signed_distances.append(signed_distance)

    # Convert JAX array to numpy array for plotting
    signed_distances = np.array([float(sd) for sd in signed_distances])

    plt.figure(figsize=(8, 4))
    plt.plot(np.linspace(0, 100, num_frames), signed_distances, 'b.-')
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.title("Kiwan SAT - Distance vs. Time")
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)
    ax.set_aspect('equal')
    ax.set_title("Kiwan SAT Distance Animation")

    obstacle_patch, = ax.plot([], [], 'r-', lw=2, label='Obstacle')
    robot_patch, = ax.plot([], [], 'b-', lw=2, label='Robot')
    text_distance = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='upper right')

    def init():
        obstacle_patch.set_data([], [])
        robot_patch.set_data([], [])
        text_distance.set_text('')
        return obstacle_patch, robot_patch, text_distance

    def animate(i):
        translation = translations[i]
        robot_poly = base_robot_poly + translation
        
        dist = compute_distance(robot_poly, obstacle_poly)
        dist_val = float(dist)
        
        robot_poly_np = np.array(robot_poly)
        obstacle_poly_np = np.array(obstacle_poly)
        robot_poly_plot = np.vstack([robot_poly_np, robot_poly_np[0]])
        obstacle_poly_plot = np.vstack([obstacle_poly_np, obstacle_poly_np[0]])
        
        robot_patch.set_data(robot_poly_plot[:, 0], robot_poly_plot[:, 1])
        obstacle_patch.set_data(obstacle_poly_plot[:, 0], obstacle_poly_plot[:, 1])
        text_distance.set_text(f"Distance = {dist_val:.3f}")
        
        return obstacle_patch, robot_patch, text_distance

    ani = animation.FuncAnimation(fig, animate, frames=num_frames,
                                  init_func=init, interval=100, blit=True)
    # Construct the save path for your Desktop with a GIF extension
    desktop_path = os.path.expanduser("~/Desktop")
    save_path = os.path.join(desktop_path, "animation_rotational.gif")

    # Save the animation as a GIF using the Pillow writer
    ani.save(save_path, writer='pillow', fps=30)
    plt.show()

# --- Define Kiwan Distancing ---

def Kiwan_Distancing():

    def get_normals(vertices):
        edges = jnp.roll(vertices, -1, axis=0) - vertices
        normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
        norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
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
        """
        Compute the 2D cross product (scalar) for vectors a and b.
        Supports vectorized inputs; a and b can have shape (..., 2).
        """
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def ray_polygon_intersection(O, d, vertices, eps=1e-8):
        """
        Compute the intersection of a ray starting at point O in direction d with a polygon's edges.
        The polygon is defined by its vertices (assumed to be ordered).
        
        Parameters:
        O: Origin of the ray, shape (2,).
        d: Ray direction (unit vector), shape (2,).
        vertices: Array of polygon vertices, shape (N, 2).
        eps: Tolerance to check for near-zero denominators (parallelism).
        
        Returns:
        The smallest positive t value (distance along the ray) for which the ray
        intersects any of the polygon's edges. If no valid intersection is found, returns jnp.inf.
        """
        # Create edge endpoints: A is each vertex, and B is the next vertex (with wrapping)
        A = vertices
        B = jnp.concatenate([vertices[1:], vertices[:1]], axis=0)
        BA = B - A  # Direction vectors for each edge

        # Compute the denominator for the intersection formula for each edge
        denom = cross2D(d, BA)

        # Vector from ray origin O to each vertex A
        A_minus_O = A - O

        # Calculate ray parameter t and segment parameter u for each edge:
        # The intersection is given by: O + t*d = A + u*(B - A)
        t = cross2D(A_minus_O, BA) / denom
        u = cross2D(A_minus_O, d) / denom

        # Determine valid intersections:
        # - Denom must be significantly non-zero.
        # - t must be non-negative (intersection is along the ray).
        # - u must be between 0 and 1 (intersection lies on the segment).
        valid = (jnp.abs(denom) > eps) & (t >= 0) & (u >= 0) & (u <= 1)

        # Replace invalid intersection t values with infinity so they are ignored when taking the minimum
        t_valid = jnp.where(valid, t, jnp.inf)

        # Return the smallest t value among all valid intersections
        return jnp.min(t_valid)

    def compute_gap_along_centers(vertices1, vertices2, eps=1e-8):
        """
        Compute the gap between two polygons along the line connecting their centroids:
        
        gap = (distance between centroids) - (radius of polygon1 in the given direction +
                                                radius of polygon2 in the opposite direction)
        
        The "radius" of a polygon is determined by finding the intersection between a ray
        emanating from the polygon's centroid and the polygon's boundary.
        
        Note: This function assumes that the centroid (computed by compute_polygon_centroid)
        is located inside the polygon.
        """
        # Assume compute_polygon_centroid is defined elsewhere to compute the centroid (shape (2,))
        C1 = compute_polygon_centroid(vertices1)
        C2 = compute_polygon_centroid(vertices2)
        
        # Compute the vector between centroids, its magnitude, and the unit direction vector d
        d_vec = C2 - C1
        d_norm = jnp.linalg.norm(d_vec)
        d = d_vec / d_norm

        # Compute the "radius" for each polygon along the specified directions
        # For polygon1, along the direction d; for polygon2, along the opposite direction -d.
        r1 = ray_polygon_intersection(C1, d, vertices1, eps)
        r2 = ray_polygon_intersection(C2, -d, vertices2, eps)
        
        # The gap is the center-to-center distance minus the sum of the two radii
        gap = d_norm - (r1 + r2)
        return gap

    # -- Plot for translational motion --
    num_sides = 8
    angles = jnp.linspace(0, 2 * jnp.pi, num_sides + 1)[:-1] 
    center = jnp.array([1.5, 1.5])
    radius = 1.0
    obstacle_poly = jnp.stack([center[0] + radius * jnp.cos(angles),
                               center[1] + radius * jnp.sin(angles)], axis=1)

    base_robot_poly = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    num_frames = 100
    angles = np.linspace(0, 2 * np.pi, num_frames)
    radius = 1.0  

    signed_distances = []  
    translations = []

    for angle in angles:
        # Calculate the translation vector corresponding to the current angle (rotation around the origin)
        translation = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        translations.append(translation)
        # Translate the moving polygon
        moved_polygon = base_robot_poly + translation

        # Compute the signed distance and penetration vector
        # Note: If the two shapes collide, signed_distance is negative; if not, it is positive.
        signed_distance = compute_gap_along_centers(obstacle_poly, moved_polygon)
        signed_distances.append(signed_distance)

    # Convert JAX array to numpy array for plotting
    signed_distances = np.array([float(sd) for sd in signed_distances])

    plt.figure(figsize=(8, 4))
    plt.plot(np.linspace(0, 100, num_frames), signed_distances, 'b.-')
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.title("Kiwan Distancing - Distance vs. Time")
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)
    ax.set_aspect('equal')
    ax.set_title("Kiwan Distancing Distance Animation")

    obstacle_patch, = ax.plot([], [], 'r-', lw=2, label='Obstacle')
    robot_patch, = ax.plot([], [], 'b-', lw=2, label='Robot')
    text_distance = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='upper right')

    def init():
        obstacle_patch.set_data([], [])
        robot_patch.set_data([], [])
        text_distance.set_text('')
        return obstacle_patch, robot_patch, text_distance

    def animate(i):
        translation = translations[i]
        robot_poly = base_robot_poly + translation
        
        dist = compute_gap_along_centers(robot_poly, obstacle_poly)
        dist_val = float(dist)
        
        robot_poly_np = np.array(robot_poly)
        obstacle_poly_np = np.array(obstacle_poly)
        robot_poly_plot = np.vstack([robot_poly_np, robot_poly_np[0]])
        obstacle_poly_plot = np.vstack([obstacle_poly_np, obstacle_poly_np[0]])
        
        robot_patch.set_data(robot_poly_plot[:, 0], robot_poly_plot[:, 1])
        obstacle_patch.set_data(obstacle_poly_plot[:, 0], obstacle_poly_plot[:, 1])
        text_distance.set_text(f"Distance = {dist_val:.3f}")
        
        return obstacle_patch, robot_patch, text_distance

    ani = animation.FuncAnimation(fig, animate, frames=num_frames,
                                  init_func=init, interval=100, blit=True)
    # Construct the save path for your Desktop with a GIF extension
    desktop_path = os.path.expanduser("~/Desktop")
    save_path = os.path.join(desktop_path, "animation_rotational.gif")

    # Save the animation as a GIF using the Pillow writer
    ani.save(save_path, writer='pillow', fps=30)
    plt.show()


# --- Define Regular SAT ---
def Regular_SAT():
    def project_points(points, axis):
        projections = jnp.dot(points, axis)
        return jnp.min(projections), jnp.max(projections)

    def get_normals(vertices):
        edges = jnp.roll(vertices, -1, axis=0) - vertices
        normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
        norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
        return normals / norms

    def compute_distance(robot_vertices, polygon_vertices):
        robot_normals = get_normals(robot_vertices)
        poly_normals  = get_normals(polygon_vertices)
        candidate_axes = jnp.concatenate([robot_normals, poly_normals], axis=0)
        
        proj_robot = robot_vertices @ candidate_axes.T
        proj_poly  = polygon_vertices @ candidate_axes.T
        
        min_R = jnp.min(proj_robot, axis=0)
        max_R = jnp.max(proj_robot, axis=0)
        min_P = jnp.min(proj_poly, axis=0)
        max_P = jnp.max(proj_poly, axis=0)
        
        separation_left  = min_P - max_R  
        separation_right = min_R - max_P  
        
        separated_mask = (max_R < min_P) | (max_P < min_R)
        
        sep_distances = jnp.where(max_R < min_P, separation_left,
                          jnp.where(max_P < min_R, separation_right, jnp.inf))
        
        penetration = jnp.minimum(max_R, max_P) - jnp.maximum(min_R, min_P)
        
        def separated_case(_):
            gap = jnp.min(sep_distances)
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

    # -- Plot for translational motion --
    num_sides = 8
    angles = jnp.linspace(0, 2 * jnp.pi, num_sides + 1)[:-1] 
    center = jnp.array([1.5, 1.5])
    radius = 1.0
    obstacle_poly = jnp.stack([center[0] + radius * jnp.cos(angles),
                               center[1] + radius * jnp.sin(angles)], axis=1)

    base_robot_poly = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    num_frames = 100
    angles = np.linspace(0, 2 * np.pi, num_frames)
    radius = 1.0  

    signed_distances = []  
    translations = []

    for angle in angles:
        # Calculate the translation vector corresponding to the current angle (rotation around the origin)
        translation = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        translations.append(translation)
        # Translate the moving polygon
        moved_polygon = base_robot_poly + translation

        # Compute the signed distance and penetration vector
        # Note: If the two shapes collide, signed_distance is negative; if not, it is positive.
        signed_distance = compute_distance(obstacle_poly, moved_polygon)
        signed_distances.append(signed_distance)

    # Convert JAX array to numpy array for plotting
    signed_distances = np.array([float(sd) for sd in signed_distances])

    plt.figure(figsize=(8, 4))
    plt.plot(np.linspace(0, 100, num_frames), signed_distances, 'b.-')
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.title("Regular SAT - Distance vs. Time")
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)
    ax.set_aspect('equal')
    ax.set_title("Regular SAT Distance Animation")

    obstacle_patch, = ax.plot([], [], 'r-', lw=2, label='Obstacle')
    robot_patch, = ax.plot([], [], 'b-', lw=2, label='Robot')
    text_distance = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='upper right')

    def init():
        obstacle_patch.set_data([], [])
        robot_patch.set_data([], [])
        text_distance.set_text('')
        return obstacle_patch, robot_patch, text_distance

    def animate(i):
        translation = translations[i]
        robot_poly = base_robot_poly + translation
        
        dist = compute_distance(robot_poly, obstacle_poly)
        dist_val = float(dist)
        
        robot_poly_np = np.array(robot_poly)
        obstacle_poly_np = np.array(obstacle_poly)
        robot_poly_plot = np.vstack([robot_poly_np, robot_poly_np[0]])
        obstacle_poly_plot = np.vstack([obstacle_poly_np, obstacle_poly_np[0]])
        
        robot_patch.set_data(robot_poly_plot[:, 0], robot_poly_plot[:, 1])
        obstacle_patch.set_data(obstacle_poly_plot[:, 0], obstacle_poly_plot[:, 1])
        text_distance.set_text(f"Distance = {dist_val:.3f}")
        
        return obstacle_patch, robot_patch, text_distance

    ani = animation.FuncAnimation(fig, animate, frames=num_frames,
                                  init_func=init, interval=100, blit=True)

    plt.show()

if __name__ == "__main__":
    # Kiwan_SAT()
    Kiwan_Distancing()
    # Regular_SAT()
