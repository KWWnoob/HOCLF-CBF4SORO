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
    def project_points(points, axis):
        projections = jnp.dot(points, axis)
        return jnp.min(projections), jnp.max(projections)

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

    def compute_gap_along_centers(vertices1, vertices2):
        # Compute the centroids of both polygons
        C1 = compute_polygon_centroid(vertices1)
        C2 = compute_polygon_centroid(vertices2)
        
        # Compute the vector between centroids and its unit direction
        d_vec = C2 - C1
        d_norm = jnp.linalg.norm(d_vec)
        d = d_vec / d_norm
        
        # Project vertices along the direction d for the first polygon
        projections1 = jnp.dot(vertices1 - C1, d)
        r1 = jnp.max(projections1)
        
        # Project vertices along the opposite direction for the second polygon
        projections2 = jnp.dot(vertices2 - C2, -d)
        r2 = jnp.max(projections2)
        
        # Calculate the gap between the polygons
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
    Regular_SAT()
