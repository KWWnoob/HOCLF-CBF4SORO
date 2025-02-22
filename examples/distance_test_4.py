'''
Ellipse
'''

import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax
import jax.numpy as jnp
from jax import jit, grad
import matplotlib.pyplot as plt
import numpy as np  # Only for plotting convenience

def mvee(P, num_iters=100):
    """
    Compute the Minimum Volume Enclosing Ellipsoid (MVEE) for a set of 2D points P using a fixed number of iterations.
    
    P: Array of shape (n_points, 2)
    num_iters: Number of iterations for the Khachiyan algorithm
    
    Returns:
      center: Array of shape (2,)
      A: Array (2x2) such that the ellipse is given by { x : (x-center)^T A (x-center) <= 1 }.
    """
    n, d = P.shape
    # Build Q: shape (d+1, n)
    Q = jnp.vstack([P.T, jnp.ones((1, n))])
    # initial uniform weights
    u0 = jnp.ones(n) / n

    def body_fun(i, u):
        # Compute X = Q * diag(u) * Q^T, shape (d+1, d+1)
        X = Q @ jnp.diag(u) @ Q.T
        X_inv = jnp.linalg.inv(X)
        # Compute M: for each index j, M_j = q_j^T X_inv q_j.
        # q_j is the j-th column of Q.
        # We can compute this vectorized.
        M = jnp.sum(Q * (X_inv @ Q), axis=0)
        # find index with maximum M
        j = jnp.argmax(M)
        # Compute step size (using d+1 because we have d+1 dimensions in Q)
        step_size = (M[j] - (d+1)) / ((d+1) * (M[j] - 1.0))
        # update u: new_u = (1-step_size)*u; then add step_size to index j.
        new_u = (1 - step_size) * u
        new_u = new_u.at[j].add(step_size)
        return new_u

    u_final = jax.lax.fori_loop(0, num_iters, body_fun, u0)

    # Compute the center of the ellipse
    center = P.T @ u_final  # shape (2,)
    # Compute the A matrix: let diff = P - center; then A = inv(diff^T diag(u) diff)/d
    diff = P - center
    # Using weighted covariance:
    A = jnp.linalg.inv(diff.T @ jnp.diag(u_final) @ diff) / d

    return center, A

### 2. Extract ellipse parameters

def ellipse_parameters(center, A):
    """
    From the ellipse representation { x : (x-center)^T A (x-center) <= 1 },
    extract the semi-axis lengths and the rotation angle.
    
    Returns:
      semi_axes: (a, b) with a >= b.
      angle: rotation angle (radians) corresponding to the larger semi-axis.
    """
    eigvals, eigvecs = jnp.linalg.eig(A)
    # The semi-axis lengths are 1/sqrt(eigval)
    semi_axes = 1.0 / jnp.sqrt(eigvals)
    # Ensure ordering so that a is the larger axis
    def reorder(semi_axes, eigvecs):
        # If first axis is smaller than second, swap.
        a, b = semi_axes[0], semi_axes[1]
        def swap():
            return jnp.array([b, a]), jnp.column_stack([eigvecs[:,1], eigvecs[:,0]])
        def noswap():
            return semi_axes, eigvecs
        return jax.lax.cond(a < b, swap, noswap)
    
    semi_axes, eigvecs = reorder(semi_axes, eigvecs)
    # Compute the rotation angle from the eigenvector associated with the larger semi-axis.
    angle = jnp.arctan2(eigvecs[1,0], eigvecs[0,0])
    return semi_axes, angle

### 3. Ellipse parameterization

def ellipse_point(t, center, semi_axes, angle):
    """
    Given a parameter t (radians), return the corresponding point on the ellipse.
    """
    a, b = semi_axes
    x = a * jnp.cos(t)
    y = b * jnp.sin(t)
    # rotation matrix:
    R = jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                   [jnp.sin(angle),  jnp.cos(angle)]])
    point = center + R @ jnp.array([x, y])
    return point

### 4. Optimize separation distance using gradient descent in JAX

def optimize_separation(center1, semi_axes1, angle1, center2, semi_axes2, angle2,
                          num_steps=500, lr=1e-2):
    """
    Minimize the distance between a point on ellipse1 (parameterized by t) and ellipse2 (parameterized by s)
    using a simple gradient descent routine.
    
    Returns:
      final distance, and the parameters (t, s) that yield that distance.
    """
    # The loss function: distance between the two ellipse points.
    def loss_fn(params):
        t, s = params
        p1 = ellipse_point(t, center1, semi_axes1, angle1)
        p2 = ellipse_point(s, center2, semi_axes2, angle2)
        return jnp.linalg.norm(p1 - p2)
    
    grad_loss = grad(loss_fn)
    
    # Initialize parameters randomly in [0, 2pi).
    key = jax.random.PRNGKey(0)
    init_params = jax.random.uniform(key, (2,), minval=0.0, maxval=2*jnp.pi)

    def step(params, _):
        g = grad_loss(params)
        new_params = params - lr * g
        # Keep parameters in [0, 2pi) by modulo operation.
        new_params = jnp.mod(new_params, 2*jnp.pi)
        return new_params, None

    final_params, _ = jax.lax.scan(step, init_params, None, length=num_steps)
    final_loss = loss_fn(final_params)
    return final_loss, final_params

def get_normals(vertices):
    # Compute edge vectors and then compute normals by swapping components and negating one
    edges = jnp.roll(vertices, -1, axis=0) - vertices
    normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
    norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
    return normals / norms
    
def compute_distance(robot_vertices, polygon_vertices):

    # Compute the normals for both shapes
    robot_normals = get_normals(robot_vertices)
    poly_normals  = get_normals(polygon_vertices)
    candidate_axes = jnp.concatenate([robot_normals, poly_normals], axis=0)
    
    center1, A1 = mvee(robot_vertices, num_iters=150)
    center2, A2 = mvee(polygon_vertices, num_iters=150)

    semi_axes1, angle1 = ellipse_parameters(center1, A1)
    semi_axes2, angle2 = ellipse_parameters(center2, A2)

    # Project both shapes on the candidate axes
    proj_robot = robot_vertices @ candidate_axes.T
    proj_poly  = polygon_vertices @ candidate_axes.T
    
    min_R = jnp.min(proj_robot, axis=0)
    max_R = jnp.max(proj_robot, axis=0)
    min_P = jnp.min(proj_poly, axis=0)
    max_P = jnp.max(proj_poly, axis=0)
    
    # Determine if there is a separation along any axis
    separated_mask = (max_R < min_P) | (max_P < min_R)
    
    # Compute penetration depth (overlap) on each axis
    penetration = jnp.minimum(max_R, max_P) - jnp.maximum(min_R, min_P)
    
    def separated_case(_):
        # If separated, compute gap based on centroids
        gap, _ = optimize_separation(center1, semi_axes1, angle1,
                                         center2, semi_axes2, angle2,
                                         num_steps=500, lr=1e-2)
        return gap
    
    def overlapping_case(_):
        # If overlapping, return the negative minimum penetration
        pen = -jnp.min(penetration)
        return pen

    overall_distance = jax.lax.cond(
        jnp.any(separated_mask),
        separated_case,
        overlapping_case,
        operand=None
    )
    
    return overall_distance
    
# Example usage:
if __name__ == '__main__':
    # Define two example polygons as arrays of vertices.
    # Polygon 1: a rough rectangle (could be any shape)
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
