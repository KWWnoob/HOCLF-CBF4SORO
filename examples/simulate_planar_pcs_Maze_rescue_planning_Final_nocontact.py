'''
SAT MULTI AXIS VERSION
'''

import csv
import diffrax as dx
from functools import partial
import jax
from cbfpy.cbfs.clf_cbf import CLFCBF, CLFCBFConfig
import inspect
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, config, debug, jacfwd, jit, vmap, lax
from jax import numpy as jnp
import jsrm
from jsrm.systems import planar_pcs
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict, Tuple

from src.img_animation import animate_images_cv2
from src.planar_pcs_rendering_rescue_sequential_sepobscontact import draw_image

# define the outputs directory
outputs_dir = Path("outputs") / "planar_pcs_simulation"
outputs_dir.mkdir(parents=True, exist_ok=True)

# load symbolic expressions
num_segments = 2
# filepath to symbolic expressions
sym_exp_filepath = Path(jsrm.__file__).parent / "symbolic_expressions" / f"planar_pcs_ns-{num_segments}.dill"

# set soft robot parameters
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
robot_length = 1.3e-1
robot_radius = 2e-2
robot_params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": robot_length * jnp.ones((num_segments,)),
    "r": robot_radius * jnp.ones((num_segments,)),
    "rho": rho,
    "g": jnp.array([0.0, 9.81]),
    "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
    "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
}
# damping matrix
damping_array = jnp.array([1e0, 1e3, 1e3])
multiplier = [1.5**m * damping_array for m in range(num_segments)]
robot_params["D"] = 5e-5 * jnp.diag(jnp.concatenate(multiplier)) * robot_length #depend on the num of segments

# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

# call the factory function for the planar PCS
strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = planar_pcs.factory(sym_exp_filepath, strain_selector)

kinetic_energy_fn = jit(auxiliary_fns["kinetic_energy_fn"])
potential_energy_fn = jit(auxiliary_fns["potential_energy_fn"])
jacobian_fn = jit(auxiliary_fns["jacobian_fn"])

# construct batched forward kinematics function
batched_forward_kinematics_fn = vmap(
    forward_kinematics_fn, in_axes=(None, None, 0)
)

# segmenting params
num_points = 20*num_segments
# Compute indices: equivalent to
# [num_points * (i+1)//num_segments - 1 for i in range(num_segments)]
end_p_ps_indices = (jnp.arange(1, num_segments+1) * num_points // num_segments) - 1

# ---Polygon matters---

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

def ray_polygon_intersection(O, d, vertices, eps=1e-3):
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

def compute_gap_along_centers(vertices1, vertices2, eps=1e-3):
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

'''V5'''

def connect_project(cA, cB, polyB):
    """
    1) Compute centroids of polyA and polyB.
    2) Draw the ray from centroid A to centroid B.
    3) Find the first intersection of that ray with polyB's boundary.
      hit_point  -- point on B where the cA→cB ray first intersects
    """

    v = cB - cA  # direction vector from A to B

    # -- Ray-segment intersection loop --
    best_t = jnp.inf
    hit = None
    for i in range(len(polyB)):
        A_edge = polyB[i]
        B_edge = polyB[(i+1) % len(polyB)]
        d = B_edge - A_edge

        M = jnp.column_stack((v, -d))
        if jnp.linalg.matrix_rank(M) < 2:
            continue

        t, u = jnp.linalg.solve(M, A_edge - cA)
        # t >=0 → along ray direction; 0<=u<=1 → within the segment
        if (t >= 0) and (0 <= u <= 1) and (t < best_t):
            best_t = t
            hit = cA + t*v

    return hit

@jax.jit
def compute_distance(robot, poly, alpha=1000.0):
    def get_normals(v):
        e = jnp.roll(v, -1, axis=0) - v
        n = jnp.stack([-e[:, 1], e[:, 0]], axis=1)
        return n / jnp.linalg.norm(n, axis=1, keepdims=True)

    Rn = get_normals(robot)
    Pn = get_normals(poly)
    axes = jnp.concatenate([Rn, Pn], axis=0)

    proj_R = robot @ axes.T
    proj_P = poly   @ axes.T
    R_min, R_max = jnp.min(proj_R, axis=0), jnp.max(proj_R, axis=0)
    P_min, P_max = jnp.min(proj_P, axis=0), jnp.max(proj_P, axis=0)

    gaps = jnp.concatenate([P_min - R_max, R_min - P_max], axis=0)
    h_olsat = (1.0 / alpha) * logsumexp(alpha * gaps)

    error_bound = jnp.log(2.0 * axes.shape[0]) / alpha
    
    h = h_olsat-error_bound
    separation_flag = jnp.where(h > -0.0, 1, 0)
    
    return h, separation_flag

def segmented_polygon(current_point, next_point, forward_direction, robotic_radius):
    '''
    Feed in soft body consecutive centered positions and directions and formulate a rectangular body for detecting collisions.
    The current point remains unchanged, but the displacement from current_point to next_point is scaled by 1.3.
    '''
    # Compute the new next point by scaling the difference
    new_next_point = current_point + 1 * (next_point - current_point)
    
    d = (next_point - current_point)/jnp.linalg.norm(next_point - current_point)
    # Compute the directional vector rotated by 90 degrees (for width of the robot)
    # d = jnp.array([
    #     jnp.cos(forward_direction + jnp.pi/2),
    #     jnp.sin(forward_direction + jnp.pi/2)
    # ])
    n1 = jnp.array([-d[1], d[0]])
    n2 = jnp.array([d[1], -d[0]])
    
    # Form the vertices using current_point and new_next_point
    vertices = [
        current_point + n1 * robotic_radius,
        new_next_point + n1 * robotic_radius,
        new_next_point + n2 * robotic_radius,
        current_point + n2 * robotic_radius
    ]
    
    return jnp.array(vertices)

def half_circle_to_polygon(center, forward_direction, radius, num_arc_points=30):
    """
    Create a convex polygon that approximates a half circle using JAX.
    
    Parameters:
      center: jnp.array with shape (2,) representing the center of the half circle.
      d: jnp.array with shape (2,) representing the apex direction of the half circle 
         (should be normalized).
      radius: scalar, the radius of the half circle.
      num_arc_points: number of points to sample along the arc (excluding the chord endpoints 
                      if desired). The total number of points returned will be num_arc_points + 2.
    
    Returns:
      polygon: jnp.array of shape (N, 2) with vertices of the polygon, ordered counter-clockwise.
               The polygon includes the arc points, and the chord between the endpoints closes
               the half circle.
    """
    # Compute the angle of the apex direction using arctan2.
    apex_angle = forward_direction
    # The half circle spans π radians, centered on the forward direction.
    start_angle = apex_angle - jnp.pi / 2
    end_angle   = apex_angle + jnp.pi / 2
    
    # Generate linearly spaced angles between the start and end angles.
    # This produces points in increasing order (counter-clockwise).
    angles = jnp.linspace(start_angle, end_angle, num_arc_points + 2)
    
    # Compute the coordinates of the arc points using the circle's parametric equations.
    arc_points = center + radius * jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=1)
    
    # Return the vertices of the polygon.
    # When these vertices are connected in order (and the last vertex is connected back to the first),
    # the shape is a convex polygon approximating the half circle with a straight chord closing the shape.
    return arc_points


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
def find_closest_segment_point_and_direction(robot: jnp.ndarray, obs: jnp.ndarray, flag: bool = True):
    poly1 = robot
    poly2 = obs
    seg1_start = poly1
    seg1_end = jnp.roll(poly1, -1, axis=0)
    seg2_start = poly2
    seg2_end = jnp.roll(poly2, -1, axis=0)

    def one_edge_pair(a0, a1):
        def inner(b0, b1):
            return segment_segment_closest_points(a0, a1, b0, b1)
        return jax.vmap(inner)(seg2_start, seg2_end)

    # Apply vmap over poly1 edges
    p_closest, q_closest, dists = jax.vmap(one_edge_pair)(seg1_start, seg1_end)  # each (N1, N2, 2) or (N1, N2)

    dists_flat = dists.reshape(-1)
    p_flat = p_closest.reshape(-1, 2)  # closest on poly1
    q_flat = q_closest.reshape(-1, 2)  # closest on poly2

    idx = jnp.argmin(dists_flat)
    p_poly1 = p_flat[idx]
    q_poly2 = q_flat[idx]

    # Vector and norm
    vec = p_poly1 - q_poly2
    norm = jnp.linalg.norm(vec) + 1e-8
    dir_vec = vec / norm

    # Conditionally flip the direction
    dir_vec = lax.cond(flag, lambda x: x, lambda x: -x, dir_vec)

    return p_poly1,q_poly2, dir_vec


def compute_contact_jacobian_fn(q: jnp.ndarray, p_c: jnp.ndarray, s_c: float) -> jnp.ndarray:
    """
    Compute the 2×3N positional Jacobian J_c(q) of the contact point p_c,
    defined using orientation-aware correction.

    Args:
        q: (3N,) robot configuration
        p_c: (2,) contact point in workspace
        s_c: float, arc-length along the backbone

    Returns:
        J_c: (2, 3N) positional Jacobian of the contact point
    """
    # 1. Full Jacobian at s_c: rows = [J_x, J_y, J_theta]
    J_full = jacobian_fn(robot_params, q, s_c)  # (3, 3N)
    J_xy = J_full[0:2, :]     # (2, 3N)
    J_theta = J_full[2:3, :]  # (1, 3N), ensure it's a row vector

    # 2. Forward kinematics position at arc-length s_c
    p_fk = forward_kinematics_fn(robot_params, q, s_c)[:2]  # (2,)

    # 3. Compute correction term
    delta = p_c - p_fk  # (2,)
    rotated = jnp.diag(jnp.array([-1.0, 1.0])) @ delta  # (2,)
    correction = rotated[:, None]  # (2,1)

    # 4. Final Jacobian (2,3N)
    J_c = J_xy + correction @ J_theta
    return J_c

@jax.jit
def compute_contact_torque(
    q: jnp.ndarray,
    robot_params,
    s_ps: jnp.ndarray,
    obs_poly: jnp.ndarray,
    robot_radius: float,
    k: float,
    eps: float = 1e-4,
) -> jnp.ndarray:

    # FK
    p = batched_forward_kinematics_fn(robot_params, q, s_ps)  # (N, 3)
    p_ps = p[:, :2]
    p_theta = p[:, 2]
    seg_starts = p_ps[:-1]
    seg_ends = p_ps[1:]
    last_vec = seg_ends[-1] - seg_starts[-1]
    new_end = seg_starts[-1] + 2.0 * last_vec
    seg_ends = seg_ends.at[-1].set(new_end)
    seg_orient = p_theta[:-1]

    robot_poly = jax.vmap(segmented_polygon, in_axes=(0, 0, 0, None))(
        seg_starts, seg_ends, seg_orient, robot_radius
    )

    num_segments = robot_poly.shape[0]
    num_obstacles = obs_poly.shape[0]

    seg_ids, obs_ids = jnp.meshgrid(jnp.arange(num_segments), jnp.arange(num_obstacles), indexing="ij")
    pair_indices = jnp.stack([seg_ids.reshape(-1), obs_ids.reshape(-1)], axis=1)

    def interact(pair_idx):
        i, j = pair_idx
        poly_seg = robot_poly[i]
        poly_obs = obs_poly[j]
        s_i = s_ps[i]

        # d, flag = compute_distance(poly_seg, poly_obs)
        # f_mag = k * jax.nn.softplus(-d / eps)
        # p_c, n_hat = find_closest_segment_point_and_direction(poly_seg, poly_obs, flag=flag)
        # n_hat = n_hat / (jnp.linalg.norm(n_hat) + 1e-6)
        # f_vec = f_mag * n_hat

        d, _ = compute_distance(poly_seg, poly_obs) 
        p_c, q_c, _ = find_closest_segment_point_and_direction(poly_seg, poly_obs)  # ← 我们改一下这个函数接口

        # Step 2: compute direction from obs → robot
        vec = p_c - q_c
        dir_vec = vec / (jnp.linalg.norm(vec) + 1e-8)

        # Step 3: compute repulsive force (only significant if d < 0)
        # f_mag = k * jax.nn.softplus(-d / eps)
        f_mag = (k * jax.nn.elu(-d / eps) + k) * eps
        f_vec = f_mag * dir_vec  
        J_c = compute_contact_jacobian_fn(q, p_c, s_i)
        return J_c.T @ f_vec  # (3N,)

    tau_all = jax.vmap(interact)(pair_indices)  # shape: (num_pairs, 3N)
    tau = tau_all.sum(axis=0)
    return tau

def soft_robot_with_safety_contact_CBFCLF_example():
    
    # define the ODE function
    class SoRoConfig(CLFCBFConfig):
        '''Config for soft robot'''

        def __init__(self):

            self.robot_params = robot_params
            self.robot_radius = robot_radius
            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

            '''Polygon Obstacle Parameter'''
            char_length = robot_length
            self.poly_obstacle_shape_1 = jnp.array([[0.0, 0.0],
                                                [0.0, char_length*1.7],
                                                [char_length*0.5, char_length*1.7],
                                                [char_length*0.5, 0.0]])
            
            self.poly_obstacle_shape_2 = jnp.array([[0.0, 0.0],
                                                [0.0, char_length*0.9],
                                                [char_length*1.7, char_length*0.9],
                                                [char_length*1.7, 0.0]])

            
            # self.poly_obstacle_pos = self.poly_obstacle_shape/4 + jnp.array([-0.08,0.04])
            self.poly_obstacle_pos_1 = self.poly_obstacle_shape_1 + jnp.array([-0.102,0])
            self.poly_obstacle_pos_2 = self.poly_obstacle_shape_2 + jnp.array([0.030,0])

            self.poly_obstacle_pos_3 = self.poly_obstacle_pos_1[2,:] + self.poly_obstacle_shape_2

            self.poly_obstacle_pos_4 = self.poly_obstacle_pos_2[2,:] + self.poly_obstacle_shape_1

            self.poly_obstacle_pos = jnp.stack([self.poly_obstacle_pos_1,self.poly_obstacle_pos_2,self.poly_obstacle_pos_3,self.poly_obstacle_pos_4])
            # self.poly_obstacle_pos = jnp.stack([self.poly_obstacle_pos_2,self.poly_obstacle_pos_3,self.poly_obstacle_pos_4])
            '''Characteristic of robot'''
            self.s_ps = jnp.linspace(0, robot_length * num_segments, num_points) # segmented

            self.p_des_1_1 = jnp.array([0.00, 0.15234353*0.65, -jnp.pi*1.8*robot_length])
            self.p_des_1_2 = jnp.array([0.06, 0.18234353, 0])

            self.p_des_2_1 = jnp.array([0.05951909*1.5, 0.15234353*0.85, -jnp.pi*1.8*robot_length])
            self.p_des_2_2 = jnp.array([0.13, 0.20234353, 0])
            self.p_des_2_3= jnp.array([0.21, 0.38234353, 0])

            self.p_des_1 = jnp.stack([self.p_des_1_1,self.p_des_2_2])
            self.p_des_2 = jnp.stack([self.p_des_1_2,self.p_des_2_2])
            self.p_des_3 = jnp.stack([self.p_des_2_2,self.p_des_2_3])
            
            self.p_des_all = jnp.stack([self.p_des_1,self.p_des_3]) # shape (num_waypoints, num_of_segments, 3)
            
            '''Select the end of each segment'''
            self.indices = end_p_ps_indices

            '''Contact model Parameter'''
            self.contact_spring_constant = 3000
            self.maximum_withhold_force = 200
            
            self.contact_torque_fn = partial(
                    compute_contact_torque,
                    robot_params=self.robot_params,
                    s_ps=self.s_ps,
                    obs_poly=self.poly_obstacle_pos,
                    robot_radius=self.robot_radius,
                    k=self.contact_spring_constant,
                    eps=2e-4,
                )
            
            super().__init__(
                n=6 * num_segments, # number of states
                m=3 * num_segments, # number of inputs
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=False,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                cbf_relaxation_penalty=1e8,
                clf_relaxation_penalty=10
            )

        # def f(self, z) -> Array:
        #     q, q_d = jnp.split(z, 2)  # Split state z into q (position) and q_d (velocity)
        #     B, C, G, K, D, alpha = dynamical_matrices_fn(self.robot_params, q, q_d)

        #     # Drift term (f(x))
        #     drift = (
        #         -jnp.linalg.inv(B) @ (C @ q_d + D @ q_d + G + K)
        #     )
            
        #     return jnp.concatenate([q_d, drift])
        
        def f(self, z) -> Array:
            q, q_d = jnp.split(z, 2)
            B, C, G, K, D, alpha = dynamical_matrices_fn(self.robot_params, q, q_d)

            # contact torque
            tau_contact = self.contact_torque_fn(q)  

            drift = -jnp.linalg.inv(B) @ (C @ q_d + D @ q_d + G + K - tau_contact)

            return jnp.concatenate([q_d, drift])

        def g(self, z) -> Array:
            q, q_d = jnp.split(z, 2)
            B, _, _, _, _, _ = dynamical_matrices_fn(self.robot_params, q, q_d)

            # Control matrix g(x)
            control_matrix = jnp.linalg.inv(B)

            # Match dimensions for concatenation
            zero_block = jnp.zeros((q.shape[0], control_matrix.shape[1]))

            return jnp.concatenate([zero_block, control_matrix], axis=0)
   
        def V_2(self, z, z_des) -> jnp.ndarray:
            # Split state into positions (q) and velocities (q_d)
            # z_des is in shape of (num_segments * 3 * 2)
            q, q_d = jnp.split(z, 2)
            z_des, _ = jnp.split(z_des, 2) # get the desired position
            z_des = jnp.stack(jnp.split(z_des,num_segments))
            # Compute forward kinematics for the current configuration.
            p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            
            p_list = p[self.indices, :]

            # Compute the tracking errors.
            # For the "middle" points, use the first two coordinates.
            error_middle = jnp.concatenate([jnp.sqrt((p_list[i,:2]- z_des[i,:2])**2) for i in range(num_segments-1)])
            # # For the "tip" point, use all coordinates and scale the error by 10.
            error_tip = jnp.sqrt((p_list[num_segments-1, :] - z_des[num_segments-1,:])**2)

            # Concatenate the errors into one vector.
            error = jnp.concatenate([error_middle, error_tip]).reshape(-1)
            
            return error
        
        def h_2(self, z) -> jnp.ndarray:
            """
            Computes the safety force (barrier function output) for the robot given its state 'z',
            considering both polygonal and circular obstacles.
            
            Args:
                z: A JAX array representing the state, typically a concatenation of positions q and velocities q_d.
            
            Returns:
                A JAX array representing the combined smooth force.
            """
            # Split the state vector into positions (q) and velocities (q_d)
            q, q_d = jnp.split(z, 2)
            
            # Compute the forward kinematics for the robot
            p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            
            # Extract positions and orientations from the forward kinematics output.
            p_ps = p[:, :2]         # Positions, shape: (N, 2)
            p_orientation = p[:, 2]   # Orientations, shape: (N,)
            
            # -------- Process the polygon obstacles --------
            # Assuming self.poly_obstacle_pos is now a stacked array of obstacles
            # with shape (num_obstacles, num_vertices, 2)
            
            # Consider segments between consecutive points (excluding the last point for segments)
            current_points = p_ps[:-1]
            next_points = jnp.concatenate([p_ps[1:-1] * 1, p_ps[-1][None, :]])
            orientations = p_orientation[:-1]
            
            def segment_robot(current,next,orientation):
                seg_poly = segmented_polygon(current, next, orientation, robot_radius)
                return(seg_poly)
            
            robot_poly = jax.vmap(segment_robot)(current_points, next_points, orientations)

            pairwise_penetration,_ = jax.vmap( lambda poly: jax.vmap(lambda obs: compute_distance(poly, obs))(self.poly_obstacle_pos)
                                            )(robot_poly)
            
            end_start = p_ps[-2]
            end_end = p_ps[-1]
            d = (end_end - end_start)/jnp.linalg.norm(end_end - end_start)
            angle = jnp.arctan2(d[1], d[0])

            robot_tip = half_circle_to_polygon(p_ps[-1],angle, robot_radius)
            tip_penetration,_ = jax.vmap(compute_distance, in_axes=(0, None))(self.poly_obstacle_pos, robot_tip)
            tip_penetration = tip_penetration[None,...]

            penetration_depth_poly = jnp.concatenate([pairwise_penetration,tip_penetration])
            penetration_depth_poly = penetration_depth_poly.reshape(-1)

        
            # safe_distance = jnp.concatenate([penetration_depth_poly], axis=0)
            # force = penetration_to_contact_force(penetration_depth_poly,self.contact_spring_constant)
            # safety = self.maximum_withhold_force + penetration_depth_poly*self.contact_spring_constant
            return penetration_depth_poly - 0.005
            # print(safety.shape)
            # return safe_distance*self.contact_spring_constant
            # return jnp.array([1.0])
                    
        def alpha_2(self, h_2):
            return h_2*100 #constant, increase for smaller affected zone
        
        def gamma_2(self, v_2):
            return v_2*100

    config = SoRoConfig()
    clf_cbf = CLFCBF.from_config(config)

    def closed_loop_ode_fn(t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        z_des = args
        q, q_d = jnp.split(y, 2)
        # Create the full desired state (assume desired velocity is zero)
        u = clf_cbf.controller(y, z_des)
        
        # Compute the dynamical matrices.
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)
        contact_torque = config.contact_torque_fn(q)
        u += contact_torque
        # Compute the acceleration.
        q_dd = jnp.linalg.inv(B) @ (u - C @ q_d - G - K - D @ q_d)
        
        # Return the full state derivative.
        return jnp.concatenate([q_d, q_dd])

    # define the initial condition
    q0_arary = jnp.array([0, 0.0, -0.5])
    multiplier = [q0_arary for m in range(num_segments)]
    q0 = jnp.concatenate(multiplier)

    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # z_des is in shape of (num_segments * 3 * 2)
    # p_des_all = (num_waypoints, num_segments, 3)
    p_des_all = jnp.stack([
        jnp.concatenate([p.flatten(), jnp.zeros(3 * num_segments)])
        for p in config.p_des_all
        ])  # shape (num_waypoints, num_segments * 3* 2)

# Time settings.
    t0 = 0.0
    tf = 16.0
    dt = 2e-3     # integration step (for manual stepping)
    sim_dt = 1e-3 # simulation dt used by the solver

    # dt = 5e-3     # integration step (for manual stepping)
    # sim_dt = 2e-3 # simulation dt used by the solver

    def simulation_step(carry, _):
        t, y_current, current_index, track_indices = carry
        current_z_des = p_des_all[current_index]

        sol = dx.diffeqsolve(
            dx.ODETerm(closed_loop_ode_fn),
            dx.Tsit5(),
            t0=t,
            t1=t + dt,
            dt0=sim_dt,
            y0=y_current,
            args=current_z_des,
        )

        y_next = sol.ys[-1]
        u = clf_cbf.controller(y_current, current_z_des)

        # tracking update
        current_z_des, _ = jnp.split(current_z_des, 2)
        current_z_des = jnp.stack(jnp.split(current_z_des, num_segments))
        q, q_d = jnp.split(y_next, 2)
        p = batched_forward_kinematics_fn(config.robot_params, q, config.s_ps)
        end_p_ps = p[track_indices, :2]

        tracking_error = jnp.sum(jnp.stack([
            jnp.linalg.norm(end_p_ps[i, :2] - current_z_des[i, :2])
            for i in range(num_segments)
        ]))

        new_index = jnp.where(tracking_error < 0.05,
                            jnp.minimum(current_index + 1, p_des_all.shape[0] - 1),
                            current_index)

        t_next = t + dt
        new_carry = (t_next, y_next, new_index, track_indices)
        output = (t_next, y_next, u)
        return new_carry, output

    # @jax.jit
    # def run_simulation():
    #     # Determine number of steps
    #     num_steps = int((tf - t0) / dt)
        
    #     # Initial carry state: time, initial state, desired position index, and the indicies for tip of each segment
    #     init_carry = (t0, y0, 0, end_p_ps_indices)
        
    #     # Use jax.lax.scan to perform the simulation steps
    #     final_carry, (ts, ys) = jax.lax.scan(simulation_step, init_carry, None, length=num_steps)
    #     return ts, ys
    
    @jax.jit
    def run_simulation():
        num_steps = int((tf - t0) / dt)
        init_carry = (t0, y0, 0, end_p_ps_indices)
        final_carry, (ts, ys, us) = jax.lax.scan(simulation_step, init_carry, None, length=num_steps)
        return ts, ys, us

    # Run the simulation
    # ts, ys = run_simulation()
    ts, ys, us = run_simulation()
    # Optionally, split ys if needed (e.g., into q_ts and q_d_ts)
    q_ts, q_d_ts = jnp.split(ys, 2, axis=1)

    # Downsample for plotting and saving
    sampled_ts = ts[::20]
    sampled_q_ts = q_ts[::20]

    tracking_error_list = []
    contact_force_list = []

    for q in sampled_q_ts:
        p = batched_forward_kinematics_fn(robot_params, q, config.s_ps)
        p_ps = p[:, :2]

        cur_pts = p_ps[:-1]
        nxt_pts = p_ps[1:]
        ors = jnp.arctan2((nxt_pts - cur_pts)[:, 1], (nxt_pts - cur_pts)[:, 0])

        segs = jax.vmap(segmented_polygon, in_axes=(0, 0, 0, None))(
            cur_pts, nxt_pts, ors, robot_radius
        )

        def distance_to_all_obstacles(seg_poly):
            dists, _ = jax.vmap(lambda obs_poly: compute_distance(seg_poly, obs_poly))(
                config.poly_obstacle_pos
            )
            return dists

        d_segs = jax.vmap(distance_to_all_obstacles)(segs)
        d_segs = d_segs.reshape(-1)  # Flatten to (num_segments * num_obstacles,)

        d_segs = jnp.where(d_segs > 0, 0.0, d_segs)
        h_tail = - jnp.min(d_segs) * config.contact_spring_constant

        tracking_error = jnp.linalg.norm(p_ps[-1, :2] - config.p_des_all[-1, -1, :2])
        tracking_error_list.append(float(tracking_error))
        contact_force_list.append(float(h_tail))

    times_np = onp.array(sampled_ts)
    tracking_error_np = onp.array(tracking_error_list)
    contact_force_np = onp.array(contact_force_list)

    data = onp.concatenate([
        times_np[:, None],     # (N, 1)
        tracking_error_np[:, None],         # (N, 1)
        contact_force_np[:, None],         # (N, 1)
    ], axis=1)

    # Create header
    header = "time,tracking_error,contact_force"

    # Save to CSV
    # alpha = config.maximum_withhold_force
    filename = f"HOCLF_notouch.csv"
    onp.savetxt(filename, data, delimiter=",", header=header, comments='', fmt="%.6f")


    def get_contact_matrix(full_distance_array, threshold=0.0002):
        """Return a boolean contact matrix where distance < threshold."""
        return full_distance_array < threshold

    # ————————————————————————————————————————
    # 1. Configuration and initialization
    times = onp.array(ts[::20])
    full_distance_list = []
    chi_ps_list = []
    segs_list =[]

    # ————————————————————————————————————————
    # 2. Forward kinematics, segmentation, and distance computation
    for q in q_ts[::20]:
        p = batched_forward_kinematics_fn(robot_params, q, config.s_ps)
        p_ps = p[:, :2]
        chi_ps_list.append(onp.array(p_ps))

        cur_pts = p_ps[:-1]
        nxt_pts = p_ps[1:]
        ors = jnp.arctan2((nxt_pts - cur_pts)[:, 1], (nxt_pts - cur_pts)[:, 0])

        segs = jax.vmap(segmented_polygon, in_axes=(0, 0, 0, None))(
            cur_pts, nxt_pts, ors, robot_radius
        )
        segs_list.append(onp.array(segs))

        def distance_to_all_obstacles(seg_poly):
            dists, _ = jax.vmap(lambda obs_poly: compute_distance(seg_poly, obs_poly))(
                config.poly_obstacle_pos
            )
            return dists

        d_segs = jax.vmap(distance_to_all_obstacles)(segs)

        # Then still add to full_distance_list
        full_distance_list.append(d_segs)

    # ————————————————————————————————————————
    # 3. Stack all distances into a full distance array
    full_distance_array = jnp.stack(full_distance_list)  # (timestamp, segment, obstacle)

    # ————————————————————————————————————————
    # 4. Detect collisions (contacts)
    # threshold = 10/3000
    # red_contacts_by_time = get_contact_matrix(full_distance_array, threshold=config.maximum_withhold_force/config.contact_spring_constant)
    # blue_contacts_by_time = get_contact_matrix(full_distance_array, threshold=config.maximum_withhold_force/config.contact_spring_constant*2)

    red_contacts_by_time = get_contact_matrix(full_distance_array, threshold=-0.005) # shape (T, N_seg, N_obs)
    blue_contacts_by_time = get_contact_matrix(full_distance_array, threshold=0)  # shape (T, N_seg, N_obs)
    # ————————————————————————————————————————
    # 5. Extract valid contact points
    T, N_seg, N_obs = red_contacts_by_time.shape
    red_contact_points_list = []
    blue_contact_points_list = []

    for t in range(T):
        for seg_id in range(N_seg):
            for obs_id in range(N_obs):
                if red_contacts_by_time[t, seg_id, obs_id]:
                    # seg_poly_center = chi_ps_list[t][seg_id] 
                    seg_poly = segs_list[t][seg_id]
                    obs_poly = config.poly_obstacle_pos[obs_id]

                    # contact_pt = connect_project(seg_poly_center, jnp.mean(obs_poly, axis=0), obs_poly)
                    _, contact_pt, _ = find_closest_segment_point_and_direction(seg_poly, obs_poly, flag=True)

                    if not jnp.isnan(contact_pt).any():
                        red_contact_points_list.append((
                            t, seg_id, obs_id,
                            float(contact_pt[0]), float(contact_pt[1])
                        ))

    T, N_seg, N_obs = blue_contacts_by_time.shape
    for t in range(T):
        for seg_id in range(N_seg):
            for obs_id in range(N_obs):
                if blue_contacts_by_time[t, seg_id, obs_id]:
                    # seg_poly_center = chi_ps_list[t][seg_id]
                    seg_poly = segs_list[t][seg_id]
                    obs_poly = config.poly_obstacle_pos[obs_id]

                    _, contact_pt, _ = find_closest_segment_point_and_direction(seg_poly, obs_poly, flag=True)

                    if not jnp.isnan(contact_pt).any():
                        blue_contact_points_list.append((
                            t, seg_id, obs_id,
                            float(contact_pt[0]), float(contact_pt[1])
                        ))

    blue_contact_points_array = onp.array(blue_contact_points_list)
    red_contact_points_array = onp.array(red_contact_points_list)
    # Collect chi_ps values
    chi_ps_list = []

    # Animate the motion and collect chi_ps
    img_ts = []
    # pos = batched_forward_kinematics_fn(config.robot_params, config.q_des, config.s_ps)
    pos = jnp.stack([
        p[:, :2]  # reshapes p to (n, 3) and takes the first two columns from each row
        for p in config.p_des_all
        ]) # shape (num_waypoints,num_semgnets,2)

    batched_segment_robot = jax.vmap(segmented_polygon,in_axes=(0, 0, 0, None))

    def extract_multiple_contact_points_by_obs(contact_points_array, t_idx, num_obs):
        # Case 1: If empty input, return list of empty lists
        if contact_points_array.ndim < 2 or contact_points_array.shape[0] == 0:
            return [[] for _ in range(num_obs)]

        # Case 2: Filter for this timestep
        mask = contact_points_array[:, 0] == t_idx
        selected_points = contact_points_array[mask]

        result = [[] for _ in range(num_obs)]
        for row in selected_points:
            obs_idx = int(row[2])
            contact_point = row[3:]
            if 0 <= obs_idx < num_obs:
                result[obs_idx].append(contact_point)
            else:
                print(f"⚠️ obs_idx={obs_idx} out of range [0, {num_obs}) at time {t_idx}")
        return result


    pos = jnp.stack([
        p[:, :2]  # reshapes p to (n, 3) and takes the first two columns from each row
        for p in config.p_des_all
        ]) # shape (num_waypoints,num_semgnets,2)
    
    current_index = 0
    for t_idx, q in enumerate(q_ts[::20]):
        num_obs = len(config.poly_obstacle_pos)
        blue_frame_contact_points = extract_multiple_contact_points_by_obs(blue_contact_points_array, t_idx, num_obs)
        red_frame_contact_points = extract_multiple_contact_points_by_obs(red_contact_points_array, t_idx, num_obs) 

        current_z_des = p_des_all[current_index]
        current_z_des, _ = jnp.split(current_z_des,2)
        current_z_des = jnp.stack(jnp.split(current_z_des,num_segments)) #shape(num_segments,3)

        p = batched_forward_kinematics_fn(config.robot_params, q, config.s_ps)
        end_p_ps = p[end_p_ps_indices, :2] # the end of every segement
        
        # Update flag based on tracking error
        tracking_error = jnp.sum(jnp.stack([
            jnp.linalg.norm(end_p_ps[i, :2] - current_z_des[i, :2])
            for i in range(num_segments)
        ]))

        current_index = jnp.where(tracking_error < 0.1,
                            jnp.minimum(current_index + 1, p_des_all.shape[0]-1),
                            current_index)

        img = draw_image(
            batched_forward_kinematics_fn,
            batched_segment_robot,
            half_circle_to_polygon,
            auxiliary_fns,
            robot_params,
            num_segments,
            q,
            poly_points=config.poly_obstacle_pos,
            p_des_all=config.p_des_all[1, 1, :2],
            index=current_index,
            blue_contact_points=blue_frame_contact_points,
            red_contact_points=red_frame_contact_points,
            enable_contact=True,
        )
        img_ts.append(img)

        chi_ps = batched_forward_kinematics_fn(config.robot_params, q, config.s_ps)
        chi_ps_list.append(onp.array(chi_ps))

    # Save chi_ps to a CSV using the csv module
    with open(outputs_dir / "chi_ps_values.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header (optional, depending on chi_ps dimensions)
        writer.writerow([f"Chi_{i}_{j}" for i in range(chi_ps.shape[0]) for j in range(chi_ps.shape[1])])
        # Write data rows
        for chi_ps in chi_ps_list:
            writer.writerow(chi_ps.flatten())

    # Animate the images
    img_ts = onp.stack(img_ts, axis=0)
    animate_images_cv2(
        onp.array(ts[::20]), img_ts, outputs_dir / "planar_pcs_safe_closed_loop_simulation.mp4"
    )

if __name__ == "__main__":
    soft_robot_with_safety_contact_CBFCLF_example()