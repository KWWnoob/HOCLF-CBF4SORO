'''
SAT MULTI AXIS VERSION
'''

import csv
import diffrax as dx
from functools import partial
import jax
from cbfpy.cbfpy.cbfs.clf_cbf import CLFCBF, CLFCBFConfig
import inspect

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, debug, jacfwd, jit, vmap, lax
from jax import numpy as jnp
import jsrm
from jsrm.systems import planar_pcs
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict, Tuple

from src.img_animation import animate_images_cv2
from src.planar_pcs_rendering_medical_sequential import draw_image

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

def compute_distance(robot_vertices, polygon_vertices, epsilon=1e-6):
    robot_normals = get_normals(robot_vertices)
    poly_normals  = get_normals(polygon_vertices)
    candidate_axes = jnp.concatenate([robot_normals, poly_normals], axis=0)
    
    proj_robot = robot_vertices @ candidate_axes.T
    proj_poly  = polygon_vertices @ candidate_axes.T
    
    min_R = jnp.min(proj_robot, axis=0)
    max_R = jnp.max(proj_robot, axis=0)
    min_P = jnp.min(proj_poly, axis=0)
    max_P = jnp.max(proj_poly, axis=0)
    
    separated_mask = (max_R < min_P - epsilon) | (max_P < min_R - epsilon)
    
    penetration = jnp.minimum(max_R, max_P) - jnp.maximum(min_R, min_P)
    
    def separated_case(_):
        gap = compute_gap_along_centers(robot_vertices, polygon_vertices)
        return gap

    def overlapping_case(_):
        pen = -jnp.min(penetration)
        return pen

    is_separated = jnp.any(separated_mask)
    overall_distance = jax.lax.cond(
        is_separated,
        separated_case,
        overlapping_case,
        operand=None
    )
    
    flag = jax.lax.cond(is_separated, lambda _: 1, lambda _: 0, operand=None)
    
    return overall_distance, flag

def compute_distance_circle(robot_center, circle_center):
    d2o_ps = jnp.linalg.norm((robot_center - circle_center), ord=2)
    return d2o_ps

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

def circle_to_polygon_cw(center, radius, num_points):
    """
    Approximate a circle as a polygon with vertices ordered in clockwise (CW) order.

    Parameters:
      center (array-like): The (x, y) coordinates of the circle center.
      radius (float): The radius of the circle.
      num_points (int): Number of vertices of the polygon.

    Returns:
      jnp.ndarray: An array of shape (num_points, 2) containing the polygon vertices in CW order.
    """
    # Generate evenly spaced angles between 0 and 2*pi (excluding the endpoint)
    angles = jnp.linspace(0, 2 * jnp.pi, num_points, endpoint=False)
    # To get CW order, reverse the order of angles
    angles = angles[::-1]
    
    # Compute x and y coordinates of the vertices
    x_coords = center[0] + radius * jnp.cos(angles)
    y_coords = center[1] + radius * jnp.sin(angles)
    
    # Stack the coordinates to get a (num_points, 2) array
    vertices = jnp.stack([x_coords, y_coords], axis=-1)
    return vertices

from jax.scipy.special import logsumexp

def sabs(x: jnp.ndarray, alpha_abs: float) -> jnp.ndarray:
    """
      xtanh    |x|：f_sabs(x) = x * tanh(alpha_abs * x)
    """
    return x * jnp.tanh(alpha_abs * x)

def compute_distance(
    robot_vertices: jnp.ndarray,
    polygon_vertices: jnp.ndarray,
    alpha_abs:   float = 6000,  #
    alpha_pair:  float = 2080, 
    alpha_axes:  float = 9000.0, 
) -> tuple[jnp.ndarray, jnp.ndarray]:


    Rn = get_normals(robot_vertices)    # (Nr,2)
    Pn = get_normals(polygon_vertices)  # (Np,2)
    axes = jnp.concatenate([Rn, Pn], axis=0)  # (K,2)

    prj_R = robot_vertices   @ axes.T  # (Nr, K)
    prj_P = polygon_vertices @ axes.T  # (Np, K)
    min_R = jnp.min(prj_R, axis=0)     # (K,)
    max_R = jnp.max(prj_R, axis=0)
    min_P = jnp.min(prj_P, axis=0)
    max_P = jnp.max(prj_P, axis=0)

    d1 = min_P - max_R  # (K,)
    d2 = min_R - max_P  # (K,)

    d1_s = sabs(d1, alpha_abs)
    d2_s = sabs(d2, alpha_abs)

    pair_stack = jnp.stack([d1_s, d2_s], axis=0)  # (2, K)
    gap_axes = (1.0/alpha_pair) * logsumexp(alpha_pair * pair_stack,
                                            axis=0)   # (K,)

    # 6) max -> h
    h = (1.0/alpha_axes) * logsumexp(alpha_axes * gap_axes)

    # 7) flag
    flag = jnp.where(h > 0, 1, 0)

    return h, flag

def soft_robot_with_safety_contact_CBFCLF_example():
    


    # define the ODE function
    class SoRoConfig(CLFCBFConfig):
        '''Config for soft robot'''

        def __init__(self):

            self.robot_params = robot_params

            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

            '''Polygon Obstacle Parameter'''
            char_length = robot_length
            self.circle_pos = jnp.array([char_length*8/jnp.pi + 1.5*robot_radius, 0])
            self.inner_circle_radius = char_length*8/jnp.pi - 0.5*robot_radius
            self.outer_circle_radius = char_length*8/jnp.pi + 4*robot_radius

            self.circle_obs_1_pos = jnp.array([-robot_radius, char_length*1.3])
            self.circle_obs_1_radius = robot_radius*2.3
            self.circle_obs_2_pos = jnp.array([char_length*1.2, char_length*2])
            self.circle_obs_2_radius = robot_radius*1.8

            self.obs_center = jnp.stack([self.circle_obs_1_pos, self.circle_obs_2_pos])
            self.obs_radius = jnp.stack([self.circle_obs_1_radius, self.circle_obs_2_radius])

            '''destination of robot'''
            self.s_ps = jnp.linspace(0, robot_length * num_segments, 20 * num_segments) # segmented

            self.p_des_1_1 = jnp.array([0.05951909*0.7, 0.15234353*1, -jnp.pi*1.8*robot_length])
            self.p_des_2_1 = jnp.array([0.05951909*1.6, 0.15234353*1.9,-jnp.pi/3 ])

            self.p_des_1_2 = jnp.array([0.05951909*1.1, 0.15234353*1.6, -jnp.pi*1.8*robot_length])
            self.p_des_2_2 = jnp.array([0.05951909*5, 0.15234353*2.5,-jnp.pi/2 ])
            
            self.p_des_1_3 = jnp.array([0.05951909*1.7, 0.15234353*2.0,-jnp.pi/3 ])
            self.p_des_2_3 = jnp.array([0.05951909*5.5, 0.15234353*2.5,-jnp.pi/2 ])

            self.p_des_1 = jnp.stack([self.p_des_1_1,self.p_des_2_1]) # shape (num_of_segments, 3)
            self.p_des_2 = jnp.stack([self.p_des_2_1,self.p_des_2_2])
            self.p_des_3 = jnp.stack([self.p_des_1_3,self.p_des_2_3])
            
            self.p_des_all = jnp.stack([self.p_des_1, self.p_des_2, self.p_des_3]) # shape (num_waypoints, num_of_segments, 3)
            self.num_waypoints = self.p_des_all.shape[0]

            '''Select the end of each segment'''
            self.indices = end_p_ps_indices

            '''Contact model Parameter'''
            self.contact_spring_constant = 2000 #contact force model
            self.maximum_withhold_force = 0

            super().__init__(
                n=6 * num_segments, # number of states + flag
                m=3 * num_segments, # number of inputs
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=False,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                cbf_relaxation_penalty=1e3,
                clf_relaxation_penalty=10
            )

        def f(self, z) -> Array:
            q, q_d = jnp.split(z, 2)  # Split state z into q (position) and q_d (velocity)
            B, C, G, K, D, alpha = dynamical_matrices_fn(self.robot_params, q, q_d)

            # Drift term (f(x))
            drift = (
                -jnp.linalg.inv(B) @ (C @ q_d + D @ q_d + G + K)
            )
            return jnp.concatenate([q_d, drift])

        def g(self, z) -> Array:
            q, q_d = jnp.split(z, 2)
            B, _, _, _, _, _ = dynamical_matrices_fn(self.robot_params, q, q_d)

            # Control matrix g(x)
            control_matrix = jnp.linalg.inv(B)

            # Match dimensions for concatenation
            zero_block = jnp.zeros((q.shape[0], control_matrix.shape[1]))

            return jnp.concatenate([zero_block, control_matrix], axis=0)
        
        def V_2(self, z, z_des) -> tuple[jnp.ndarray, jnp.ndarray]:
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
            
            # -------- Process the polygon obstacles --------
            # inner circle avoid
            penetration_inner = jax.vmap(compute_distance_circle,in_axes=(0,None))(p_ps,self.circle_pos) - self.inner_circle_radius - robot_radius
            penetration_outer = - jax.vmap(compute_distance_circle,in_axes=(0,None))(p_ps,self.circle_pos) + self.outer_circle_radius - robot_radius

            penetration_obs_1 = jax.vmap(compute_distance_circle,in_axes=(0,None))(p_ps,self.obs_center[0]) - self.obs_radius[0] - robot_radius
            penetration_obs_2 = jax.vmap(compute_distance_circle,in_axes=(0,None))(p_ps,self.obs_center[1]) - self.obs_radius[1] - robot_radius
            
            safe_distance = jnp.concatenate([penetration_inner,penetration_outer,penetration_obs_1,penetration_obs_2])
            safe_distance = safe_distance.reshape(-1)
            return safe_distance
                    
        def alpha_2(self, h_2):
            return h_2*200 #constant, increase for smaller affected zone
        
        def gamma_2(self, v_2):
            return v_2*200

    config = SoRoConfig()
    clf_cbf = CLFCBF.from_config(config)

    def closed_loop_ode_fn(t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        z_des = args
        q, q_d = jnp.split(y, 2)
        # Create the full desired state (assume desired velocity is zero)
        u = clf_cbf.controller(y, z_des)
        
        # Compute the dynamical matrices.
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)
        
        # Compute the acceleration.
        q_dd = jnp.linalg.inv(B) @ (u - C @ q_d - G - K - D @ q_d)
        
        # Return the full state derivative.
        return jnp.concatenate([q_d, q_dd])

    # define the initial condition
    q0_arary_0 = jnp.array([-jnp.pi*3/3, 0.0, -0.5])
    q0_arary_1 = jnp.array([-jnp.pi*1/3, 0.0, -0.5])
    multiplier = [q0_arary_0,q0_arary_1]
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
    dt = 1e-3     # integration step (for manual stepping)
    sim_dt = 5e-4 # simulation dt used by the solver

    def simulation_step(carry, _):
        """
        Performs one simulation step.
        
        carry: a tuple (t, y_current, current_flag)
        _    : placeholder for scan (unused)
        """
        t, y_current, current_index, track_indices = carry
        # Choose z_des based on current_index
        current_z_des = p_des_all[current_index]

        # Integrate the ODE from t to t + dt
        sol = dx.diffeqsolve(
            dx.ODETerm(closed_loop_ode_fn),
            dx.Tsit5(),
            t0=t,
            t1=t + dt,
            dt0=sim_dt,
            y0=y_current,
            args=current_z_des,
        )

        current_z_des, _ = jnp.split(current_z_des, 2) # get the desired position
        current_z_des = jnp.stack(jnp.split(current_z_des,num_segments)) # in the shape of (num_segments, 3)
        
        y_next = sol.ys[-1]
        q, q_d = jnp.split(y_next, 2)
        p = batched_forward_kinematics_fn(config.robot_params, q, config.s_ps)
        
        end_p_ps = p[track_indices, :2] # the end of every segement
        
        # Update flag based on tracking error
        tracking_error = jnp.sum(jnp.stack([
            jnp.linalg.norm(end_p_ps[i, :2] - current_z_des[i, :2])
            for i in range(num_segments)
        ]))

        new_index = jnp.where(tracking_error < 0.05,
                            jnp.minimum(current_index + 1, p_des_all.shape[0]-1),
                            current_index)
        
        # Update time
        t_next = t + dt

        # New carry for next iteration
        new_carry = (t_next, y_next, new_index, track_indices)
        # Output for storage (time and state)
        output = (t_next, y_next)
        return new_carry, output

    @jax.jit
    def run_simulation():
        # Determine number of steps
        num_steps = int((tf - t0) / dt)
        
        # Initial carry state: time, initial state, and index (starting at 0)
        init_carry = (t0, y0, 0, end_p_ps_indices)
        
        # Use jax.lax.scan to perform the simulation steps
        final_carry, (ts, ys) = jax.lax.scan(simulation_step, init_carry, None, length=num_steps)
        return ts, ys

    # Run the simulation
    ts, ys = run_simulation()

    # Optionally, split ys if needed (e.g., into q_ts and q_d_ts)
    q_ts, q_d_ts = jnp.split(ys, 2, axis=1)
    # Collect chi_ps values
    chi_ps_list = []

    # Animate the motion and collect chi_ps
    img_ts = []
    
    pos = jnp.stack([
        p[:, :2]  # reshapes p to (n, 3) and takes the first two columns from each row
        for p in config.p_des_all
        ]) # shape (num_waypoints,num_semgnets,2)

    circle_center = jnp.stack([config.circle_pos, config.circle_pos])
    obs_center = config.obs_center
    obs_radius = config.obs_radius
    batched_segment_robot = jax.vmap(segmented_polygon,in_axes=(0, 0, 0, None))

    current_index = 0
    for q in q_ts[::20]:
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
            circle_center = circle_center,
            inner_radius = config.inner_circle_radius,
            outer_radius = config.outer_circle_radius,
            obs_center = obs_center,
            obs_radius = obs_radius,
            p_des_all = None, # orgianlly was "pos"
            index = current_index
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