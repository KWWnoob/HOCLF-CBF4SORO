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
from src.planar_pcs_rendering_manipulations import draw_image

# define the outputs directory
outputs_dir = Path("outputs") / "planar_pcs_simulation"
outputs_dir.mkdir(parents=True, exist_ok=True)

# load symbolic expressions
num_segments = 1
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
# print(inspect.getsource(batched_forward_kinematics_fn.__wrapped__))
# get the jacobian of the forward kinematics
jacobian_fk_fn = jax.jacfwd(batched_forward_kinematics_fn, argnums=1)

# segmenting params
num_points = 20*num_segments
# Compute indices: equivalent to
# [num_points * (i+1)//num_segments - 1 for i in range(num_segments)]
end_p_ps_indices = (jnp.arange(1, num_segments+1) * num_points // num_segments) - 1


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

def ideal_point(x_des, y_des, x_obj, y_obj, r_robot, r_obj, offset = -0.001):

    # Compute the difference between centers and distance
    dx = x_des - x_obj
    dy = y_des - y_obj
    d = jnp.sqrt(dx**2 + dy**2)
    
    # Compute the unit vector from destination to object center
    ux = dx / d
    uy = dy / d
    
    # Compute the intersection points along the line connecting the centers
    hit_dist = r_robot + r_obj + offset
    x_hit = x_obj - hit_dist * ux
    y_hit = y_obj - hit_dist * uy
    return jnp.array([x_hit, y_hit])

def ideal_point_V2(x_des, y_des, x_obj, y_obj, r_robot, r_obj, max_distance, max_offset = -0.001, min_offset = 0.0):

    # Compute the difference between centers and distance
    dx = x_des - x_obj
    dy = y_des - y_obj
    d = jnp.sqrt(dx**2 + dy**2)

    scale = jnp.clip(d/max_distance, 0.0, 1.0)

    offset = scale * max_offset + (1.0 - scale) * min_offset
    
    # Compute the unit vector from destination to object center
    ux = dx / d
    uy = dy / d
    
    # Compute the intersection points along the line connecting the centers
    hit_dist = r_robot + r_obj + offset
    x_hit = x_obj - hit_dist * ux
    y_hit = y_obj - hit_dist * uy
    return jnp.array([x_hit, y_hit])

def compute_contact_force(p_robot_end, p_obs, v_robot_end, v_obs, r_robot, r_obj, c_damp = 50):
    #TODO: check the force formulation
    diff = p_robot_end - p_obs
    distance = jnp.linalg.norm(diff)
    
    eps = 1e-6
    safe_distance = jnp.maximum(distance, eps) # ensuring minimum distance does not go to zero 
    
    penetration = jnp.maximum((r_robot + r_obj) - safe_distance, 0.0)
    # jax.debug.print("Cooridnates = {}{}", p_robot_end,p_obs)
    normal = diff / safe_distance
    
    F_spring = 600 * penetration  #
    
    # Compute the relative velocity.
    v_rel = v_robot_end - v_obs
    # Project relative velocity along the normal.
    v_normal = jnp.dot(v_rel, normal)
    
    # Damping force opposing the normal component of the relative velocity.
    F_damp = - c_damp * v_normal
    
    F_contact = jnp.where(penetration > 0, (F_spring + F_damp) * normal, jnp.zeros_like(normal))
    
    return F_contact

def soft_robot_with_safety_contact_CBFCLF_example():
    
    # define the ODE function
    class SoRoConfig(CLFCBFConfig):
        '''Config for soft robot'''

        def __init__(self):

            self.robot_params = robot_params

            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

            '''Handling Object, circle'''
            self.object_mass = 1
            self.object_radius = 0.015
            self.object_friction = 10
            self.object_initial_pos = [0.03, 0.155]
            self.object_initial_velocity = [0, 0]

            self.p_des = jnp.array([0.15, 0.25])
            self.p_des_all = jnp.array([self.p_des])

            '''destination of robot'''
            self.s_ps = jnp.linspace(0, robot_length * num_segments, 20 * num_segments) # segmented

            '''Select the end of each segment'''
            self.indices = end_p_ps_indices

            '''Contact model Parameter'''
            self.contact_spring_constant = 2000 #contact force model
            self.maximum_withhold_force = 20

            super().__init__(
                n=6 * num_segments + 2 * 2, # number of states + the position of the objects
                m=3 * num_segments, # number of inputs
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=False,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                cbf_relaxation_penalty=1e3,
                clf_relaxation_penalty=10
            )

        def f(self, z) -> Array:
            # Split the augmented state into positions and velocities.
            q, q_d = jnp.split(z, 2)
            # Assume the last 2 entries of q are the object's 2D position.
            q_robot = q[:-2]    # robot's positions
            p_obs   = q[-2:]    # object's position

            # Similarly, assume the last 2 entries of q_d are the object's velocity.
            q_d_robot = q_d[:-2]  # robot's velocities
            v_obs     = q_d[-2:]  # object's velocity

            # Compute the robot dynamics (for its own state only).
            B, C, G, K, D, alpha = dynamical_matrices_fn(self.robot_params, q_robot, q_d_robot)
            # Compute the nominal robot drift term.
            drift_robot = -jnp.linalg.inv(B) @ (C @ q_d_robot + D @ q_d_robot + G + K)

            # Get the Contact Force
            p = batched_forward_kinematics_fn(self.robot_params, q_robot, self.s_ps)
            v = batched_forward_kinematics_fn(self.robot_params, q_d_robot, self.s_ps)
            
            p_tip = p[-1, :2]
            v_tip = v[-1, :2]

            F_contact = compute_contact_force(p_tip, p_obs, v_tip, v_obs, robot_radius, self.object_radius)
            
            # Get Jacobian
            Jacobian = jacobian_fk_fn(self.robot_params, q_robot, self.s_ps)
            Jacobian_tip = Jacobian[-1, :2, :]
            tau_contact = Jacobian_tip.T @ F_contact

            # Modify the robot's drift with the effect of the contact force.
            drift_robot = drift_robot - jnp.linalg.inv(B) @ tau_contact

            # --- Object Dynamics ---
            # The object's dynamics: dp_obs/dt = v_obs, and dv_obs/dt = F_contact / m_obj.
            dp_obs = v_obs
            friction_force = - self.object_friction * v_obs
            dv_obs = (-F_contact+friction_force) / self.object_mass  # Ensure self.m_obj is defined as the object's mass. TODO: check sign

            # --- Combine Derivatives ---
            # Derivative of positions: robot positions derivative and object positions derivative.
            dq = jnp.concatenate([q_d_robot, v_obs])
            # Derivative of velocities: robot acceleration and object acceleration.
            dq_d = jnp.concatenate([drift_robot, dv_obs])
            # jax.debug.print("dv_obs = {}", dv_obs)
            # Return the concatenated derivative in the same ordering as the state.
            return jnp.concatenate([dq, dq_d])

        def g(self, z) -> Array:
            # Split the augmented state into positions and velocities.
            q, q_d = jnp.split(z, 2)
            # Assume the last 2 entries of q are the object's 2D position.
            q_robot = q[:-2]    # robot's positions (dimension: n_r)
            p_obs   = q[-2:]    # object's position (2D)

            # Similarly, assume the last 2 entries of q_d are the object's velocity.
            q_d_robot = q_d[:-2]  # robot's velocities (dimension: n_r)
            v_obs     = q_d[-2:]  # object's velocity (2D)

            # Compute the robot dynamics using only the robot's states.
            B, _, _, _, _, _ = dynamical_matrices_fn(self.robot_params, q_robot, q_d_robot)
            # Compute the control matrix for the robot: B^{-1} (size: (n_r, m)).
            control_matrix = jnp.linalg.inv(B)

            # Determine the dimensions.
            n_r = q_robot.shape[0]           # Robot configuration dimension.
            m   = control_matrix.shape[1]    # Control input dimension.
            
            # Top block: positions derivatives have no control influence.
            # The full position part of the state includes both the robot and object positions.
            pos_dim = q.shape[0]  # This is n_r + 2.
            zeros_top = jnp.zeros((pos_dim, m))
            
            # Bottom block: velocities derivative.
            # For the robot (first n_r entries), we use control_matrix.
            # For the object (last 2 entries), control input is zero.
            zeros_object = jnp.zeros((2, m))
            bottom_block = jnp.concatenate([control_matrix, zeros_object], axis=0)  # Shape: ((n_r+2), m)
            
            # Concatenate the top and bottom blocks vertically.
            # The overall g has shape: ((pos_dim + pos_dim) x m) = ((2*(n_r+2)) x m).
            return jnp.concatenate([zeros_top, bottom_block], axis=0)
    
        def V(self, z: jnp.ndarray, z_des: jnp.ndarray) -> jnp.ndarray:

            q, q_d = jnp.split(z, 2)
            p_obs = q[-2:]  
            v_obs = q_d[-2:]  
            
            q_des, _ = jnp.split(z_des, 2)
            p_des = q_des[-2:]
            
            term = p_obs - p_des
            term = jnp.dot(term, term)
            return term
        
        def V_d(self, z: jnp.ndarray, z_des: jnp.ndarray) -> jnp.ndarray:

            q, q_d = jnp.split(z, 2)
            p_obs = q[-2:]  
            v_obs = q_d[-2:]  
            
            q_des, _ = jnp.split(z_des, 2)
            p_des = q_des[-2:]

            term = 2 * jnp.dot(p_obs - p_des, v_obs)
            return term

        def V_dd(self, z: jnp.ndarray, z_des: jnp.ndarray) -> jnp.ndarray:

            q, q_d = jnp.split(z, 2)
            p_obs = q[-2:]  
            v_obs = q_d[-2:]  
            
            q_des, _ = jnp.split(z_des, 2)
            p_des = q_des[-2:]

            a_obs = self.f(z)[-2:]

            term = 2 * jnp.dot(v_obs, v_obs) + 2 * jnp.dot(p_obs - p_des, a_obs)
            return term
        
        def V_approach(self, z: jnp.ndarray, z_des: jnp.ndarray) -> jnp.ndarray:
            
            q, q_d = jnp.split(z, 2)
            q_robot = q[:-2]
            p_obs = q[-2:]  

            q_des, _ = jnp.split(z_des, 2)
            p_des = q_des[-2:]

            p = batched_forward_kinematics_fn(self.robot_params, q_robot, self.s_ps)
            p_tip = p[-1, :2]

            max_distance = jnp.linalg.norm(jnp.array(p_des) - jnp.array(self.object_initial_pos))

            desired_point = ideal_point_V2(
                x_des=p_des[0], 
                y_des=p_des[1], 
                x_obj=p_obs[0], 
                y_obj=p_obs[1], 
                r_robot=robot_radius, 
                r_obj=self.object_radius,
                max_distance = max_distance
            )
            distance = desired_point - p_tip
            lyapunov_value = jnp.sqrt(distance **2) 

            return lyapunov_value


        def V_2(self, z: jnp.ndarray, z_des: jnp.ndarray,
                c0: float = 5.8, c1: float = 5.0) -> jnp.ndarray:

            V_val = self.V(z, z_des)         
            V_d_val = self.V_d(z, z_des)        
            V_dd_val = self.V_dd(z, z_des)     
            
            V1 = V_d_val + c0 * V_val
            V1_dot = V_dd_val + c0 * V_d_val
            V2 = V1_dot + c1 * V1
            
            V1_approach = self.V_approach(z, z_des)
            
            psi = jnp.concatenate([V1_approach, jnp.atleast_1d(V2)*4])
            return V1_approach

        def h_2(self, z) -> jnp.ndarray:
            q, q_d = jnp.split(z, 2)
            q_robot = q[:-2]
            p_obs = q[-2:]  

            q_des, _ = jnp.split(z_des, 2)
            p_des = q_des[-2:]

            p = batched_forward_kinematics_fn(self.robot_params, q_robot, self.s_ps)
            p_tip = p[-1, :2]

            max_distance = jnp.linalg.norm(jnp.array(p_des) - jnp.array(self.object_initial_pos))

            desired_point = ideal_point_V2(
                x_des=p_des[0], 
                y_des=p_des[1], 
                x_obj=p_obs[0], 
                y_obj=p_obs[1], 
                r_robot=robot_radius, 
                r_obj=self.object_radius,
                max_distance = max_distance
            )
            distance = desired_point - p_tip
            lyapunov_value = jnp.sqrt(distance **2) 
            
            return jnp.array([1.0])
                    
        def alpha_2(self, h_2):
            return h_2*1 #constant, increase for smaller affected zone
        
        def gamma_2(self, v_2):
            return v_2*30

    config = SoRoConfig()
    clf_cbf = CLFCBF.from_config(config)

    def closed_loop_ode_fn(t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        z_des = args
        q, q_d = jnp.split(y, 2)
        
        # Extract the robot state from q (assume the last 2 entries are for the object).
        q_robot = q[:-2]
        q_d_robot = q_d[:-2]
        
        # Use your controller to compute u (the control input).
        u = clf_cbf.controller(y, z_des)
        
        # Compute robot dynamics matrices using the robot's configuration.
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q_robot, q_d_robot)
        
        # --- Contact Force Incorporation ---
        # Compute the forward kinematics to get the tip position.
        p = batched_forward_kinematics_fn(robot_params, q_robot, config.s_ps)
        v = batched_forward_kinematics_fn(robot_params, q_d_robot, config.s_ps)
        p_tip = p[-1, :2]
        v_tip = v[-1, :2]
        
        # Assume the object's position is given by the last two entries in q.
        p_obj = q[-2:]
        v_obj = q_d[-2:]
        
        # Compute the contact force using your contact model.
        F_contact = compute_contact_force(p_tip, p_obj, v_tip, v_obj, robot_radius, config.object_radius)
        
        # Compute the Jacobian at the tip position.
        Jacobian = jacobian_fk_fn(robot_params, q_robot, config.s_ps)
        Jacobian_tip = Jacobian[-1, :2, :]  # This extracts the relevant rows for the 2D tip
        
        # Map the contact force into joint space (torque).
        tau_contact = Jacobian_tip.T @ F_contact
        
        # --- Update Robot Acceleration ---
        # Subtract the torque effect due to the contact force.
        q_dd_robot = jnp.linalg.inv(B) @ (u - C @ q_d_robot - G - K - D @ q_d_robot - tau_contact)
        
        # For the object, if not controlled, assume zero acceleration.
        friction_force = - config.object_friction * q_d[-2:]
        q_dd_object = (-F_contact+friction_force) / config.object_mass
        
        # Combine derivatives:
        dq = jnp.concatenate([q_d_robot, q_d[-2:]])   # positions derivative remains same
        dq_d = jnp.concatenate([q_dd_robot, q_dd_object])  # velocities derivative includes new acceleration
        
        return jnp.concatenate([dq, dq_d])

    # define the initial condition
    q0_arary_0 = jnp.array([0.0, 0.0, 0.0])
    object_pos = jnp.array(config.object_initial_pos)  # Convert list to JAX array
    q0_arary_0 = jnp.concatenate([q0_arary_0, object_pos])
    # q0_arary_1 = jnp.array([-jnp.pi*1/3, 0.0, -0.5])
    multiplier = [q0_arary_0]
    q0 = jnp.concatenate(multiplier)
    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # z_des is in shape of (num_segments * 3 * 2)
    # p_des_all = (num_waypoints, num_segments, 3)
    p_des_all = jnp.stack([
        jnp.concatenate([jnp.zeros(3*num_segments), p.flatten(), jnp.zeros(3 * num_segments +2)])
        for p in config.p_des_all
        ])  # shape (num_waypoints, num_segments * 3* 2 + 4 )
    
    # Time settings.
    t0 = 0.0
    tf = 12.0
    dt = 1e-3     # integration step (for manual stepping)
    sim_dt = 5e-4 # simulation dt used by the solver

    def simulation_step(carry, _):
        """
        Performs one simulation step.
        
        carry: a tuple (t, y_current, current_flag)
        _    : placeholder for scan (unused)
        """
        t, y_current, current_index = carry
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
        
        y_next = sol.ys[-1]

        new_index = 0
        
        # Update time
        t_next = t + dt

        # New carry for next iteration
        new_carry = (t_next, y_next, new_index)
        # Output for storage (time and state)
        output = (t_next, y_next)
        return new_carry, output

    @jax.jit
    def run_simulation():
        # Determine number of steps
        num_steps = int((tf - t0) / dt)
        
        # Initial carry state: time, initial state, and index (starting at 0)
        init_carry = (t0, y0, 0)
        
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

    batched_segment_robot = jax.vmap(segmented_polygon,in_axes=(0, 0, 0, None))

    current_index = 0
    for q in q_ts[::20]:
        q_robot = q[:-2]
        p_obs = q[-2:]
        img = draw_image(
            batched_forward_kinematics_fn,
            batched_segment_robot,
            half_circle_to_polygon,
            auxiliary_fns,
            robot_params,
            num_segments,
            q_robot,
            x_obs = p_obs,
            R_obs = config.object_radius,
            p_des_all = config.p_des_all,
            index = current_index
        )
        img_ts.append(img)

    # Animate the images
    img_ts = onp.stack(img_ts, axis=0)
    animate_images_cv2(
        onp.array(ts[::20]), img_ts, outputs_dir / "planar_pcs_safe_closed_loop_simulation.mp4"
    )

if __name__ == "__main__":
    soft_robot_with_safety_contact_CBFCLF_example()