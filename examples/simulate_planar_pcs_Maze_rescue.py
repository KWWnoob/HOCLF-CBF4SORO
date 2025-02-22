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
from src.planar_pcs_rendering_rescue import draw_image

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

def soft_robot_with_safety_contact_CBFCLF_example():
    
    # define the ODE function
    class SoRoConfig(CLFCBFConfig):
        '''Config for soft robot'''

        def __init__(self):

            self.robot_params = robot_params

            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

            '''Polygon Obstacle Parameter'''
            char_length = robot_length
            self.poly_obstacle_shape_1 = jnp.array([[0.0, 0.0],
                                                [0.0, char_length*2.3],
                                                [char_length*0.5, char_length*2.3],
                                                [char_length*0.5, 0.0]])
            
            self.poly_obstacle_shape_2 = jnp.array([[0.0, 0.0],
                                                [0.0, char_length*0.45],
                                                [char_length*0.5, char_length*0.45],
                                                [char_length*0.5, 0.0]])

            self.poly_obstacle_shape_3 = jnp.array([[0.0, 0.0],
                                                    [0.0, char_length*0.45],
                                                    [char_length*1.2, char_length*0.45],
                                                    [char_length*1.2, 0.0]])
            
            # self.poly_obstacle_pos = self.poly_obstacle_shape/4 + jnp.array([-0.08,0.04])
            self.poly_obstacle_pos_1 = self.poly_obstacle_shape_1 + jnp.array([-0.09,0])
            self.poly_obstacle_pos_2 = self.poly_obstacle_shape_2 + jnp.array([0.07,0])

            self.poly_obstacle_pos_3 = self.poly_obstacle_pos_1[2,:] + self.poly_obstacle_shape_3
            
            self.poly_obstacle_pos_4 = self.poly_obstacle_pos_2[1,:] + self.poly_obstacle_shape_3
            
            self.poly_obstacle_pos_5 = self.poly_obstacle_pos_3[2,:] + self.poly_obstacle_shape_2+ jnp.array([-char_length*0.5, 0])

            self.poly_obstacle_pos_6 = self.poly_obstacle_pos_4[2,:] + self.poly_obstacle_shape_1

            self.poly_obstacle_pos = jnp.stack([self.poly_obstacle_pos_1,self.poly_obstacle_pos_2,self.poly_obstacle_pos_3,self.poly_obstacle_pos_4,self.poly_obstacle_pos_5,self.poly_obstacle_pos_6])

            '''Characteristic of robot'''
            self.s_ps = jnp.linspace(0, robot_length * num_segments, 20 * num_segments) # segmented
            self.q_des_arary_1 = jnp.array([-jnp.pi*1.8, 0.3, 0.4])
            # self.q_des_array_0 = jnp.array([-jnp.pi*1.8, 0.3, 0.4])/2
            self.q_des_arary_3 = jnp.array([jnp.pi*2, 0.3, 0.4])
            # self.q_des_arary_2 = jnp.array([jnp.pi*2, 0.3, 0.4])/2
            self.q_des = jnp.concatenate([self.q_des_arary_3])# destination

            self.p_des_1 = jnp.array([0.05951909*1.5, 0.15234353*0.85, -jnp.pi*1.8*robot_length])
            self.p_des_2 = jnp.array([0.18, 0.40234353, 0])

            self.p_des = jnp.stack([self.p_des_1,self.p_des_2])
            print(self.p_des)

            '''Contact model Parameter'''
            self.contact_spring_constant = 2000 #contact force model
            self.maximum_withhold_force = 0

            super().__init__(
                n=6 * num_segments, # number of states
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
        
        # def V_2(self, z) -> jnp.ndarray:
        #     # CLF: tracking error for different segments with priority weighting
            
        #     # split
        #     q, q_d = jnp.split(z, 2)
            
        #     # FK
        #     p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            
        #     # Track the index points
        #     num_points = p.shape[0]
        #     index = [num_points * (i+1) // num_segments - 1 for i in range(num_segments)]
            
        #     # Get the indexed state
        #     p_list = [p[i, :] for i in index]
            
        #     # Destination states
        #     p_des = batched_forward_kinematics_fn(self.robot_params, self.q_des, self.s_ps)
        #     p_des_list = [p_des[i, :] for i in index]
            
        #     # weighting of errors
        #     weights = [1.2] + [1.0] * (num_segments - 1)
            
        #     # Distance
        #     error = [weights[i] * jnp.linalg.norm(p_list[i] - p_des_list[i]) for i in range(num_segments)]

        #     # Summarize errors
        #     V_total = jnp.sum(jnp.array(error))
        #     V_total = V_total[None,...]
        #     return V_total         
        def V_2(self, z, z_des) -> jnp.ndarray:
            # CLF: tracking error for both the middle point and the tip (last point)
            
            # Split state into positions (q) and velocities (q_d)
            q, q_d = jnp.split(z, 2)
            
            # Compute forward kinematics for the current configuration.
            p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            
            # Determine indices: use the middle point and the tip.
            num_points = p.shape[0]
            index = [num_points * (i+1)//num_segments-1 for i in range(num_segments)]

            p_list = [p[i, :] for i in index]
            
            # Compute forward kinematics for the desired configuration.
            # p_des = batched_forward_kinematics_fn(self.robot_params, self.q_des, self.s_ps)
            # p_des_list = [p_des[i, :2] for i in index]
            p_des_list = [self.p_des[i, :] for i in range(num_segments)]

            p_list = jnp.array(p_list)
            p_des_list = jnp.array(p_des_list)
            
            # Compute the element-wise absolute differences (note that sqrt((x)^2) equals |x|).
            # This returns a vector for each point.
            error_1 = jnp.sqrt((p_list[0,:2] - p_des_list[0,:2])**2)
            error_2 = jnp.sqrt((p_list[1,:] - p_des_list[1,:])**2)
            error = jnp.concatenate([error_1,error_2])
            error = error.reshape(-1)
            # Option: Return the errors as a single vector by concatenating the two.
            # V_total = jnp.concatenate(error)
            return error   

        # def h_2(self, z) -> jnp.ndarray:
        #     '''
        #     Constrain the orientation difference between segments using a smooth absolute value approximation.
        #     This function assumes at least two segments are present.
        #     '''
        #     # Split the state vector into segments (currently fixed to 2 segments)
        #     q, _ = jnp.split(z, 2)

        #     # Compute forward kinematics for all segments
        #     p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            
        #     num_points = p.shape[0]
        #     # Compute the index of the last point in each segment
        #     index = [num_points * (i + 1) // num_segments - 1 for i in range(num_segments)]
            
        #     # Extract the orientations from these indices (assuming orientation is at index 2)
        #     p_orientation = [p[i, 2] for i in index]
            
        #     # Compute the smooth absolute error using a small epsilon for smoothness
        #     diff = p_orientation[1] - p_orientation[0]
            
        #     error = - jnp.sqrt(diff**2 + 1e-6) + (1 * jnp.pi / 6)
        #     error = error[None,...]
        #     jax.debug.print("error is{}", error)
            
        #     return error    
        
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

            penetration_depth_poly = jnp.concatenate([pairwise_penetration,tip_penetration])-0.01
            penetration_depth_poly = penetration_depth_poly.reshape(-1)
            
            # # Extract the orientations from these indices (assuming orientation is at index 2)
            # num_points = p.shape[0]
            # index = [num_points * (i+1)//num_segments-1 for i in range(num_segments)]

            # p_orientation = [p[i, -1] for i in index]
            
            # # Compute the smooth absolute error using a small epsilon for smoothness
            # diff = p_orientation[1] - p_orientation[0]

            # # jax.debug.print("p_orientation is{}", p_orientation[1])
            # error = (-jnp.sqrt(diff**2))+ (1 * jnp.pi / 3)
            # error = error[None,...]
            # # jax.debug.print("error is{}", error)
        
            force_smooth = jnp.concatenate([penetration_depth_poly], axis=0)
            
            return force_smooth
                    
        def alpha_2(self, h_2):
            return h_2*300 #constant, increase for smaller affected zone
        
        def gamma_2(self, v_2):
            return v_2*300

    config = SoRoConfig()
    clf_cbf = CLFCBF.from_config(config)

    def closed_loop_ode_fn(t: float, y: Array, q_des: Array) -> Array:
        # split the state vector into the configuration and velocity
        q, q_d = jnp.split(y, 2)
        q_d_des = jnp.zeros_like(q_des)
        q_des = jnp.concatenate([q_des, q_d_des])
        
        # evaluate the control policy
        u = clf_cbf.controller(y,q_des)

        # compute the dynamical matrices
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)

        # compute the acceleration
        q_dd = jnp.linalg.inv(B) @ (u - C @ q_d - G - K - D @ q_d)

        # concatenate the velocity and acceleration
        y_d = jnp.concatenate([q_d, q_dd])

        return y_d

    # define the initial condition
    q0_arary = jnp.array([0, 0.0, 0.0])
    multiplier = [q0_arary for m in range(num_segments)]
    q0 = jnp.concatenate(multiplier)

    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the desired configuration
    q_des = config.q_des

    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 5e-4
    # dt = 5e-2
    # sim_dt = 1e-3
    # dt = 5e-4
    # sim_dt = 1e-4


    # define the time steps
    ts = jnp.arange(0.0, 10.0, dt) # original is 7

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, q_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)

    q_des_ts = jnp.tile(q_des, (ts.shape[0], 1))
    print("q_des_ts", q_des_ts.shape)
    # Compute tau_ts using vmap
    tau_ts = vmap(clf_cbf.controller)(sol.ys, q_des_ts) #TODO: understand this
    print("tau_ts", tau_ts.shape)

    flag_list = []     
    for q in q_ts[::20]:
        p = batched_forward_kinematics_fn(robot_params, q, config.s_ps)
        p_ps = p[:, :2]       
        p_orientation = p[:, 2]  
        current_points = p_ps[:-1]
        next_points = p_ps[1:]
        orientations = p_orientation[:-1]

        def segment_robot(current,next,orientation):
                seg_poly = segmented_polygon(current, next, orientation, robot_radius)
                return(seg_poly)
        
        seg_poly = jax.vmap(segment_robot)(current_points, next_points, orientations)
        _,flag = jax.vmap(lambda poly: compute_distance(poly, config.poly_obstacle_pos[0]))(seg_poly)
        
        flag_list.append(int(min(flag)))

    flag_list = onp.array(flag_list)

    # Plot the motion and tau_ts
    fig, axes = plt.subplots(5, 1, figsize=(8,10), sharex=True, num="Regulation example")

    # Plot strains
    # plot the reference strain evolution
    for i in range(num_segments):
        axes[0].plot(ts, q_des_ts[:, i], linewidth=3.0, linestyle=":", label=r"$\kappa_\mathrm{be}^\mathrm{d}$")
        axes[1].plot(ts, q_des_ts[:, i+1], linewidth=3.0, linestyle=":", label=r"$\sigma_\mathrm{sh}^\mathrm{d}$")
        axes[2].plot(ts, q_des_ts[:, i+2], linewidth=3.0, linestyle=":", label=r"$\sigma_\mathrm{ax}^\mathrm{d}$")

    # reset the color cycle
    axes[0].set_prop_cycle(None)
    axes[1].set_prop_cycle(None)
    axes[2].set_prop_cycle(None)
    # plot the actual strain evolution
    for i in range(num_segments):
        axes[0].plot(ts, q_ts[:, i], linewidth=2.0, label=r"$\kappa_\mathrm{be}$")
        axes[1].plot(ts, q_ts[:, i+1], linewidth=2.0, label=r"$\sigma_\mathrm{sh}$")
        axes[2].plot(ts, q_ts[:, i+2], linewidth=2.0, label=r"$\sigma_\mathrm{ax}$")

    # Plot control inputs tau_ts
    for i in range(tau_ts.shape[1]):  # Assuming tau_ts has multiple dimensions (e.g., torques for each actuator)
        axes[3].plot(ts, tau_ts[:, i], label=f"Control Input {i+1}")


    axes[4].plot(ts[::20], flag_list, linewidth=1.0, label=r"$\sigma_\mathrm{ax}^\mathrm{d}$")

    # Set labels and legends
    axes[0].set_ylabel(r"Bending strain $\kappa_\mathrm{be}$")
    axes[1].set_ylabel(r"Shear strain $\sigma_\mathrm{sh}$")
    axes[2].set_ylabel(r"Axial strain $\sigma_\mathrm{ax}$")
    axes[3].set_ylabel(r"Control inputs $\tau$")
    axes[4].set_ylabel(r"Flags")
    axes[4].set_xlabel("Time [s]")

    # Add legends and grid
    for ax in axes:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Collect chi_ps values
    chi_ps_list = []

    # Animate the motion and collect chi_ps
    img_ts = []
    # pos = batched_forward_kinematics_fn(config.robot_params, config.q_des, config.s_ps)
    pos1 = config.p_des[0,:2]
    pos2 = config.p_des[1,:2]
    pos = jnp.stack([pos1,pos2])
    print("pos", pos2)

    batched_segment_robot = jax.vmap(segmented_polygon,in_axes=(0, 0, 0, None))

    for q, flag in zip(q_ts[::20], flag_list):
        img = draw_image(
            batched_forward_kinematics_fn,
            batched_segment_robot,
            half_circle_to_polygon,
            auxiliary_fns,
            robot_params,
            num_segments,
            q,
            p_des=pos,
            poly_points=config.poly_obstacle_pos,
            flag=flag  # Pass the flag to the function
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