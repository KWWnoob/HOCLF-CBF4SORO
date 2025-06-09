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
from jax import Array, debug, jacfwd, jit, vmap, lax, grad
from jax import numpy as jnp
import jsrm
from jsrm.systems import planar_pcs
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict, Tuple
from jax.scipy.special import logsumexp

from src.img_animation import animate_images_cv2
from src.planar_pcs_rendering_rescue_obscontact import draw_image

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


# ---Polygon matters---
def compute_distance(
    robot_vertices: jnp.ndarray,
    polygon_vertices: jnp.ndarray,
    alpha_axes: float = 9000.0 
) -> tuple[jnp.ndarray, jnp.ndarray]:

    def get_normals(vertices):
        vertices = jnp.asarray(vertices) 
        edges = jnp.roll(vertices, -1, axis=0) - vertices
        normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
        return normals / jnp.linalg.norm(normals, axis=1, keepdims=True)

    Rn = get_normals(robot_vertices)
    Pn = get_normals(polygon_vertices)
    axes = jnp.concatenate([Rn, Pn], axis=0)


    proj_R = robot_vertices @ axes.T
    proj_P = polygon_vertices @ axes.T
    R_min, R_max = jnp.min(proj_R, axis=0), jnp.max(proj_R, axis=0)
    P_min, P_max = jnp.min(proj_P, axis=0), jnp.max(proj_P, axis=0)

    d1 = P_min - R_max
    d2 = R_min - P_max

    pair_gaps = jnp.stack([d1, d2], axis=0)   # shape (2, K)
    flat_gaps = pair_gaps.reshape(-1)         # shape (2*K,)
    h = (1.0 / alpha_axes) * logsumexp(alpha_axes * flat_gaps)

    separation_flag = jnp.where(h > 0, 1, 0)
    
    return h, separation_flag

@jax.jit
def compute_distance(robot_vertices, polygon_vertices, alpha=100):
    rv = robot_vertices
    
    def get_normals(verts):
        edges = jnp.roll(verts, -1, axis=0) - verts
        normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
        return normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
    
    Rn = get_normals(rv)
    Pn = get_normals(polygon_vertices)
    axes = jnp.vstack((Rn, Pn))  # |𝓐| = number of axes
    proj_R = rv @ axes.T
    proj_P = polygon_vertices @ axes.T
    R_min, R_max = proj_R.min(axis=0), proj_R.max(axis=0)
    P_min, P_max = proj_P.min(axis=0), proj_P.max(axis=0)
    
    gaps = jnp.hstack((P_min - R_max, R_min - P_max))  # all signed separations
    h_olsat = (1.0 / alpha) * logsumexp(alpha * gaps)

    # Add error bound term: log(2|𝓐|) / alpha
    error_bound = jnp.log(2 * axes.shape[0]) / alpha 
    h = h_olsat - error_bound
    separation_flag = jnp.where(h > 0, 1, 0)
    
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

def penetration_to_contact_force(penetration_depth: jnp.ndarray, k_contact: float, eps: float = 1e-8) -> jnp.ndarray:
    """
    Smooth contact force using LSE-based spring model (soft-max approximation of -d if d<0).
    
    Args:
        penetration_depth: (n,) array
        k_contact: contact stiffness
        eps: smoothing parameter
    
    Returns:
        (n,) array of contact force magnitudes
    """
    # Use logsumexp for better numerical stability
    scaled = jnp.stack([jnp.zeros_like(penetration_depth), -penetration_depth / eps], axis=0)
    smooth_penetration = eps * logsumexp(scaled, axis=0)
    return k_contact * smooth_penetration

def compute_contact_force(p_robot_end, p_obs, r_robot, r_obj, c_damp = 50):
    #TODO: check the force formulation
    diff = p_robot_end - p_obs
    distance = jnp.linalg.norm(diff)
    
    eps = 1e-6
    safe_distance = jnp.maximum(distance, eps) # ensuring minimum distance does not go to zero 
    
    penetration = jnp.maximum((r_robot + r_obj) - safe_distance, 0.0)
    # jax.debug.print("Cooridnates = {}{}", p_robot_end,p_obs)
    normal = diff / safe_distance
    
    F_spring = 600 * penetration  
    
    F_contact = jnp.where(penetration > 0, (F_spring) * normal, jnp.zeros_like(normal))
    
    return F_contact

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
                                                [0.0, char_length*1.0],
                                                [char_length*0.5, char_length*1.0],
                                                [char_length*0.5, 0.0]])
            self.poly_obstacle_shape_2 = jnp.array([[0.0, char_length*0.5],
                                    [0.0, char_length*1.0],
                                    [char_length*0.5, char_length*1.0],
                                    [char_length*0.5, char_length*0.5]])

            self.poly_obstacle_shape_3 = jnp.array([[0.0, 0.0],
                                                    [0.0, char_length*0.45],
                                                    [char_length*1.2, char_length*0.45],
                                                    [char_length*1.2, 0.0]])
            
            # self.poly_obstacle_pos = self.poly_obstacle_shape/4 + jnp.array([-0.08,0.04])
            self.poly_obstacle_pos_1 = self.poly_obstacle_shape_1 + jnp.array([-0.09,0])
            self.poly_obstacle_pos_2 = self.poly_obstacle_shape_2 + jnp.array([0.07,-0.02])

            self.poly_obstacle_pos_3 = self.poly_obstacle_pos_1[2,:] + self.poly_obstacle_shape_3
            
            self.poly_obstacle_pos_4 = self.poly_obstacle_pos_2[1,:] + self.poly_obstacle_shape_3
            
            self.poly_obstacle_pos_5 = self.poly_obstacle_pos_3[2,:] + self.poly_obstacle_shape_2+ jnp.array([-char_length*0.5, 0])

            self.poly_obstacle_pos_6 = self.poly_obstacle_pos_4[2,:] + self.poly_obstacle_shape_1

            self.poly_obstacle_pos = jnp.stack([self.poly_obstacle_pos_2])

            '''Characteristic of robot'''
            self.s_ps = jnp.linspace(0, robot_length * num_segments, 20 * num_segments) # segmented
            self.q_des_arary_1 = jnp.array([jnp.pi*2,0.5,1.0])
            # self.q_des_array_0 = jnp.array([-jnp.pi*1.8, 0.3, 0.4])/2
            self.q_des_arary_3 = jnp.array([jnp.pi*2, 0.3, 0.4])
            # self.q_des_arary_2 = jnp.array([jnp.pi*2, 0.3, 0.4])/2
            self.q_des = jnp.concatenate([self.q_des_arary_1])# destination

            '''Contact model Parameter'''
            self.contact_spring_constant = 2000 #contact force model
            self.maximum_withhold_force = 20

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
         
        # def V_2(self, z, z_des) -> jnp.ndarray:
        #     # CLF: tracking error for both the middle point and the tip (last point)
            
        #     # Split state into positions (q) and velocities (q_d)
        #     q, q_d = jnp.split(z, 2)
            
        #     # Compute forward kinematics for the current configuration.
        #     p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            
        #     # Determine indices: use the middle point and the tip.
        #     num_points = p.shape[0]
        #     index = [num_points * (i+1)//num_segments-1 for i in range(num_segments)]

        #     p_list = [p[i, :] for i in index]
            
        #     # Compute forward kinematics for the desired configuration.
        #     # p_des = batched_forward_kinematics_fn(self.robot_params, self.q_des, self.s_ps)
        #     # p_des_list = [p_des[i, :2] for i in index]
        #     p_des_list = [self.p_des[i, :] for i in range(num_segments)]

        #     p_list = jnp.array(p_list)
        #     p_des_list = jnp.array(p_des_list)
            
        #     # Compute the element-wise absolute differences (note that sqrt((x)^2) equals |x|).
        #     # This returns a vector for each point.
        #     error_1 = jnp.sqrt((p_list[0,:2] - p_des_list[0,:2])**2)
        #     error_2 = jnp.sqrt((p_list[1,:] - p_des_list[1,:])**2)
        #     error = jnp.concatenate([error_1,error_2])
        #     error = error.reshape(-1)
        #     # Option: Return the errors as a single vector by concatenating the two.
        #     # V_total = jnp.concatenate(error)
        #     return error   
        
        def V_2(self, z, z_des) -> jnp.ndarray:
            """
            Potential energy-based HOCLF function:
            V(q) = U(q) - U(q_des) + (G(q_des) + K q_des)^T (q_des - q)
            """
            q, q_d = jnp.split(z, 2)  
            q_des = self.q_des_arary_1
            q_d_des = jnp.zeros_like(q_des)

            # compute the kinetic energy at the current configuration
            T = kinetic_energy_fn(robot_params, q, q_d)

            # compute the potential energy at the current configuration
            U = potential_energy_fn(robot_params, q)
            # compute the potential energy at the desired configuration
            U_des = potential_energy_fn(robot_params, self.q_des)
            # compute the dynamical matrices at the desired configuration
            B_des, C_des, G_des, K_des, D_des, alpha_des = dynamical_matrices_fn(self.robot_params, self.q_des, jnp.zeros_like(self.q_des))
            # shaped potential energy
            U_shaped = U - U_des + (G_des + K_des).T @ (self.q_des - q)

            # compute the control Lyapunov function
            # V = T + U_shaped
            V = U_shaped
            V = V[None,...] * 3e-2
            return V

        
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


            contact_force = -penetration_to_contact_force(penetration_depth_poly,self.contact_spring_constant)+self.maximum_withhold_force
            
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
        
            # force_smooth = jnp.concatenate([penetration_depth_poly], axis=0)
            
            return penetration_depth_poly
                    
        def alpha_2(self, h_2):
            return h_2*10#constant, increase for smaller affected zone
        
        def gamma_2(self, v_2):
            return v_2*200

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
    dt = 2e-3
    sim_dt = 1e-3
    # dt = 2e-3
    # sim_dt = 1e-3
    # dt = 5e-4
    # sim_dt = 1e-4


    # define the time steps
    ts = jnp.arange(0.0, 6.0, dt) # original is 7

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, q_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, _ = jnp.split(sol.ys, 2, axis=1)

    i_focus = 10

    def get_contact_matrix(full_distance_array, threshold=0.0002):
        """Return a boolean contact matrix where distance < threshold."""
        return full_distance_array < threshold

    # ————————————————————————————————————————
    # 1. Configuration and initialization
    times = onp.array(ts[::20])
    full_distance_list = []
    chi_ps_list = []
    distance_list = []
    force_list = []
    flag_list = []

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

        def distance_to_all_obstacles(seg_poly):
            dists, flags = jax.vmap(lambda obs_poly: compute_distance(seg_poly, obs_poly))(
                config.poly_obstacle_pos
            )
            return dists

        d_segs = jax.vmap(distance_to_all_obstacles)(segs)

        d_focus = float(d_segs[0, 0])
        distance_list.append(d_focus)
        force_list.append(config.maximum_withhold_force - penetration_to_contact_force(d_segs[10,1],config.contact_spring_constant))
        # Then still add to full_distance_list
        full_distance_list.append(d_segs)
    force_array = onp.array(force_list)
    nonzero_indices = onp.argwhere(force_array != 0)
    print(nonzero_indices)
    # ————————————————————————————————————————
    # 3. Stack all distances into a full distance array
    full_distance_array = jnp.stack(full_distance_list)  # (timestamp, segment, obstacle)

    # ————————————————————————————————————————
    # 4. Detect collisions (contacts)
    contacts_by_time = get_contact_matrix(full_distance_array, threshold=0.002)

    # ————————————————————————————————————————
    # 5. Extract valid contact points
    T, N_seg, N_obs = contacts_by_time.shape
    contact_points_list = []

    for t in range(T):
        for seg_id in range(N_seg):
            for obs_id in range(N_obs):
                if contacts_by_time[t, seg_id, obs_id]:
                    seg_poly_center = chi_ps_list[t][seg_id]
                    obs_poly = config.poly_obstacle_pos[obs_id]

                    contact_pt = connect_project(seg_poly_center, jnp.mean(obs_poly, axis=0), obs_poly)

                    if not jnp.isnan(contact_pt).any():
                        contact_points_list.append((
                            t, seg_id, obs_id,
                            float(contact_pt[0]), float(contact_pt[1])
                        ))

    contact_points_array = onp.array(contact_points_list)  # Now works perfectly
    # print(f"contact_points_array shape = {contact_points_array.shape}")
    # print(contact_points_array)

    plt.figure(figsize=(6,4))
    plt.plot(times, force_list, linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel(f"Signed distance of segment {i_focus}")
    plt.title("Force to Obstacle Over Time (Focused)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Animate the motion and collect chi_ps
    img_ts = []
    # pos = batched_forward_kinematics_fn(config.robot_params, config.q_des, config.s_ps)
    pos1 = config.q_des_arary_1
    pos1 = batched_forward_kinematics_fn(robot_params, pos1, config.s_ps)
    print(pos1.shape)
    pos1 = pos1[-1,:2]
    pos = jnp.stack([pos1])

    batched_segment_robot = jax.vmap(segmented_polygon,in_axes=(0, 0, 0, None))

    for t_idx, q in enumerate(q_ts[::20]):
        # 1. Find all contact points at this timestamp
        # frame_contact_points = contact_points_array[contact_points_array[:, 0] == t_idx, 3:5]

        # 2. Draw
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
            flag=None,  # explicitly disable flag indicator
            contact_points=None  # pass collision points
        )
        img_ts.append(img)

    # Animate the images
    img_ts = onp.stack(img_ts, axis=0)
    animate_images_cv2(
        onp.array(ts[::20]), img_ts, outputs_dir / "planar_pcs_safe_closed_loop_simulation.mp4"
    )

if __name__ == "__main__":
    soft_robot_with_safety_contact_CBFCLF_example()