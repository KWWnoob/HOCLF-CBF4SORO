'''
SAT MULTI AXIS VERSION
'''

import csv
import diffrax as dx
from functools import partial
import jax
from cbfpy.cbfs.clf_cbf import CLFCBF, CLFCBFConfig
import inspect

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, debug, jacfwd, jit, vmap, lax, grad
from jax import numpy as jnp

import jsrm
# from jsrm.systems import planar_pcs
from jsrm.systems import tendon_actuated_planar_pcs as planar_pcs

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
    "E": 3.5e4 * jnp.ones((num_segments,)) ,  # Elastic modulus [Pa]
    "G": 2e4 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
    "d": 2e-2 * jnp.array([[1.0, -1.0]]).repeat(num_segments, axis=0),  # distance of tendons from the central axis [m]
}
# damping matrix
damping_array = jnp.array([1e0, 1e3, 1e3]) * 50
multiplier = [1.5**m * damping_array for m in range(num_segments)]
robot_params["D"] = 5e-5 * jnp.diag(jnp.concatenate(multiplier)) * robot_length #depend on the num of segments

# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)
# actuation selector for the segments
segment_actuation_selector = jnp.ones((num_segments,), dtype=bool)

# call the factory function for the planar PCS
strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
        planar_pcs.factory(num_segments, sym_exp_filepath, strain_selector, segment_actuation_selector=segment_actuation_selector)
    )

actuation_mapping_fn = auxiliary_fns["actuation_mapping_fn"]
xi_eq = jnp.array([0.0, 0.0, 1.0])[None].repeat(num_segments, axis=0).flatten()

# jit the functions
dynamical_matrices_fn = jax.jit(partial(dynamical_matrices_fn))
batched_forward_kinematics_fn = vmap(
    forward_kinematics_fn, in_axes=(None, None, 0)
)

jacobian_fk_fn = jax.jacfwd(batched_forward_kinematics_fn, argnums=1)

# segmenting params
num_points = 20*num_segments
# Compute indices: equivalent to
# [num_points * (i+1)//num_segments - 1 for i in range(num_segments)]
end_p_ps_indices = (jnp.arange(1, num_segments+1) * num_points // num_segments) - 1

@jax.jit
def compute_distance(robot_vertices, polygon_vertices, alpha=1000):
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
            # self.poly_obstacle_pos_2 = self.poly_obstacle_shape_2 + jnp.array([0.07,-0.02])
            self.poly_obstacle_pos_2 = self.poly_obstacle_shape_2 + jnp.array([-0.12,0.08])

            self.poly_obstacle_pos_3 = self.poly_obstacle_pos_1[2,:] + self.poly_obstacle_shape_3
            
            self.poly_obstacle_pos_4 = self.poly_obstacle_pos_2[1,:] + self.poly_obstacle_shape_3
            
            self.poly_obstacle_pos_5 = self.poly_obstacle_pos_3[2,:] + self.poly_obstacle_shape_2+ jnp.array([-char_length*0.5, 0])

            self.poly_obstacle_pos_6 = self.poly_obstacle_pos_4[2,:] + self.poly_obstacle_shape_1

            self.poly_obstacle_pos = jnp.stack([self.poly_obstacle_pos_2])

            '''Characteristic of robot'''
            self.s_ps = jnp.linspace(0, robot_length * num_segments, 20 * num_segments) # segmented
            # self.q_des_arary_1 = jnp.array([-jnp.pi*2,0.5,1.0])
            # # self.q_des_array_0 = jnp.array([-jnp.pi*1.8, 0.3, 0.4])/2
            # self.q_des_arary_3 = jnp.array([jnp.pi*2, 0.3, 0.4])
            # # self.q_des_arary_2 = jnp.array([jnp.pi*2, 0.3, 0.4])/2
            # self.q_des = jnp.concatenate([self.q_des_arary_1])# destination

            self.p_des_array_1 = jnp.array([-0.1, 0.12, +jnp.pi/3 ])
            self.p_des_array_2 = jnp.array([-0.1*1.3, 0.12*2.3, +jnp.pi/2])

            self.p_des = jnp.stack([self.p_des_array_1, self.p_des_array_2]) #shape 2,2
            '''Select the end of each segment'''
            self.indices = end_p_ps_indices

            '''Contact model Parameter'''
            self.contact_spring_constant = 2000 #contact force model
            self.maximum_withhold_force = 10

            '''Real Input'''
            # self.actuation_mask = jnp.array([1, 0, 1])
            # self.active_idx = jnp.where(self.actuation_mask == 1)[0]  

            super().__init__(
                n=6 * num_segments, # number of states
                m=2 * num_segments, # _number of inputs
                u_min = jnp.zeros(2*num_segments), 
                u_max = 5 * jnp.ones(2*num_segments),
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=False,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                cbf_relaxation_penalty=1e6,
                clf_relaxation_penalty=10
            )

        def f(self, z) -> Array:
            q, q_d = jnp.split(z, 2)  # Split state z into q (position) and q_d (velocity)
            B, C, G, K, D, alpha = dynamical_matrices_fn(self.robot_params, q, q_d)
            # Drift term (f(x))
            drift = (
                -jnp.linalg.inv(B) @ (C @ q_d + D @ q_d + G + K)
            )

            result = jnp.concatenate([q_d, drift])
            # result = result.at[1].set(0)
            # result = result.at[4].set(0)
            
            return result

        def g(self, z) -> Array:
            """
            Control influence matrix g(z) for dz/dt = f(z) + g(z)·u
            """
            q, q_d = jnp.split(z, 2)
            B, _, _, _, _, _ = dynamical_matrices_fn(self.robot_params, q, q_d)

            A = actuation_mapping_fn(
                forward_kinematics_fn,
                auxiliary_fns["jacobian_fn"],
                self.robot_params,
                strain_basis,
                xi_eq,  # xi_eq
                q           # for dynamic compensation
            )
            # A = A.at[:, [1, 4]].set(0)
            g_mat = jnp.linalg.inv(B) @ A
            zero_block = jnp.zeros((q.shape[0], g_mat.shape[1]))

            return jnp.concatenate([zero_block, g_mat], axis=0)  # shape (2n, m)
        
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
            error_middle = jnp.concatenate([jnp.sqrt((p_list[i,:]- z_des[i,:])**2) for i in range(num_segments-1)])
            # # For the "tip" point, use all coordinates and scale the error by 10.
            error_tip = jnp.sqrt((p_list[num_segments-1, :] - z_des[num_segments-1,:])**2)

            # Concatenate the errors into one vector.
            error = jnp.concatenate([error_middle, error_tip]).reshape(-1)
            
            return error
        
        def h_2(self, z) -> jnp.ndarray:
            """
            # Computes the safety force (barrier function output) for the robot given its state 'z',
            # considering both polygonal and circular obstacles.
            
            # Args:
            #     z: A JAX array representing the state, typically a concatenation of positions q and velocities q_d.
            
            # Returns:
            #     A JAX array representing the combined smooth force.
            # """
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

            return penetration_depth_poly
            # return jnp.array([10.0])
        
        def alpha_2(self, h_2):
            return h_2*100  #constant, increase for smaller affected zone
        
        def gamma_2(self, v_2):
            return v_2*100

    config = SoRoConfig()
    clf_cbf = CLFCBF.from_config(config)

    def closed_loop_ode_fn(t: float, y: Array, z_des: Array) -> Array:
        # split the state vector into the configuration and velocity
        q, q_d = jnp.split(y, 2)
        
        # evaluate the control policy
        u_reduced = clf_cbf.controller(y,z_des)
        # u_full = jnp.zeros_like(q)  # or jnp.zeros((3,))
        # u_full = u_full.at[config.active_idx].set(u_reduced)

        # compute the dynamical matrices
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)
        
        A = actuation_mapping_fn(
                forward_kinematics_fn,
                auxiliary_fns["jacobian_fn"],
                config.robot_params,
                strain_basis,
                xi_eq,  # xi_eq
                q            # for dynamic compensation
            )
        
        # compute the acceleration
        q_dd = jnp.linalg.inv(B) @ (A@u_reduced - C @ q_d - G - K - D @ q_d)

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
    z_des = config.p_des.reshape(-1)
    z_dot_des = jnp.zeros_like(z_des)
    z_des = jnp.concatenate([z_des, z_dot_des])

    # z_des = jnp.stack([
    #     jnp.concatenate([p.flatten(), jnp.zeros(3 * num_segments)])
    #     for p in config.p_des
    #     ])  # shape (num_waypoints, num_segments * 3* 2)


    # define the sampling and simulation time step
    # dt = 2e-4
    # sim_dt = 5e-5
    dt = 2e-3
    sim_dt = 1e-4
    # dt = 5e-4
    # sim_dt = 1e-4


    # define the time steps
    ts = jnp.arange(0.0, 6.0, dt) # original is 7

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, z_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, _ = jnp.split(sol.ys, 2, axis=1)
    
    # u_list = []
    # for y in sol.ys:
    #     u = clf_cbf.controller(y, z_des)
    #     u_list.append(onp.array(u))
    # u_array = onp.stack(u_list)

    # # Plot control inputs over time
    # plt.figure(figsize=(7, 4))
    # for i in range(u_array.shape[1]):
    #     plt.plot(ts, u_array[:, i], label=f'$u_{i}$')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Control Input $u$")
    # plt.title("Control Inputs Over Time")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # === Plot V2 over time (scalar V_2) ===
    # V2_fn = jax.jit(lambda y: config.V_2(y, z_des))          # y: (n,)
    # V2_vals = jax.vmap(V2_fn)(sol.ys)                        # (T,)

    # plt.figure(figsize=(7,4))
    # plt.plot(ts, onp.array(V2_vals))
    # plt.xlabel("Time [s]")
    # plt.ylabel(r"$V_2(z(t))$")
    # plt.title("CLF $V_2$ along trajectory")
    # plt.grid(True, ls="--", alpha=0.5)
    # plt.tight_layout()
    # plt.show()
    
    # Animate the motion and collect chi_ps
    img_ts = []
    # pos = batched_forward_kinematics_fn(config.robot_params, config.q_des, config.s_ps)
    pos = jnp.stack([config.p_des])

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
            p_des=config.p_des_array_2[:2],
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