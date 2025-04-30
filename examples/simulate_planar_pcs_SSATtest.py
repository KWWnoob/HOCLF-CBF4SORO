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
from src.planar_pcs_rendering_rescue import draw_image

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
def smooth_abs(x, alpha):
    return x * jnp.tanh(alpha * x)

def compute_distance_smooth_xtanh(robot, poly, alpha_axes=9000., alpha_sabs=1000.):
    """
    Distance using smooth absolute value replacement.
    """
    def get_normals(v):
        e = jnp.roll(v, -1, axis=0) - v
        n = jnp.stack([-e[:,1], e[:,0]], axis=1)
        return n / jnp.linalg.norm(n, axis=1, keepdims=True)
    
    # Centroids
    p_robot = robot.mean(axis=0)
    p_poly  = poly.mean(axis=0)
    delta_p = p_poly - p_robot

    # Normals
    axes = jnp.concatenate([get_normals(robot), get_normals(poly)], axis=0)

    def half_extent(v, axis):
        projections = (v - v.mean(axis=0)) @ axis
        return smooth_abs(projections, alpha_sabs).max()

    axis_gaps = []
    for ni in axes:
        proj_dist = smooth_abs(ni @ delta_p, alpha_sabs)
        rA = half_extent(robot, ni)
        rB = half_extent(poly, ni)
        gap_i = proj_dist - (rA + rB)
        axis_gaps.append(gap_i)

    axis_gaps = jnp.stack(axis_gaps)
    h = (1 / alpha_axes) * logsumexp(alpha_axes * axis_gaps)
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
            self.poly_obstacle_pos_2 = self.poly_obstacle_shape_2 + jnp.array([0.07,0])

            self.poly_obstacle_pos_3 = self.poly_obstacle_pos_1[2,:] + self.poly_obstacle_shape_3
            
            self.poly_obstacle_pos_4 = self.poly_obstacle_pos_2[1,:] + self.poly_obstacle_shape_3
            
            self.poly_obstacle_pos_5 = self.poly_obstacle_pos_3[2,:] + self.poly_obstacle_shape_2+ jnp.array([-char_length*0.5, 0])

            self.poly_obstacle_pos_6 = self.poly_obstacle_pos_4[2,:] + self.poly_obstacle_shape_1

            self.poly_obstacle_pos = jnp.stack([self.poly_obstacle_pos_2])

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

            pairwise_penetration,_ = jax.vmap( lambda poly: jax.vmap(lambda obs: compute_distance_smooth_xtanh(poly, obs))(self.poly_obstacle_pos)
                                            )(robot_poly)
            
            end_start = p_ps[-2]
            end_end = p_ps[-1]
            d = (end_end - end_start)/jnp.linalg.norm(end_end - end_start)
            angle = jnp.arctan2(d[1], d[0])

            robot_tip = half_circle_to_polygon(p_ps[-1],angle, robot_radius)
            tip_penetration,_ = jax.vmap(compute_distance_smooth_xtanh, in_axes=(0, None))(self.poly_obstacle_pos, robot_tip)
            tip_penetration = tip_penetration[None,...]

            penetration_depth_poly = jnp.concatenate([pairwise_penetration,tip_penetration])
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
    ts = jnp.arange(0.0, 6.0, dt) # original is 7

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, q_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)

    q_des_ts = jnp.tile(q_des, (ts.shape[0], 1))
    # Compute tau_ts using vmap
    tau_ts = vmap(clf_cbf.controller)(sol.ys, q_des_ts) #TODO: understand this

    i_focus = 10 

    times = onp.array(ts[::20])
    distance_list = []
    flag_list     = []

    for q in q_ts[::20]:
        # 1) forward kinematics
        p    = batched_forward_kinematics_fn(robot_params, q, config.s_ps)
        p_ps = p[:, :2]
        cur_pts = p_ps[:-1]
        nxt_pts = p_ps[1:]
        ors     = jnp.arctan2((nxt_pts-cur_pts)[:,1],
                            (nxt_pts-cur_pts)[:,0])
        segs = jax.vmap(segmented_polygon, in_axes=(0,0,0,None))(
                cur_pts, nxt_pts, ors, robot_radius)

        d_segs, flags = jax.vmap(lambda poly:
                                compute_distance_smooth_xtanh(poly,
                                                config.poly_obstacle_pos[0]))(segs)

        d_focus = float(d_segs[i_focus])
        flag_focus = int(flags[i_focus])
        print(d_segs.shape)
        distance_list.append(d_focus)
        flag_list.append(flag_focus)

    distance_list = onp.array(distance_list)
    flag_list     = onp.array(flag_list)

    plt.figure(figsize=(6,4))
    plt.plot(times, distance_list, linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel(f"Signed distance of segment {i_focus}")
    plt.title("Distance to Obstacle Over Time (Focused)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    @jit
    def dist_focus_for_q(q):
        p_ps = batched_forward_kinematics_fn(robot_params, q, config.s_ps)[:, :2]
        cur, nxt = p_ps[:-1], p_ps[1:]
        ors      = jnp.arctan2((nxt-cur)[:,1], (nxt-cur)[:,0])
        segs     = jax.vmap(segmented_polygon, in_axes=(0,0,0,None))(
                    cur, nxt, ors, robot_radius)
        d_segs, _ = jax.vmap(lambda poly:
                            compute_distance_smooth_xtanh(poly,
                                            config.poly_obstacle_pos[0]))(segs)
        return d_segs[i_focus]

    grad_dist_focus = jit(grad(dist_focus_for_q))

    qs    = q_ts[::20]
    grads = onp.stack([onp.array(grad_dist_focus(q)) for q in qs])  # shape=(len(qs), dim_q)

    grad_norm = onp.linalg.norm(grads, axis=1)
    plt.figure(figsize=(6,3))
    plt.plot(times, grad_norm, marker='.', linestyle='-')
    plt.xlabel("Time [s]")
    plt.ylabel(f"‖∇₍q₎ distance_seg{i_focus}‖")
    plt.title("Gradient Norm Over Time (Focused Segment)")
    plt.grid(True)
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