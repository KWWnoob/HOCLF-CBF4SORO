import csv
import diffrax as dx
from functools import partial
import jax
from cbfpy.cbfpy.cbfs.clf_cbf import CLFCBF, CLFCBFConfig
from cbfpy.cbfpy.cbfs.cbf import CBF, CBFConfig
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
from src.planar_pcs_rendering_one_segment import draw_image

# define the outputs directory
outputs_dir = Path("outputs") / "planar_pcs_simulation"
outputs_dir.mkdir(parents=True, exist_ok=True)

# load symbolic expressions
num_segments = 2
# filepath to symbolic expressions
sym_exp_filepath = Path(jsrm.__file__).parent / "symbolic_expressions" / f"planar_pcs_ns-{num_segments}.dill"

# set soft robot parameters
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
robot_length = 2.6e-1
robot_radius = 2e-2
robot_params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": robot_length * jnp.ones((num_segments,))/num_segments,
    "r": robot_radius * jnp.ones((num_segments,)),
    "rho": rho,
    "g": jnp.array([0.0, 0.0]), # used to be 0，9.81
    "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
    "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
}
# damping matrix
damping_array = jnp.array([1e0, 1e3, 1e3])
multiplier = [1 * damping_array for m in range(num_segments)]
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

jacobian_fk_fn = jax.jacfwd(batched_forward_kinematics_fn, argnums=1)

# segmenting params
num_points = 20*num_segments
# Compute indices: equivalent to
# [num_points * (i+1)//num_segments - 1 for i in range(num_segments)]
end_p_ps_indices = (jnp.arange(1, num_segments+1) * num_points // num_segments) - 1

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

def compute_contact_force(p_robot_end, p_obs, v_robot_end, v_obs, r_robot, r_obj, c_damp = 50):
    #TODO: check the force formulation
    diff = p_robot_end - p_obs
    distance = jnp.linalg.norm(diff)
    
    eps = 1e-6
    safe_distance = jnp.maximum(distance, eps) # ensuring minimum distance does not go to zero 
    
    penetration = jnp.maximum((r_robot + r_obj) - safe_distance, 0.0)
    # jax.debug.print("Cooridnates = {}{}", p_robot_end,p_obs)
    normal = diff / safe_distance
    
    F_spring = 2000 * penetration  #
    
    # Compute the relative velocity.
    v_rel = v_robot_end - v_obs
    # Project relative velocity along the normal.
    v_normal = jnp.dot(v_rel, normal)
    
    # Damping force opposing the normal component of the relative velocity.
    F_damp = - c_damp * v_normal
    
    F_contact = jnp.where(penetration > 0, (F_spring + F_damp) * normal, jnp.zeros_like(normal))
    
    return F_contact

def closest_point_eval(p_selected, obs):
    # Compute the Euclidean distance between each point and the target obs
    distances = jnp.linalg.norm(p_selected - obs, axis=1)
    # Return the index of the point with the minimum distance
    return distances.argmin()

def soft_robot_with_safety_contact_CBFCLF_example():
    
    # define the ODE function
    class SoRoConfig(CLFCBFConfig):
        '''Config for soft robot'''

        def __init__(self):

            self.robot_params = robot_params

            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

            '''Polygon Obstacle Parameter'''
            char_length = robot_length/2

            self.circle_obs_1_pos = jnp.array([robot_radius*6, char_length*1.0])
            self.circle_obs_1_radius = robot_radius*1
            # self.circle_obs_2_pos = jnp.array([char_length*1.2, char_length*1.2])
            # self.circle_obs_2_radius = robot_radius*1.3
            
            self.obs_center = jnp.stack([self.circle_obs_1_pos])
            self.obs_radius = jnp.stack([self.circle_obs_1_radius])

            '''destination of robot'''
            self.s_ps = jnp.linspace(0, robot_length * num_segments, 20 * num_segments) # segmented

            # self.p_des = jnp.array([-0.04, 0.15234353*1.2,0, 0.3, 0.15234353*1.5, -jnp.pi*1.8*robot_length])
            self.p_des = jnp.array([0.13, 0.15234353*1.5,0, 0.15, 0.15234353*1.5, -jnp.pi*1.8*robot_length])
            # self.p_des = jnp.array([0.15, 0.15234353*2.4, -jnp.pi*1.8*robot_length])

            self.p_des_all = [self.p_des]  
            '''Select the end of each segment'''
            self.indices = end_p_ps_indices

            '''Contact model Parameter'''
            self.contact_spring_constant = 2000 #contact force model
            self.maximum_withhold_force = 10


            super().__init__(
                n=6 * num_segments, # number of states + flag
                m=3, # number of inputs
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=True,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                cbf_relaxation_penalty=1e8,
                clf_relaxation_penalty=10
            )

        def f(self, z) -> Array:
            q, q_d = jnp.split(z, 2)  # Split state z into q (position) and q_d (velocity)
            B, C, G, K, D, alpha = dynamical_matrices_fn(self.robot_params, q, q_d)

            p = batched_forward_kinematics_fn(robot_params, q, self.s_ps)
            v = batched_forward_kinematics_fn(robot_params, q_d, self.s_ps)

            p_selected = p[1:, :2]
            v_selected = v[1:, :2]

            obs_1 = self.circle_obs_1_pos

            closest_point_index_obs_1 = closest_point_eval(p_selected, obs_1)

            F_contact_1 = compute_contact_force(
                p_selected[closest_point_index_obs_1],
                self.circle_obs_1_pos,
                v_selected[closest_point_index_obs_1],
                jnp.zeros_like(self.circle_obs_1_pos),
                robot_radius,
                self.circle_obs_1_radius,
                )

            Jacobian = jacobian_fk_fn(robot_params, q, self.s_ps)
            Jacobian_selected = Jacobian[closest_point_index_obs_1, :2, :]  # This extracts the relevant rows for the 2D tip

            # Map the contact force into joint space (torque).
            tau_contact = Jacobian_selected.T @ (F_contact_1)
            # jax.debug.print("tau contact is {}", tau_contact)
            # Compute the acceleration.
            q_dd = jnp.linalg.inv(B) @ (C @ q_d - G - K - D @ q_d+tau_contact)
            return jnp.concatenate([q_d, q_dd])

        def g(self, z) -> Array:
            q, q_d = jnp.split(z, 2)
            B, _, _, _, _, _ = dynamical_matrices_fn(self.robot_params, q, q_d)
            control_matrix = jnp.linalg.inv(B)
            
            # Create a mapping matrix that duplicates the 3 inputs for both segments
            R = jnp.vstack([jnp.eye(3), jnp.eye(3)])  # For 2 segments
            full_control_matrix = control_matrix @ R
            
            zero_block = jnp.zeros((q.shape[0], full_control_matrix.shape[1]))
            return jnp.concatenate([zero_block, full_control_matrix], axis=0)
        
        def V_2(self, z, z_des) -> tuple[jnp.ndarray, jnp.ndarray]:
            q, q_d = jnp.split(z, 2)
            z_des, _ = jnp.split(z_des, 2)  # now z_des is of shape (num_segments * 3,)
            
            # Reshape to (num_segments, 3)
            z_des = z_des.reshape((num_segments, -1))
            
            p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            p_ps_selected = p[16,:2]
            p_list = p[self.indices, :]
            error_selected = jnp.sqrt((p_ps_selected -self.p_des[0:2])**2)
            error_middle = jnp.sqrt((p_list[0, :3] - z_des[num_segments-1, :3]/2)**2)
            error_tip = jnp.sqrt((p_list[num_segments-1, :3] - z_des[num_segments-1, :3])**2)
            error = jnp.concatenate([error_tip]).reshape(-1)
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

            p_ps_selected = p[16,:2]
            error_selected = - jnp.linalg.norm(p_ps_selected -self.obs_center) + self.obs_radius - self.maximum_withhold_force/self.contact_spring_constant

            p_ps_without_selected = jnp.delete(p_ps, 16, axis=0)
            # -------- Process the polygon obstacles --------
            # inner circle avoid
            tolerance = self.maximum_withhold_force/self.contact_spring_constant

            # force_selected = compute_contact_force()
            penetration_obs_1_first = jax.vmap(compute_distance_circle,in_axes=(0,None))(p_ps_without_selected,self.circle_obs_1_pos) - self.obs_radius[0] - robot_radius
            # penetration_obs_2_first = jax.vmap(compute_distance_circle,in_axes=(0,None))(p_ps_first_segment,self.obs_center[1]) - self.obs_radius[1] - robot_radius+tolerance

            safe_distance = jnp.concatenate([penetration_obs_1_first])
            safe_distance = safe_distance.reshape(-1)
            return safe_distance
    
                    
        def alpha_2(self, h_2):
            return h_2*800 #constant, increase for smaller affected zone
        
        def gamma_2(self, v_2):
            return v_2*800
        
    config = SoRoConfig()
    clf_cbf = CLFCBF.from_config(config)

    def closed_loop_ode_fn(t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        z_des = args
        q, q_d = jnp.split(y, 2)

        # Create the full desired state (assume desired velocity is zero)
        u = clf_cbf.controller(y, z_des)
        jax.debug.print("input is {}", u)
        # Compute the dynamical matrices.
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)
        
        # --- Contact Force Incorporation ---
        # Compute the forward kinematics to get the tip position.
        p = batched_forward_kinematics_fn(robot_params, q, config.s_ps)
        v = batched_forward_kinematics_fn(robot_params, q_d, config.s_ps)
        
        p_selected = p[1:, :2]
        v_selected = v[1:, :2]

        obs_1 = config.circle_obs_1_pos
        # obs_2 = config.circle_obs_2_pos

        closest_point_index_obs_1 = closest_point_eval(p_selected, obs_1)
        # closest_point_index_obs_2 = closest_point_eval(p_selected, obs_2)

        F_contact_1 = compute_contact_force(
                p_selected[closest_point_index_obs_1],
                config.circle_obs_1_pos,
                v_selected[closest_point_index_obs_1],
                jnp.zeros_like(config.circle_obs_1_pos),
                robot_radius,
                config.circle_obs_1_radius,
                )

        # Compute the Jacobian at the tip position.
        Jacobian = jacobian_fk_fn(robot_params, q, config.s_ps)
        Jacobian_selected = Jacobian[closest_point_index_obs_1, :2, :]  # This extracts the relevant rows for the 2D tip

        # Map the contact force into joint space (torque).
        tau_contact = Jacobian_selected.T @ (F_contact_1)
        # jax.debug.print("tau contact is {}", tau_contact)
        # Compute the acceleration.
        q_dd = jnp.linalg.inv(B) @ (jnp.concatenate([u,u]) - C @ q_d - G - K - D @ q_d+tau_contact)
        
        # jax.debug.print("Segment 1 torque: {}", jnp.concatenate([u, u])[:3])
        # jax.debug.print("Segment 2 torque: {}", jnp.concatenate([u, u])[3:6])
        # Return the full state derivative.
        return jnp.concatenate([q_d, q_dd])


    # define the initial condition
    q0_arary_0 = jnp.array([0.0, 0.0, -0.5])
    q0_arary_1 = jnp.array([jnp.pi*7/3, 0.0, -0.5])
    multiplier = [q0_arary_0,q0_arary_0]
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
    tf = 8.0
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
        
        u = clf_cbf.controller(y_current, current_z_des)

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
        output = (t_next, y_next, u)
        return new_carry, output

    @jax.jit
    def run_simulation():
        # Determine number of steps
        num_steps = int((tf - t0) / dt)
        
        # Initial carry state: time, initial state, and index (starting at 0)
        init_carry = (t0, y0, 0, end_p_ps_indices)
        
        # Use jax.lax.scan to perform the simulation steps
        final_carry, (ts, ys, us) = jax.lax.scan(simulation_step, init_carry, None, length=num_steps)
        return ts, ys, us

    # Run the simulation
    ts, ys, us = run_simulation()

    # Convert the control inputs to a NumPy array for plotting (if needed).
    us_plot = onp.array(us)
    # Suppose us_plot has shape (num_steps, m); here we plot each control channel on a separate subplot.
    num_steps, m = us_plot.shape

    plt.figure(figsize=(12, 8))
    for i in range(m):
        plt.subplot(m, 1, i+1)
        plt.plot(ts, us_plot[:, i], label=f'$u_{i}$')
        plt.ylabel(f'$u_{i}$')
        plt.xlabel('Time [s]')
        plt.legend()
    plt.tight_layout()
    plt.show()

    # Optionally, split ys if needed (e.g., into q_ts and q_d_ts)
    q_ts, q_d_ts = jnp.split(ys, 2, axis=1)
    # Collect chi_ps values
    chi_ps_list = []

    # Animate the motion and collect chi_ps
    img_ts = []
    
    pos = [jnp.concatenate([config.p_des[0:2], config.p_des[3:5]])]
    # print(pos)
    # pos = pos.reshape(-1)

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
            circle_center = None,
            inner_radius = None,
            outer_radius = None,
            obs_center = obs_center,
            obs_radius = obs_radius,
            p_des_all = pos,
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