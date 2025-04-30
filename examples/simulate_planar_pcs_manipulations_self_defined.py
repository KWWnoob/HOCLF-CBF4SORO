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

def compute_contact_force(p_robot_end, p_obs, r_robot, r_obj):
    diff = p_robot_end - p_obs
    distance = jnp.linalg.norm(diff)
    
    eps = 1e-6
    safe_distance = jnp.maximum(distance, eps) 
    
    penetration = jnp.maximum((r_robot + r_obj) - safe_distance, 0.0)
    
    normal = diff / safe_distance
    
    F_spring = 2000 * jnp.tanh(penetration)  #
    
    F_contact = jnp.where(penetration > 0, F_spring * normal, jnp.zeros_like(normal))
    
    return F_contact

def soft_robot_with_safety_contact_CBFCLF_example():
    
    robot_params = robot_params
    strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

    '''Define the handling object'''
    object_mass = 1
    object_radius = 0.01
    object_initial_pos = [0.04, 0.08]
    object_initial_velocity = [0, 0]

    '''Destination of robots'''
    p_des_all = jnp.stack([jnp.stack([jnp.array([0.05951909*0.7, 0.15234353*1])])])

    '''Segmentation'''
    s_ps = jnp.linspace(0, robot_length * num_segments, 20 * num_segments)

    '''Contact model Parameter'''
    contact_spring_constant = 2000
    maximum_withhold_force = 20

    '''state matrix'''
    

    '''Barrier Function'''
    term = v_obs + k * (p_obs - p_des)
    barrier = jnp.dot(term, term)

    Lf2b = 
    LgLfbu = 

    # define the ODE function
    class SoRoConfig(CLFCBFConfig):
        '''Config for soft robot'''

        def __init__(self):

            self.robot_params = robot_params

            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

            '''Handling Object, circle'''
            self.object_mass = 1
            self.object_radius = 0.01
            self.object_initial_pos = [0.04, 0.08]
            self.object_initial_velocity = [0, 0]
            
            self.p_des_1_1 = jnp.array([0.05951909*0.7, 0.15234353*1])

            self.p_des_1 = jnp.stack([self.p_des_1_1]) # shape (2)

            self.p_des_all = jnp.stack([self.p_des_1])

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
            
            p_tip = p[-1, :2]

            F_contact = compute_contact_force(p_tip, p_obs, robot_radius, self.object_radius)
        
            # Get Jacobian
            Jacobian = jacobian_fk_fn(self.robot_params, q_robot, self.s_ps)
            Jacobian_tip = Jacobian[-1, :2, :]
            tau_contact = Jacobian_tip.T @ F_contact

            # Modify the robot's drift with the effect of the contact force.
            drift_robot = drift_robot - jnp.linalg.inv(B) @ tau_contact

            # --- Object Dynamics ---
            # The object's dynamics: dp_obs/dt = v_obs, and dv_obs/dt = F_contact / m_obj.
            dp_obs = v_obs
            dv_obs = -F_contact / self.object_mass  # Ensure self.m_obj is defined as the object's mass. TODO: check sign

            # --- Combine Derivatives ---
            # Derivative of positions: robot positions derivative and object positions derivative.
            dq = jnp.concatenate([q_d_robot, v_obs])
            # Derivative of velocities: robot acceleration and object acceleration.
            dq_d = jnp.concatenate([drift_robot, dv_obs])
            
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
    
        def V(self, z: jnp.ndarray, z_des: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:

            q, q_d = jnp.split(z, 2)
            p_obs = q[-2:]  
            v_obs = q_d[-2:]  
            
            q_des, _ = jnp.split(z_des, 2)
            p_des = q_des[-2:]
            
            term = v_obs + k * (p_obs - p_des)
            return jnp.dot(term, term)


        def compute_dot_V(self, z: jnp.ndarray, z_des: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
            grad_V = jax.grad(lambda z_: self.V(z_, z_des, k))(z)
            f_val = self.f(z) 
            return jnp.dot(grad_V, f_val)


        def V_1(self, z: jnp.ndarray, z_des: jnp.ndarray,
                                k: float = 1.0, c0: float = 10.0, c1: float = 5.0) -> tuple[jnp.ndarray, jnp.ndarray]:
        
            V_val = self.V(z, z_des, k)
            
            V1 = self.compute_dot_V(z, z_des, k)
            
            grad_V1 = jax.grad(lambda z_: self.compute_dot_V(z_, z_des, k))(z)
            f_val = self.f(z)
            V2 = jnp.dot(grad_V1, f_val)
            
            psi = V2 + c1 * V1 + c0 * V_val
            psi = jnp.reshape(psi, (1,))  
            
            return psi


        def h_2(self, z) -> jnp.ndarray:
            # Split state into positions (q) and velocities (q_d)
            q, _ = jnp.split(z, 2)
            # Extract robot configuration and object position.
            q_robot = q[:-2]  # Robot's configuration.
            p_obj   = q[-2:]  # Object's (ball's) 2D position.
            
            # Compute the forward kinematics for the robot using the robot's configuration.
            # self.s_ps is the set of arc-length parameters along the robot's body.
            p = batched_forward_kinematics_fn(self.robot_params, q_robot, self.s_ps)
            # Extract the tip position (assumed pose is [p_x, p_y, theta]).
            p_tip = p[-1, :2]
            
            # Define a small safety margin delta.
            delta = 0.05  # Tune this parameter based on your system.
            
            # Compute the contact barrier function.
            h = (robot_radius + self.object_radius + delta) - jnp.linalg.norm(p_tip - p_obj)
            h = h[None,...]
            
            return h
                    
        def alpha_2(self, h_2):
            return h_2*30 #constant, increase for smaller affected zone
        
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
        
        # Compute robot dynamics using only the robot's configuration.
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q_robot, q_d_robot)
        q_dd_robot = jnp.linalg.inv(B) @ (u - C @ q_d_robot - G - K - D @ q_d_robot)
        
        # For the object, if you're not controlling it via u,
        # you can set its acceleration to zero (or use your contact dynamics).
        # Here, we'll assume zero acceleration for the object.
        q_dd_object = jnp.zeros(2)
        
        # Combine the derivatives:
        # For positions: robot velocities and object's velocities remain as in y.
        dq = jnp.concatenate([q_d_robot, q_d[-2:]])
        # For velocities: robot acceleration and object acceleration.
        dq_d = jnp.concatenate([q_dd_robot, q_dd_object])
        
        return jnp.concatenate([dq, dq_d])

    # define the initial condition
    q0_arary_0 = jnp.array([-jnp.pi*3/3, 0.0, -0.5])
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
    tf = 8.0
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
    print(ys)
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
            p_des_all = None,
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