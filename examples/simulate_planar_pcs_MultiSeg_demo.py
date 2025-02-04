import csv
import diffrax as dx
from functools import partial
import jax
from cbfpy import CBF, CBFConfig
from cbfpy.cbfs.clf_cbf import CLFCBF, CLFCBFConfig

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, debug, jacfwd, jit, vmap
from jax import numpy as jnp
import jsrm
from jsrm.systems import planar_pcs
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict, Tuple

from src.img_animation import animate_images_cv2
from src.planar_pcs_rendering_multiSeg import draw_image

# define the outputs directory
outputs_dir = Path("outputs") / "planar_pcs_simulation"
outputs_dir.mkdir(parents=True, exist_ok=True)

# load symbolic expressions
num_segments = 1
# filepath to symbolic expressions
sym_exp_filepath = Path(jsrm.__file__).parent / "symbolic_expressions" / f"planar_pcs_ns-{num_segments}.dill"

# set soft robot parameters
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
robot_length = 1e-1
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

def soft_robot_with_safety_contact_CBF_example():

    # define the ODE function
    class SoRoConfig(CBFConfig):
        '''Config for soft robot'''

        def __init__(self):

            self.robot_params = robot_params

            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

            self.obstacle_pos = jnp.array([-5e-2, 0.07])
            self.obstacle_radius = 1e-2
            self.s_ps = jnp.linspace(0, robot_length, 20)

            self.q_des_arary = jnp.array([jnp.pi*3, 0.0, 0.2])
            multiplier = [1.1**m * self.q_des_arary for m in range(num_segments)]
            self.q_des = jnp.concatenate(multiplier)# destination

            self.safety_margin_norm_factor = 1

            super().__init__(
                n=2*3*num_segments, # number of states
                m=3*num_segments, # number of inputs
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=False,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                # cbf_relaxation_penalty=5e-6,
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

        def h_2(self, z):
            # regulating "pose space"

            # Split input into positions (q) and velocities (q_d)
            q, q_d = jnp.split(z, 2)

            # Compute positions of all robotic segments
            chi_ps = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            # Ignore orientation, keep x-y positions
            p_ps = chi_ps[:, :2]

            # Remove the first point (base) from the list of points as its not controllable
            p_ps = p_ps[1:]
            s_ps = self.s_ps[1:]

            # Compute the distance to the obstacle center
            d2o_ps = jnp.linalg.norm((p_ps - self.obstacle_pos), ord=2, axis=1)
            # Compute safety margin for each segment
            safety_margins = d2o_ps - self.obstacle_radius - robot_radius # minimal distance
            
            # debug.print("Safety Margins: {safety_margins}", safety_margins=safety_margins)

            # normalize the safety margin
            # normalized_safety_margins = safety_margins / robot_length * safety_margins.shape[0] ** 4 * self.safety_margin_norm_factor
            normalized_safety_margins = safety_margins * s_ps / robot_length * safety_margins.shape[0] ** 2 * self.safety_margin_norm_factor
            return normalized_safety_margins

            # the "min-approach" seems to be unstable
            # minimal_safety_margin = jnp.min(safety_margins / robot_length)[None] * 2
            # return minimal_safety_margin
        
        def alpha_2(self, h_2):
            return h_2*10 #constant

    def control_policy_fn(t: float, y: Array, q_des: Array) -> Array:
        """
        Control policy that regulates the configuration to a desired configuration q_des.
        Args:
            t: time
            y: state vector
            q_des: desired configuration
        Returns:
            tau: generalized torque
        """
        # compute the dynamical matrices at the desired configuration
        B_des, C_des, G_des, K_des, D_des, alpha_des = dynamical_matrices_fn(robot_params, q_des, jnp.zeros_like(q_des))

        # the torque is equal to the potential forces at the desired configuration
        tau = G_des + K_des

        return tau

    config = SoRoConfig()
    cbf = CBF.from_config(config)

    def safety_filtered_control_policy_fn(t: float, y: Array, q_des: Array):
        # evaluate the control policy
        tau = control_policy_fn(t, y, q_des)

        # evaluate the safe control polic
        tau_filtered = cbf.safety_filter(y, tau)

        # tau_filtered = tau

        return tau_filtered

    def closed_loop_ode_fn(t: float, y: Array, q_des: Array) -> Array:
        # split the state vector into the configuration and velocity
        q, q_d = jnp.split(y, 2)

        tau_filtered = safety_filtered_control_policy_fn(t, y, q_des)

        # compute the dynamical matrices
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)

        # compute the acceleration
        q_dd = jnp.linalg.inv(B) @ (tau_filtered - C @ q_d - G - K - D @ q_d)

        # concatenate the velocity and acceleration
        y_d = jnp.concatenate([q_d, q_dd])

        return y_d

    # define the initial condition
    q0_arary = jnp.array([jnp.pi, 0.01, 0.05])
    multiplier = [q0_arary for m in range(num_segments)]
    q0 = jnp.concatenate(multiplier)

    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 5e-5

    # define the time steps
    ts = jnp.arange(0.0, 7.0, dt)

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, config.q_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)

    q_des_ts = jnp.tile(config.q_des, (ts.shape[0], 1))
    # Compute tau_ts using vmap
    tau_ts = vmap(safety_filtered_control_policy_fn)(ts, sol.ys, q_des_ts)
    print("tau_ts", tau_ts.shape)

    # Plot the motion and tau_ts
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, num="Regulation example")

    # Plot strains
    # plot the reference strain evolution
    axes[0].plot(ts, q_des_ts[:, 0], linewidth=3.0, linestyle=":", label=r"$\kappa_\mathrm{be}^\mathrm{d}$")
    axes[1].plot(ts, q_des_ts[:, 1], linewidth=3.0, linestyle=":", label=r"$\sigma_\mathrm{sh}^\mathrm{d}$")
    axes[2].plot(ts, q_des_ts[:, 2], linewidth=3.0, linestyle=":", label=r"$\sigma_\mathrm{ax}^\mathrm{d}$")
    # reset the color cycle
    axes[0].set_prop_cycle(None)
    axes[1].set_prop_cycle(None)
    axes[2].set_prop_cycle(None)
    # plot the actual strain evolution
    axes[0].plot(ts, q_ts[:, 0], linewidth=2.0, label=r"$\kappa_\mathrm{be}$")
    axes[1].plot(ts, q_ts[:, 1], linewidth=2.0, label=r"$\sigma_\mathrm{sh}$")
    axes[2].plot(ts, q_ts[:, 2], linewidth=2.0, label=r"$\sigma_\mathrm{ax}$")

    # Plot control inputs tau_ts
    for i in range(tau_ts.shape[1]):  # Assuming tau_ts has multiple dimensions (e.g., torques for each actuator)
        axes[3].plot(ts, tau_ts[:, i], label=f"Control Input {i+1}")

    # Set labels and legends
    axes[0].set_ylabel(r"Bending strain $\kappa_\mathrm{be}$")
    axes[1].set_ylabel(r"Shear strain $\sigma_\mathrm{sh}$")
    axes[2].set_ylabel(r"Axial strain $\sigma_\mathrm{ax}$")
    axes[3].set_ylabel(r"Control inputs $\tau$")
    axes[3].set_xlabel("Time [s]")

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
    pos = batched_forward_kinematics_fn(config.robot_params, config.q_des, config.s_ps)
    pos = pos[-1,:2]
    print("pos", pos)
    for q in q_ts[::20]:
        img = draw_image(batched_forward_kinematics_fn, auxiliary_fns, robot_params, num_segments, q, x_obs=config.obstacle_pos, R_obs=config.obstacle_radius, p_des = pos)
        img_ts.append(img)

        chi_ps = batched_forward_kinematics_fn(robot_params, q, config.s_ps)
        # Store chi_ps as a list for each timestep
        chi_ps_list.append(onp.array(chi_ps))  # Convert to numpy array for easier handling

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
def soft_robot_with_safety_contact_CBFCLF_example():
    # define the ODE function
    class SoRoConfig(CLFCBFConfig):
        '''Config for soft robot'''

        def __init__(self):

            self.robot_params = robot_params

            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

            self.obstacle_pos = jnp.array([-9e-2, 0.1]) # radius postion
            self.obstacle_radius = 1e-2 # radius obstacle
            self.s_ps = jnp.linspace(0, robot_length * num_segments, 7 * num_segments) # segmented

            self.q_des_arary = jnp.array([jnp.pi*2, 0.1, 0.1])
            multiplier = [1.1**m * self.q_des_arary for m in range(num_segments)]
            self.q_des = jnp.concatenate(multiplier)# destination

            super().__init__(
                n=6 * num_segments, # number of states
                m=3 * num_segments, # number of inputs
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=False,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                cbf_relaxation_penalty=1e5,
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
        

        def V_2(self, z) -> jnp.ndarray:
            # CLF: tracking error for both the middle point and the tip (last point)
            
            # Split state into positions (q) and velocities (q_d)
            q, q_d = jnp.split(z, 2)
            
            # Compute forward kinematics for the current configuration.
            # Assume p has shape (num_points, 2)
            p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            
            # Determine indices: use the middle point and the tip.
            num_points = p.shape[0]
            index = [num_points * (i+1)//num_segments for i in range(num_segments)]

            p_list = [p[i, :2] for i in index]
            
            # Compute forward kinematics for the desired configuration.
            p_des = batched_forward_kinematics_fn(self.robot_params, self.q_des, self.s_ps)
            p_des_list = [p_des[i, :2] for i in index]
            
            # Compute the element-wise absolute differences (note that sqrt((x)^2) equals |x|).
            # This returns a vector for each point.
            error = [jnp.sqrt((p_list[i]-p_des_list[i])**2) for i in range(num_segments)]
            
            # Option: Return the errors as a single vector by concatenating the two.
            V_total = jnp.concatenate(error)
            
            # V_total now is a 1D array containing the element-wise errors
            # for the middle point followed by those for the tip.
            return V_total

        # def V_2(self, z) -> jnp.ndarray:
        #     # CLF: tracking error for both the middle point and the tip (last point)
            
        #     # Split state into positions (q) and velocities (q_d)
        #     q, q_d = jnp.split(z, 2)
            
        #     # Compute forward kinematics for the current configuration.
        #     # Assume p has shape (num_points, 2)
        #     p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            
        #     # Determine indices: use the middle point and the tip.
        #     num_points = p.shape[0]
        #     index = [num_points * (i+1)/num_segments for i in range(num_segments)]
        #     mid_index = num_points // num  # middle index (adjust if needed)
        #     tip_index = -1              # tip (last point)
            
        #     # Extract the positions for the current configuration.
        #     p_mid = p[mid_index, :2]  # middle point
        #     p_tip = p[tip_index, :2]  # tip of the second segment
            
        #     # Compute forward kinematics for the desired configuration.
        #     p_des = batched_forward_kinematics_fn(self.robot_params, self.q_des, self.s_ps)
        #     p_des_mid = p_des[mid_index, :2]  # desired middle point
        #     p_des_tip = p_des[tip_index, :2]    # desired tip
            
        #     # Compute the element-wise absolute differences (note that sqrt((x)^2) equals |x|).
        #     # This returns a vector for each point.
        #     error_mid = jnp.sqrt((p_mid - p_des_mid) ** 2)
        #     error_tip = jnp.sqrt((p_tip - p_des_tip) ** 2)
            
        #     # Option: Return the errors as a single vector by concatenating the two.
        #     V_total = jnp.concatenate([error_mid, error_tip])
            
        #     # V_total now is a 1D array containing the element-wise errors
        #     # for the middle point followed by those for the tip.
        #     return V_total
        
        # def V_2(self, z) -> jnp.ndarray:
        # # CLF: distance from tip to destination
        #     q, q_d = jnp.split(z, 2)    

        #     p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
        #     p_1 = p[9, :2]
        #     p_2 = p[-1, :2]
        #     p_concat = jnp.concatenate([p_1, p_2])

        #     p_des = batched_forward_kinematics_fn(self.robot_params, self.q_des, self.s_ps)
        #     p_des_1 = p_des[9, :2]
        #     p_des_2 = p_des[-1, :2]
        #     p_des_concat = jnp.concatenate([p_des_1, p_des_2])

        #     Lyapnov_function = ((p_concat - p_des_concat) ** 2)
        #     # debug.print("{}",squared_differences)

        #     return Lyapnov_function * 1
        
        # def V_2(self, z) -> jnp.ndarray:
        #     # Split state into positions and velocities
        #     q, q_d = jnp.split(z, 2)    

        #     # Compute forward kinematics for the current configuration
        #     p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)  # shape: (num_points, 2)
            
        #     # Choose indices for the endpoints
        #     # For example, if self.s_ps has 20 points, you might choose:
        #     i_mid = 9  # index for the endpoint of the first segment (adjust as needed)
        #     i_tip = -1  # index for the tip (end of the second segment)

        #     # Extract the endpoints
        #     p_1 = p[i_mid, :2]  # first segment's endpoint
        #     p_2 = p[i_tip, :2]  # second segment's endpoint (tip)

        #     # Compute forward kinematics for the desired configuration
        #     p_des = batched_forward_kinematics_fn(self.robot_params, self.q_des, self.s_ps)
        #     p_des_1 = p_des[i_mid, :2]  # desired position for the first segment's endpoint
        #     p_des_2 = p_des[i_tip, :2]  # desired position for the tip

        #     # Define weights (you can adjust these to prioritize one endpoint over the other)
        #     w1 = 1.0
        #     w2 = 1.0

        #     # Compute the squared errors at both endpoints
        #     error1 = jnp.linalg.norm(p_1 - p_des_1) ** 2
        #     error2 = jnp.linalg.norm(p_2 - p_des_2) ** 2

        #     # Compute the overall Lyapunov function
        #     V_total = 0.5 * (w1 * error1 + w2 * error2)

        #     V_total = V_total[None, ...]
        #     return V_total*0.5


        def h_2(self, z):
            # regulating "pose space"

            # Split input into positions (q) and velocities (q_d)
            q, q_d = jnp.split(z, 2)

            # Compute positions of all robotic segments
            chi_ps = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            # Ignore orientation, keep x-y positions
            p_ps = chi_ps[:, :2]

            # Remove the first point (base) from the list of points as its not controllable
            p_ps = p_ps[1:]
            s_ps = self.s_ps[1:]

            # Compute the distance to the obstacle center
            d2o_ps = jnp.linalg.norm((p_ps - self.obstacle_pos), ord=2, axis=1)
            # Compute safety margin for each segment
            safety_margins = d2o_ps - self.obstacle_radius - robot_radius # minimal distance
            
            # debug.print("Safety Margins: {safety_margins}", safety_margins=safety_margins)

            # normalize the safety margin
            # normalized_safety_margins = safety_margins / robot_length * safety_margins.shape[0] ** 4 * self.safety_margin_norm_factor
            normalized_safety_margins = safety_margins * s_ps / robot_length * safety_margins.shape[0] ** 2
            return normalized_safety_margins

            # the "min-approach" seems to be unstable
            # minimal_safety_margin = jnp.min(safety_margins / robot_length)[None] * 2
            # return minimal_safety_margin
        
        def alpha_2(self, h_2):
            return h_2*90 #constant, increase for smaller affected zone
        
        def gamma_2(self, v_2):
            return v_2*85

    config = SoRoConfig()
    clf_cbf = CLFCBF.from_config(config)

    def closed_loop_ode_fn(t: float, y: Array, q_des: Array) -> Array:
        # split the state vector into the configuration and velocity
        q, q_d = jnp.split(y, 2)
        q_d_des = jnp.zeros_like(q_des)
        q_des = jnp.concatenate([q_des, q_d_des])

        # evaluate the control policy
        u = clf_cbf.controller(y, q_des)

        # compute the dynamical matrices
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)

        # compute the acceleration
        q_dd = jnp.linalg.inv(B) @ (u - C @ q_d - G - K - D @ q_d)

        # concatenate the velocity and acceleration
        y_d = jnp.concatenate([q_d, q_dd])

        return y_d

    # define the initial condition
    q0_arary = jnp.array([jnp.pi, 0.01, 0.05])
    multiplier = [q0_arary for m in range(num_segments)]
    q0 = jnp.concatenate(multiplier)

    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the desired configuration
    q_des = config.q_des

    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 1e-4

    # define the time steps
    ts = jnp.arange(0.0, 7.0, dt)

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, q_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)

    q_des_ts = jnp.tile(q_des, (ts.shape[0], 1))
    print("q_des_ts", q_des_ts.shape)
    # Compute tau_ts using vmap
    tau_ts = vmap(clf_cbf.controller)(sol.ys, q_des_ts)
    print("tau_ts", tau_ts.shape)

    # Plot the motion and tau_ts
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, num="Regulation example")

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
    for i in range(tau_ts.shape[1]//2):  # Assuming tau_ts has multiple dimensions (e.g., torques for each actuator)
        axes[3].plot(ts, tau_ts[:, i], label=f"Control Input {i+1}")

    # Set labels and legends
    axes[0].set_ylabel(r"Bending strain $\kappa_\mathrm{be}$")
    axes[1].set_ylabel(r"Shear strain $\sigma_\mathrm{sh}$")
    axes[2].set_ylabel(r"Axial strain $\sigma_\mathrm{ax}$")
    axes[3].set_ylabel(r"Control inputs $\tau$")
    axes[3].set_xlabel("Time [s]")

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
    pos = batched_forward_kinematics_fn(config.robot_params, config.q_des, config.s_ps)
    pos = pos[-1,:2]
    print("pos", pos)
    for q in q_ts[::20]:
        img = draw_image(batched_forward_kinematics_fn, auxiliary_fns, robot_params, num_segments, q, x_obs=config.obstacle_pos, R_obs=config.obstacle_radius, p_des = pos)
        img_ts.append(img)

        chi_ps = batched_forward_kinematics_fn(config.robot_params, q, config.s_ps)
        # Store chi_ps as a list for each timestep
        chi_ps_list.append(onp.array(chi_ps))  # Convert to numpy array for easier handling

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
    # soft_robot_with_safety_contact_CBF_example()
    soft_robot_with_safety_contact_CBFCLF_example()