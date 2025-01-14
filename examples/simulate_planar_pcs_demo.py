import diffrax as dx
from functools import partial
import jax
from cbfpy import CBF, CBFConfig
from cbfpy.cbfs.clf_cbf import CLFCBF
from cbfpy.config.clf_cbf_config import CLFCBFConfig

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
from src.planar_pcs_rendering import draw_image

# define the outputs directory
outputs_dir = Path("outputs") / "planar_pcs_simulation"
outputs_dir.mkdir(parents=True, exist_ok=True)

# load symbolic expressions
num_segments = 1
# filepath to symbolic expressions
sym_exp_filepath = Path(jsrm.__file__).parent / "symbolic_expressions" / f"planar_pcs_ns-{num_segments}.dill"

# set soft robot parameters
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
robot_params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 1e-1 * jnp.ones((num_segments,)),
    "r": 2e-2 * jnp.ones((num_segments,)),
    "rho": rho,
    "g": jnp.array([0.0, 9.81]),
    "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
    "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
}
# damping matrix
robot_params["D"] = 5e-5 * jnp.diag(jnp.array([1e0, 1e3, 1e3]) * robot_params["l"])

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

def soft_robot_ode_example():
    # define the ODE function

    def ode_fn(t: float, y: Array, tau: Array) -> Array:
        # split the state vector into the configuration and velocity
        q, q_d = jnp.split(y, 2)

        # compute the dynamical matrices (B is M)
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)

        # compute the acceleration
        q_dd = jnp.linalg.inv(B) @ (tau - C @ q_d - G - K - D @ q_d)

        # concatenate the velocity and acceleration
        y_d = jnp.concatenate([q_d, q_dd])

        return y_d

    # define the initial condition
    q0 = jnp.array([jnp.pi, 0.01, 0.05])
    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the (constant) generalized torque
    tau = jnp.array([-2e-4, 0.0, 1e-2])

    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 5e-5

    # define the time steps
    ts = jnp.arange(0.0, 7.0, dt)

    # setup the diffrax ode term
    ode_term = dx.ODETerm(ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, tau, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)

    # plot the motion
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, num="Soft robot open-loop simulation")
    axes[0].plot(ts, q_ts[:, 0])
    axes[1].plot(ts, q_ts[:, 1])
    axes[2].plot(ts, q_ts[:, 2])
    axes[0].set_ylabel(r"Bending strain $\kappa_\mathrm{be}$")
    axes[1].set_ylabel(r"Shear strain $\sigma_\mathrm{sh}$")
    axes[2].set_ylabel(r"Axial strain $\sigma_\mathrm{ax}$")
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    # animate the motion
    img_ts = []
    for q in q_ts[::20]:
        img = draw_image(batched_forward_kinematics_fn, auxiliary_fns, robot_params, q)
        img_ts.append(img)
    img_ts = onp.stack(img_ts, axis=0)
    animate_images_cv2(
        onp.array(ts[::20]), img_ts, outputs_dir / "planar_pcs_open_loop_simulation.mp4"
    )

def soft_robot_regulation_example():
    # define the ODE function

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

    def closed_loop_ode_fn(t: float, y: Array, q_des: Array) -> Array:
        # split the state vector into the configuration and velocity
        q, q_d = jnp.split(y, 2)

        # evaluate the control policy
        tau = control_policy_fn(t, y, q_des)

        # compute the dynamical matrices
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)

        # compute the acceleration
        q_dd = jnp.linalg.inv(B) @ (tau - C @ q_d - G - K - D @ q_d)

        # concatenate the velocity and acceleration
        y_d = jnp.concatenate([q_d, q_dd])

        return y_d

    # define the initial condition
    q0 = jnp.array([jnp.pi, 0.01, 0.05])
    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the desired configuration
    q_des = jnp.array([-jnp.pi, 0.0, 0.2])

    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 5e-5

    # define the time steps
    ts = jnp.arange(0.0, 7.0, dt)

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, q_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)
    q_des_ts = jnp.tile(q_des, (len(ts), 1))

    # plot the motion
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, num="Regulation example")
    axes[0].plot(ts, q_ts[:, 0])
    axes[1].plot(ts, q_ts[:, 1])
    axes[2].plot(ts, q_ts[:, 2])
    axes[0].set_ylabel(r"Bending strain $\kappa_\mathrm{be}$")
    axes[1].set_ylabel(r"Shear strain $\sigma_\mathrm{sh}$")
    axes[2].set_ylabel(r"Axial strain $\sigma_\mathrm{ax}$")
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    # animate the motion
    img_ts = []
    for q in q_ts[::20]:
        img = draw_image(batched_forward_kinematics_fn, auxiliary_fns, robot_params, q)
        img_ts.append(img)
    img_ts = onp.stack(img_ts, axis=0)
    animate_images_cv2(
        onp.array(ts[::20]), img_ts, outputs_dir / "planar_pcs_closed_loop_simulation.mp4"
    )

def regulation_objective_example():
    # cost function weights
    Q = jnp.diag(jnp.array([1/jnp.pi, 10.0, 5.0]))

    def cost_fn(q: Array, q_des: Array) -> Array:
        """
        Cost function that penalizes the deviation from the desired configuration.
        Args:
            q: configuration
            q_des: desired configuration
        Returns:
            cost: cost
        """
        error = q - q_des
        cost = 0.5 * error.T @ Q @ error
        return cost

    q = jnp.array([1.0, 0.01, 0.05])
    q_des = jnp.array([0.0, 0.0, 0.2])
    cost = cost_fn(q, q_des)
    print(f"Cost for q = {q} and q_des = {q_des} is {cost}")

def soft_robot_with_safety_example():
    # define the ODE function
    class SoRoConfig(CBFConfig):
        '''Config for soft robot'''

        def __init__(self):
            self.robot_params = {
                "th0": jnp.array(0.0),  # initial orientation angle [rad]
                "l": 1e-1 * jnp.ones((num_segments,)),
                "r": 2e-2 * jnp.ones((num_segments,)),
                "rho": 1070 * jnp.ones((num_segments,)),
                "g": jnp.array([0.0, 9.81]),
                "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
                "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
            }   
            self.robot_params["D"] = 5e-5 * jnp.diag(jnp.array([1e0, 1e3, 1e3]) * robot_params["l"])
            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)
            super().__init__(
                n=6, # number of states
                m=3, # number of inputs
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=True,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                cbf_relaxation_penalty=1e6,
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
        
        # def V_2(self, z) -> float:
        # # CLF: general energy
        #     q, q_d = jnp.split(z, 2)
        #     T = kinetic_energy_fn(robot_params, q, q_d)
        #     U = potential_energy_fn(robot_params, q)
        #     return T + U
        
        def h_2(self, z):
        # CBF: robotic segment should not go to the right
            q, q_d = jnp.split(z, 2)
            return - q[0:1]*5

        
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

    def closed_loop_ode_fn(t: float, y: Array, q_des: Array) -> Array:
        # split the state vector into the configuration and velocity
        q, q_d = jnp.split(y, 2)

        # evaluate the control policy
        tau = control_policy_fn(t, y, q_des)

        # evaluate the safe control polic
        tau_filtered = cbf.safety_filter(y, tau)

        # compute the dynamical matrices
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)

        # compute the acceleration
        q_dd = jnp.linalg.inv(B) @ (tau_filtered - C @ q_d - G - K - D @ q_d)

        # concatenate the velocity and acceleration
        y_d = jnp.concatenate([q_d, q_dd])

        return y_d


    # define the initial condition
    q0 = jnp.array([jnp.pi, 0.01, 0.05])
    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the desired configuration
    q_des = jnp.array([jnp.pi * 1.5, 0.0, 0.2])

    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 5e-5

    # define the time steps
    ts = jnp.arange(0.0, 7.0, dt)

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, q_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)


    q_des_ts = jnp.tile(q_des, (len(ts), 1))
    # Compute tau_ts using vmap
    tau_ts = vmap(closed_loop_ode_fn)(ts, sol.ys, q_des_ts)

    # Plot the motion and tau_ts
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, num="Regulation example")

    # Plot strains
    axes[0].plot(ts, q_ts[:, 0], label=r"$\kappa_\mathrm{be}$")
    axes[1].plot(ts, q_ts[:, 1], label=r"$\sigma_\mathrm{sh}$")
    axes[2].plot(ts, q_ts[:, 2], label=r"$\sigma_\mathrm{ax}$")

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

    import csv

    # Collect chi_ps values
    chi_ps_list = []

    s_ps = jnp.linspace(0, robot_params["l"], 20)

    # Animate the motion and collect chi_ps
    img_ts = []
    for q in q_ts[::20]:
        img = draw_image(batched_forward_kinematics_fn, auxiliary_fns, robot_params, q)
        img_ts.append(img)
        
        chi_ps = batched_forward_kinematics_fn(robot_params, q, s_ps)
        print(chi_ps)
        # Store chi_ps as a list for each timestep
        chi_ps_list.append(onp.array(chi_ps))  # Convert to numpy array for easier handling

    # Save chi_ps to a CSV using the csv module
    with open("chi_ps_values.csv", mode="w", newline="") as file:
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

def soft_robot_with_safety_2_example():
    # define the ODE function
    class SoRoConfig(CLFCBFConfig):
        '''Config for soft robot'''

        def __init__(self):
            self.robot_params = {
                "th0": jnp.array(0.0),  # initial orientation angle [rad]
                "l": 1e-1 * jnp.ones((num_segments,)),
                "r": 2e-2 * jnp.ones((num_segments,)),
                "rho": 1070 * jnp.ones((num_segments,)),
                "g": jnp.array([0.0, 9.81]),
                "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
                "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
            }   
            self.robot_params["D"] = 5e-5 * jnp.diag(jnp.array([1e0, 1e3, 1e3]) * robot_params["l"])
            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)
            super().__init__(
                n=6, # number of states
                m=3, # number of inputs
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=True,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                cbf_relaxation_penalty=1e6,
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
        # CLF: general energy
            q, q_d = jnp.split(z, 2)
            T = kinetic_energy_fn(robot_params, q, q_d)
            U = potential_energy_fn(robot_params, q)
            return jnp.array([T+U])
        
        def h_2(self, z):
        # CBF: robotic segment should not go to the right
            q, q_d = jnp.split(z, 2)
            return - q[0:1]*5

    
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
    q0 = jnp.array([jnp.pi, 0.01, 0.05])
    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the desired configuration
    q_des = jnp.array([jnp.pi * 1.5, 0.0, 0.2])

    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 5e-5

    # define the time steps
    ts = jnp.arange(0.0, 7.0, dt)

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, q_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)
    q_des_ts = jnp.tile(q_des, (len(ts), 1))
    # Compute tau_ts using vmap
    tau_ts = vmap(closed_loop_ode_fn)(ts, sol.ys, q_des_ts)

    # Plot the motion and tau_ts
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, num="Regulation example")

    # Plot strains
    axes[0].plot(ts, q_ts[:, 0], label=r"$\kappa_\mathrm{be}$")
    axes[1].plot(ts, q_ts[:, 1], label=r"$\sigma_\mathrm{sh}$")
    axes[2].plot(ts, q_ts[:, 2], label=r"$\sigma_\mathrm{ax}$")

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

    # animate the motion
    img_ts = []
    for q in q_ts[::20]:
        img = draw_image(batched_forward_kinematics_fn, auxiliary_fns, robot_params, q)
        img_ts.append(img)
    img_ts = onp.stack(img_ts, axis=0)
    animate_images_cv2(
        onp.array(ts[::20]), img_ts, outputs_dir / "planar_pcs_safe_closed_loop_simulation.mp4"
    )

def soft_robot_with_safety_contact_example():
    # define the ODE function
    class SoRoConfig(CBFConfig):
        '''Config for soft robot'''

        def __init__(self):

            self.robot_params = {
                "th0": jnp.array(0.0),  # initial orientation angle [rad]
                "l": 1e-1 * jnp.ones((num_segments,)),
                "r": 2e-2 * jnp.ones((num_segments,)),
                "rho": 1070 * jnp.ones((num_segments,)),
                "g": jnp.array([0.0, 9.81]),
                "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
                "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
            }   

            self.robot_params["D"] = 5e-5 * jnp.diag(jnp.array([1e0, 1e3, 1e3]) * self.robot_params["l"])
            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

            self.obstacle_pos = jnp.array([-5e-2, 0.09])
            self.obstacle_radius = 1e-2
            self.s_ps = jnp.linspace(0, self.robot_params["l"], 10)

            super().__init__(
                n=6, # number of states
                m=3, # number of inputs
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=True,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                cbf_relaxation_penalty=1e6,
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
            pos = forward_kinematics_fn(self.robot_params, q, self.s_ps)  
            # Ignore orientation, keep x-y positions
            pos = pos[:2, -1] 

            # Compute the distance to the obstacle center
            distance_to_obstacle = jnp.linalg.norm((pos.T - self.obstacle_pos), ord=2, axis=0) 

            # Compute safety margin for each segment
            safety_margins = distance_to_obstacle - (self.obstacle_radius + (2e-2)) # minimal distance
            return jnp.array([safety_margins])
            # return jnp.array([1])

        # def h_1(self, z):
        #     #regulating "strain space"
        #     q, q_d = jnp.split(z, 2)
        #     pos_max = jnp.array([jnp.pi * 4, 0.1, 0.3])

            return jnp.concatenate([pos_max - q])
        
        def alpha(self, h):
            return 0.00001*h
        
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

    def closed_loop_ode_fn(t: float, y: Array, q_des: Array) -> Array:
        # split the state vector into the configuration and velocity
        q, q_d = jnp.split(y, 2)

        # evaluate the control policy
        tau = control_policy_fn(t, y, q_des)

        # evaluate the safe control polic
        tau_filtered = cbf.safety_filter(y, tau)

        # compute the dynamical matrices
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)

        # compute the acceleration
        q_dd = jnp.linalg.inv(B) @ (tau_filtered - C @ q_d - G - K - D @ q_d)

        # concatenate the velocity and acceleration
        y_d = jnp.concatenate([q_d, q_dd])

        return y_d


    # define the initial condition
    q0 = jnp.array([jnp.pi, 0.01, 0.05])
    print("p0:",forward_kinematics_fn(robot_params, q0, jnp.linspace(0, robot_params["l"], 10)))
    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the desired configuration
    q_des = jnp.array([jnp.pi * 3, 0.0, 0.2])
    print("pdes:",forward_kinematics_fn(robot_params, q_des, jnp.linspace(0, robot_params["l"], 10)))
    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 5e-5

    # define the time steps
    ts = jnp.arange(0.0, 7.0, dt)

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, q_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)


    q_des_ts = jnp.tile(q_des, (len(ts), 1))
    # Compute tau_ts using vmap
    tau_ts = vmap(closed_loop_ode_fn)(ts, sol.ys, q_des_ts)

    # Plot the motion and tau_ts
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, num="Regulation example")

    # Plot strains
    axes[0].plot(ts, q_ts[:, 0], label=r"$\kappa_\mathrm{be}$")
    axes[1].plot(ts, q_ts[:, 1], label=r"$\sigma_\mathrm{sh}$")
    axes[2].plot(ts, q_ts[:, 2], label=r"$\sigma_\mathrm{ax}$")

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

    import csv

    # Collect chi_ps values
    chi_ps_list = []

    s_ps = jnp.linspace(0, robot_params["l"], 20)

    # Animate the motion and collect chi_ps
    img_ts = []
    for q in q_ts[::20]:
        img = draw_image(batched_forward_kinematics_fn, auxiliary_fns, robot_params, q, x_obs = jnp.array([-0.02, 0.9]), R_obs= 0.002)
        img_ts.append(img)
        
        chi_ps = batched_forward_kinematics_fn(robot_params, q, s_ps)
        # Store chi_ps as a list for each timestep
        chi_ps_list.append(onp.array(chi_ps))  # Convert to numpy array for easier handling

    # Save chi_ps to a CSV using the csv module
    with open("chi_ps_values.csv", mode="w", newline="") as file:
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


def soft_robot_with_safety_contact_2_example():
    # define the ODE function
    class SoRoConfig(CLFCBFConfig):
        '''Config for soft robot'''

        def __init__(self):

            self.robot_params = {
                "th0": jnp.array(0.0),  # initial orientation angle [rad]
                "l": 1e-1 * jnp.ones((num_segments,)),
                "r": 2e-2 * jnp.ones((num_segments,)),
                "rho": 1070 * jnp.ones((num_segments,)),
                "g": jnp.array([0.0, 9.81]),
                "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
                "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
            }   

        
            self.robot_params["D"] = 5e-5 * jnp.diag(jnp.array([1e0, 1e3, 1e3]) * self.robot_params["l"])
            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

            self.obstacle_pos = jnp.array([0.1,0.1])
            self.obstacle_radius = 0.02
            self.s_ps = jnp.linspace(0, self.robot_params["l"], 20)
            self.q_des = jnp.array([0.2,0.1])

            super().__init__(
                n=6, # number of states
                m=3, # number of inputs
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=True,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                cbf_relaxation_penalty=1e6,
            )
    
        def f(self, z) -> Array:
            q, q_d = jnp.split(z, 2)  # Split state z into q (position) and q_d (velocity)
            B, C, G, K, D, alpha = dynamical_matrices_fn(self.robot_params, q, q_d)

            q_dd = (
                -jnp.linalg.inv(B) @ (C @ q_d + D @ q_d + G + K)
            )
    
            return jnp.concatenate([q_d, q_dd])
        
        def g(self, z) -> Array:
            q, q_d = jnp.split(z, 2)
            B, _, _, _, _, _ = dynamical_matrices_fn(self.robot_params, q, q_d)
            
            # Control matrix g(x)
            control_matrix = jnp.linalg.inv(B)
            
            # Match dimensions for concatenation
            zero_block = jnp.zeros((q.shape[0], control_matrix.shape[1]))
            
            return jnp.concatenate([zero_block, control_matrix], axis=0)
        
        def V_2(self, z) -> jnp.ndarray:
        # CLF: distance from tip to destination
            q, q_d = jnp.split(z, 2)
            pos = forward_kinematics_fn(robot_params, q, self.s_ps) 
            pos = pos[:2,-1]

            squared_differences = (pos - self.q_des) ** 2
            return squared_differences
        
        def h_2(self, z):
            # Split input into positions (q) and velocities (q_d)
            q, q_d = jnp.split(z, 2)

            # Get the position of the robotic segments
            pos = forward_kinematics_fn(robot_params, q, self.s_ps) 
            pos = pos[:2,:] # ignore the orientation

            repeated_obstacle_pos = jnp.tile(self.obstacle_pos, (len(self.s_ps), 1)) 

            squared_differences = (pos.T - repeated_obstacle_pos) ** 2 

            # Sum over the coordinate dimensions (2D -> scalar per segment) and then sum over segments
            diff = jnp.sum(squared_differences)  # First sum over coordinates, then over segments

            return jnp.array([diff])

    
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
    q0 = jnp.array([jnp.pi, 0.01, 0.05])
    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the desired configuration
    q_des = jnp.array([jnp.pi * 1.5, 0.0, 0.2])
    pos = forward_kinematics_fn(robot_params, q_des, jnp.linspace(0, robot_params["l"], 2)) 
    q_des = pos[-3:-2]

    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 5e-5

    # define the time steps
    ts = jnp.arange(0.0, 7.0, dt)

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, q_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)
    q_des_ts = jnp.tile(q_des, (len(ts), 1))
    # Compute tau_ts using vmap
    tau_ts = vmap(closed_loop_ode_fn)(ts, sol.ys, q_des_ts)

    # Plot the motion and tau_ts
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, num="Regulation example")

    # Plot strains
    axes[0].plot(ts, q_ts[:, 0], label=r"$\kappa_\mathrm{be}$")
    axes[1].plot(ts, q_ts[:, 1], label=r"$\sigma_\mathrm{sh}$")
    axes[2].plot(ts, q_ts[:, 2], label=r"$\sigma_\mathrm{ax}$")

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

    # animate the motion
    img_ts = []
    for q in q_ts[::20]:
        img = draw_image(batched_forward_kinematics_fn, auxiliary_fns, robot_params, q)
        img_ts.append(img)
    img_ts = onp.stack(img_ts, axis=0)
    animate_images_cv2(
        onp.array(ts[::20]), img_ts, outputs_dir / "planar_pcs_safe_closed_loop_simulation.mp4"
    )

def control_lyapunov_function_example():
    def control_lyapunov_fn(params: Dict[str, Array], q: Array, q_d: Array, q_des: Array) -> Array:
        """
        Control Lyapunov Function for a setpoint regulator of form 
            tau = K(q_des) + G(q_des)
        The Control Lyapunov Function is given in Eq. (19) of the paper
            Della Santina, C., Duriez, C., & Rus, D. (2023). Model-based control of soft robots: 
            A survey of the state of the art and open challenges. IEEE Control Systems Magazine, 43(3), 30-65.
        Args:
            params: robot parameters
            q: configuration
            q_d: configuration velocity
            q_des: desired configuration
        Returns:
            V: value of the control Lyapunov function
        """
        # compute the kinetic energy at the current configuration
        T = kinetic_energy_fn(robot_params, q, q_d)
        # compute the potential energy at the current configuration
        U = potential_energy_fn(robot_params, q)
        # compute the potential energy at the desired configuration
        U_des = potential_energy_fn(robot_params, q_des)
        # compute the dynamical matrices at the desired configuration
        B_des, C_des, G_des, K_des, D_des, alpha_des = dynamical_matrices_fn(params, q_des, jnp.zeros_like(q_des))

        # compute the control Lyapunov function
        V = T + U - U_des + (G_des + K_des).T @ (q_des - q)

        return V

    # define the desired configuration
    q_des = jnp.array([1.0, 0.0, -0.05])
    # set the configuration velocity to zero
    q_d = jnp.zeros_like(q_des)

    # define a grid of configurations
    kappa_be_grid, sigma_ax_grid = jnp.meshgrid(jnp.linspace(-jnp.pi, jnp.pi, 200), jnp.linspace(-0.1, 0.1, 200))

    # compute the control Lyapunov function on the grid
    kappa_be_pts, sigma_ax_pts = kappa_be_grid.flatten(), sigma_ax_grid.flatten()
    q_pts = jnp.column_stack([kappa_be_pts, jnp.zeros_like(kappa_be_pts), sigma_ax_pts])
    V_pts = vmap(
        control_lyapunov_fn, 
        in_axes=(None, 0, None, None)
    )(robot_params, q_pts, q_d, q_des)

    # reshape the results
    V_grid = V_pts.reshape(kappa_be_grid.shape)

    # plot the control Lyapunov function
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), num="Control Lyapunov function")
    cs = ax.contourf(kappa_be_grid, sigma_ax_grid, V_grid, levels=100)
    fig.colorbar(cs, ax=ax, label="Control Lyapunov Function")
    # plot the contour lines
    ax.contour(kappa_be_grid, sigma_ax_grid, V_grid, levels=20, colors="black", alpha=0.5)
    ax.set_xlabel(r"Bending strain $\kappa_\mathrm{be}$")
    ax.set_ylabel(r"Axial strain $\sigma_\mathrm{ax}$")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # soft_robot_ode_example()
    # soft_robot_regulation_example()
    # regulation_objective_example()
    # control_lyapunov_function_example()

    # soft_robot_with_safety_example()
    # soft_robot_with_safety_2_example()
    soft_robot_with_safety_contact_example()
    # soft_robot_with_safety_contact_2_example()
