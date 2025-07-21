import diffrax as dx
from functools import partial
import jax

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
        print(tau)
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
    kappa_be_grid, sigma_ax_grid = jnp.meshgrid(jnp.linspace(-3*jnp.pi, 3*jnp.pi, 500), jnp.linspace(-0.3, 0.3, 500))

    # compute the control Lyapunov function on the grid
    kappa_be_pts, sigma_ax_pts = kappa_be_grid.flatten(), sigma_ax_grid.flatten()
    q_pts = jnp.column_stack([kappa_be_pts, jnp.zeros_like(kappa_be_pts), sigma_ax_pts])
    U_pts = vmap(
        control_lyapunov_fn, 
        in_axes=(None, 0, None, None)
    )(robot_params, q_pts, q_d, q_des)
    T_pts = vmap(
        control_lyapunov_fn, 
        in_axes=(None, None, 0, None)
    )(robot_params, jnp.zeros_like(q_des), q_pts, q_des)

    # reshape the results
    U_grid = U_pts.reshape(kappa_be_grid.shape)
    T_grid = T_pts.reshape(kappa_be_grid.shape)

    # plot the potential energy
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), num="CLF: Potential energy")
    cs = ax.contourf(kappa_be_grid, sigma_ax_grid, U_grid, levels=100)
    fig.colorbar(cs, ax=ax, label="Control Lyapunov Function")
    # plot the contour lines
    ax.contour(kappa_be_grid, sigma_ax_grid, U_grid, levels=20, colors="black", alpha=0.5)
    ax.set_xlabel(r"Bending strain $\kappa_\mathrm{be}$ [rad/m]")
    ax.set_ylabel(r"Axial strain $\sigma_\mathrm{ax}$ [-]")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # plot the kinetic energy
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), num="CLF: Kinetic energy")
    cs = ax.contourf(kappa_be_grid, sigma_ax_grid, T_grid, levels=100)
    fig.colorbar(cs, ax=ax, label="Kinetic Energy")
    # plot the contour lines
    ax.contour(kappa_be_grid, sigma_ax_grid, T_grid, levels=20, colors="black", alpha=0.5)
    ax.set_xlabel(r"Bending strain velocity $\dot{\kappa}_\mathrm{be}$ [rad/(m s)]")
    ax.set_ylabel(r"Axial strain velocity $\dot{\sigma}_\mathrm{ax}$ [1/s]")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # soft_robot_ode_example()
    soft_robot_regulation_example()
    # regulation_objective_example()
    # control_lyapunov_function_example()
