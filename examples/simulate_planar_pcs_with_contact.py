import diffrax as dx
from functools import partial
import jax

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, jacfwd, jit, vmap
from jax import numpy as jnp
import jsrm
from jsrm.systems import planar_pcs
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from srsm.planar_pcs_injury_severity_criterion import planar_pcs_injury_severity_criterion_factory
from typing import Callable, Dict, Tuple

from src.contact_aware_planar_pcs_dynamics import contact_torque_fn, ode_with_contact_fn
from src.img_animation import animate_images_cv2
from src.planar_pcs_rendering import draw_image

# define the outputs directory
outputs_dir = Path("outputs") / "planar_pcs_simulation_with_contact"
outputs_dir.mkdir(parents=True, exist_ok=True)

# specify the number of segments
num_segments = 1

# set soft robot parameters
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
D = 5e-6 * jnp.diag(jnp.array([1e0, 1e3, 1e3]))  # Damping coefficient
robot_params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 1e-1 * jnp.ones((num_segments,)),
    "r": 2e-2 * jnp.ones((num_segments,)),
    "rho": rho,
    "g": jnp.array([0.0, 9.81]),
    "E": 2e2 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
    "G": 1e2 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
    "D": D,
}
# define the contact characteristic
contact_characteristic = dict(
    k_H=jnp.array(150 * 1e3),  # N/m the spring constant of the skull and forehead (ISO TS 15066 - 2016)
    A_c=jnp.array(1e-4),  # m^2 = 1 cm^2
)

# call the factory for the injury severity criterion
isc_callables = planar_pcs_injury_severity_criterion_factory(num_segments=num_segments)
contact_stiffness_fn = isc_callables["contact_stiffness_fn"]

# extract the planar pcs functions
forward_kinematics_fn = isc_callables["forward_kinematics_fn"]
dynamical_matrices_fn = isc_callables["dynamical_matrices_fn"]
auxiliary_fns = isc_callables["auxiliary_fns"]

# construct batched forward kinematics function
batched_forward_kinematics_fn = vmap(
    forward_kinematics_fn, in_axes=(None, None, 0)
)

# define the ODE function
ode_fn = partial(
    ode_with_contact_fn,
    forward_kinematics_fn,
    dynamical_matrices_fn,
    auxiliary_fns,
    contact_stiffness_fn,
    robot_params,
    contact_characteristic,
)

def simulate_open_loop_planar_pcs_with_contact():
    # define the initial condition
    q0 = jnp.array([0.0, 0.0, 0.0])
    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the (constant) generalized torque
    tau = jnp.array([0.0, 0.0, 1e-2])

    # define the obstacle position and radius
    x_obs = jnp.array([0.00, 0.15])
    R_obs = jnp.array(0.01)

    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 5e-5

    # define the time steps
    ts = jnp.arange(0.0, 7.0, dt)

    # define the local ode fn
    local_ode_fn = partial(ode_fn, x_obs=x_obs, R_obs=R_obs)

    # test ode_fn
    y_d = local_ode_fn(0.0, y0, tau)
    print(f"y_d = {y_d}")

    # setup the diffrax ode term
    ode_term = dx.ODETerm(local_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, tau, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)

    # evaluate the contact force along the trajectory
    tau_c_ts, aux_contact_ts = vmap(
        partial(
            contact_torque_fn,
            forward_kinematics_fn,
            auxiliary_fns,
            contact_stiffness_fn,
            robot_params,
            contact_characteristic,
            x_obs=x_obs,
            R_obs=R_obs
        ),
    )(q_ts, q_d_ts)
    f_c_ts = aux_contact_ts["f_c"]

    # plot the motion
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, num="Soft robot open-loop simulation: Configuration")
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
    plt.savefig(outputs_dir / "planar_pcs_with_contact_open_loop_simulation_configuration.pdf")
    plt.show()

    # plot the contact force
    fig, ax = plt.subplots(num="Soft robot open-loop simulation: Contact force")
    ax.plot(ts, f_c_ts)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Contact force [N]")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(outputs_dir / "planar_pcs_with_contact_open_loop_simulation_contact_force.pdf")
    plt.show()

    # animate the motion
    img_ts = []
    for q in q_ts[::20]:
        img = draw_image(batched_forward_kinematics_fn, auxiliary_fns, robot_params, q, x_obs=x_obs, R_obs=R_obs)
        img_ts.append(img)
    img_ts = onp.stack(img_ts, axis=0)
    animate_images_cv2(
        onp.array(ts[::20]), img_ts, outputs_dir / "planar_pcs_with_contact_open_loop_simulation.mp4"
    )

if __name__ == "__main__":
    simulate_open_loop_planar_pcs_with_contact()
