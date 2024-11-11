from functools import partial
import jax
import os

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
os.environ["DDE_BACKEND"] = "jax"
from jax import Array, jacfwd, jit, vmap
from jax import numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from pathlib import Path
from srsm.planar_pcs_injury_severity_criterion import planar_pcs_injury_severity_criterion_factory
from typing import Callable, Dict

from src.contact_aware_planar_pcs_injury_severity_criterion import (
    injury_severity_criterion_with_contact_geometry_fn as injury_severity_criterion_with_contact_geometry
)

# define the outputs directory
outputs_dir = Path("outputs") / "injury_severity_criterion_with_contact_geometry"
outputs_dir.mkdir(parents=True, exist_ok=True)

# define the number of segments
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

# set configuration bounds
q_min = jnp.array([-jnp.pi, -0.2, -0.2])
q_max = jnp.array([jnp.pi, 0.2, 0.2])

# call the factory for the injury severity criterion
isc_callables = planar_pcs_injury_severity_criterion_factory(num_segments=num_segments)
injury_severity_criterion_with_contact_geometry_fn = partial(
    injury_severity_criterion_with_contact_geometry, isc_callables, robot_params, contact_characteristic,
    num_backbone_samples=500
)

# jacobian function of the injury severity criterion w.r.t. the configuration
disc_dq_fn = jacfwd(injury_severity_criterion_with_contact_geometry_fn, argnums=0, has_aux=True)
# jacobian function of the injury severity criterion w.r.t. the configuration velocity
disc_dq_d_fn = jacfwd(injury_severity_criterion_with_contact_geometry_fn, argnums=1, has_aux=True)


def sweep_obstacle_vertically_along_straight_backbone():
    # define the obstacle
    num_points = 250
    x1_obs_pts = 2.5e-2 * jnp.ones((num_points, ))
    x2_obs_pts = jnp.linspace(0, 0.1, num_points)
    x_obs_pts = jnp.stack([x1_obs_pts, x2_obs_pts], axis=-1)
    R_obs = jnp.array(0.01)

    # define the configuration
    q = jnp.zeros((3,))
    q_d = jnp.zeros_like(q)
    # define the maximum actuation torque
    tau_max = isc_callables["dynamical_matrices_fn"](robot_params, q_max, jnp.zeros_like(q_max))[3]

    isc_pts, aux_isc_pts = vmap(
        injury_severity_criterion_with_contact_geometry_fn, in_axes=(None, None, None, 0, None)
    )(q, q_d, tau_max, x_obs_pts, R_obs)

    disc_dq_pts, _ = vmap(
        disc_dq_fn, in_axes=(None, None, None, 0, None)
    )(q, q_d, tau_max, x_obs_pts, R_obs)

    # plot the injury severity criterion vs. the y-coordinate of the obstacle
    plt.figure(num="Sweep vertical obstacle position for straight backbone: Injury Severity Criterion")
    plt.plot(x2_obs_pts, isc_pts)
    plt.xlabel(r"$y_\mathrm{obs}$ [m]")
    plt.ylabel(r"Injury Severity Criterion [Pa]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(outputs_dir / "vertical_obstacle_position_sweep_straight_backbone.pdf")
    plt.show()

    # plot the derivative w.r.t. to the configuration
    fig, axes = plt.subplots(
        3, 1,
        figsize=(5, 10),
        num="Sweep vertical obstacle position for straight backbone: Derivative w.r.t. q"
    )
    axes[0].plot(
        x2_obs_pts, disc_dq_pts[:, 0], linewidth=2.5, label=r"$\frac{\partial \mathrm{ISC}}{\partial \kappa_\mathrm{be}}$"
    )
    axes[1].plot(x2_obs_pts, disc_dq_pts[:, 1], linewidth=2.5, label=r"$\frac{\partial \mathrm{ISC}}{\partial \sigma_\mathrm{sh}}$")
    axes[2].plot(x2_obs_pts, disc_dq_pts[:, 1], linewidth=2.5, label=r"$\frac{\partial \mathrm{ISC}}{\partial \sigma_\mathrm{ax}}$")
    axes[0].set_ylabel(r"$\frac{\partial \mathrm{ISC}}{\partial \kappa_\mathrm{be}}$")
    axes[1].set_ylabel(r"$\frac{\partial \mathrm{ISC}}{\partial \sigma_\mathrm{sh}}$")
    axes[2].set_ylabel(r"$\frac{\partial \mathrm{ISC}}{\partial \sigma_\mathrm{ax}}$")
    for ax in axes:
        ax.set_xlabel(r"$y_\mathrm{obs}$ [m]")
        ax.grid()
        ax.legend()
    plt.tight_layout()
    plt.savefig(outputs_dir / "vertical_obstacle_position_sweep_straight_backbone_derivative.pdf")
    plt.show()

def sweep_obstacle_on_planar_surface_for_straight_backbone():
    # define the obstacle
    x1_obs_grid, x2_obs_grid = jnp.meshgrid(
        jnp.linspace(-4e-2, 4e-2, 100), jnp.linspace(-0.02, 0.12, 100)
    )
    x_obs_pts = jnp.stack([x1_obs_grid.flatten(), x2_obs_grid.flatten()], axis=-1)
    R_obs = jnp.array(0.01)

    # define the configuration
    q = jnp.zeros((3,))
    q_d = jnp.zeros_like(q)

    # define the maximum actuation torque
    tau_max = isc_callables["dynamical_matrices_fn"](robot_params, q_max, jnp.zeros_like(q_max))[3]

    isc_pts, aux_isc_pts = vmap(
        injury_severity_criterion_with_contact_geometry_fn, in_axes=(None, None, None, 0, None)
    )(q, q_d, tau_max, x_obs_pts, R_obs)

    # reshape the arrays
    isc_grid = isc_pts.reshape(x1_obs_grid.shape)
    isc_log_grid = jnp.log(isc_grid)

    # plot the injury severity criterion as contour plot
    plt.figure(num="Sweep obstacle on planar surface for straight backbone: Injury Severity Criterion")
    plt.contourf(x1_obs_grid, x2_obs_grid, isc_log_grid, levels=100)
    # # plot the contour lines
    # plt.contour(x1_obs_grid, x2_obs_grid, isc_log_grid, levels=100, colors="k", linewidths=0.5)
    plt.xlabel(r"$x_\mathrm{obs}$ [m]")
    plt.ylabel(r"$y_\mathrm{obs}$ [m]")
    plt.colorbar(label=r"$\log(\mathrm{ISC})$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(outputs_dir / "obstacle_on_planar_surface_sweep_straight_backbone.pdf")
    plt.show()


def rotate_obstacle_around_tip_straight_backbone():
    # define the obstacle
    num_points = 1000
    varphi_pts = jnp.linspace(-jnp.pi, jnp.pi, num_points)
    r = 2.5e-2
    x_obs_pts = jnp.array([0.0, 0.1])[None, :] + r * jnp.stack([
        jnp.cos(varphi_pts), jnp.sin(varphi_pts)
    ], axis=-1)
    R_obs = jnp.array(0.01)

    # define the configuration
    q = jnp.zeros((3,))
    q_d = jnp.zeros_like(q)
    # define the maximum actuation torque
    tau_max = isc_callables["dynamical_matrices_fn"](robot_params, q_max, jnp.zeros_like(q_max))[3]

    isc_pts, aux_isc_pts = vmap(
        injury_severity_criterion_with_contact_geometry_fn, in_axes=(None, None, None, 0, None)
    )(q, q_d, tau_max, x_obs_pts, R_obs)

    # plot the injury severity criterion vs. the y-coordinate of the obstacle
    plt.figure(num="Rotate obstacle around tip for straight backbone: Injury Severity Criterion")
    plt.plot(varphi_pts, isc_pts)
    plt.xlabel(r"Obstacle polar angle around tip $y_\mathrm{obs}$ [m]")
    plt.ylabel(r"Injury Severity Criterion [Pa]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(outputs_dir / "rotate_obstacle_around_tip_straight_backbone.pdf")
    plt.show()

    # plot the details of the injury severity criterion
    fig, axes = plt.subplots(1, 1, num="Rotate obstacle around tip for straight backbone: Details")
    axes.plot(varphi_pts, aux_isc_pts["F_c0"], linewidth=3.5, label=r"$F_{c0}$ [N]")
    axes.plot(varphi_pts, aux_isc_pts["F_c0_el"], linewidth=3.0, label=r"$F_{c0,\mathrm{el}}$ [N]")
    axes.plot(varphi_pts, aux_isc_pts["F_c0_tau"], linewidth=2.5, label=r"$F_{c0,\tau}$ [N]")
    axes.plot(varphi_pts, aux_isc_pts["F_c0_vel"], linewidth=2.0, label=r"$F_{c0,\mathrm{vel}}$ [N]")
    axes.set_xlabel(r"Obstacle polar angle around tip $\varphi$ [rad]")
    axes.set_ylabel(r"Force [N]")
    axes.legend()
    axes.grid()
    plt.tight_layout()
    plt.savefig(outputs_dir / "rotate_obstacle_around_tip_straight_backbone_details.pdf")
    plt.show()

    plt.plot(varphi_pts, aux_isc_pts["s_min_dist"], label="s_min_dist")
    plt.grid(True)
    plt.xlabel(r"Obstacle polar angle around tip $\varphi$ [rad]")
    plt.ylabel(r"Backbone coordinate with minimum distance $s_\mathrm{min}$ [m]")
    plt.show()

    plt.plot(varphi_pts, aux_isc_pts["n_c_min_dist"][:, 0], label=r"$n_\mathrm{c}(0)$")
    plt.plot(varphi_pts, aux_isc_pts["n_c_min_dist"][:, 1], label=r"$n_\mathrm{c}(1)$")
    plt.legend()
    plt.grid(True)
    plt.xlabel(r"Obstacle polar angle around tip $\varphi$ [rad]")
    plt.ylabel(r"Collision surface normal $n_\mathrm{c}$")
    plt.show()

    # compute the estimated polar angle
    varphi_est = jnp.arctan2(aux_isc_pts["n_c_min_dist"][:, 1], aux_isc_pts["n_c_min_dist"][:, 0])
    plt.plot(varphi_pts, varphi_est, linewidth=3.5, label=r"$\hat{\varphi}$")
    plt.plot(varphi_pts, varphi_pts, label=r"$\varphi$")
    plt.legend()
    plt.grid(True)
    plt.show()

def sweep_configuration_space_static_obstacle():
    # define the configuration space samples
    kappa_be_grid, sigma_ax_grid = jnp.meshgrid(
        jnp.linspace(q_min[0], q_max[0], 100), jnp.linspace(0.0, q_max[2], 100)
    )
    sigma_sh_grid = jnp.zeros_like(kappa_be_grid)
    # define the configuration-space grid
    q_grid = jnp.stack([kappa_be_grid, sigma_sh_grid, sigma_ax_grid], axis=-1)
    # reshape to points
    q_pts = q_grid.reshape(-1, q_grid.shape[-1])
    q_d_pts = jnp.zeros_like(q_pts)

    # define the obstacle
    x_obs = jnp.array([0.0, 0.11])
    R_obs = jnp.array(0.01)

    # define the maximum actuation torque
    tau_max = isc_callables["dynamical_matrices_fn"](robot_params, q_max, jnp.zeros_like(q_max))[3]

    isc_pts, aux_isc_pts = vmap(
        injury_severity_criterion_with_contact_geometry_fn, in_axes=(0, 0, None, None, None)
    )(q_pts, q_d_pts, tau_max, x_obs, R_obs)

    # reshape the arrays
    isc_grid = isc_pts.reshape(kappa_be_grid.shape)
    isc_log_grid = jnp.log(isc_grid)

    # plot the injury severity criterion as contour plot
    plt.figure(num="Sweep configuration space for static obstacle: Injury Severity Criterion")
    plt.contourf(kappa_be_grid, sigma_ax_grid, isc_log_grid, levels=100)
    # # plot the contour lines
    # plt.contour(kappa_be_grid, sigma_ax_grid, isc_log_grid, levels=100, colors="k", linewidths=0.5)
    plt.xlabel(r"$\kappa_\mathrm{be}$ [rad]")
    plt.ylabel(r"$\sigma_\mathrm{ax}$ [m]")
    plt.colorbar(label=r"$\log(\mathrm{ISC})$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(outputs_dir / "sweep_configuration_space_static_obstacle.pdf")
    plt.show()



if __name__ == "__main__":
    # define the configuration
    # q = jnp.array([jnp.pi, 0.1, 0.1])
    q = jnp.zeros((3, ))
    q_d = jnp.zeros_like(q)
    # q_d = jnp.array([jnp.pi, 0.1, 0.1])

    # define the maximum actuation torque
    tau_max = isc_callables["dynamical_matrices_fn"](robot_params, q_max, jnp.zeros_like(q_max))[3]

    # define the obstacle
    x_obs = jnp.array([2e-2, 0.05])
    R_obs = jnp.array(0.01)

    # compute the injury severity criterion
    isc, aux_isc = injury_severity_criterion_with_contact_geometry_fn(
        q, q_d, tau_max, x_obs, R_obs
    )
    print("Injury severity criterion:", isc, "\n", aux_isc)

    # jacobian of the injury severity criterion w.r.t. the configuration
    disc_dq, _ = disc_dq_fn(q, q_d, tau_max, x_obs, R_obs)
    print("disc_dq:\n", disc_dq)
    # jacobian of the injury severity criterion w.r.t. the configuration velocity
    disc_dq_d, _ = disc_dq_d_fn(q, q_d, tau_max, x_obs, R_obs)
    print("disc_dq_d:\n", disc_dq_d)

    sweep_obstacle_vertically_along_straight_backbone()
    sweep_obstacle_on_planar_surface_for_straight_backbone()
    rotate_obstacle_around_tip_straight_backbone()
    sweep_configuration_space_static_obstacle()
