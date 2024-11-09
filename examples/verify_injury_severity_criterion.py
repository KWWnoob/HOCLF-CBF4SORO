from functools import partial
import jax
import os

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
os.environ["DDE_BACKEND"] = "jax"
from jax import Array, jacfwd, jit, vmap
from jax import numpy as jnp
from srsm.planar_pcs_injury_severity_criterion import planar_pcs_injury_severity_criterion_factory
from typing import Callable, Dict

from src.contact_aware_planar_pcs_injury_severity_criterion import (
    injury_severity_criterion_with_contact_geometry_fn as injury_severity_criterion_with_contact_geometry
)

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

# call the factory for the injury severity criterion
isc_callables = planar_pcs_injury_severity_criterion_factory(num_segments=num_segments)
injury_severity_criterion_with_contact_geometry_fn = partial(
    injury_severity_criterion_with_contact_geometry, isc_callables, robot_params, contact_characteristic
)

# jacobian function of the injury severity criterion w.r.t. the configuration
disc_dq_fn = jacfwd(injury_severity_criterion_with_contact_geometry_fn, argnums=0, has_aux=True)
# jacobian function of the injury severity criterion w.r.t. the configuration velocity
disc_dq_d_fn = jacfwd(injury_severity_criterion_with_contact_geometry_fn, argnums=1, has_aux=True)


if __name__ == "__main__":
    # define the configuration
    q = jnp.array([jnp.pi, 0.1, 0.1])
    # q_d = jnp.zeros_like(q)
    q_d = jnp.array([jnp.pi, 0.1, 0.1])

    # define the maximum actuation torque
    tau_max = isc_callables["dynamical_matrices_fn"](robot_params, q, q_d)[3]

    # define the obstacle
    x_obs = jnp.array([0.0, 0.1])
    R_obs = jnp.array(0.01)

    # compute the injury severity criterion
    isc, aux_isc = injury_severity_criterion_with_contact_geometry_fn(
        q, q_d, tau_max, x_obs, R_obs, num_backbone_samples=100
    )
    print("Injury severity criterion:", isc, "\n", aux_isc)

    # jacobian of the injury severity criterion w.r.t. the configuration
    disc_dq, _ = disc_dq_fn(q, q_d, tau_max, x_obs, R_obs)
    print("disc_dq:\n", disc_dq)
    # jacobian of the injury severity criterion w.r.t. the configuration velocity
    disc_dq_d, _ = disc_dq_d_fn(q, q_d, tau_max, x_obs, R_obs)
    print("disc_dq_d:\n", disc_dq_d)
