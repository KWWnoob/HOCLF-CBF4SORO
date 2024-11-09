from functools import partial
import jax
from jax import Array, jacfwd, jit, lax, vmap
from jax import numpy as jnp
from typing import Callable, Dict

from .planar_contact_geometry import compute_planar_contact_geometry


def injury_severity_criterion_with_contact_geometry_fn(
    isc_callables: Dict[str, Callable],
    robot_params: Dict[str, Array],
    contact_characteristic: Dict[str, Array],
    q: Array,
    q_d: Array,
    tau_max: Array,
    x_obs: Array,
    R_obs: Array,
    num_backbone_samples: int = 100,
):
    """
    Compute the injury severity criterion by considering contact geometry.
    For now, we assume that there exists a single, circular, and stationary obstacle.
    Arguments:
        robot_params: dictionary containing the robot parameters
        contact_characteristic: dictionary containing the contact characteristics
        q: generalized coordinates
        q_d: generalized velocities
        tau_max: maximum actuation torque
        x_obs: position of the obstacle
        R_obs: radius of the obstacle
        num_backbone_samples: number of points to discretize the backbone
    Returns:
        isc: injury severity criterion
        aux_isc: auxiliary variables
    """
    # evaluate the contact geometry
    d_min, s_min_dist, n_c_min_dist, aux_contact_geometry = compute_planar_contact_geometry(
        isc_callables["forward_kinematics_fn"], robot_params, q, x_obs, R_obs, num_backbone_samples=num_backbone_samples
    )

    # evaluate the injury severity criterion
    isc, aux_isc = isc_callables["injury_severity_criterion_fn"](
        robot_params, contact_characteristic, q, q_d, s_min_dist, n_c_min_dist, tau_max, apply_actuation_norm=True
    )

    aux_isc = aux_isc | dict(
        d_min=d_min,
        s_min_dist=s_min_dist,
        n_c_min_dist=n_c_min_dist,
    )

    isc = lax.select(d_min <= 0.0, isc, jnp.zeros_like(isc))

    return isc, aux_isc
