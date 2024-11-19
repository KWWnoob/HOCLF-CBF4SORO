__all__ = ["injury_severity_criterion_with_contact_geometry_fn"]
from functools import partial
import jax
from jax import Array, jacfwd, jit, lax, vmap
from jax import numpy as jnp
from typing import Callable, Dict, Optional

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
    v_obs: Optional[Array] = None,
    num_backbone_samples: int = 250,
    contact_boundary_fn: Optional[Callable] = jax.nn.sigmoid,
    contact_boundary_compression_factor: float = 2e3,
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
        x_obs: position of the obstacle as array of shape (2,)
        R_obs: radius of the obstacle
        v_obs: velocity of the obstacle as array of shape (2,)
        num_backbone_samples: number of points to discretize the backbone
        contact_boundary_fn: function to compute the collision boundary.
            If None, use a hard boundary (i.e., discontinuous). The boundary function should be in the range [0, 1].
        contact_boundary_compression_factor: compression factor for the contact boundary function.
            The robot-obstacle distance is multiplied by this factor before applying the boundary function.
    Returns:
        isc: injury severity criterion
        aux_isc: auxiliary variables
    """
    # set the velocity of the obstacle to zero if not provided
    if v_obs is None:
        v_obs = jnp.zeros_like(x_obs)

    # evaluate the contact geometry
    d_min, s_min_dist, n_c_min_dist, aux_contact_geometry = compute_planar_contact_geometry(
        isc_callables["forward_kinematics_fn"], isc_callables["auxiliary_fns"], robot_params, q, x_obs, R_obs,
        num_backbone_samples=num_backbone_samples
    )
    # s_min_dist and n_c_min_dist are equal to s_c and n_c
    s_c, n_c = s_min_dist, n_c_min_dist

    # compute the penetration depth at the beginning of the collision
    delta_c0 = jnp.maximum(0.0, -d_min)

    # compute the velocity of the obstacle in the direction of the contact
    v_obs_c = jnp.dot(v_obs, n_c)

    # evaluate the injury severity criterion
    isc, aux_isc = isc_callables["injury_severity_criterion_fn"](
        robot_params, contact_characteristic, q, q_d, s_c, n_c, tau_max, delta_c0=delta_c0, v_H=-v_obs_c,
        apply_actuation_norm=True
    )

    aux_isc = aux_isc | dict(
        delta_c0=delta_c0,
        v_obs_c=v_obs_c,
        d_min=d_min,
        s_c=s_c,
        n_c=n_c,
    )

    if contact_boundary_fn is None:
        isc = lax.select(-d_min >= 0.0, isc, jnp.zeros_like(isc))
    else:
        isc = contact_boundary_fn(-d_min * contact_boundary_compression_factor) * isc

    return isc, aux_isc
