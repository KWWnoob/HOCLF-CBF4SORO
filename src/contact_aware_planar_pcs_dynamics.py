from functools import partial
import jax
from jax import Array, jacfwd, jit, lax, vmap
from jax import numpy as jnp
from typing import Callable, Dict

from src.planar_contact_geometry import compute_planar_contact_geometry

def contact_torque_fn(
    forward_kinematics_fn: Callable,
    auxiliary_fns: Dict[str, Callable],
    contact_stiffness_fn: Callable,
    robot_params: Dict[str, Array],
    contact_characteristic: Dict[str, Array],
    q: Array,
    q_d: Array,
    x_obs: Array,
    R_obs: Array,
    num_backbone_samples: int = 250,
):
    """
    Compute the contact force by considering contact geometry.
    For now, we assume that there exists a single, circular, and stationary obstacle.
    Arguments:
        forward_kinematics_fn: function to compute the forward kinematics
        auxiliary_fns: dictionary containing the auxiliary functions
        contact_stiffness_fn: function to compute the contact stiffness
        robot_params: dictionary containing the robot parameters
        contact_characteristic: Dictionary with contact characteristics
        q: generalized coordinates
        q_d: generalized velocities
        x_obs: position of the obstacle as array of shape (2,)
        R_obs: radius of the obstacle
        num_backbone_samples: number of points to discretize the backbone
    Returns:
        tau_c: contact torque
        aux: dictionary with auxiliary information
    """
    # evaluate the contact geometry
    d_min, s_c, n_c, aux_contact_geometry = compute_planar_contact_geometry(
        forward_kinematics_fn, auxiliary_fns, robot_params, q, x_obs, R_obs,
        num_backbone_samples=num_backbone_samples
    )

    # compute the contact stiffness
    k_c = contact_stiffness_fn(robot_params, **contact_characteristic)

    # compute the penetration depth
    delta_c = jnp.maximum(0.0, -d_min)

    # compute the contact force
    f_c = k_c * delta_c

    # compute the Jacobian of the contact point
    J_c = n_c[None, :] @ auxiliary_fns["jacobian_fn"](robot_params, q, s_c)[:2]

    # compute the contact torque
    tau_c = lax.select(
        d_min <= 0.0,
        -J_c.T @ f_c[None],
        jnp.zeros_like(q)
    )

    aux = dict(
        f_c=f_c,
        d_min=d_min,
        s_c=s_c,
        n_c=n_c,
        J_c=J_c,
    )

    return tau_c, aux

def ode_with_contact_fn(
    forward_kinematics_fn: Callable,
    dynamical_matrices_fn: Callable,
    auxiliary_fns: Dict[str, Callable],
    contact_stiffness_fn: Callable,
    robot_params: Dict[str, Array],
    contact_characteristic: Dict[str, Array],
    t: Array,
    y: Array,
    tau: Array,
    x_obs: Array,
    R_obs: Array,
    num_backbone_samples: int = 250,
):
    """
    Compute the time derivative of the generalized coordinates by considering contact geometry.
    Args:
        forward_kinematics_fn: function to compute the forward kinematics
        dynamical_matrices_fn: function to compute the dynamical matrices
        auxiliary_fns: dictionary containing the auxiliary functions
        contact_stiffness_fn: function to compute the contact stiffness
        robot_params: dictionary containing the robot parameters
        contact_characteristic: dictionary containing the contact characteristics
        t: time
        y: state vector
        tau: actuation torque
        x_obs: position of the obstacle
        R_obs: radius of the obstacle
        num_backbone_samples: number of points to discretize the backbone

    Returns:
        y_d: time derivative of the state vector
    """
    # split the state vector into the configuration and velocity
    q, q_d = jnp.split(y, 2)

    # compute the dynamical matrices
    B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)

    # compute the contact torque
    tau_c, contact_torque_aux = contact_torque_fn(
        forward_kinematics_fn, auxiliary_fns, contact_stiffness_fn, robot_params, contact_characteristic,
        q, q_d, x_obs, R_obs, num_backbone_samples
    )

    # compute the acceleration
    q_dd = jnp.linalg.inv(B) @ (tau + tau_c - C @ q_d - G - K - D @ q_d)

    # concatenate the velocity and acceleration
    y_d = jnp.concatenate([q_d, q_dd])

    return y_d
