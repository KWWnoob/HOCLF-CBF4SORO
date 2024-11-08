from jax import Array, jacfwd, jit, lax, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Optional, Tuple, Union


def compute_planar_contact_geometry(
    forward_kinematics_fn: Callable,
    robot_params: Dict[str, Array],
    q: Array,
    x_obs: Array,
    R_obs: Union[float, Array],
    num_backbone_samples: int = 100,
) -> Tuple[Array, Array, Array, Dict[str, Array]]:
    """
    Compute the minimum distance of the backbone from the obstacle.
    Args:
        forward_kinematics_fn: function to compute the forward kinematics
        robot_params: dictionary with robot parameters
        q: joint angles
        x_obs: planar position of the obstacle
        R_obs: radius of the obstacle
        num_backbone_samples: number of points along the backbone
    Returns:
        d_min: minimum distance of the backbone from the obstacle
        s_min_dist: backbone coordinate of the closest point
        n_c_min_dist: collision surface normal at the closest point
        aux: dictionary with auxiliary information
    """
    # get the backbone points
    s_pts = onp.linspace(0.0, 1.0, num_backbone_samples) * jnp.sum(robot_params["l"])
    chi_pts = vmap(forward_kinematics_fn, in_axes=(None, None, 0))(robot_params, q, s_pts)

    # compute the distances of the backbone points from the obstacle
    d_pts = jnp.linalg.norm(chi_pts[:, :2] - x_obs, axis=-1) - R_obs

    # determine the minimum distance and the associated backbone coordinate
    d_min = jnp.min(d_pts)
    min_distance_idx = jnp.argmin(d_pts)
    s_min_dist = lax.dynamic_slice(s_pts, [min_distance_idx], [1]).squeeze(0)
    chi_min_dist = lax.dynamic_slice(
        chi_pts, [min_distance_idx, 0], [1, chi_pts.shape[-1]]
    ).squeeze(0)

    # compute the collision surface normal at the closest point
    n_c_min_dist = (x_obs - chi_min_dist[:2]) / jnp.linalg.norm(x_obs - chi_min_dist[:2])

    aux = dict(
        chi_min_dist=chi_min_dist,
        chi_pts=chi_pts,
        s_pts=s_pts,
        d_pts=d_pts,
    )

    return d_min, s_min_dist, n_c_min_dist, aux
