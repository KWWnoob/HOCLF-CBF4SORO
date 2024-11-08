import cv2  # importing cv2
from jax import Array, jacfwd, jit, lax, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Optional, Tuple, Union


def draw_image(
    batched_forward_kinematics_fn: Callable,
    robot_params: Dict[str, Array],
    q: Array,
    x_obs: Array,
    R_obs: Union[float, Array],
    img_width: int = 700,
    img_height: int = 700,
    num_points: int = 50,
) -> onp.ndarray:
    """
    Draw the robot in a 2D image.
    Arguments:
        batched_forward_kinematics_fn: function to compute the forward kinematics with interface (params, q, s_pts)
        robot_params: dictionary with robot parameters
        q: joint angles
        x_obs: planar position of the obstacle
        R_obs: radius of the obstacle
        img_width: image width
        img_height: image height
        num_points: number of points along the robot to plot
    """
    # plotting in OpenCV
    h, w = img_height, img_width  # img height and width
    ppm = h / (2.0 * jnp.sum(robot_params["l"]))  # pixel per meter
    base_color = (0, 0, 0)  # black robot_color in BGR
    robot_color = (255, 0, 0)  # black robot_color in BGR

    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, jnp.sum(robot_params["l"]), num_points)

    # poses along the robot of shape (3, N)
    chi_ps = batched_forward_kinematics_fn(robot_params, q, s_ps)

    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    curve_origin = onp.array(
        [w // 2, 0.1 * h], dtype=onp.int32
    )  # in x-y pixel coordinates

    # draw base
    cv2.rectangle(img, (0, h - curve_origin[1]), (w, h), color=base_color, thickness=-1)
    # transform robot poses to pixel coordinates
    # should be of shape (N, 2)
    curve = onp.array((curve_origin + chi_ps[:2, :].T * ppm), dtype=onp.int32)
    # invert the v pixel coordinate
    curve[:, 1] = h - curve[:, 1]
    cv2.polylines(img, [curve], isClosed=False, color=robot_color, thickness=10)

    # # draw the obstacle
    # uv_obs = onp.array((curve_origin + x_obs * ppm), dtype=onp.int32)
    # # invert the v pixel coordinate
    # uv_obs[1] = h - uv_obs[1]
    # cv2.circle(img, tuple(uv_obs), int(R_obs * ppm), (0, 255, 0), thickness=-1)

    return img
