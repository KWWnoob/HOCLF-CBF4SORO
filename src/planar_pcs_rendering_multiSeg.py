__all__ = ["draw_image"]
import cv2  # importing cv2
from jax import Array, jacfwd, jit, lax, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Optional, Tuple, Union

def draw_image(
    batched_forward_kinematics_fn: Callable,
    auxiliary_fns: Dict[str, Callable],
    robot_params: Dict[str, Array],
    num_segments: int,
    q: Array,
    x_obs: Optional[Array] = None,
    R_obs: Optional[Union[float, Array]] = None,
    p_des: Optional[Array] = None,
    poly_points: Optional[Array] = None,
    img_width: int = 700,
    img_height: int = 700,
    num_points: int = 50,
) -> onp.ndarray:
    """
    Draw the robot in a 2D image with different colors for each segment.
    Arguments:
        batched_forward_kinematics_fn: function to compute the forward kinematics with interface (params, q, s_pts)
        robot_params: dictionary with robot parameters
        q: joint angles
        x_obs: planar position of the obstacle
        R_obs: radius of the obstacle
        p_des: target position
        img_width: image width
        img_height: image height
        num_points: number of points along the robot to plot
    """
    # compute the total length of the robot
    L = jnp.sum(robot_params["l"])

    # plotting in OpenCV
    h, w = img_height, img_width  # img height and width
    ppm = h / (2.0 * L)  # pixel per meter
    base_color = (0, 0, 0)  # black base color in BGR

    # dynamically generate a color palette for each segment
    segment_colors = [
        (int(255 * (i / num_segments)), int(255 * ((i + 1) / num_segments)), int(255 * ((i + 2) / num_segments)))
        for i in range(num_segments)
    ]

    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, L, num_points)

    from jax import vmap
    classify_segment_vmap = vmap(auxiliary_fns["classify_segment"], in_axes=(None, 0))
    segment_idx_ps, _ = classify_segment_vmap(robot_params, s_ps)

    # poses along the robot of shape (3, N)
    chi_ps = batched_forward_kinematics_fn(robot_params, q, s_ps)

    # the equilibrium distance between the backbone and the obstacle is the sum of the radii of the obstacle and the backbone
    r_backbone_ps = robot_params["r"][segment_idx_ps]

    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    curve_origin = onp.array(
        [w // 2, 0.1 * h], dtype=onp.int32
    )  # in x-y pixel coordinates

    # draw base
    cv2.rectangle(img, (0, h - curve_origin[1]), (w, h), color=base_color, thickness=-1)

    # transform robot poses to pixel coordinates
    curve = onp.array((curve_origin + chi_ps[:, :2] * ppm), dtype=onp.int32)
    # invert the v pixel coordinate
    curve[:, 1] = h - curve[:, 1]

    # draw each segment with its assigned color
    for segment_idx in jnp.unique(segment_idx_ps):
        segment_ps_selector = segment_idx_ps == segment_idx
        segment_radius = r_backbone_ps[segment_idx].item()  # Use segment index directly
        segment_thickness = int(2 * segment_radius * ppm)
        segment_color = segment_colors[segment_idx]  # Get color for this segment

        # draw the robot segment
        cv2.polylines(
            img, [curve[segment_ps_selector]], isClosed=False, color=segment_color, thickness=segment_thickness
        )

    if x_obs is not None and R_obs is not None:
        # draw the obstacle
        uv_obs = onp.array((curve_origin + x_obs * ppm), dtype=onp.int32)
        # invert the v pixel coordinate
        uv_obs[1] = h - uv_obs[1]
        cv2.circle(img, tuple(uv_obs), int(R_obs * ppm), (0, 255, 0), thickness=-1)

    if p_des is not None:
        # draw the desired position as a cross
        uv_des = onp.array((curve_origin + p_des * ppm), dtype=onp.int32)
        # invert the v pixel coordinate
        uv_des[1] = h - uv_des[1]
        cross_size = int(10)  # size of the cross in pixels
        cv2.line(img, (uv_des[0] - cross_size, uv_des[1]), (uv_des[0] + cross_size, uv_des[1]), (0, 0, 255), 2)
        cv2.line(img, (uv_des[0], uv_des[1] - cross_size), (uv_des[0], uv_des[1] + cross_size), (0, 0, 255), 2)

    if poly_points is not None:
        poly_points_pixel = []
        for point in poly_points:
            uv_poly = onp.array((curve_origin + point * ppm), dtype=onp.int32)
            uv_poly[1] = h - uv_poly[1]

            poly_points_pixel.append(uv_poly)

        poly_points_pixel = onp.array(poly_points_pixel, dtype=onp.int32).reshape((-1, 1, 2))

        cv2.polylines(img, [poly_points_pixel], isClosed=True, color=(255,0,0), thickness=3)
        cv2.fillPoly(img, [poly_points_pixel], color=(255, 0, 0))
    return img