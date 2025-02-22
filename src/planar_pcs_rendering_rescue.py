__all__ = ["draw_image"]
import cv2  # importing cv2
from jax import Array, jacfwd, jit, lax, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Optional, Tuple, Union

def draw_image(
    batched_forward_kinematics_fn: Callable,
    batched_segment_robot: Callable,
    half_circle_to_polygon: Callable,
    auxiliary_fns: Dict[str, Callable],
    robot_params: Dict[str, Array],
    num_segments: int,
    q: Array,
    x_obs: Optional[Array] = None,
    R_obs: Optional[Union[float, Array]] = None,
    p_des: Optional[Array] = None,
    poly_points: Optional[Union[Array, list]] = None,  # supports list or single polygon
    img_width: int = 700,
    img_height: int = 700,
    num_points: int = 50, # original:50
    flag: Optional[int] = 0  # New parameter for collision flag
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
        poly_points: list of arrays, each array is a set of points (N,2) representing a polygon,
        img_width: image width
        img_height: image height
        num_points: number of points along the robot to plot
    """
    # compute the total length of the robot
    L = jnp.sum(robot_params["l"])

    # plotting in OpenCV
    h, w = img_height, img_width  # img height and width
    ppm = h / (2 * L)  # pixel per meter
    base_color = (0, 0, 0)  # black base color in BGR

    # dynamically generate a color palette for each segment
    segment_colors = [
        (int(255 * (i / num_segments)), int(255 * ((i + 1) / num_segments)), int(255 * ((i + 2) / num_segments)))
        for i in range(num_segments)
    ]

    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, L, num_points)
    robot_radius = 2e-2 #TODO: pass value direcly

    classify_segment_vmap = vmap(auxiliary_fns["classify_segment"], in_axes=(None, 0))
    segment_idx_ps, _ = classify_segment_vmap(robot_params, s_ps)

    # poses along the robot of shape (3, N)
    chi_ps = batched_forward_kinematics_fn(robot_params, q, s_ps)
    p_ps = chi_ps[:, :2]         # Positions, shape: (N, 2)
    p_orientation = chi_ps[:, 2]  # Orientation, shape: (N, 1)

    current_points = p_ps[:-1]
    next_points = jnp.concatenate([p_ps[1:-1] * 1, p_ps[-1][None, :]])
    orientations = p_orientation[:-1]
    
    # segmenting robots
    end_start = p_ps[-2]
    end_end = p_ps[-1]
    d = (end_end - end_start)/jnp.linalg.norm(end_end - end_start)
    angle = jnp.arctan2(d[1], d[0])

    robot_tip = half_circle_to_polygon(p_ps[-1],angle, robot_radius)
    robot_poly = batched_segment_robot(current_points,next_points,orientations,robot_radius)

    # the equilibrium distance between the backbone and the obstacle is the sum of the radii of the obstacle and the backbone
    r_backbone_ps = robot_params["r"][segment_idx_ps]

    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    curve_origin = onp.array(
        [w // 2, int(0.1 * h)], dtype=onp.int32
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

    # draw the segment robot
    robot_tip_pix = onp.array(robot_tip * ppm, dtype=onp.int32) + curve_origin
    robot_tip_pix[:, 1] = h - robot_tip_pix[:, 1]
    robot_tip_color = [255 , 0, 0]
    cv2.fillPoly(img, [robot_tip_pix], color=robot_tip_color)

    segment_poly_color = (0, 255, 0)  # For instance, green
    for poly in robot_poly:
        # Convert each polygon to pixel coordinates
        poly_pix = onp.array(poly * ppm, dtype=onp.int32) + curve_origin
        poly_pix[:, 1] = h - poly_pix[:, 1]
        # You can choose to fill or outline; here we fill the polygon:
        # cv2.fillPoly(img, [poly_pix], color=segment_poly_color)
        # Alternatively, to draw an outline use:
        cv2.polylines(img, [poly_pix], isClosed=True, color=segment_poly_color, thickness=1)
        
    if x_obs is not None and R_obs is not None:
        # draw the obstacle
        uv_obs = onp.array((curve_origin + x_obs * ppm), dtype=onp.int32)
        # invert the v pixel coordinate
        uv_obs[1] = h - uv_obs[1]
        cv2.circle(img, tuple(uv_obs), int(R_obs * ppm), (0, 255, 0), thickness=-1)

    if p_des is not None:
        p_des_np = onp.array(p_des)
        print(p_des_np)

        if p_des_np.ndim == 2:
            num_pd = p_des_np.shape[0]
            # Loop over each desired point with its index
            for i, pd in enumerate(p_des_np):
                # Compute a value between 0 and 1 based on the index.
                # When there is only one point, set the value to 0.
                alpha = i / (num_pd - 1) if num_pd > 1 else 0
                
                # Create a color gradient between red and blue.
                # Red in BGR is (0, 0, 255) and blue is (255, 0, 0).
                # When alpha = 0, the color is red; when alpha = 1, the color is blue.
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                
                uv_des = onp.array(curve_origin + pd * ppm, dtype=onp.int32)
                uv_des[1] = h - uv_des[1]
                cross_size = 10 
                cv2.line(img,
                        (uv_des[0] - cross_size, uv_des[1]),
                        (uv_des[0] + cross_size, uv_des[1]),
                        color, 2)
                cv2.line(img,
                        (uv_des[0], uv_des[1] - cross_size),
                        (uv_des[0], uv_des[1] + cross_size),
                        color, 2)
                print("Desired position:", pd)

        else:
            uv_des = onp.array(curve_origin + p_des_np * ppm, dtype=onp.int32)
            uv_des[1] = h - uv_des[1]
            cross_size = 10
            cv2.line(img,
                    (uv_des[0] - cross_size, uv_des[1]),
                    (uv_des[0] + cross_size, uv_des[1]),
                    (0, 0, 255), 2)
            cv2.line(img,
                    (uv_des[0], uv_des[1] - cross_size),
                    (uv_des[0], uv_des[1] + cross_size),
                    (0, 0, 255), 2)
            print("Desired position:", p_des_np)

    if poly_points is not None:
        # Transfering jnp array to onp
        if isinstance(poly_points, jnp.ndarray):
            poly_points = onp.array(poly_points)
        
        if not isinstance(poly_points, list):
            poly_points = [poly_points]
        
        colors = (255, 0, 0)
        alpha = 0.3

        for i, poly in enumerate(poly_points):
            if poly.ndim > 2:
                polys = poly
            else:
                polys = [poly]
            
            for poly_idx, single_poly in enumerate(polys):
                poly_points_pixel = []
                for point in single_poly:
                    uv_poly = onp.array(curve_origin + point * ppm, dtype=onp.int32)
                    uv_poly[1] = h - uv_poly[1]
                    poly_points_pixel.append(uv_poly)
                
                poly_points_pixel = onp.array(poly_points_pixel, dtype=onp.int32).reshape((-1, 1, 2))
                

                cv2.polylines(img, [poly_points_pixel], isClosed=True, color=colors, thickness=1)
                
                overlay = img.copy()
                cv2.fillPoly(overlay, [poly_points_pixel], color=colors)

                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    if flag is not None:
        indicator_radius = 20   # Radius of the indicator circle (in pixels)
        margin = 30            # Margin from the edges

        indicator_color = (255, 0, 0) if flag == 0 else (0, 255, 0) #red if collide, green if it is safe

        # Determine the top-right corner position (remembering that OpenCV uses (x, y) order)
        indicator_center = (img_width - indicator_radius - margin, indicator_radius + margin)
        cv2.circle(img, indicator_center, indicator_radius, indicator_color, thickness=-1)

    return img
