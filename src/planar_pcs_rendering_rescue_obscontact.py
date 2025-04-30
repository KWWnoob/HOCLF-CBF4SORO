import cv2
from jax import Array, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Optional, Tuple, Union

# Compute normalized position along rectangle perimeter
def compute_marker_fraction(
    rect: Tuple[int, int, int, int],
    marker: Tuple[int, int]
) -> float:
    x0, y0, w, h = rect
    mx, my = marker
    L = 2 * (w + h)
    if my == y0:
        d0 = mx - x0
    elif mx == x0 + w:
        d0 = w + (my - y0)
    elif my == y0 + h:
        d0 = w + h + (x0 + w - mx)
    else:
        d0 = 2 * w + h + (y0 + h - my)
    return (d0 % L) / L

# Draw red→blue gradient along full rectangle contour
def draw_gradient_rectangle_contour_cv2(
    img: onp.ndarray,
    rect: Tuple[int, int, int, int],
    fade_distance: float,
    fade_value: float = 1.0,
    marker_fraction: float = 0.0,
    num_points: int = 400,
    point_radius: int = 1,
    thickness: int = -1
) -> None:
    x0, y0, w, h = rect
    L = 2 * (w + h)
    ds = onp.linspace(0, L, num_points, endpoint=False)
    marker_d = (marker_fraction % 1) * L
    delta = onp.abs(ds - marker_d)
    delta = onp.minimum(delta, L - delta)
    colors = onp.zeros((num_points, 3), dtype=onp.float32)
    colors[:, 2] = 1.0  # default: red

    mask = (delta <= fade_distance)
    t = (delta[mask] / fade_distance) ** fade_value

    # blue → red
    colors[mask, 0] = 1.0 - t  # blue
    colors[mask, 2] = t        # red
    for d, (b, g, r) in zip(ds, colors):
        d_mod = d % L
        if d_mod < w:
            px, py = x0 + d_mod, y0
        elif d_mod < w + h:
            px, py = x0 + w, y0 + (d_mod - w)
        elif d_mod < 2*w + h:
            px = x0 + (w - (d_mod - w - h)); py = y0 + h
        else:
            px, py = x0, y0 + (h - (d_mod - 2*w - h))
        cv2.circle(
            img,
            (int(px), int(py)),
            radius=point_radius,
            color=(int(b*255), int(g*255), int(r*255)),
            thickness=thickness
        )


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
    poly_points: Optional[Union[Array, list]] = None,
    img_width: int = 700,
    img_height: int = 700,
    num_points: int = 10,
    flag: Optional[int] = 0,
    contact_points: Optional[Array] = None 
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
    ppm = h / (4 * L)  # pixel per meter
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

    if poly_points is not None:
        contact_px = []
        if contact_points is not None:
            for pt in onp.array(contact_points):
                uv_f = curve_origin + pt * float(ppm)
                uv = onp.array(uv_f, dtype=onp.int32)
                uv[1] = img_height - uv[1]
                contact_px.append((int(uv[0]), int(uv[1])))

        for poly in poly_points:
            single = onp.array(poly).reshape(-1, 2)

            pix_f = curve_origin + single * float(ppm)
            pix = onp.array(pix_f, dtype=onp.int32)
            pix[:, 1] = img_height - pix[:, 1]
            pts = pix.reshape(-1, 1, 2)

            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color=(255, 0, 0))
            img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)

            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

            if single.shape[0] == 4:
                xs, ys = pix[:, 0], pix[:, 1]
                x0, y0 = int(xs.min()), int(ys.min())
                w_rect, h_rect = int(xs.max() - xs.min()), int(ys.max() - ys.min())
                rect = (x0, y0, w_rect, h_rect)
                for mx, my in contact_px:

                    on_edge = (
                        (my == y0 and x0 <= mx <= x0 + w_rect) or
                        (my == y0 + h_rect and x0 <= mx <= x0 + w_rect) or
                        (mx == x0 and y0 <= my <= y0 + h_rect) or
                        (mx == x0 + w_rect and y0 <= my <= y0 + h_rect)
                    )
                    if on_edge:
                        draw_gradient_rectangle_contour_cv2(
                            img,              # your BGR image
                            rect,             # (x0, y0, w, h)
                            100,    # how many pixels along the entire perimeter to fade
                            fade_value=1.0,
                            marker_fraction=0.0,  # fraction of the perimeter at which pure red sits
                            num_points=4000,
                            point_radius=1,
                            thickness=-1
                        )
            else:
                pix = onp.array(curve_origin + single * ppm, dtype=onp.int32)
                pix = pix.copy()
                pix[:,1] = img_height - pix[:,1]
                arr = pix.reshape(-1,1,2)
                cv2.fillPoly(img, [arr], color=(255,0,0))
    

    if flag is not None:
        indicator_radius = 20   # Radius of the indicator circle (in pixels)
        margin = 30            # Margin from the edges

        indicator_color = (255, 0, 0) if flag == 0 else (0, 255, 0) #red if collide, green if it is safe

        # Determine the top-right corner position (remembering that OpenCV uses (x, y) order)
        indicator_center = (img_width - indicator_radius - margin, indicator_radius + margin)
        cv2.circle(img, indicator_center, indicator_radius, indicator_color, thickness=-1)

    #  # ——— Draw contact points as red "X" markers ———
    # if contact_points is not None:
    #     cp = onp.array(contact_points)
    #     for pt in cp:
    #         uv = onp.array(curve_origin + pt * ppm, dtype=onp.int32)
    #         uv[1] = h - uv[1]
    #         cross_size = 8  # size of the "X"
    #         color = (255, 0, 0)  # red color in BGR
    #         thickness = 2
    #         # draw \
    #         cv2.line(img,
    #                  (uv[0] - cross_size, uv[1] - cross_size),
    #                  (uv[0] + cross_size, uv[1] + cross_size),
    #                  color, thickness)
    #         # draw /
    #         cv2.line(img,
    #                  (uv[0] - cross_size, uv[1] + cross_size),
    #                  (uv[0] + cross_size, uv[1] - cross_size),
    #                  color, thickness)
            
    return img
