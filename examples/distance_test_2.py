import numpy as np
import matplotlib.pyplot as plt

def draw_localized_gradient_contour(size, center, radius,
                                    marker_angle=90, fade_width=30,
                                    fade_value=1.0, num_points=360,
                                    point_size=4):
    """
    Draw only the circular contour with a local gradient around a marker point:
    - Pure red at marker_angle
    - Fades to pure blue within ±fade_width degrees
    - Rest of the circle remains pure blue
    """
    H, W = size
    # Sample full circle angles
    angles = np.linspace(0, 360, num_points, endpoint=False)
    xs = center[0] + radius * np.cos(np.radians(angles))
    ys = center[1] + radius * np.sin(np.radians(angles))
    
    # Compute minimal angular distance (0–180)
    delta = np.abs((angles - marker_angle + 180) % 360 - 180)
    
    # Initialize all points to pure blue
    colors = np.tile(np.array([0, 0, 1.0]), (num_points, 1))
    
    # Identify points within fade_width and blend red -> blue
    mask = delta <= fade_width
    # normalized fade parameter (0 at marker, 1 at edge of fade width)
    norm = (delta[mask] / fade_width).clip(0, 1)
    alpha = norm**fade_value  # apply exponent curve
    colors[mask, 0] = 1 - alpha  # red channel
    colors[mask, 2] = alpha      # blue channel
    
    # Plot
    plt.figure(figsize=(5,5))
    plt.scatter(xs, ys, c=colors, s=point_size, marker='o')
    plt.xlim(0, W)
    plt.ylim(H, 0)
    plt.axis('off')
    plt.show()


def draw_gradient_rectangle_contour(size, rect, fade_distance, fade_value=1.0,
                                    marker_fraction=0.0, num_points=400, point_size=4):
    """
    Draw the contour of a rectangle with a localized gradient from red to blue.
    
    - size: (H, W) canvas size in pixels
    - rect: (x0, y0, width, height) rectangle parameters
    - fade_distance: distance (in pixels) along perimeter to fade red→blue
    - fade_value: exponent controlling fade curve (1 = linear)
    - marker_fraction: fraction [0,1) of the perimeter where pure red appears
    - num_points: total points sampled around the perimeter
    - point_size: marker size
    """
    H, W = size
    x0, y0, w, h = rect
    # Total perimeter
    L = 2 * (w + h)
    # Sample cumulative distances along perimeter
    ds = np.linspace(0, L, num_points, endpoint=False)
    
    # Map cumulative distance to (x,y) on rectangle boundary
    xs = np.empty(num_points)
    ys = np.empty(num_points)
    for i, d in enumerate(ds):
        d_mod = d % L
        if d_mod < w:
            xs[i] = x0 + d_mod;   ys[i] = y0
        elif d_mod < w + h:
            xs[i] = x0 + w;       ys[i] = y0 + (d_mod - w)
        elif d_mod < 2*w + h:
            xs[i] = x0 + (w - (d_mod - w - h)); ys[i] = y0 + h
        else:
            xs[i] = x0;           ys[i] = y0 + (h - (d_mod - 2*w - h))
    
    # Marker location along perimeter
    marker_d = (marker_fraction % 1) * L
    # Compute minimal distance along the loop
    delta = np.abs(ds - marker_d)
    delta = np.minimum(delta, L - delta)
    
    # Initialize blue color
    colors = np.tile(np.array([0, 0, 1.0]), (num_points, 1))
    # Fade mask
    mask = delta <= fade_distance
    norm = (delta[mask] / fade_distance).clip(0, 1)
    alpha = norm**fade_value
    # Blend red⊻blue
    colors[mask, 0] = 1 - alpha  # red channel
    colors[mask, 2] = alpha      # blue channel
    
    # Plot
    plt.figure(figsize=(5,5))
    plt.scatter(xs, ys, c=colors, s=point_size, marker='o')
    plt.xlim(0, W)
    plt.ylim(H, 0)
    plt.axis('off')
    plt.show()

# # Example usage:
# draw_localized_gradient_contour(
#     size=(400,400), center=(200,200), radius=120,
#     marker_angle=0, fade_width=100, fade_value=1.0, num_points=360)

# Example usage:
canvas_size = (400, 400)
# rect at (100,100) of width 200, height 120
rect_params = (100, 100, 200, 120)
draw_gradient_rectangle_contour(
    size=canvas_size,
    rect=rect_params,
    fade_distance=50,    # fade within 50 pixels each side
    fade_value=1.0,      # linear fade
    marker_fraction=0,# red at 25% along perimeter
    num_points=500,
    point_size=3
)
