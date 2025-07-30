import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.spatial import ConvexHull
from skimage import measure
import jax.numpy as jnp
import jax
plt.rcParams['pdf.fonttype'] = 42

# 1) Define two convex polygons: A (robot) and B (obstacle)
A = np.array([[-0.5, -0.5],
              [ 0.5, -0.5],
              [ 0.5,  0.5],
              [-0.5,  0.5]])
B = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [2.0, 2.0],
    [0.0, 2.0]
])
B = np.array([
    [0.2, 0.1],
    [1.0, 0.0],
    [1.8, 0.3],
    [2.0, 1.0],
    [1.5, 1.8],
    [0.7, 2.0],
    [0.1, 1.5],
    [0.0, 0.5]
])

# 2) Shrink the robot polygon
scale = 0.5
A_shrunk = A * scale

# 3) Compute Minkowski sum of B and -A_shrunk for contact reference
A_reflect = -A_shrunk
sum_pts_shrunk = np.array([b + a for b in B for a in A_reflect])
hull_shrunk = ConvexHull(sum_pts_shrunk)
mink_shrunk_pts = sum_pts_shrunk[hull_shrunk.vertices]
mink_shrunk_closed = np.vstack([mink_shrunk_pts, mink_shrunk_pts[0]])
def compute_sat_distance(A: np.ndarray, B: np.ndarray) -> float:
    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        norms = np.linalg.norm(edges, axis=1)
        valid = norms > 1e-8  # 忽略长度为 0 的边
        edges = edges[valid]
        if edges.shape[0] == 0:
            raise ValueError("Degenerate polygon with no valid edges.")
        normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        return normals / np.linalg.norm(normals, axis=1, keepdims=True)

    if A.shape[0] < 3 or B.shape[0] < 3:
        raise ValueError("Polygons must have at least 3 vertices.")

    axes = np.vstack([get_normals(A), get_normals(B)])
    max_penetration = -np.inf
    found_valid_axis = False

    for axis in axes:
        if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-8:
            continue

        proj_A = A @ axis
        proj_B = B @ axis

        if not np.isfinite(proj_A).all() or not np.isfinite(proj_B).all():
            continue

        A_min, A_max = proj_A.min(), proj_A.max()
        B_min, B_max = proj_B.min(), proj_B.max()
        sep = max(B_min - A_max, A_min - B_max)

        found_valid_axis = True

        if sep > 0:
            return sep
        else:
            max_penetration = max(max_penetration, sep)

    if not found_valid_axis:
        raise ValueError("No valid axis found: check for NaNs or degenerate edges.")

    return max_penetration





# 4) One-log distance computation (NumPy)
def compute_distance_onelog_np(pt, robot_vertices, polygon_vertices, alpha):
    rv = robot_vertices + pt
    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.stack([-edges[:,1], edges[:,0]], axis=1)
        return normals / np.linalg.norm(normals, axis=1, keepdims=True)
    Rn = get_normals(rv)
    Pn = get_normals(polygon_vertices)
    axes = np.vstack((Rn, Pn))
    proj_R = rv @ axes.T
    proj_P = polygon_vertices @ axes.T
    R_min, R_max = proj_R.min(axis=0), proj_R.max(axis=0)
    P_min, P_max = proj_P.min(axis=0), proj_P.max(axis=0)
    gaps = np.hstack((P_min - R_max, R_min - P_max))
    return (1.0 / alpha) * logsumexp(alpha * gaps)


def compute_distance_onelog_np(pt, robot_vertices, polygon_vertices, alpha):
    rv = robot_vertices + pt
    
    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        return normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    Rn = get_normals(rv)
    Pn = get_normals(polygon_vertices)
    axes = np.vstack((Rn, Pn))  # |𝓐| = number of axes
    proj_R = rv @ axes.T
    proj_P = polygon_vertices @ axes.T
    R_min, R_max = proj_R.min(axis=0), proj_R.max(axis=0)
    P_min, P_max = proj_P.min(axis=0), proj_P.max(axis=0)
    
    gaps = np.hstack((P_min - R_max, R_min - P_max))  # all signed separations
    h_olsat = (1.0 / alpha) * logsumexp(alpha * gaps)

    # Add error bound term: log(2|𝓐|) / alpha
    error_bound = np.log(2 * axes.shape[0]) / alpha 
    
    return h_olsat - error_bound


# from jax.scipy.special import logsumexp
# def smooth_abs(x, alpha):
#     return x * jnp.tanh(alpha * x)
# @jax.jit
# def compute_distance_onelog_np(pt,robot, poly, alpha_axes, alpha_sabs=1000.):
#     robot += pt
#     """
#     Distance using smooth absolute value replacement.
#     """
#     def get_normals(v):
#         e = jnp.roll(v, -1, axis=0) - v
#         n = jnp.stack([-e[:,1], e[:,0]], axis=1)
#         return n / jnp.linalg.norm(n, axis=1, keepdims=True)
    
#     # Centroids
#     p_robot = robot.mean(axis=0)
#     p_poly  = poly.mean(axis=0)
#     delta_p = p_poly - p_robot

#     # Normals
#     axes = jnp.concatenate([get_normals(robot), get_normals(poly)], axis=0)

#     def half_extent(v, axis):
#         projections = (v - v.mean(axis=0)) @ axis
#         return smooth_abs(projections, alpha_sabs).max()

#     axis_gaps = []
#     for ni in axes:
#         proj_dist = smooth_abs(ni @ delta_p, alpha_sabs)
#         rA = half_extent(robot, ni)
#         rB = half_extent(poly, ni)
#         gap_i = proj_dist - (rA + rB)
#         axis_gaps.append(gap_i)

#     axis_gaps = jnp.stack(axis_gaps)
#     h = (1 / alpha_axes) * logsumexp(alpha_axes * axis_gaps)
#     return h

# 5) Sampling grid
N = 300
x_min, x_max = -0.8, 2.8
y_min, y_max = -0.8, 2.8
xs = np.linspace(x_min, x_max, N)
ys = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(xs, ys)

# 6) Extract contours for multiple alphas using the specified colors
alphas = [5,10,50]
palette = [
    (90/255, 174/255, 52/255),
    (109/255, 131/255, 250/255),
    (206/255, 101/255, 95/255)
]
contours_dict = {}
for alpha, col in zip(alphas, palette):
    Z = np.zeros_like(X)
    for i in range(N):
        for j in range(N):
            Z[i, j] = compute_distance_onelog_np(np.array([X[i, j], Y[i, j]]), A_shrunk, B, alpha)
    segs = measure.find_contours(Z, level=0.0)
    if segs:
        seg = max(segs, key=lambda seg: seg.shape[0])
        cont = np.column_stack([
            np.interp(seg[:,1], np.arange(N), xs),
            np.interp(seg[:,0], np.arange(N), ys)
        ])
        contours_dict[alpha] = (cont, col)

# 7) Choose rightmost Minkowski point for contact
contact_idx = np.argmax(mink_shrunk_pts[:,0])
contact_t = mink_shrunk_pts[contact_idx]
A_contact = A_shrunk + contact_t

# 8) Plot
plt.figure(figsize=(6,6))
plt.plot(mink_shrunk_closed[:,0], mink_shrunk_closed[:,1], '--', color='gray', label='SAT Path')
for alpha, (cont, col) in contours_dict.items():
    plt.plot(cont[:,0], cont[:,1], label=f'α={alpha}', color=col, linewidth=1.5)
# Obstacle (no legend entry)
plt.plot(*np.vstack([B, B[0]]).T, '-', color='black', linewidth=2)
# Robot (no legend entry)
plt.plot(*np.vstack([A_contact, A_contact[0]]).T, '-.', color='black', linewidth=2)

# Legend with larger font
legend = plt.legend(loc='upper right', fontsize=14)
legend.set_frame_on(False)

plt.axis('equal')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
# Zoom into the top-right region
plt.xlim(-0.5, 3.2)
plt.ylim(-0.5, 3.2)
plt.savefig("sat_contour_plot.pdf", format='pdf', bbox_inches='tight')


import matplotlib.animation as animation

# === Step 1: choose the contour to follow ===
alpha_to_follow = 50
contour, _ = contours_dict[alpha_to_follow]

# Downsample the contour to make animation faster
num_frames = 100
indices = np.linspace(0, contour.shape[0]-1, num_frames).astype(int)
path_points = contour[indices]  # (num_frames, 2)

# === Step 2: setup animation ===
fig, ax = plt.subplots(figsize=(6, 6))

# Plot background: obstacle, contours, and Minkowski sum
ax.plot(mink_shrunk_closed[:,0], mink_shrunk_closed[:,1], '--', color='gray', label='SAT Path')
ax.plot(contour[:, 0], contour[:, 1], label=f'α={alpha_to_follow}', color='red', linewidth=1.5)
ax.plot(*np.vstack([B, B[0]]).T, '-', color='black', linewidth=2)  # Obstacle

# Legend and axes
legend = ax.legend(loc='upper right', fontsize=13)
legend.set_frame_on(False)
ax.axis('equal')
ax.set_xlim(-0.5, 3.2)
ax.set_ylim(-0.5, 3.2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)

# === Step 3: animated robot and text ===
robot_patch, = ax.plot([], [], '-.', color='red', linewidth=2)
text_handle = ax.text(3.1, 2.7, '', ha='right', va='top', fontsize=12, weight='bold')

# === Step 4: SAT distance function ===
def compute_sat_distance(A: np.ndarray, B: np.ndarray) -> float:
    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        norms = np.linalg.norm(edges, axis=1)
        valid = norms > 1e-8
        edges = edges[valid]
        if edges.shape[0] == 0:
            raise ValueError("Degenerate polygon with no valid edges.")
        normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        return normals / np.linalg.norm(normals, axis=1, keepdims=True)

    if A.shape[0] < 3 or B.shape[0] < 3:
        raise ValueError("Polygons must have at least 3 vertices.")

    axes = np.vstack([get_normals(A), get_normals(B)])
    max_penetration = -np.inf
    found_valid_axis = False

    for axis in axes:
        if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-8:
            continue

        proj_A = A @ axis
        proj_B = B @ axis

        if not np.isfinite(proj_A).all() or not np.isfinite(proj_B).all():
            continue

        A_min, A_max = proj_A.min(), proj_A.max()
        B_min, B_max = proj_B.min(), proj_B.max()
        sep = max(B_min - A_max, A_min - B_max)

        found_valid_axis = True

        if sep > 0:
            return sep
        else:
            max_penetration = max(max_penetration, sep)

    if not found_valid_axis:
        raise ValueError("No valid axis found: input polygons may be degenerate or contain NaNs.")

    return max_penetration

# === Step 5: update function ===
def update(frame):
    t = path_points[frame]
    A_moved = A_shrunk + t
    closed = np.vstack([A_moved, A_moved[0]])
    robot_patch.set_data(closed[:, 0], closed[:, 1])

    # SAT distance
    dist = compute_sat_distance(A_moved, B)
    color = 'green' if dist > 0 else 'red'
    text_handle.set_text(f"SAT distance: {dist:.2f}")
    text_handle.set_color(color)

    return robot_patch, text_handle

# === Step 6: run animation ===
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
plt.show()

# === Step 7: save as GIF ===
ani.save("SASAT.gif", writer="pillow", fps=10, dpi=100)
