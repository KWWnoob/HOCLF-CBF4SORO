import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.spatial import ConvexHull
from skimage import measure

# 1) Define two convex polygons: A (robot) and B (obstacle)
A = np.array([[-0.5, -0.5],
              [ 0.5, -0.5],
              [ 0.5,  0.5],
              [-0.5,  0.5]])
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

# 5) Sampling grid
N = 300
x_min, x_max = -0.8, 2.8
y_min, y_max = -0.8, 2.8
xs = np.linspace(x_min, x_max, N)
ys = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(xs, ys)

# 6) Extract contours for multiple alphas using the specified colors
alphas = [3, 7, 50]
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
plt.plot(mink_shrunk_closed[:,0], mink_shrunk_closed[:,1], '--', color='gray', label='Reference Path')
for alpha, (cont, col) in contours_dict.items():
    plt.plot(cont[:,0], cont[:,1], label=f'α={alpha}', color=col, linewidth=1.5)
# Obstacle (no legend entry)
plt.plot(*np.vstack([B, B[0]]).T, '-', color='black', linewidth=2)
# Robot (no legend entry)
plt.plot(*np.vstack([A_contact, A_contact[0]]).T, '-.', color='black', linewidth=2)

# Legend with larger font
legend = plt.legend(loc='upper right', fontsize=13)
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
plt.show()


