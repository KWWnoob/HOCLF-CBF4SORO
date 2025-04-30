import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.spatial import ConvexHull
from skimage import measure
from shapely.geometry import Polygon
import jax
from jax import lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import matplotlib.pyplot as plt


# 1) Define two convex polygons: A (robot) and B (obstacle)
# your original robot polygon (centered on origin)
A = np.array([[-0.5, -0.5],
              [ 0.5, -0.5],
              [ 0.5,  0.5],
              [-0.5,  0.5]])

# 1) compute a 10°‐rotation matrix
θ = np.deg2rad(0.0)
R = np.array([[ np.cos(θ), -np.sin(θ)],
              [ np.sin(θ),  np.cos(θ)]])

# 2) rotate A about the origin
A = A @ R.T
B = np.array([[ 0.3,  0.2],
              [ 1.3,  0.2],
              [ 1.3,  1.2],
              [ 0.3,  1.2]])

B= np.array([
    [0.2, 0.1],
    [1.0, 0.0],
    [1.8, 0.3],
    [2.0, 1.0],
    [1.5, 1.8],
    [0.7, 2.0],
    [0.1, 1.5],
    [0.0, 0.5]
])

# 2) Compute Minkowski sum of B and -A for reference
A_reflect = -A
sum_pts = np.array([b + a for b in B for a in A_reflect])
hull = ConvexHull(sum_pts)
mink_pts = sum_pts[hull.vertices]
mink_closed = np.vstack([mink_pts, mink_pts[0]])  # closed loop

# 2) Helper: edge normals

def compute_distance_twolog(
    robot_vertices: jnp.ndarray,
    polygon_vertices: jnp.ndarray,
    alpha_pair: float = 2080,   
    alpha_axes: float = 9000.0  
) -> tuple[jnp.ndarray, jnp.ndarray]:

    def get_normals(vertices):
        edges = jnp.roll(vertices, -1, axis=0) - vertices
        normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
        return normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
    
    Rn = get_normals(robot_vertices)     
    Pn = get_normals(polygon_vertices)  
    axes = jnp.concatenate([Rn, Pn], axis=0)  

    proj_R = robot_vertices @ axes.T    
    proj_P = polygon_vertices @ axes.T   
    
    R_min, R_max = jnp.min(proj_R, axis=0), jnp.max(proj_R, axis=0)  # (K,)
    P_min, P_max = jnp.min(proj_P, axis=0), jnp.max(proj_P, axis=0)  # (K,)

    d1 = P_min - R_max  
    d2 = R_min - P_max  
    
    pair_gaps = jnp.stack([d1, d2], axis=0)  # (2, K)
    axis_gaps = (1.0/alpha_pair) * logsumexp(alpha_pair * pair_gaps, axis=0)  # (K,)

    h = (1.0/alpha_axes) * logsumexp(alpha_axes * axis_gaps)
    
    separation_flag = jnp.where(h > 0, 1, 0)
    
    return h, separation_flag


@jax.jit
def compute_distance_onelog(
    pt,
    robot_vertices: jnp.ndarray,
    polygon_vertices: jnp.ndarray,
    alpha_axes: float = 9000.0  
) -> tuple[jnp.ndarray, jnp.ndarray]:
    robot_vertices += pt

    def get_normals(vertices):
        edges = jnp.roll(vertices, -1, axis=0) - vertices
        normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
        return normals / jnp.linalg.norm(normals, axis=1, keepdims=True)

    Rn = get_normals(robot_vertices)
    Pn = get_normals(polygon_vertices)
    axes = jnp.concatenate([Rn, Pn], axis=0)
      
    proj_R = robot_vertices @ axes.T
    proj_P = polygon_vertices @ axes.T
    R_min, R_max = jnp.min(proj_R, axis=0), jnp.max(proj_R, axis=0)
    P_min, P_max = jnp.min(proj_P, axis=0), jnp.max(proj_P, axis=0)

    d1 = P_min - R_max
    d2 = R_min - P_max

    pair_gaps = jnp.stack([d1, d2], axis=0)   # shape (2, K)
    flat_gaps = pair_gaps.reshape(-1)         # shape (2*K,)
    h = (1.0 / alpha_axes) * logsumexp(alpha_axes * flat_gaps)
    
    return h


# 4) Sampling grid  
N = 300
x_min, x_max = -2.0, 4.0
y_min, y_max = -2.0, 4.0
xs = np.linspace(x_min, x_max, N)
ys = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(xs, ys)

# 5) Evaluate both metrics on the grid
Z1 = np.zeros((N, N))
Z2 = np.zeros((N, N))
Z3 = np.zeros((N, N))
Z4 = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        pt = np.array([X[i,j], Y[i,j]])
        # Z1[i,j] = sat_distance_np(pt, A, B)
        # Z2[i,j] = xtanh_signed_distance_np(pt, A, B)
        # Z3[i,j] = SKSAT_compute_distance(pt,A,B)
        Z4[i,j] = compute_distance_onelog(pt,A,B)


# 6) Print ranges to confirm zero‐crossing
# print(f"SAT range:   [{Z1.min():.3f}, {Z1.max():.3f}]")
# print(f"XTANH range: [{Z2.min():.3f}, {Z2.max():.3f}]")
# print(f"SKSAT range: [{Z3.min():.3f}, {Z3.max():.3f}]")

# 7) Extract zero‐level contours
# contours1 = measure.find_contours(Z1, level=0.0)
# contours2 = measure.find_contours(Z2, level=0.0)
# contours3 = measure.find_contours(Z3, level=0.0)
contours4 = measure.find_contours(Z4, level=0.0)
# if not contours1 or not contours2:
#     raise RuntimeError("No zero‐level contour found; adjust sampling bounds or parameters.")
# seg1 = max(contours1, key=lambda seg: seg.shape[0])
# seg2 = max(contours2, key=lambda seg: seg.shape[0])
# seg3 = max(contours3, key=lambda seg: seg.shape[0])
seg4 = max(contours4, key=lambda seg: seg.shape[0])


# 8) Convert to (x,y) coords
# cont1 = np.stack([np.interp(seg1[:,1], np.arange(N), xs),
#                   np.interp(seg1[:,0], np.arange(N), ys)], axis=1)
# cont2 = np.stack([np.interp(seg2[:,1], np.arange(N), xs),
#                   np.interp(seg2[:,0], np.arange(N), ys)], axis=1)
# cont3 = np.stack([np.interp(seg3[:,1], np.arange(N), xs),
#                   np.interp(seg3[:,0], np.arange(N), ys)], axis=1)
cont4 = np.stack([np.interp(seg4[:,1], np.arange(N), xs),
                  np.interp(seg4[:,0], np.arange(N), ys)], axis=1)


# 9) Plot overlay
plt.figure(figsize=(6,6))
# plt.plot(cont1[:,0], cont1[:,1], 'C0-', label='SAT contour')
# plt.plot(cont2[:,0], cont2[:,1], 'C1-', label='XTANH contour')
# plt.plot(cont3[:,0], cont3[:,1], 'C2-', label='SKSAT contour')
plt.plot(cont4[:,0], cont4[:,1], 'C3-', label='KSAT contour')
plt.plot(mink_closed[:,0], mink_closed[:,1], 'k--', label='Minkowski sum boundary')
plt.plot(*np.vstack([B, B[0]]).T, 'k-', label='Obstacle B')
plt.plot(*np.vstack([A, A[0]]).T, 'k:', label='Robot A')
plt.legend()
plt.axis('equal')
plt.xlabel('x'); plt.ylabel('y')
plt.title("Zero‐Level Contour Comparison")
plt.show()

# 10) Quantitative metrics
poly1 = Polygon(cont1)
poly2 = Polygon(cont2)
area1, area2 = poly1.area, poly2.area
iou = poly1.intersection(poly2).area / poly1.union(poly2).area
haus = poly1.hausdorff_distance(poly2)

print(f"Area SAT = {area1:.4f}, XTANH = {area2:.4f}, Δ = {abs(area1-area2):.4f}")
print(f"IoU = {iou:.4f}, Hausdorff = {haus:.4f}")
