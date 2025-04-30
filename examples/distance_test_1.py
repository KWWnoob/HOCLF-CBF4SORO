import numpy as np
import matplotlib.pyplot as plt

def connect_centroids_project(polyA: np.ndarray, polyB: np.ndarray):
    """
    1) Compute centroids of polyA and polyB.
    2) Draw the ray from centroid A to centroid B.
    3) Find the first intersection of that ray with polyB's boundary.
    Returns:
      cA, cB     -- centroids of A and B
      hit_point  -- point on B where the cA→cB ray first intersects
    """
    # -- Compute centroid of a polygon --
    def centroid(poly: np.ndarray) -> np.ndarray:
        area = 0.0
        cx = 0.0
        cy = 0.0
        N = len(poly)
        for i in range(N):
            x0, y0 = poly[i]
            x1, y1 = poly[(i+1) % N]
            cross = x0*y1 - x1*y0
            area += cross
            cx += (x0 + x1)*cross
            cy += (y0 + y1)*cross
        area *= 0.5
        cx /= (6*area)
        cy /= (6*area)
        return np.array([cx, cy])

    # centroids
    cA = centroid(polyA)
    cB = centroid(polyB)
    v = cB - cA  # direction vector from A to B

    # -- Ray-segment intersection loop --
    best_t = np.inf
    hit = None
    for i in range(len(polyB)):
        A_edge = polyB[i]
        B_edge = polyB[(i+1) % len(polyB)]
        d = B_edge - A_edge

        M = np.column_stack((v, -d))
        if np.linalg.matrix_rank(M) < 2:
            continue

        t, u = np.linalg.solve(M, A_edge - cA)
        # t >=0 → along ray direction; 0<=u<=1 → within the segment
        if (t >= 0) and (0 <= u <= 1) and (t < best_t):
            best_t = t
            hit = cA + t*v

    return cA, cB, hit

# ——— Example usage & visualization ———
polyA = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
polyB = polyA + np.array([1.5, 0.5])

cA, cB, hit_pt = connect_centroids_project(polyA, polyB)

plt.figure(figsize=(6,6))
plt.plot(*np.vstack([polyA, polyA[0]]).T, '-', label='Polygon A', color='C0')
plt.plot(*np.vstack([polyB, polyB[0]]).T, '-', label='Polygon B', color='C1')
plt.scatter(*cA, s=100, marker='o', color='C0', label='Centroid A')
plt.scatter(*cB, s=100, marker='o', color='C1', label='Centroid B')
plt.plot([cA[0], cB[0]], [cA[1], cB[1]], '--', color='gray', label='Centroid Ray')
if hit_pt is not None:
    plt.scatter(*hit_pt, s=100, marker='x', color='red', label='Hit on B')
plt.axis('equal')
plt.legend()
plt.title('Centroid A → Centroid B Ray Intersecting Polygon B')
plt.show()
