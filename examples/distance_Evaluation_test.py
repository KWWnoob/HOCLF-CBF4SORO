    import time
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    # ——— Distance function implementations ———

    def smooth_abs(x, alpha):
        return x * jnp.tanh(alpha * x)
    @jax.jit
    def compute_distance_smoothabs(robot, poly, alpha_axes=9000., alpha_sabs=1000.):
        """
        Distance using smooth absolute value replacement.
        """
        def get_normals(v):
            e = jnp.roll(v, -1, axis=0) - v
            n = jnp.stack([-e[:,1], e[:,0]], axis=1)
            return n / jnp.linalg.norm(n, axis=1, keepdims=True)
        
        # Centroids
        p_robot = robot.mean(axis=0)
        p_poly  = poly.mean(axis=0)
        delta_p = p_poly - p_robot

        # Normals
        axes = jnp.concatenate([get_normals(robot), get_normals(poly)], axis=0)

        def half_extent(v, axis):
            projections = (v - v.mean(axis=0)) @ axis
            return smooth_abs(projections, alpha_sabs).max()

        axis_gaps = []
        for ni in axes:
            proj_dist = smooth_abs(ni @ delta_p, alpha_sabs)
            rA = half_extent(robot, ni)
            rB = half_extent(poly, ni)
            gap_i = proj_dist - (rA + rB)
            axis_gaps.append(gap_i)

        axis_gaps = jnp.stack(axis_gaps)
        h = (1 / alpha_axes) * logsumexp(alpha_axes * axis_gaps)
        return h

    def compute_distance_twolog(robot, poly, alpha_pair=2080., alpha_axes=9000.):
        """
        Two-step smooth distance using nested LogSumExp:
        1) Smooth max between penetration and separation on each axis.
        2) Smooth max across all axes.
        """
        def get_normals(v):
            # Compute outward normals of each edge
            e = jnp.roll(v, -1, axis=0) - v
            n = jnp.stack([-e[:,1], e[:,0]], axis=1)
            return n / jnp.linalg.norm(n, axis=1, keepdims=True)
        Rn = get_normals(robot)
        Pn = get_normals(poly)
        axes = jnp.concatenate([Rn, Pn], axis=0)
        prj_R = robot @ axes.T
        prj_P = poly  @ axes.T
        Rmin, Rmax = prj_R.min(0), prj_R.max(0)
        Pmin, Pmax = prj_P.min(0), prj_P.max(0)
        d1 = Pmin - Rmax  # separation in forward direction
        d2 = Rmin - Pmax  # separation in reverse direction
        # Smooth max over d1, d2 for each axis
        axis_gaps = (1/alpha_pair) * logsumexp(alpha_pair * jnp.stack([d1, d2]), axis=0)
        # Smooth max over all axes
        h = (1/alpha_axes) * logsumexp(alpha_axes * axis_gaps)
        return h

    @jax.jit
    def compute_distance_onelog(robot, poly, alpha_axes=9000.):
        """
        Single-step smooth distance using one LogSumExp over all axis gaps.
        JIT-compiled for efficiency.
        """
        def get_normals(v):
            e = jnp.roll(v, -1, axis=0) - v
            n = jnp.stack([-e[:,1], e[:,0]], axis=1)
            return n / jnp.linalg.norm(n, axis=1, keepdims=True)
        Rn = get_normals(robot)
        Pn = get_normals(poly)
        axes = jnp.concatenate([Rn, Pn], axis=0)
        prj_R = robot @ axes.T
        prj_P = poly  @ axes.T
        d1 = prj_P.min(0) - prj_R.max(0)
        d2 = prj_R.min(0) - prj_P.max(0)
        flat = jnp.stack([d1, d2]).reshape(-1)
        # Smooth max across all gaps
        return (1.0/alpha_axes) * logsumexp(alpha_axes * flat)

    @jax.jit
    def compute_distance_onelogminus(robot, poly, alpha=9000.0):
        def get_normals(v):
            e = jnp.roll(v, -1, axis=0) - v
            n = jnp.stack([-e[:, 1], e[:, 0]], axis=1)
            return n / jnp.linalg.norm(n, axis=1, keepdims=True)

        Rn = get_normals(robot)
        Pn = get_normals(poly)
        axes = jnp.concatenate([Rn, Pn], axis=0)

        proj_R = robot @ axes.T
        proj_P = poly   @ axes.T
        R_min, R_max = jnp.min(proj_R, axis=0), jnp.max(proj_R, axis=0)
        P_min, P_max = jnp.min(proj_P, axis=0), jnp.max(proj_P, axis=0)

        gaps = jnp.concatenate([P_min - R_max, R_min - P_max], axis=0)
        h_olsat = (1.0 / alpha) * logsumexp(alpha * gaps)

        error_bound = jnp.log(2.0 * axes.shape[0]) / alpha
        return h_olsat - error_bound
    @jax.jit
    def compute_distance_onelogminus(robot_vertices, polygon_vertices, alpha_outer=500, alpha_inner=500):
        """
        Fully smoothed OLSAT: smooth min/max for d1, d2 and smooth outer max.

        Parameters:
        - pt: (2,) np.ndarray, shift of robot.
        - robot_vertices: (N, 2) np.ndarray.
        - polygon_vertices: (M, 2) np.ndarray.
        - alpha_outer: float, outer sharpness parameter (for outer max).
        - alpha_inner: float, inner sharpness parameter (for min/max in d1/d2).

        Returns:
        - smooth OLSAT distance (float)
        """
        rv = robot_vertices

        def get_normals(verts):
            edges = jnp.roll(verts, -1, axis=0) - verts
            normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
            return normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
        
        # Get all axes from robot and polygon
        Rn = get_normals(rv)
        Pn = get_normals(polygon_vertices)
        axes = jnp.vstack((Rn, Pn))  # Shape (K, 2)

        # Project onto each axis
        proj_R = rv @ axes.T  # (N, K)
        proj_P = polygon_vertices @ axes.T  # (M, K)

        # Smooth min/max along axis dimension (for each axis separately)
        R_max = (1.0 / alpha_inner) * logsumexp(alpha_inner * proj_R, axis=0)
        R_min = -(1.0 / alpha_inner) * logsumexp(-alpha_inner * proj_R, axis=0)
        P_max = (1.0 / alpha_inner) * logsumexp(alpha_inner * proj_P, axis=0)
        P_min = -(1.0 / alpha_inner) * logsumexp(-alpha_inner * proj_P, axis=0)

        # Smooth d1 and d2
        d1 = P_min - R_max  # (K,)
        d2 = R_min - P_max  # (K,)
        gap_soft = (1.0 / alpha_inner) * logsumexp(alpha_inner * jnp.stack([d1, d2]), axis=0)  # shape (K,)

        # Final outer smooth max
        h_olsat = (1.0 / alpha_outer) * logsumexp(alpha_outer * gap_soft)

        # Optional: subtract outer error bound
        err_bound = jnp.log(gap_soft.shape[0]) / alpha_outer
        return h_olsat - err_bound
    @jax.jit
    def compute_distance_olsat_fast(robot_vertices, polygon_vertices, alpha=500.):
        """
        Efficient and fully smooth OLSAT distance metric using one-layer LogSumExp.

        Parameters:
        - robot_vertices: (N, 2) jnp.ndarray
        - polygon_vertices: (M, 2) jnp.ndarray
        - alpha: float, smoothness parameter

        Returns:
        - h_shifted: smooth over-approximation of SAT distance (float)
        """

        def get_normals(verts):
            edges = jnp.roll(verts, -1, axis=0) - verts
            normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
            return normals / jnp.linalg.norm(normals, axis=1, keepdims=True)

        # 1. Get all separating axes from robot and polygon
        axes = jnp.concatenate([get_normals(robot_vertices), get_normals(polygon_vertices)], axis=0)  # (K, 2)

        # 2. Project both polygons onto axes: shapes (N, K), (M, K)
        proj_robot = robot_vertices @ axes.T
        proj_poly  = polygon_vertices @ axes.T

        # 3. Compute min/max projections (true max/min for speed)
        R_max = jnp.max(proj_robot, axis=0)
        R_min = jnp.min(proj_robot, axis=0)
        P_max = jnp.max(proj_poly, axis=0)
        P_min = jnp.min(proj_poly, axis=0)

        # 4. Signed separations
        d1 = P_min - R_max  # poly left of robot
        d2 = R_min - P_max  # robot left of poly

        # 5. Smooth one-log version: combine all signed gaps
        all_d = jnp.concatenate([d1, d2])  # shape (2K,)
        h_raw = (1. / alpha) * logsumexp(alpha * all_d)

        # 6. Subtract overestimation upper bound (optional but recommended)
        h_shifted = h_raw - (jnp.log(all_d.shape[0]) / alpha)

        return h_shifted

    def compute_distance_sat(robot, poly):
        """
        Basic SAT distance: maximum of axis-wise penetration/separation.
        Non-smooth, JIT-compiled separately.
        """
        def get_normals(v):
            e = jnp.roll(v, -1, axis=0) - v
            n = jnp.stack([-e[:,1], e[:,0]], axis=1)
            return n / jnp.linalg.norm(n, axis=1, keepdims=True)
        Rn = get_normals(robot)
        Pn = get_normals(poly)
        axes = jnp.concatenate([Rn, Pn], axis=0)
        prj_R = robot @ axes.T
        prj_P = poly  @ axes.T
        d1 = prj_P.min(0) - prj_R.max(0)
        d2 = prj_R.min(0) - prj_P.max(0)
        return jnp.max(jnp.maximum(d1, d2))

    # JIT-compile the two-step and SAT versions
    xtanh_jit = compute_distance_smoothabs
    twolog_jit = jax.jit(compute_distance_twolog)
    sat_jit    = jax.jit(compute_distance_sat)

    # ——— Helper functions ———

    def regular_ngon(N, radius=1.0):
        """
        Generate vertices of a regular N-gon centered at origin.
        """
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        return np.stack([radius * np.cos(angles),
                        radius * np.sin(angles)], axis=1)

    def benchmark(fn, *args, n=1000):
        """
        Measure average latency (ms) of fn(*args) over n runs.
        Uses .block_until_ready() to ensure synchronous timing.
        """
        t0 = time.perf_counter()
        for _ in range(n):
            out = fn(*args)
            try:
                out.block_until_ready()
            except AttributeError:
                pass
        t1 = time.perf_counter()
        return (t1 - t0) / n * 1e3

    # ——— Main experiment: repeat trials and average ———

    if __name__ == "__main__":
        polygon_sizes = [4, 8, 16, 32]
        outer_repeats = 10  # number of outer average loops

        print("    N   | lse-xtanh(ms) | one-log(ms) | SAT(ms)  | speedup")
        print("--------+-------------+-------------+----------+---------")
        for N in polygon_sizes:
            # Create robot (square) and obstacle (N-gon)
            A = regular_ngon(4, radius=0.5)
            B = regular_ngon(N, radius=1.0)
            A_j = jnp.array(A)
            B_j = jnp.array(B)

            # Warm up JIT
            # _ = twolog_jit(A_j, B_j).block_until_ready()
            _ = compute_distance_onelogminus(A_j, B_j).block_until_ready()
            _ = sat_jit(A_j, B_j).block_until_ready()
            _ = xtanh_jit(A_j, B_j).block_until_ready()

            # Accumulate timings
            sum_two = sum_one = sum_sat = sum_xtanh = 0.0
            for _ in range(outer_repeats):
                # sum_two += benchmark(twolog_jit, A_j, B_j, n=1000)
                sum_one += benchmark(compute_distance_onelogminus, A_j, B_j, n=1000)
                sum_sat += benchmark(sat_jit, A_j, B_j, n=1000)
                sum_xtanh += benchmark(xtanh_jit, A_j, B_j, n=1000)

            # Compute average
            avg_two = sum_two / outer_repeats
            avg_one = sum_one / outer_repeats
            avg_sat = sum_sat / outer_repeats
            avg_xtanh = sum_xtanh / outer_repeats
            speedup = avg_xtanh / avg_one if avg_one > 0 else float('nan')

            print(f"{N:7d} | {avg_xtanh:11.3f} | {avg_one:11.3f} | {avg_sat:8.3f} | {speedup:7.2f}")

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
