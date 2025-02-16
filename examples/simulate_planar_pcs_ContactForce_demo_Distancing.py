'''
SAT MULTI AXIS VERSION
'''

import csv
import diffrax as dx
from functools import partial
import jax
from cbfpy import CBF, CBFConfig
from cbfpy.cbfs.clf_cbf import CLFCBF, CLFCBFConfig
import inspect

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, debug, jacfwd, jit, vmap, lax
from jax import numpy as jnp
import jsrm
from jsrm.systems import planar_pcs
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict, Tuple

from src.img_animation import animate_images_cv2
from src.planar_pcs_rendering_multiSeg import draw_image

# define the outputs directory
outputs_dir = Path("outputs") / "planar_pcs_simulation"
outputs_dir.mkdir(parents=True, exist_ok=True)

# load symbolic expressions
num_segments = 1
# filepath to symbolic expressions
sym_exp_filepath = Path(jsrm.__file__).parent / "symbolic_expressions" / f"planar_pcs_ns-{num_segments}.dill"

# set soft robot parameters
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
robot_length = 1e-1
robot_radius = 2e-2
robot_params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": robot_length * jnp.ones((num_segments,)),
    "r": robot_radius * jnp.ones((num_segments,)),
    "rho": rho,
    "g": jnp.array([0.0, 9.81]),
    "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
    "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
}
# damping matrix
damping_array = jnp.array([1e0, 1e3, 1e3])
multiplier = [1.5**m * damping_array for m in range(num_segments)]
robot_params["D"] = 5e-5 * jnp.diag(jnp.concatenate(multiplier)) * robot_length #depend on the num of segments

# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

# call the factory function for the planar PCS
strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = planar_pcs.factory(sym_exp_filepath, strain_selector)

kinetic_energy_fn = jit(auxiliary_fns["kinetic_energy_fn"])
potential_energy_fn = jit(auxiliary_fns["potential_energy_fn"])

# construct batched forward kinematics function
batched_forward_kinematics_fn = vmap(
    forward_kinematics_fn, in_axes=(None, None, 0)
)

# ---Polygon matters---

def get_normals(vertices):
    edges = jnp.roll(vertices, -1, axis=0) - vertices
    normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
    norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
    return normals / norms

def compute_polygon_centroid(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_next = jnp.roll(x, -1)
    y_next = jnp.roll(y, -1)
    cross = x * y_next - x_next * y
    area = jnp.sum(cross) / 2.0
    Cx = jnp.sum((x + x_next) * cross) / (6.0 * area)
    Cy = jnp.sum((y + y_next) * cross) / (6.0 * area)
    return jnp.array([Cx, Cy])

def cross2D(a, b):
    """
    Compute the 2D cross product (scalar) for vectors a and b.
    Supports vectorized inputs; a and b can have shape (..., 2).
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def ray_polygon_intersection(O, d, vertices, eps=1e-8):
    """
    Compute the intersection of a ray starting at point O in direction d with a polygon's edges.
    The polygon is defined by its vertices (assumed to be ordered).
    
    Parameters:
      O: Origin of the ray, shape (2,).
      d: Ray direction (unit vector), shape (2,).
      vertices: Array of polygon vertices, shape (N, 2).
      eps: Tolerance to check for near-zero denominators (parallelism).
      
    Returns:
      The smallest positive t value (distance along the ray) for which the ray
      intersects any of the polygon's edges. If no valid intersection is found, returns jnp.inf.
    """
    # Create edge endpoints: A is each vertex, and B is the next vertex (with wrapping)
    A = vertices
    B = jnp.concatenate([vertices[1:], vertices[:1]], axis=0)
    BA = B - A  # Direction vectors for each edge

    # Compute the denominator for the intersection formula for each edge
    denom = cross2D(d, BA)

    # Vector from ray origin O to each vertex A
    A_minus_O = A - O

    # Calculate ray parameter t and segment parameter u for each edge:
    # The intersection is given by: O + t*d = A + u*(B - A)
    t = cross2D(A_minus_O, BA) / denom
    u = cross2D(A_minus_O, d) / denom

    # Determine valid intersections:
    # - Denom must be significantly non-zero.
    # - t must be non-negative (intersection is along the ray).
    # - u must be between 0 and 1 (intersection lies on the segment).
    valid = (jnp.abs(denom) > eps) & (t >= 0) & (u >= 0) & (u <= 1)

    # Replace invalid intersection t values with infinity so they are ignored when taking the minimum
    t_valid = jnp.where(valid, t, jnp.inf)

    # Return the smallest t value among all valid intersections
    return jnp.min(t_valid)

def compute_gap_along_centers(vertices1, vertices2, eps=1e-8):
    """
    Compute the gap between two polygons along the line connecting their centroids:
    
    gap = (distance between centroids) - (radius of polygon1 in the given direction +
                                            radius of polygon2 in the opposite direction)
    
    The "radius" of a polygon is determined by finding the intersection between a ray
    emanating from the polygon's centroid and the polygon's boundary.
    
    Note: This function assumes that the centroid (computed by compute_polygon_centroid)
    is located inside the polygon.
    """
    # Assume compute_polygon_centroid is defined elsewhere to compute the centroid (shape (2,))
    C1 = compute_polygon_centroid(vertices1)
    C2 = compute_polygon_centroid(vertices2)
    
    # Compute the vector between centroids, its magnitude, and the unit direction vector d
    d_vec = C2 - C1
    d_norm = jnp.linalg.norm(d_vec)
    d = d_vec / d_norm

    # Compute the "radius" for each polygon along the specified directions
    # For polygon1, along the direction d; for polygon2, along the opposite direction -d.
    r1 = ray_polygon_intersection(C1, d, vertices1, eps)
    r2 = ray_polygon_intersection(C2, -d, vertices2, eps)
    
    # The gap is the center-to-center distance minus the sum of the two radii
    gap = d_norm - (r1 + r2)
    return gap

def segmented_polygon(current_point, next_point,forward_direction,robotic_radius):
    '''
    Feedin soft body consecutive centered positions and directions and formulate a rectangle body for detecting collisions
    '''
    d = jnp.array([jnp.cos(forward_direction+jnp.pi/2), jnp.sin(forward_direction+jnp.pi/2)])
    n1 = jnp.array([-d[1], d[0]])
    n2 = jnp.array([d[1], -d[0]])
    vertices = [current_point+n1*robotic_radius,
                next_point+n1*robotic_radius,
                next_point+n2*robotic_radius,
                current_point+n2*robotic_radius]

    return jnp.array(vertices)

def minkowski_sum_convex(poly1, poly2):
    """
    Computes the Minkowski sum of two convex polygons.
    
    Both poly1 and poly2 must be provided as jnp.array of shape (N,2) and 
    (M,2) respectively, with vertices ordered counter-clockwise.
    
    Returns:
        A jnp.array of shape (K,2) representing the Minkowski sum polygon.
    """
    # Convert to numpy arrays for ease of iteration (polygons are expected to be small)
    poly1_np = jnp.array(poly1)
    poly2_np = jnp.array(poly2)
    
    n = poly1_np.shape[0]
    m = poly2_np.shape[0]
    
    # Find the indices of the vertices with the smallest x (and then y) for each polygon.
    i0 = jnp.argmin(poly1_np[:, 0] + 1e-8*poly1_np[:, 1])
    j0 = jnp.argmin(poly2_np[:, 0] + 1e-8*poly2_np[:, 1])
    
    i, j = i0, j0
    result = []
    
    # Loop through each edge of both polygons
    while True:
        result.append(poly1_np[i] + poly2_np[j])
        # Compute next indices (wrap-around)
        next_i = (i + 1) % n
        next_j = (j + 1) % m
        
        # Compute edge vectors
        edge1 = poly1_np[next_i] - poly1_np[i]
        edge2 = poly2_np[next_j] - poly2_np[j]
        
        # Compute cross product (z-component)
        cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        
        # Advance in the polygon(s) based on the cross product
        if cross >= 0:
            i = next_i
        if cross <= 0:
            j = next_j
        
        # Terminate when we've looped back to the start in both polygons
        if i == i0 and j == j0:
            break

    return jnp.array(result)

def round_polygon_corners_minkowski(poly, radius, num_circle_points=16):
    """
    Rounds the corners of a polygon using the Minkowski sum with a discretized circle.
    
    Parameters:
        poly: jnp.array of shape (N, 2) representing the polygon vertices (assumed to be in CCW order).
        radius: Rounding radius.
        num_circle_points: Number of points used to approximate the circle.
    
    Returns:
        A jnp.array representing the rounded polygon.
    """
    # Create a discretized circle (approximating a disk) as a convex polygon.
    angles = jnp.linspace(0, 2 * jnp.pi, num_circle_points, endpoint=False)
    circle = jnp.stack([radius * jnp.cos(angles), radius * jnp.sin(angles)], axis=1)
    
    # Compute the Minkowski sum of the original polygon and the circle.
    # This gives the offset polygon with rounded corners.
    rounded_poly = minkowski_sum_convex(poly, circle)
    return rounded_poly

def soft_robot_with_safety_contact_CBFCLF_example():
    
    # define the ODE function
    class SoRoConfig(CLFCBFConfig):
        '''Config for soft robot'''

        def __init__(self):

            self.robot_params = robot_params

            self.strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

            '''Circular Obstacle Parameter'''
            self.cir_obstacle_center = jnp.array([-1e-2, 0.12]) # radius postion
            self.cir_obstacle_radius = 1e-2 # radius obstacle

            '''Polygon Obstacle Parameter'''
            # self.poly_obstacle_shape = jnp.array([[0.0, 0.0],
            #                                     [0.0, 0.17],
            #                                     [0.03, 0.17],
            #                                     [0.03, 0.0]])

            self.poly_obstacle_shape = jnp.array([[ 0.00000,  0.00851],
                                                    [-0.00809,  0.00263],
                                                    [-0.00500, -0.00688],
                                                    [ 0.00500, -0.00688],
                                                    [ 0.00809,  0.00264]])
            # self.rounded_poly = round_polygon_corners_minkowski(self.poly_obstacle_shape, radius=0.005, num_circle_points=16)
            self.poly_obstacle_pos = self.poly_obstacle_shape + jnp.array([-0.04,0.06])

            '''Characteristic of robot'''
            self.s_ps = jnp.linspace(0, robot_length * num_segments, 10 * num_segments) # segmented
            self.q_des_arary = jnp.array([jnp.pi*4, 0.1, 0.1])
            multiplier = [1.1**m * self.q_des_arary for m in range(num_segments)]
            self.q_des = jnp.concatenate(multiplier)# destination

            '''Contact model Parameter'''
            self.contact_spring_constant = 2000 #contact force model
            self.maximum_withhold_force = 0

            super().__init__(
                n=6 * num_segments, # number of states
                m=3 * num_segments, # number of inputs
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=False,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                cbf_relaxation_penalty=1e5,
                clf_relaxation_penalty=10
            )

        def f(self, z) -> Array:
            q, q_d = jnp.split(z, 2)  # Split state z into q (position) and q_d (velocity)
            B, C, G, K, D, alpha = dynamical_matrices_fn(self.robot_params, q, q_d)

            # Drift term (f(x))
            drift = (
                -jnp.linalg.inv(B) @ (C @ q_d + D @ q_d + G + K)
            )
            
            return jnp.concatenate([q_d, drift])

        def g(self, z) -> Array:
            q, q_d = jnp.split(z, 2)
            B, _, _, _, _, _ = dynamical_matrices_fn(self.robot_params, q, q_d)

            # Control matrix g(x)
            control_matrix = jnp.linalg.inv(B)

            # Match dimensions for concatenation
            zero_block = jnp.zeros((q.shape[0], control_matrix.shape[1]))

            return jnp.concatenate([zero_block, control_matrix], axis=0)
 
        def V_2(self, z) -> jnp.ndarray:
            # CLF: tracking error for both the middle point and the tip (last point)
            
            # Split state into positions (q) and velocities (q_d)
            q, q_d = jnp.split(z, 2)
            
            # Compute forward kinematics for the current configuration.
            p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            
            # Determine indices: use the middle point and the tip.
            num_points = p.shape[0]
            index = [num_points * (i+1)//num_segments for i in range(num_segments)]

            p_list = [p[i, :2] for i in index]
            
            # Compute forward kinematics for the desired configuration.
            p_des = batched_forward_kinematics_fn(self.robot_params, self.q_des, self.s_ps)
            p_des_list = [p_des[i, :2] for i in index]
            
            # Compute the element-wise absolute differences (note that sqrt((x)^2) equals |x|).
            # This returns a vector for each point.
            error = [jnp.sqrt((p_list[i]-p_des_list[i])**2) for i in range(num_segments)]
            
            # Option: Return the errors as a single vector by concatenating the two.
            V_total = jnp.concatenate(error)

            return V_total


        def h_2(self, z) -> jnp.ndarray:
            """
            Computes the safety force (barrier function output) for the robot given its state 'z',
            considering both polygonal and circular obstacles.
            
            Args:
                z: A JAX array representing the state, typically a concatenation of positions q and velocities q_d.
            
            Returns:
                A JAX array representing the combined smooth force.
            """
            # Split the state vector into positions (q) and velocities (q_d)
            q, q_d = jnp.split(z, 2)
            
            # Compute the forward kinematics for the robot
            p = batched_forward_kinematics_fn(self.robot_params, q, self.s_ps)
            
            # Extract positions and orientations from the forward kinematics output.
            p_ps = p[:, :2]         # Positions, shape: (N, 2)
            p_orientation = p[:, 2]   # Orientations, shape: (N,)
            
            # -------- Process the polygon obstacle --------
            obs_poly = self.poly_obstacle_pos  # Predefined polygon obstacle vertices

            # Consider segments between consecutive points (excluding the last point for segments)
            current_points = p_ps[:]
            next_points = p_ps[1:]
            top_point  = p_ps[-1] + p_orientation[-1]*robot_radius
            next_points = jnp.concatenate([next_points, top_point[None, :]], axis=0)
            orientations = p_orientation[:]

            # Define a function to compute penetration depth for a single segment against the polygon obstacle.
            def segment_penetration(current, nxt, orientation):
                # segmented_polygon should generate a polygon from the segment based on current, nxt, orientation, and robot_radius.
                seg_poly = segmented_polygon(current, nxt, orientation, robot_radius)
                # compute_distance should compute the penetration depth between the segment polygon and the obstacle polygon.
                return compute_gap_along_centers(seg_poly, obs_poly)

            # Vectorize the penetration computation over all segments.
            penetration_depth_poly = jax.vmap(segment_penetration)(current_points, next_points, orientations)
            # jax.debug.print()


            d2o_ps = jnp.linalg.norm((p_ps - self.cir_obstacle_center), ord=2, axis=1)
            penetration_depth_cir = d2o_ps - self.cir_obstacle_radius - robot_radius

            # -------- Compute the smooth force outputs --------
            contact_spring_constant = self.contact_spring_constant
            maximum_withhold_force = self.maximum_withhold_force

            # Compute the smooth force for the circular obstacle using a sigmoid to smooth the transition.
            force_smooth_cir = (penetration_depth_cir * contact_spring_constant + maximum_withhold_force) * \
                            (1 - jax.nn.sigmoid(20 * penetration_depth_cir))
            
            # Compute the smooth force for the polygon obstacle in a similar way.
            force_smooth_poly = (penetration_depth_poly * contact_spring_constant+ maximum_withhold_force) * \
                                (1 - jax.nn.sigmoid(20 * penetration_depth_poly)) 
            
            # Combine the forces from both the circular and polygon obstacles.
            # You may choose to add them instead of concatenating depending on the desired behavior.
            force_smooth = jnp.concatenate([force_smooth_cir,force_smooth_poly], axis=0)
            
            return force_smooth
            
        def alpha_2(self, h_2):
            return h_2*160 #constant, increase for smaller affected zone
        
        def gamma_2(self, v_2):
            return v_2*160

    config = SoRoConfig()
    clf_cbf = CLFCBF.from_config(config)

    def closed_loop_ode_fn(t: float, y: Array, q_des: Array) -> Array:
        # split the state vector into the configuration and velocity
        q, q_d = jnp.split(y, 2)
        q_d_des = jnp.zeros_like(q_des)
        q_des = jnp.concatenate([q_des, q_d_des])

        # evaluate the control policy
        u = clf_cbf.controller(y, q_des)

        # compute the dynamical matrices
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)

        # compute the acceleration
        q_dd = jnp.linalg.inv(B) @ (u - C @ q_d - G - K - D @ q_d)

        # concatenate the velocity and acceleration
        y_d = jnp.concatenate([q_d, q_dd])

        return y_d

    # define the initial condition
    q0_arary = jnp.array([-jnp.pi, 0.01, 0.05])
    multiplier = [q0_arary for m in range(num_segments)]
    q0 = jnp.concatenate(multiplier)

    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the desired configuration
    q_des = config.q_des

    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 5e-4

    # define the time steps
    ts = jnp.arange(0.0, 7.0, dt)

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, q_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)

    q_des_ts = jnp.tile(q_des, (ts.shape[0], 1))
    print("q_des_ts", q_des_ts.shape)
    # Compute tau_ts using vmap
    tau_ts = vmap(clf_cbf.controller)(sol.ys, q_des_ts) #TODO: understand this
    print("tau_ts", tau_ts.shape)

    force_list_cir = []
    # Getting the position of the points based on the pose
    for q in q_ts[::20]:
        p = batched_forward_kinematics_fn(robot_params, q, config.s_ps)
        p_ps = p[:, :2]
        p_orientation = p[:, 2]

        d2o_ps = jnp.linalg.norm((p_ps - config.cir_obstacle_center), ord=2, axis=1)
        penetration_depth_cir = d2o_ps - config.cir_obstacle_radius - robot_radius

        if jnp.any(penetration_depth_cir) < 0:
            worst_pen = jnp.max(penetration_depth_cir[penetration_depth_cir < 0])
        else:
            worst_pen = jnp.min(penetration_depth_cir)

        # worst_pen = jnp.min(penetration_depth_cir)
        # force = jnp.where(worst_pen >= 0, 0.0, -config.contact_spring_constant * worst_pen)
        force_list_cir.append(worst_pen)

    force_list_cir = onp.array(force_list_cir)

    force_list_poly = []

    # for q in q_ts[::20]:
    #     p = batched_forward_kinematics_fn(robot_params, q, config.s_ps)
    #     p_ps = p[:, :2]
    #     p_orientation = p[:, 2]
    #     current_points = p_ps[:-1]

    #     next_points = p_ps[1:]
    #     orientations = p_orientation[:-1]
    #     def seg_pen(current, nxt, orientation):
    #         seg_poly = segmented_polygon(current, nxt, orientation, robot_radius)
    #         return compute_distance(seg_poly, config.poly_obstacle_pos)
    #     seg_pen_vec = jax.vmap(seg_pen)(current_points, next_points, orientations)

    #     if jnp.any(seg_pen_vec) < 0:
    #         worst_pen = jnp.max(seg_pen_vec[seg_pen_vec < 0])
    #     else:
    #         worst_pen = jnp.min(seg_pen_vec)

    #     # worst_pen = jnp.min(seg_pen_vec)
    #     # force = jnp.where(worst_pen >= 0, 0.0, -config.contact_spring_constant * worst_pen)
    #     force_list_poly.append(worst_pen)

    for q in q_ts[::20]:
        p = batched_forward_kinematics_fn(robot_params, q, config.s_ps)
        p_ps = p[:, :2]
        p_orientation = p[:, 2]
        current_points = p_ps[:-1]

        next_points = p_ps[1:]
        orientations = p_orientation[:-1]
        def seg_pen(current, nxt, orientation):
            seg_poly = segmented_polygon(current, nxt, orientation, robot_radius)
            return compute_gap_along_centers(seg_poly, config.poly_obstacle_pos)
        seg_pen_vec = jax.vmap(seg_pen)(current_points, next_points, orientations)
        
        def segment_penetration(current, nxt, orientation):
            # segmented_polygon should generate a polygon from the segment based on current, nxt, orientation, and robot_radius.
            seg_poly = segmented_polygon(current, nxt, orientation, robot_radius)
            # compute_distance should compute the penetration depth between the segment polygon and the obstacle polygon.
            return compute_gap_along_centers(seg_poly, config.poly_obstacle_pos)
        gap = jax.vmap(segment_penetration)(current_points, next_points, orientations)
        gap = jnp.min(gap)
        if jnp.any(seg_pen_vec) < 0:
            worst_pen = 0
        else:
            worst_pen = 1

        # worst_pen = jnp.min(seg_pen_vec)
        # force = jnp.where(worst_pen >= 0, 0.0, -config.contact_spring_constant * worst_pen)
        force_list_poly.append(gap)

    force_list_poly = onp.array(force_list_poly)

    # Plot the motion and tau_ts
    fig, axes = plt.subplots(6, 1, figsize=(8,10), sharex=True, num="Regulation example")

    # Plot strains
    # plot the reference strain evolution
    for i in range(num_segments):
        axes[0].plot(ts, q_des_ts[:, i], linewidth=3.0, linestyle=":", label=r"$\kappa_\mathrm{be}^\mathrm{d}$")
        axes[1].plot(ts, q_des_ts[:, i+1], linewidth=3.0, linestyle=":", label=r"$\sigma_\mathrm{sh}^\mathrm{d}$")
        axes[2].plot(ts, q_des_ts[:, i+2], linewidth=3.0, linestyle=":", label=r"$\sigma_\mathrm{ax}^\mathrm{d}$")

    # reset the color cycle
    axes[0].set_prop_cycle(None)
    axes[1].set_prop_cycle(None)
    axes[2].set_prop_cycle(None)
    # plot the actual strain evolution
    for i in range(num_segments):
        axes[0].plot(ts, q_ts[:, i], linewidth=2.0, label=r"$\kappa_\mathrm{be}$")
        axes[1].plot(ts, q_ts[:, i+1], linewidth=2.0, label=r"$\sigma_\mathrm{sh}$")
        axes[2].plot(ts, q_ts[:, i+2], linewidth=2.0, label=r"$\sigma_\mathrm{ax}$")

    # Plot control inputs tau_ts
    for i in range(tau_ts.shape[1]):  # Assuming tau_ts has multiple dimensions (e.g., torques for each actuator)
        axes[3].plot(ts, tau_ts[:, i], label=f"Control Input {i+1}")

    safe_threshold = config.maximum_withhold_force
    axes[4].plot(ts[::20], force_list_cir, linewidth=1.0, label=r"$\sigma_\mathrm{ax}^\mathrm{d}$")
    axes[5].plot(ts[::20], force_list_poly, linewidth=1.0, label=r"$\sigma_\mathrm{ax}^\mathrm{d}$")

    # axes[4].fill_between(
    # ts[::20],               # x values for the entire time vector
    # force_list_cir,       # y values (force)
    # safe_threshold,   # baseline (threshold)
    # where=(force_list_cir <= safe_threshold),  # condition for safe region
    # interpolate=True,
    # color='green',
    # alpha=0.3,
    # label='Safe Region'
    # )
    # axes[4].fill_between(
    #     ts[::20],
    #     force_list_cir,
    #     safe_threshold,
    #     where=(force_list_cir > safe_threshold),  # condition for unsafe region
    #     interpolate=True,
    #     color='red',
    #     alpha=0.3,
    #     label='Unsafe Region'
    # )

    # axes[5].fill_between(
    # ts[::20],               # x values for the entire time vector
    # force_list_poly,       # y values (force)
    # safe_threshold,   # baseline (threshold)
    # where=(force_list_poly <= safe_threshold),  # condition for safe region
    # interpolate=True,
    # color='green',
    # alpha=0.3,
    # label='Safe Region'
    # )
    # axes[5].fill_between(
    #     ts[::20],
    #     force_list_poly,
    #     safe_threshold,
    #     where=(force_list_poly > safe_threshold),  # condition for unsafe region
    #     interpolate=True,
    #     color='red',
    #     alpha=0.3,
    #     label='Unsafe Region'
    # )
    
    # Set labels and legends
    axes[0].set_ylabel(r"Bending strain $\kappa_\mathrm{be}$")
    axes[1].set_ylabel(r"Shear strain $\sigma_\mathrm{sh}$")
    axes[2].set_ylabel(r"Axial strain $\sigma_\mathrm{ax}$")
    axes[3].set_ylabel(r"Control inputs $\tau$")
    axes[4].set_ylabel(r"Contact Force/Cir Newton")
    axes[5].set_ylabel(r"Contact Force/Poly Newton")
    axes[5].set_xlabel("Time [s]")

    # Add legends and grid
    for ax in axes:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Collect chi_ps values
    chi_ps_list = []

    # Animate the motion and collect chi_ps
    img_ts = []
    pos = batched_forward_kinematics_fn(config.robot_params, config.q_des, config.s_ps)
    pos = pos[-1,:2]
    print("pos", pos)
    for q in q_ts[::20]:
        img = draw_image(batched_forward_kinematics_fn, auxiliary_fns, robot_params, num_segments, q,
                          x_obs=config.cir_obstacle_center, 
                          R_obs=config.cir_obstacle_radius, 
                          p_des = pos, 
                          poly_points=config.poly_obstacle_pos
                        )
        
        img_ts.append(img)

        chi_ps = batched_forward_kinematics_fn(config.robot_params, q, config.s_ps)
        # Store chi_ps as a list for each timestep
        chi_ps_list.append(onp.array(chi_ps))  # Convert to numpy array for easier handling

    # Save chi_ps to a CSV using the csv module
    with open(outputs_dir / "chi_ps_values.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header (optional, depending on chi_ps dimensions)
        writer.writerow([f"Chi_{i}_{j}" for i in range(chi_ps.shape[0]) for j in range(chi_ps.shape[1])])
        # Write data rows
        for chi_ps in chi_ps_list:
            writer.writerow(chi_ps.flatten())

    # Animate the images
    img_ts = onp.stack(img_ts, axis=0)
    animate_images_cv2(
        onp.array(ts[::20]), img_ts, outputs_dir / "planar_pcs_safe_closed_loop_simulation.mp4"
    )

if __name__ == "__main__":
    soft_robot_with_safety_contact_CBFCLF_example()