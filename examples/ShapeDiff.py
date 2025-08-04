import csv
import diffrax as dx
import jax
from jax import Array, config, debug, jacfwd, jit, vmap, lax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU

from scipy.spatial.distance import directed_hausdorff

import jsrm
from jsrm.systems import planar_pcs
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path


'''
1) Importing necessary parameters and functions from JSRM (JAX Soft Robot Models).
'''
# load symbolic expressions
num_segments = 2
# filepath to symbolic expressions
sym_exp_filepath = Path(jsrm.__file__).parent / "symbolic_expressions" / f"planar_pcs_ns-{num_segments}.dill"

# set soft robot parameters
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
robot_length = 1.3e-1
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
jacobian_fn = jit(auxiliary_fns["jacobian_fn"])

# construct batched forward kinematics function
batched_forward_kinematics_fn = vmap(
    forward_kinematics_fn, in_axes=(None, None, 0)
)

# segmenting params
num_points = 20*num_segments
# Compute indices: equivalent to
# [num_points * (i+1)//num_segments - 1 for i in range(num_segments)]
end_p_ps_indices = (jnp.arange(1, num_segments+1) * num_points // num_segments) - 1

'''
2) Setting up the calculation of the segmented polygon for collision detection.
'''
def segmented_polygon(current_point, next_point, forward_direction, robotic_radius):
    '''
    Feed in soft body consecutive centered positions and directions and formulate a rectangular body for detecting collisions.
    The current point remains unchanged, but the displacement from current_point to next_point is scaled by 1.3.
    '''
    # Compute the new next point by scaling the difference
    new_next_point = current_point + 1 * (next_point - current_point)
    
    d = (next_point - current_point)/jnp.linalg.norm(next_point - current_point)
    # Compute the directional vector rotated by 90 degrees (for width of the robot)
    # d = jnp.array([
    #     jnp.cos(forward_direction + jnp.pi/2),
    #     jnp.sin(forward_direction + jnp.pi/2)
    # ])
    n1 = jnp.array([-d[1], d[0]])
    n2 = jnp.array([d[1], -d[0]])
    
    # Form the vertices using current_point and new_next_point
    vertices = [
        current_point + n1 * robotic_radius,
        new_next_point + n1 * robotic_radius,
        new_next_point + n2 * robotic_radius,
        current_point + n2 * robotic_radius
    ]
    
    return jnp.array(vertices)

def segment_robot(current, next, orientation):
    seg_poly = segmented_polygon(current, next, orientation, robot_radius)
    return (seg_poly)

def get_robot_polygons(q, robot_params, num_segments, resolution_per_segment):
    # Compute FK
    p_group = batched_forward_kinematics_fn(q, robot_params, resolution_per_segment * num_segments)
    p_ps = p_group[:, :2]
    orientations = p_group[:, 2]

    # Segment-wise positions and orientations
    start_pts = p_ps[:-1]
    end_pts = p_ps[1:]
    dirs = orientations[:-1]

    # Polygons per segment
    return vmap(segment_robot, in_axes=(0, 0, 0))(start_pts, end_pts, dirs)
'''
Example Scripts
'''
def soft_robot_segmentation_result_example():
    
    def ode_fn(t:float, y: Array, tau: Array) -> Array:
        q, q_d = jnp.split(y, 2)
        B, C, G, K, D, alpha = dynamical_matrices_fn(robot_params, q, q_d)

        q_dd = jnp.linalg.inv(B) @ (tau - C @ q_d - G - D @ q_d - K)

        y_d = jnp.concatenate([q_d, q_dd])

        return y_d
    
    q0 = jnp.array ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    tau = jnp.array([-2e-4, 0.0, 1e-2, -2e-4, 0.0, 1e-2])

    # time
    dt = 1e-3
    sim_dt = 5e-5

    # define the time steps
    ts = jnp.arange(0, 7.0, dt)

    ode_term = dx.ODETerm(ode_fn)

    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0,tau,
                        saveat=dx.SaveAt(ts=ts), max_steps = None)

    q_ts, _ = jnp.split(sol.ys, 2, axis=1)

    hausdorff_list = []

    for idx, q in enumerate(q_ts[::20]):
        # Get low/high resolution polygons
        robot_poly_group0 = get_robot_polygons(q, robot_params, num_segments, 20)
        robot_poly_group1 = get_robot_polygons(q, robot_params, num_segments, 1000)

        # Convert to numpy for distance calc and plotting
        points0 = onp.concatenate([onp.array(poly) for poly in robot_poly_group0], axis=0)
        points1 = onp.concatenate([onp.array(poly) for poly in robot_poly_group1], axis=0)

        # Compute Hausdorff distance
        d01 = directed_hausdorff(points0, points1)[0]
        d10 = directed_hausdorff(points1, points0)[0]
        haus_dist = max(d01, d10)
        hausdorff_list.append(haus_dist)

    # Plot Hausdorff distance over time
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(hausdorff_list)), hausdorff_list, marker='o')
    plt.title('Symmetric Hausdorff Distance over Sampled Time Steps')
    plt.xlabel('Sampled Frame Index')
    plt.ylabel('Hausdorff Distance (m)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    soft_robot_segmentation_result_example()