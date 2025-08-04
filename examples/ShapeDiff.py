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
    s_ps = jnp.linspace(0, robot_length * num_segments, resolution_per_segment * num_segments)
    p_group = batched_forward_kinematics_fn(robot_params, q, s_ps)

    p_ps = p_group[:, :2]
    orientations = p_group[:, 2]
    start_pts, end_pts = p_ps[:-1], p_ps[1:]
    dirs = orientations[:-1]

    robot_polygons = jax.vmap(segment_robot, in_axes=(0, 0, 0))(start_pts, end_pts, dirs)
    return robot_polygons

'''
Example Scripts
'''
def soft_robot_segmentation_result_example():
    key = jax.random.PRNGKey(0)

    rand_vals = jax.random.uniform(key, shape=(100, 6))
    min_vals = jnp.array([-0.5, -0.2, -0.5, -0.5, -0.2, -0.5])
    max_vals = jnp.array([ 0.5,  0.2,  0.5,  0.5,  0.2,  0.5])
    q_batch = min_vals + rand_vals * (max_vals - min_vals)

    num_polygons = jnp.arange(5, 1000, 50)
    num_q_samples = q_batch.shape[0]
    haus_records = [[] for _ in range(len(num_polygons))] 

    for q in q_batch:
        # Reference high-resolution shape
        robot_poly_ref = get_robot_polygons(q, robot_params, num_segments, resolution_per_segment=1000)
        points_ref = onp.concatenate([onp.array(poly) for poly in robot_poly_ref], axis=0)  # shape (N_ref, 2)

        for i, num in enumerate(num_polygons):
            robot_poly = get_robot_polygons(q, robot_params, num_segments, resolution_per_segment=num)
            points = onp.concatenate([onp.array(poly) for poly in robot_poly], axis=0)

            # Hausdorff distance
            d01 = directed_hausdorff(points, points_ref)[0]
            d10 = directed_hausdorff(points_ref, points)[0]
            haus = max(d01, d10)
            haus_records[i].append(haus)

    # Compute statistics
    haus_array = jnp.array([jnp.array(hs) for hs in haus_records])  # shape (num_polygons, num_q_samples)
    haus_avg = haus_array.mean(axis=1)
    haus_std = haus_array.std(axis=1)
    haus_std_log = jnp.log10(haus_std + 1e-10)
    # Add small epsilon to avoid log(0) if necessary
    haus_avg_safe = haus_avg + 1e-10

    # ---- Plot with error bars (on log y-axis) ----
    plt.figure(figsize=(8, 4))
    plt.errorbar(
        num_polygons,
        haus_avg_safe,
        yerr=haus_std_log,
        fmt='o-', capsize=3,
        color='blue',
        ecolor='gray',
        elinewidth=1.5,
        label='Average ± Std Dev'
    )

    # Optional: fill_between alternative to error bars
    # plt.fill_between(
    #     num_polygons,
    #     haus_avg_safe - haus_std,
    #     haus_avg_safe + haus_std,
    #     alpha=0.2,
    #     color='blue',
    #     label='Std Deviation Band'
    # )

    plt.xlabel("Number of Points per Segment")
    plt.ylabel("Symmetric Hausdorff Distance")
    plt.yscale("log")  # apply log scale AFTER adding epsilon
    plt.title(f"Average Shape Error with Std Dev ({num_q_samples} Samples)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    soft_robot_segmentation_result_example()