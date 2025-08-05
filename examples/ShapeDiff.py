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

from shapely.geometry import Polygon
from shapely.ops import unary_union
plt.rcParams['pdf.fonttype'] = 42
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

def polys_to_union_shapely(polygons):
    """Convert JAX/Numpy polygons to Shapely union"""
    shapely_polys = []
    for poly in polygons:
        try:
            arr = onp.array(poly)
            if not onp.allclose(arr[0], arr[-1]):
                arr = onp.concatenate([arr, arr[0:1]], axis=0)  # ensure closed
            p = Polygon(arr)
            if p.is_valid:
                shapely_polys.append(p)
        except Exception:
            continue
    return unary_union(shapely_polys)

'''
Example Scripts
'''
def soft_robot_segmentation_result_example():
    key = jax.random.PRNGKey(0)

    rand_vals = jax.random.uniform(key, shape=(100, 6))
    min_vals = jnp.array([-0.5, -0.2, -0.5, -0.5, -0.2, -0.5])
    max_vals = jnp.array([ 0.5,  0.2,  0.5,  0.5,  0.2,  0.5])
    q_batch = min_vals + rand_vals * (max_vals - min_vals)

    num_polygons = jnp.arange(5, 1000, 35)
    num_q_samples = q_batch.shape[0]
    haus_records = [[] for _ in range(len(num_polygons))]
    containment_ratios = [[] for _ in range(len(num_polygons))] 

    for q in q_batch:
        # Reference high-resolution shape
        robot_poly_ref = get_robot_polygons(q, robot_params, num_segments, resolution_per_segment=1000)
        points_ref = onp.concatenate([onp.array(poly) for poly in robot_poly_ref], axis=0)  # shape (N_ref, 2)
        union_ref = polys_to_union_shapely(robot_poly_ref)
        area_ref = union_ref.area
        for i, num in enumerate(num_polygons):
            robot_poly = get_robot_polygons(q, robot_params, num_segments, resolution_per_segment=num)
            points = onp.concatenate([onp.array(poly) for poly in robot_poly], axis=0)

            # Hausdorff distance
            d01 = directed_hausdorff(points, points_ref)[0]
            d10 = directed_hausdorff(points_ref, points)[0]
            haus = max(d01, d10)
            haus_records[i].append(haus)

            # Containment ratio
            union_test = polys_to_union_shapely(robot_poly)
            try:
                area_test = union_test.area
                # area_diff = union_test.difference(union_ref).area
                area_common = union_test.intersection(union_ref).area
                # error_ratio = area_diff / area_test  # percentage not covered by A
                containment_ratio = area_common / area_ref
            except Exception:
                containment_ratio = jnp.nan

            containment_ratios[i].append(containment_ratio)

    # Compute statistics
    haus_array = jnp.array([jnp.array(hs) for hs in haus_records])  # shape (num_polygons, num_q_samples)
    haus_avg = haus_array.mean(axis=1)
    haus_std = haus_array.std(axis=1)
    # Add small epsilon to avoid log(0) if necessary
    # ---- Plot with error bars (on log y-axis) ----
    plt.figure(figsize=(8, 8))
    plt.errorbar(
        num_polygons,
        haus_avg,
        yerr=haus_std,
        fmt='o-', capsize=3,
        color='blue',
        ecolor='gray',
        elinewidth=1.5,
        label='Average ± Std Dev'
    )

    x_target = 40
    if x_target in num_polygons:
        idx = jnp.where(num_polygons == x_target)[0].item()
        y_target = float(haus_avg[idx])
        plt.plot(x_target, y_target, marker='x', markersize=12, color='red', label='Experimental Resolution')

    plt.xlabel(r"Number of Soft Robot Convex Polygons $N_\mathrm{srpoly}$", fontsize=20)
    plt.ylabel("Symmetric Hausdorff Distance (logarithmic scale)", fontsize=20)
    plt.yscale("log")
    plt.title(f"Average Shape Error with Std Dev ({num_q_samples} Samples)", fontsize=20)
    plt.grid(True, which='both', linestyle='--', linewidth=1.0)
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.savefig("hausdorff_vs_resolution.pdf")
    # plt.show()

    containment_array = jnp.array([jnp.array(x) for x in containment_ratios])
    containment_avg = jnp.nanmean(containment_array, axis=1)
    containment_std = jnp.nanstd(containment_array, axis=1)

    plt.figure(figsize=(8, 8))
    plt.errorbar(num_polygons, containment_avg, yerr=containment_std, fmt='o-', capsize=3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.4f}"))

    x_target = 40
    if x_target in num_polygons:
        idx = jnp.where(num_polygons == x_target)[0].item()
        y_target = float(containment_avg[idx])
        plt.plot(x_target, y_target, marker='x', markersize=12, color='red', label='Experimental Resolution')

    plt.xlabel(r"Number of Soft Robot Convex Polygons $N_\mathrm{srpoly}$", fontsize=20)
    plt.ylabel("Fraction of Area not inside Reference", fontsize=20)
    plt.grid(True, which='both', linestyle='--', linewidth=1.0)
    plt.title("Soft Robot Body Not Contained in Convex Polygons", fontsize=20)
    plt.legend(loc='lower right', fontsize=14)
    plt.tight_layout()
    plt.savefig("containment_vs_resolution.pdf")

if __name__ == "__main__":
    soft_robot_segmentation_result_example()