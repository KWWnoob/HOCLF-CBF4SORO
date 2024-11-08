from functools import partial
import jax
from sympy.printing.pretty.pretty_symbology import line_width

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, jacfwd, jit, vmap
from jax import numpy as jnp
import jsrm
from jsrm.systems import planar_pcs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict, Tuple

from src.planar_contact_geometry import compute_planar_contact_geometry


# define the outputs directory
outputs_dir = Path("outputs") / "planar_contact_geometry"
outputs_dir.mkdir(parents=True, exist_ok=True)

# load symbolic expressions
num_segments = 1
# filepath to symbolic expressions
sym_exp_filepath = Path(jsrm.__file__).parent / "symbolic_expressions" / f"planar_pcs_ns-{num_segments}.dill"

# set soft robot parameters
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
D = 5e-6 * jnp.diag(jnp.array([1e0, 1e3, 1e3]))  # Damping coefficient
robot_params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 1e-1 * jnp.ones((num_segments,)),
    "r": 2e-2 * jnp.ones((num_segments,)),
    "rho": rho,
    "g": jnp.array([0.0, 9.81]),
    "E": 2e2 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
    "G": 1e2 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
    "D": D,
}

# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

# call the factory function for the planar PCS
strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = planar_pcs.factory(
    sym_exp_filepath, strain_selector
)


def static_example():
    # define the configuration
    q = jnp.array([jnp.pi, 0.1, 0.1])

    # define the position and radius of the end-effector
    x_obs = jnp.array([0.05, 0.05])
    R_obs = jnp.array(0.008)

    # compute the backbone distances
    d_min, s_min_dist, n_c_min_dist, aux = compute_planar_contact_geometry(
        forward_kinematics_fn, robot_params, q, x_obs, R_obs
    )

    # print("d_min:", d_min)
    # print("s_min_dist:", s_min_dist)
    # print("n_c_min_dist:", n_c_min_dist)
    # print("aux:", aux)

    s_pts = aux["s_pts"]
    chi_pts = aux["chi_pts"]

    fig, ax = plt.subplots()
    # ax.plot(chi_pts[:, 0], chi_pts[:, 1], "k-", linewidth=4.0)
    # scatter plot of backbone with the color visualizing the distance to the obstacle
    sc = ax.scatter(chi_pts[:, 0], chi_pts[:, 1], c=aux["d_pts"], s=20, cmap="gist_heat")
    # plot the obstacle
    ax.add_patch(mpatches.Circle((x_obs[0], x_obs[1]), R_obs, fill=True, color="g"))
    # plot the point of minimum distance
    ax.scatter(aux["chi_min_dist"][0], aux["chi_min_dist"][1], c="blue", s=200, marker="x")
    # plot the normal vector
    print("n_c_min_dist:", n_c_min_dist)
    ax.quiver(
        aux["chi_min_dist"][0],
        aux["chi_min_dist"][1],
        n_c_min_dist[0],
        n_c_min_dist[1],
        color="blue",
        scale=2e0,
        width=0.02,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    plt.colorbar(sc, label="d [m]")
    plt.grid(True)
    plt.savefig(outputs_dir / "static_example.pdf")
    plt.show()


if __name__ == "__main__":
    static_example()
