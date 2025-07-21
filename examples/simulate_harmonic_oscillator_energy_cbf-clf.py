import csv
import diffrax as dx
from functools import partial
import jax
from cbfpy.cbfs.clf_cbf import CLFCBF, CLFCBFConfig

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, debug, jacfwd, jit, vmap
from jax import numpy as jnp
import jsrm
from jsrm.systems import planar_pcs
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict, Tuple

from src.img_animation import animate_images_cv2
from src.planar_pcs_rendering import draw_image

# define the outputs directory
outputs_dir = Path("outputs") / "harmonic_oscillator_simulation"
outputs_dir.mkdir(parents=True, exist_ok=True)

# mass
M = 1.0 * jnp.eye(1)
# spring constant
K = 10.0 * jnp.eye(1)
# damping coefficient
D = 1.0 * jnp.eye(1)

# define the desired configuration
q_des = jnp.array([-0.1])

def simulate_open_loop_harmonic_oscillator():
    # define the ODE function
    def open_loop_ode_fn(t: float, y: Array, args) -> Array:
        # split the state vector into the configuration and velocity
        q, q_d = jnp.split(y, 2)

        # compute the acceleration
        q_dd = jnp.linalg.inv(M) @ (-D @ q_d - K @ q)

        # concatenate the velocity and acceleration
        y_d = jnp.concatenate([q_d, q_dd])

        return y_d

    # define the initial condition
    q0 = jnp.array([0.1])

    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 5e-5

    # define the time steps
    ts = jnp.arange(0.0, 10.0, dt)

    # setup the diffrax ode term
    ode_term = dx.ODETerm(open_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)

    # Plot time evolution of the harmonic oscillator
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, num="Harmonic oscillator open-loop simulation")
    axes[0].plot(ts, q_ts[:, 0])
    axes[1].plot(ts, q_d_ts[:, 0])
    axes[0].set_ylabel(r"Position $q$")
    axes[1].set_ylabel(r"Velocity $\dot{q}$")
    axes[1].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True)
    plt.tight_layout()
    plt.show()


def simulate_harmonic_oscillator_with_cbf_clf():
    # define the ODE function
    class HarmonicOscillatorConfig(CLFCBFConfig):
        '''Config for soft robot'''

        def __init__(self):
            super().__init__(
                n=2*M.shape[0], # number of states
                m=M.shape[0], # number of inputs
                # Note: Relaxing the CLF-CBF QP is tricky because there is an additional relaxation
                # parameter already, balancing the CLF and CBF constraints.
                relax_cbf=False,
                # If indeed relaxing, ensure that the QP relaxation >> the CLF relaxation
                # cbf_relaxation_penalty=1e5,
                # clf_relaxation_penalty=10
            )

            self.q_des = q_des

        def f(self, z) -> Array:
            q, q_d = jnp.split(z, 2)  # Split state z into q (position) and q_d (velocity)

            # Drift term (f(x))
            drift = -jnp.linalg.inv(M) @ (D @ q_d + K @ q)

            return jnp.concatenate([q_d, drift])

        def g(self, z) -> Array:
            q, q_d = jnp.split(z, 2)

            # # Control matrix g(x)
            # control_matrix = jnp.linalg.inv(M)

            # # Match dimensions for concatenation
            # zero_block = jnp.zeros((q.shape[0], control_matrix.shape[1]))
            # jnp.concatenate([zero_block, control_matrix], axis=0)

            B = jnp.block([[jnp.zeros((q.shape[0], q.shape[0]))], [jnp.linalg.inv(M)]])

            return B
        
        def V_2(self, z, z_des) -> Array:
            q, q_d = jnp.split(z, 2)

            # V = jnp.sqrt(jnp.square(q - q_des))

            # kinetic energy T(x)
            T = 0.5 * q_d.T @ M @ q_d

            # potential energy U(x)
            U = 0.5 * q.T @ K @ q
            # potential energy at desired configuration
            U_des = 0.5 * q_des.T @ K @ q_des

            # Lyapunov function V(x)
            # V = U + T
            # V = U
            V = T + U - U_des + q_des.T @ K.T @ (q_des - q)
            V = V[None]

            return V
        
        def h_2(self, z):
            q, q_d = jnp.split(z, 2)

            # return jnp.zeros(1)

            # h = 1.0 - q
            h = 1.0 - jnp.abs(q)

            return h
            
        
        def alpha_2(self, h_2):
            # return h_2 * 20
            return h_2 * 2e1
        
        def gamma_2(self, v_2):
            # return v_2 * 20
            return v_2 * 2e1

    config = HarmonicOscillatorConfig()
    clf_cbf = CLFCBF.from_config(config)

    def closed_loop_ode_fn(t: float, y: Array, q_des: Array) -> Array:
        # split the state vector into the configuration and velocity
        q, q_d = jnp.split(y, 2)

        # evaluate the control policy
        u = clf_cbf.controller(y, q_des)

        # compute the acceleration
        q_dd = jnp.linalg.inv(M) @ (u - D @ q_d - K @ q)

        # concatenate the velocity and acceleration
        y_d = jnp.concatenate([q_d, q_dd])

        return y_d

    # define the initial condition
    q0 = jnp.array([0.1])

    q_d0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, q_d0])

    # define the sampling and simulation time step
    dt = 1e-3
    sim_dt = 5e-5

    # define the time steps
    ts = jnp.arange(0.0, 10.0, dt)

    # setup the diffrax ode term
    ode_term = dx.ODETerm(closed_loop_ode_fn)

    # solve the ODE
    sol = dx.diffeqsolve(ode_term, dx.Tsit5(), ts[0], ts[-1], sim_dt, y0, q_des, saveat=dx.SaveAt(ts=ts), max_steps=None)

    # extract the results
    q_ts, q_d_ts = jnp.split(sol.ys, 2, axis=1)

    q_des_ts = jnp.tile(q_des, (ts.shape[0], 1))
    # Compute tau_ts using vmap
    tau_ts = vmap(clf_cbf.controller)(sol.ys, q_des_ts)

    # Plot time evolution of the harmonic oscillator
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, num="Harmonic oscillator closed-loop simulation")
    axes[0].plot(ts, q_ts[:, 0], label="Position $q$")
    axes[1].plot(ts, q_d_ts[:, 0], label=r"Velocity $\dot{q}$")
    axes[2].plot(ts, tau_ts[:, 0], label=r"Control input $\tau$")
    axes[0].set_ylabel(r"Position $q$")
    axes[1].set_ylabel(r"Velocity $\dot{q}$")
    axes[2].set_ylabel(r"Control input $\tau$")
    axes[2].set_xlabel("Time [s]")

    # Add legends and grid
    for ax in axes:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # simulate_open_loop_harmonic_oscillator()
    simulate_harmonic_oscillator_with_cbf_clf()
