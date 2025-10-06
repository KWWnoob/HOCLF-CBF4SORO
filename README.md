# Contact-Aware Safety in Soft Robots Using High-Order Control Barrier and Lyapunov Functions

This repository provides the official implementation of the paper:

**Contact-Aware Safety in Soft Robots Using High-Order Control Barrier and Lyapunov Functions**  
Kiwan Wong, Maximilian Stölzle, Wei Xiao, Cosimo Della Santina, Daniela Rus, Gioele Zardini  
arXiv:2505.03841 • [Preprint Link](https://arxiv.org/abs/2505.03841)

---

## 🚀 Overview

We propose a unified framework for **safety-aware control of continuum soft robots**.  
Our method leverages **High-Order Control Barrier Functions (HOCBFs)** and **High-Order Control Lyapunov Functions (HOCLFs)**, integrated with a differentiable soft robot model, to guarantee safety while maintaining task performance.

Key features:

- ✅ **Force-aware safety constraints** during environment contact  
- ✅ **Flexible objectives**: shape regulation, end-effector tracking  
- ✅ **Differentiable SAT-based metric (DCSAT)** for conservative & efficient distance computation  
- ✅ **Simulation experiments** validating safety and performance across tasks  

---

## 📦 Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/mit-zardini-lab/contact-safe-soft-robot-control.git
cd contact-safe-soft-robot-control
pip install -r requirements.txt
```

## 📦 Usage

First, source the necessary environment variables:

```bash
source ./01-configure-env-vars.sh
```

## 🧪 Examples

We provide several example scripts under the [`examples/`](examples/) folder
to demonstrate different control strategies and safety mechanisms for the
planar PCS soft robot:

- **`simulate_planar_pcs_contact.py`**  
  CLF–CBF controller with contact-aware safety constraints. Demonstrates full
  safety-critical control with DCSAT distances and smooth contact forces.

- **`simulate_planar_pcs_NOCBF.py`**  
  Pure CLF (stabilization/tracking) without barrier constraints. Useful as a
  baseline to compare the effect of safety enforcement.

- **`simulate_planar_pcs_nocontact.py`**  
  CLF–CBF controller with no force allowed. 

- **`simulate_planar_pcs_PID.py`**  
  Task-space PID controller with DCSAT-based safety barrier. A classical
  baseline combining PID tracking with modern safety metrics.

- **`simulate_planar_pcs_Potential.py`**  
  PID controller augmented with an **Artificial Potential Field (APF)** for
  obstacle avoidance, using DCSAT distances to shape repulsive forces.

Each script outputs:
- **CSV logs** (tracking error, contact/safety metrics)  
- **MP4 animations** (visualizing robot, obstacles, and contact points)  
- **Waypoint data** (forward kinematics snapshots for analysis)


## 📖 Citation
If you find this repository useful, please cite:

```bash
@article{wong2025contact,
  title   = {Contact-Aware Safety in Soft Robots Using High-Order Control Barrier and Lyapunov Functions},
  author  = {Wong, Kiwan and Stölzle, Maximilian and Xiao, Wei and Della Santina, Cosimo and Rus, Daniela and Zardini, Gioele},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2025},
  note    = {Accepted}
}
