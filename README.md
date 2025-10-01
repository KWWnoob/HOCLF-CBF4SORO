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
- ✅ **Flexible objectives**: shape regulation, end-effector tracking, manipulation  
- ✅ **Differentiable SAT-based metric (DCSAT)** for conservative & efficient distance computation  
- ✅ **Simulation experiments** validating safety and performance across tasks  

---

## 📦 Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/KWWnoob/HOCLF-HOCBF4SORO.git
cd HOCLF-HOCBF4SORO
pip install -r requirements.txt

## Usage

First, source the necessary environment variables:

```bash
source ./01-configure-env-vars.sh
```

## 📖 Citation
If you find this repository useful, please cite:

```bash
@article{wong2025contactaware,
  title   = {Contact-Aware Safety in Soft Robots Using High-Order Control Barrier and Lyapunov Functions},
  author  = {Wong, Kiwan and St{\ö}lzle, Maximilian and Xiao, Wei and Della Santina, Cosimo and Rus, Daniela and Zardini, Gioele},
  journal = {arXiv preprint arXiv:2505.03841},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.03841}
}
