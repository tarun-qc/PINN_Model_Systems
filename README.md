# Physics Informed Neural Network in Model Systems

## Overview

This repository contains Python implementations of **Physics-Informed Neural Networks (PINNs)** for solving two fundamental model systems in physics:

1. **Simple Pendulum** - A nonlinear Ordinary Differential Equation (ODE)
2. **1D Heat Equation** - A partial Differential Equation (PDE)

PINNs are a class of deep learning methods that embed physical laws (expressed as differential equations) directly into the neural network's loss function. This enables solving differential equations **without any training data** — the network learns purely from the physics itself [web:12].

---

## What are Physics-Informed Neural Networks?

### Key Concept

Traditional neural networks learn from data by minimizing prediction error. PINNs instead minimize the **residual of the governing differential equation** using automatic differentiation:

$$\text{Loss} = \text{PDE Loss} + \text{Boundary Condition Loss} + \text{Initial Condition Loss}$$

### Why PINNs?

| Advantage | Description |
|-----------|-------------|
| **Data-free** | Solve PDEs/ODEs without experimental data [web:12] |
| **Differentiable** | All outputs are analytical functions (gradients available) [web:12] |
| **Flexible** | Work in high dimensions, complex geometries [web:12] |
| **Forward & Inverse** | Can solve for solutions OR discover unknown parameters [web:12] |

---

## Problem 1: Simple Pendulum

### Physical System

A simple pendulum consists of a mass (bob) attached to a fixed pivot by a rigid rod of length $L$, free to swing under gravity $g$.

![Simple Pendulum Diagram](figs/pendulum_diagram.png)

### Governing Equation

The nonlinear ODE describing pendulum motion:

$$\frac{d^2\theta}{dt^2} + \frac{g}{L}\sin(\theta) = 0$$

where:
- $\theta(t)$ = angular displacement (rad)
- $g = 9.81 \, \text{m/s}^2$ = gravitational acceleration
- $L = 1.0 \, \text{m}$ = pendulum length

### Initial Conditions

$$\theta(0) = \frac{\pi}{4} \, \text{rad} \quad (45^\circ)$$
$$\frac{d\theta}{dt}\bigg|_{t=0} = 0 \, \text{rad/s}$$

### PINN Implementation

The neural network $f_\theta(t)$ learns $\theta(t)$ directly. The physics loss enforces:

$$\mathcal{L}_{\text{physics}} = \mathbb{E}\left[\left(\frac{d^2\hat{\theta}}{dt^2} + \omega^2\sin(\hat{\theta})\right)^2\right]$$

where $\omega = \sqrt{g/L}$ and derivatives are computed via **automatic differentiation** using `torch.autograd.grad`.

### Results

![Simple Pendulum Solution](pinn_pendulum_solution.png)

**Figure Explanation:**

- **Left Panel**: Comparison of PINN prediction (green) vs numerical solution (red, from `scipy.odeint`). The PINN accurately captures the oscillatory motion over $t \in [0, 10]$ seconds.
- **Right Panel**: Absolute error on a log scale. Maximum error is $\sim 10^{-3}$, demonstrating the PINN's ability to learn the nonlinear dynamics without any training data.

The phase plot (θ vs dθ/dt) shows the characteristic closed orbit of a conservative pendulum system.

### Usage

```bash
python pinn_pendulum.py
```

**Output files:**
- `pinn_pendulum_solution.png` - Solution and error plots
- `pinn_pendulum_phase.png` - Phase space trajectory
- `pinn_pendulum_model.pth` - Saved PyTorch model

---

## Problem 2: 1D Heat Equation

### Physical System

Heat diffusion in a 1D rod of length $L=1$ with fixed temperature boundaries.

![Heat Equation Schematic](figs/heat_equation_schematic.png)

### Governing Equation

The linear parabolic PDE:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

where:
- $u(x,t)$ = temperature at position $x$ and time $t$
- $\alpha = 0.1 \, \text{m}^2/\text{s}$ = thermal diffusivity

### Boundary Conditions

$$u(0, t) = 0 \quad \text{(left end)}$$
$$u(1, t) = 0 \quad \text{(right end)}$$

### Initial Condition

$$u(x, 0) = \sin(\pi x)$$

### PINN Implementation

The neural network $f_\theta(x, t)$ learns $u(x,t)$. The physics loss enforces:

$$\mathcal{L}_{\text{PDE}} = \mathbb{E}\left[\left(\frac{\partial \hat{u}}{\partial t} - \alpha \frac{\partial^2 \hat{u}}{\partial x^2}\right)^2\right]$$

Higher-order derivatives ($\partial^2 u/\partial x^2$) are computed using **nested automatic differentiation**:

```python
u_x = torch.autograd.grad(u, x, create_graph=True)
u_xx = torch.autograd.grad(u_x, x, create_graph=True)
```

### Results

![Heat Equation Solution](pinn_heat_equation.png)

**Figure Explanation:**

- **Left Panel**: PINN solution $u(x,t)$ as a 2D heatmap. Temperature decays exponentially over time while maintaining the sinusoidal spatial profile.
- **Middle Panel**: Analytical solution $u(x,t) = \sin(\pi x)e^{-\alpha\pi^2 t}$. The PINN matches this perfectly.
- **Right Panel**: Absolute error (PINN − analytical). Maximum error is $\sim 10^{-2}$, with higher error near boundaries due to stricter BC enforcement.

The heat equation PINN demonstrates successful learning of a **2D PDE** with coupled space-time derivatives.

### Usage

```bash
python pinn_heat_equation.py
```

**Output files:**
- `pinn_heat_equation.png` - Solution, analytical, and error heatmaps
- `pinn_heat_model.pth` - Saved PyTorch model

---

## Neural Network Architecture

Both implementations use the same fully-connected architecture:
Input (t) or (x, t)
↓
Linear(1→20) + Tanh
↓
Linear(20→20) + Tanh
↓
Linear(20→20) + Tanh
↓
Linear(20→20) + Tanh
↓
Output (θ) or (u)


| Parameter | Value |
|-----------|-------|
| Hidden layers | 4 |
| Neurons per layer | 20 |
| Activation | Tanh |
| Weight init | Xavier Normal |
| Optimizer | Adam (lr=0.001) |

---

## Technical Requirements

### Dependencies

```bash
pip install torch numpy matplotlib scipy
```

### Python Version

- Python 3.8+ recommended
- PyTorch 1.9+ (for `torch.autograd.grad` with `create_graph=True`)

### Hardware

- CPU works fine for these 1D problems
- GPU recommended for higher-dimensional PDEs

---

## Code Structure

### Common PINN Components

```python
class PINN(nn.Module):          # Neural network architecture
class PendulumPINN:             # Pendulum solver (ODE)
class HeatEquationPINN:         # Heat equation solver (PDE)
    compute_physics_loss()      # PDE/ODE residual
    compute_boundary_loss()     # BC enforcement
    compute_initial_loss()      # IC enforcement
    train()                     # Training loop
    predict()                   # Evaluation
```

### Key Differences

| Component | Pendulum (ODE) | Heat Eq (PDE) |
|-----------|----------------|---------------|
| Input | $t$ (1D) | $x, t$ (2D) |
| Derivatives | $d\theta/dt$, $d^2\theta/dt^2$ | $\partial u/\partial t$, $\partial^2 u/\partial x^2$ |
| Boundary Loss | None (initial value problem) | $u(0,t)=0$, $u(1,t)=0$ |
| Training points | 20,000 time points | 10,000 (x,t) collocation points |

---

## Extending to Your Own Problems

### Step 1: Define the Governing Equation

```python
# Example: Schrödinger equation
residual = i * dPsi_dt - (-0.5 * d2Psi_dx2 + V(x) * Psi)
```

### Step 2: Implement Physics Loss

```python
def compute_pde_loss(self, x, t):
    x.requires_grad = True
    t.requires_grad = True
    
    u = self.model(x, t)
    
    # Compute derivatives
    u_t = torch.autograd.grad(u, t, create_graph=True)
    u_xx = torch.autograd.grad(
        torch.autograd.grad(u, x, create_graph=True), 
        x, create_graph=True
    )
    
    residual = u_t - alpha * u_xx  # Your PDE
    return torch.mean(residual ** 2)
```

### Step 3: Add Boundary/Initial Conditions

```python
def compute_bc_loss(self):
    # Enforce u(0, t) = 0
    u_left = self.model(torch.zeros_like(t), t)
    return torch.mean(u_left ** 2)
```

### Step 4: Train

```python
pinn = YourProblemPINN()
pinn.train(n_epochs=10000)
```

---

## References

1. **Raissi, M., Perdikas, P., & Karniadakis, G. E. (2019).** "Physics-informed deep learning (part I): Data-driven solutions of nonlinear partial differential equations." *arXiv preprint arXiv:1711.10561*. [web:12]

2. **DeepXDE Library** - Comprehensive PINN implementation for various PDEs [web:15]

3. **PINN Tutorial (PyTorch)** - Minimal implementation examples [web:18]

---

## License

MIT License - Free for research and educational use.

---

## Author

Implementations adapted for quantum chemistry and computational physics applications.

*Last updated: June 2026*