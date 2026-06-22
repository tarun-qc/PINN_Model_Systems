# Physics Informed Neural Network in Model Systems

## Overview

This repository contains Python implementations of **Physics-Informed Neural Networks (PINNs)** for solving two fundamental model systems in physics:

1. **Simple Pendulum** - A nonlinear Ordinary Differential Equation (ODE)
2. **1D Heat Equation** - A partial Differential Equation (PDE)

PINNs are a class of deep learning methods that embed physical laws (expressed as differential equations) directly into the neural network's loss function. This enables solving differential equations **without any training data** — the network learns purely from the physics itself [web:y].

Additional Python implementations of **Fourier Embedding Physics-Informed Neural Networks (FPINNs)** for solving the spring-mass model systems in physics:

3. **Simple Pendulum** - A a second-order linear Ordinary Differential Equation (ODE)

The model learns the displacement \(x(t)\) directly from the governing physics, without requiring labeled solution data during training. Fourier features are used to improve the network's ability to represent oscillatory motion more accurately [web:x][web:x+3].


---

## What are Physics-Informed Neural Networks?

### Key Concept

Traditional neural networks learn from data by minimizing prediction error. PINNs instead minimize the **residual of the governing differential equation** using automatic differentiation:

$$\text{Loss} = \text{PDE Loss} + \text{Boundary Condition Loss} + \text{Initial Condition Loss}$$

### Why PINNs?

| Advantage | Description |
|-----------|-------------|
| **Data-free** | Solve PDEs/ODEs without experimental data [web:y] |
| **Differentiable** | All outputs are analytical functions (gradients available) [web:y] |
| **Flexible** | Work in high dimensions, complex geometries [web:y] |
| **Forward & Inverse** | Can solve for solutions OR discover unknown parameters [web:y] |

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

## Problem 3: Spring Mass System

### Physical System

A spring-mass system is a standard example of simple harmonic motion.  
If a mass \(m\) is attached to a spring with stiffness \(k\), the displacement \(x(t)\) from equilibrium satisfies a second-order linear ODE.

![Spring Mass Schema](figs/spring_mass_schema.png)

### Governing Equation

The second-order linear ODE:

\[
x''(t) + \omega^2 x(t) = 0
\]

where:

\[
\omega^2 = \frac{k}{m}
\]
This equation describes undamped periodic motion. The motion repeats with angular frequency \(\omega\), and the analytical solution is sinusoidal [web:x-1][web:x+2][web:x+5]

## Equation Parameters

The model uses the following physical parameters:

- **\(x(t)\)**: displacement of the mass from equilibrium at time \(t\).
- **\(t\)**: time variable.
- **\(m\)**: mass attached to the spring, in kilograms.
- **\(k\)**: spring constant, in newtons per meter.
- **\(\omega\)**: angular frequency, defined as \(\omega = \sqrt{k/m}\).
- **\(x_0\)**: initial displacement, i.e. \(x(0)\).
- **\(v_0\)**: initial velocity, i.e. \(x'(0)\).

For this example:

- \(m = 1.0\) kg
- \(k = 2.5\) N/m
- \(x_0 = 1.0\) m
- \(v_0 = 0.0\) m/s

The analytical solution for these initial conditions is:

\[
x(t) = x_0 \cos(\omega t) + \frac{v_0}{\omega}\sin(\omega t)
\]
Since \(v_0 = 0\), this simplifies to:

\[
x(t) = x_0 \cos(\omega t)
\]
### PINN Implementation

A Physics-Informed Neural Network solves the ODE by minimizing a loss function that combines:

1. **Physics residual loss**, which forces the network output to satisfy the differential equation.
2. **Initial condition loss**, which forces the solution to match the known starting values.

For this spring-mass system, the residual is:

\[
r(t) = \frac{d^2x}{dt^2} + \frac{k}{m}x
\]

The physics loss is the mean squared residual:

\[
L_{\mathrm{physics}} = \frac{1}{N}\sum_{i=1}^{N} r(t_i)^2
\]

The initial condition loss is:

\[
L_{\mathrm{IC}} = (x(0) - x_0)^2 + (x'(0) - v_0)^2
\]

The total training loss is:

\[
L = L_{\mathrm{physics}} + 10 \cdot L_{\mathrm{IC}}
\]

The weighting factor on the initial condition term helps the model satisfy the starting conditions more strongly [web:x+1][web:x+5].
## Why Fourier Features

Standard neural networks can struggle with oscillatory functions because of spectral bias, meaning they tend to learn low-frequency patterns first. Fourier feature embedding helps by mapping the time input into a higher-dimensional sinusoidal feature space before the main neural network processes it [web:x][web:x+3].

This improves the network’s ability to represent periodic behavior such as:

- spring oscillations,
- pendulum motion,
- wave-like solutions,
- other frequency-rich dynamics.

In this implementation, the input time \(t\) is transformed using random Fourier features:

\[
\phi(t) = [\sin(2\pi tB), \cos(2\pi tB)]
\]

where \(B\) is a fixed random projection matrix.

---

## Code Structure

### `FourierFeatures`

This class converts scalar time input into a sinusoidal embedding.  
It expands the input from 1 dimension to `2 * mapping_size` dimensions using sine and cosine transforms.

Purpose:
- improve representation of periodic functions,
- reduce spectral bias,
- make learning oscillatory motion easier.

### `FourierPINN`

This is the fully connected neural network that takes the Fourier-encoded input and predicts displacement \(x(t)\).

It uses:
- linear layers,
- `tanh` activation,
- Xavier initialization.

Purpose:
- approximate the unknown solution function \(x(t)\).

### `SpringMassFourierPINN`

This class manages the full PINN workflow.

It contains:
- the physical parameters \(m\) and \(k\),
- the Fourier feature encoder,
- the neural network,
- the optimizer,
- the loss functions,
- the training loop,
- prediction utilities.

### `physics_loss(t)`

Computes the ODE residual by:
1. predicting \(x(t)\),
2. computing first derivative \(x'(t)\),
3. computing second derivative \(x''(t)\),
4. evaluating the residual \(x''(t) + (k/m)x(t)\).

### `ic_loss(x0, v0)`

Computes the initial condition mismatch at \(t=0\).

It enforces:
- \(x(0) = x_0\),
- \(x'(0) = v_0\).

### `train(...)`

Trains the model using Adam optimization.

Training uses:
- collocation points in the time domain,
- physics loss,
- initial condition loss.

The network learns a function that satisfies the governing ODE rather than fitting labeled data.

### `predict(t)`

Evaluates the trained model at a set of time values and returns the displacement prediction.

### `analytical_solution(...)`

Computes the exact closed-form solution of the spring-mass system for comparison

### Results

The result is generated in four stages:

1. **Generate collocation points** in the time interval \([0, 10]\).
2. **Evaluate the PINN residual** using automatic differentiation to compute \(x'(t)\) and \(x''(t)\).
3. **Optimize the neural network** so that both the ODE residual and the initial conditions are satisfied.
4. **Compare the learned solution** against the analytical spring-mass solution.

After training, the code:
- predicts \(x(t)\) on a test time grid,
- computes the absolute error,
- plots the analytical and PINN solutions,
- saves the figure as `fourier_pinn_spring_mass.png`,
- saves the trained model as `fourier_pinn_spring_mass.pth`.

### Usage

```bash
python fourier_pinn_spring_mass.py
```
During training, the script prints:
- total loss,
- physics loss,
- initial condition loss.

After training, it displays and saves the solution plot.

**Output files:**
After running the script, the following files are created:

- `fourier_pinn_spring_mass.png`: comparison plot of analytical vs PINN solution and error.
- `fourier_pinn_spring_mass.pth`: saved PyTorch model weights.
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

1. **Raissi, M., Perdikas, P., & Karniadakis, G. E. (2019).** "Physics-informed deep learning (part I): Data-driven solutions of nonlinear partial differential equations." *arXiv preprint arXiv:1711.10561*. [web:y]

2. **DeepXDE Library** - Comprehensive PINN implementation for various PDEs [web:y+3]

3. **PINN Tutorial (PyTorch)** - Minimal implementation examples [web:y+6]
4. **Spring-Mass System** - Classical spring-mass motion and angular frequency relations [web:x-1][web:x+2][web:x+5].
5. **Mass-Spring-Damper Model** - PINN loss construction and physics residual formulation [web:x+1][web:x+6][web:x+7].
6. **PINN/Foruier Feature References** - Fourier feature embedding for improved oscillatory learning [web:x][web:x+3].

---

## License

BSD3 License - Don't worry, Feel Free to use for research and educational use.

---

## Author
Many People, Many factors, I am grateful to effort of everyone who made learning this possible. 
Implementations adapted for quantum chemistry and computational physics applications.

*Last updated: June 2026*
