"""
Physics-Informed Neural Network (PINN) for solving the Simple Pendulum ODE
d²θ/dt² + (g/L) * sin(θ) = 0

Parameters:
- g = 9.81 m/s² (gravity)
- L = 1.0 m (pendulum length)
- θ₀ = π/4 rad (initial angle)
- dθ/dt|₀ = 0 (initial angular velocity)
"""

"""
Physics-Informed Neural Network (PINN) for solving the Simple Pendulum ODE
d²θ/dt² + (g/L) * sin(θ) = 0
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

torch.set_default_dtype(torch.float32)
torch.manual_seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PINN(nn.Module):
    def __init__(self, layers=[1, 20, 20, 20, 20, 1]):
        super(PINN, self).__init__()
        
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)
        ])
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)
    
    def forward(self, t):
        for i in range(len(self.linears) - 1):
            t = self.linears[i](t)
            t = self.activation(t)
        
        return self.linears[-1](t)


class PendulumPINN:
    def __init__(self, g=9.81, L=1.0, layers=[1, 20, 20, 20, 20, 1]):
        self.g = g
        self.L = L
        self.omega_sq = g / L
        self.model = PINN(layers).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, amsgrad=False
        )
    
    def compute_physics_loss(self, t):
        t = t.clone()
        t.requires_grad = True
        
        theta = self.model(t)
        
        d_theta_dt = torch.autograd.grad(
            theta, t, grad_outputs=torch.ones_like(theta), 
            create_graph=True, retain_graph=True
        )[0]
        
        d2_theta_dt2 = torch.autograd.grad(
            d_theta_dt, t, grad_outputs=torch.ones_like(d_theta_dt), 
            create_graph=True, retain_graph=True
        )[0]
        
        residual = d2_theta_dt2 + self.omega_sq * torch.sin(theta)
        
        return torch.mean(residual ** 2)
    
    def compute_initial_condition_loss(self, theta0, dtheta0):
        t0 = torch.zeros(1, 1).to(device)
        
        theta_pred = self.model(t0)
        
        t0.requires_grad = True
        d_theta_dt = torch.autograd.grad(
            self.model(t0), t0, grad_outputs=torch.ones_like(theta_pred), 
            create_graph=True, retain_graph=True
        )[0]
        
        theta_loss = (theta_pred - theta0) ** 2
        dtheta_loss = (d_theta_dt - dtheta0) ** 2
        
        return theta_loss + dtheta_loss
    
    def train(self, n_epochs=10000, print_freq=1000, 
              theta0=np.pi/4, dtheta0=0.0):
        
        t_physics = torch.linspace(0, 10, 500).to(device).view(-1, 1)
        t_physics = t_physics[torch.randint(0, 500, (20000,)).to(device)]
        
        print("Starting training...")
        print(f"Initial conditions: θ₀ = {theta0:.4f} rad, dθ/dt₀ = {dtheta0:.4f} rad/s")
        
        for epoch in range(n_epochs):
            physics_loss = self.compute_physics_loss(t_physics)
            ic_loss = self.compute_initial_condition_loss(theta0, dtheta0)
            
            total_loss = physics_loss + 10 * ic_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # FIX: Use .item() to extract scalar from tensor [web:31][web:32]
            if epoch % print_freq == 0:
                print(f"Epoch {epoch:5d} | Total: {total_loss.item():.6e} | "
                      f"Physics: {physics_loss.item():.6e} | IC: {ic_loss.item():.6e}")
        
        print("Training complete!")
    
    def predict(self, t):
        t = torch.tensor(t, dtype=torch.float32).to(device).view(-1, 1)
        
        with torch.no_grad():
            theta = self.model(t)
        
        return theta.cpu().numpy()


def numerical_solution_pendulum(t, theta0, dtheta0=0.0, g=9.81, L=1.0):
    omega_sq = g / L
    
    def pendulum_ode(state, t, omega_sq):
        theta, v = state
        dtheta_dt = v
        dv_dt = -omega_sq * np.sin(theta)
        return [dtheta_dt, dv_dt]
    
    initial_state = [theta0, dtheta0]
    states = odeint(pendulum_ode, initial_state, t, args=(omega_sq,))
    
    return states[:, 0]


if __name__ == "__main__":
    g = 9.81
    L = 1.0
    theta0 = np.pi / 4
    dtheta0 = 0.0
    
    pinn = PendulumPINN(g=g, L=L)
    pinn.train(n_epochs=10000, print_freq=1000, 
               theta0=theta0, dtheta0=dtheta0)
    
    t_test = np.linspace(0, 10, 500)
    theta_pinn = pinn.predict(t_test)
    theta_numerical = numerical_solution_pendulum(t_test, theta0, dtheta0, g, L)
    
    error = np.abs(theta_pinn - theta_numerical)
    max_error = np.max(error)
    
    # FIX: Use .item() for tensor values [web:31]
    print(f"\nMaximum absolute error: {max_error:.6e}")
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(t_test, theta_numerical, 'r-', linewidth=2, label='Numerical')
    plt.plot(t_test, theta_pinn, 'g-', linewidth=2, label='PINN', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('θ (rad)')
    plt.title('Simple Pendulum: θ(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(t_test, error, 'g-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute Error')
    plt.title('Prediction Error')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('pinn_pendulum_solution.png', dpi=150)
    plt.show()
    
    torch.save(pinn.model.state_dict(), 'pinn_pendulum_model.pth')
    print("Model saved to pinn_pendulum_model.pth")