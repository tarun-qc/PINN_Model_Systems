"""
Physics-Informed Neural Network (PINN) for solving the 1D Heat Equation
u_t = alpha * u_xx

Domain: x ∈ [0, 1], t ∈ [0, 1]
Boundary conditions: u(0, t) = 0, u(1, t) = 0
Initial condition: u(x, 0) = sin(πx)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set default dtype to float32
torch.set_default_dtype(torch.float32)
torch.manual_seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# Neural Network Architecture
# =============================================================================
class PINN(nn.Module):
    """Fully-connected feedforward neural network for PINN"""
    
    def __init__(self, layers=[2, 20, 20, 20, 20, 1]):
        super(PINN, self).__init__()
        
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)
        ])
        
        # Xavier Normal initialization
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)
    
    def forward(self, x, t):
        """
        Inputs: x (spatial), t (time) - both tensors of shape [N, 1]
        Output: u(x, t) - tensor of shape [N, 1]
        """
        inputs = torch.cat([x, t], dim=1)  # [N, 2]
        
        for i in range(len(self.linears) - 1):
            inputs = self.linears[i](inputs)
            inputs = self.activation(inputs)
        
        return self.linears[-1](inputs)  # [N, 1]


# =============================================================================
# PINN Solver
# =============================================================================
class HeatEquationPINN:
    """PINN solver for the 1D heat equation"""
    
    def __init__(self, alpha=0.1, layers=[2, 20, 20, 20, 20, 1]):
        self.alpha = alpha
        self.model = PINN(layers).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, amsgrad=False
        )
    
    def compute_pde_loss(self, x, t):
        """
        Compute PDE residual: u_t - alpha * u_xx
        Uses automatic differentiation
        """
        x = x.clone()
        t = t.clone()
        x.requires_grad = True
        t.requires_grad = True
        
        u = self.model(x, t)  # [N, 1]
        
        # First derivatives
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), 
            create_graph=True, retain_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), 
            create_graph=True, retain_graph=True
        )[0]
        
        # Second derivative u_xx
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), 
            create_graph=True, retain_graph=True
        )[0]
        
        # PDE residual: u_t - alpha * u_xx
        pde_residual = u_t - self.alpha * u_xx
        
        return torch.mean(pde_residual ** 2)
    
    def compute_boundary_loss(self):
        """Enforce u(0, t) = 0 and u(1, t) = 0"""
        t_bc = torch.linspace(0, 1, 100).to(device).view(-1, 1)
        
        # Left boundary: x = 0
        x_left = torch.zeros_like(t_bc)
        u_left = self.model(x_left, t_bc)
        loss_left = torch.mean(u_left ** 2)
        
        # Right boundary: x = 1
        x_right = torch.ones_like(t_bc)
        u_right = self.model(x_right, t_bc)
        loss_right = torch.mean(u_right ** 2)
        
        return loss_left + loss_right
    
    def compute_initial_loss(self):
        """Enforce u(x, 0) = sin(πx)"""
        x_ic = torch.linspace(0, 1, 100).to(device).view(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        
        u_pred = self.model(x_ic, t_ic)
        u_true = torch.sin(np.pi * x_ic)
        
        return torch.mean((u_pred - u_true) ** 2)
    
    def train(self, n_epochs=5000, print_freq=500):
        """Train the PINN"""
        
        # Collocation points for PDE loss
        x_pde = torch.linspace(0, 1, 500).to(device).view(-1, 1)
        t_pde = torch.linspace(0, 1, 500).to(device).view(-1, 1)
        
        # Random sampling for PDE points
        x_pde = x_pde[torch.randint(0, 500, (10000,)).to(device)]
        t_pde = t_pde[torch.randint(0, 500, (10000,)).to(device)]
        
        print("Starting training...")
        
        for epoch in range(n_epochs):
            # Compute losses
            pde_loss = self.compute_pde_loss(x_pde, t_pde)
            bc_loss = self.compute_boundary_loss()
            ic_loss = self.compute_initial_loss()
            
            # Total loss
            total_loss = pde_loss + bc_loss + ic_loss
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Print progress
            if epoch % print_freq == 0:
                print(f"Epoch {epoch:5d} | Total: {total_loss:.6e} | "
                      f"PDE: {pde_loss:.6e} | BC: {bc_loss:.6e} | IC: {ic_loss:.6e}")
        
        print("Training complete!")
    
    def predict(self, x, t):
        """Predict u(x, t) at given points"""
        x = torch.tensor(x, dtype=torch.float32).to(device).view(-1, 1)
        t = torch.tensor(t, dtype=torch.float32).to(device).view(-1, 1)
        
        with torch.no_grad():
            u = self.model(x, t)
        
        return u.cpu().numpy()


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Create and train PINN
    pinn = HeatEquationPINN(alpha=0.1)
    pinn.train(n_epochs=5000, print_freq=500)
    
    # Create grid for visualization
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0, 1, 100)
    
    u_pred = np.zeros((len(t_grid), len(x_grid)))
    
    for i, t_val in enumerate(t_grid):
        for j, x_val in enumerate(x_grid):
            u_pred[i, j] = pinn.predict([x_val], [t_val])[0, 0]
    
    # Analytical solution for comparison
    def analytical_solution(x, t, alpha=0.1):
        return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)
    
    u_analytical = np.zeros((len(t_grid), len(x_grid)))
    for i, t_val in enumerate(t_grid):
        for j, x_val in enumerate(x_grid):
            u_analytical[i, j] = analytical_solution(x_val, t_val, alpha=0.1)
    
    # Compute error
    error = np.abs(u_pred - u_analytical)
    max_error = np.max(error)
    print(f"\nMaximum absolute error: {max_error:.6e}")
    
    # Plotting
    plt.figure(figsize=(12, 4))
    
    # PINN solution
    plt.subplot(1, 3, 1)
    plt.imshow(u_pred, extent=[0, 1, 0, 1], origin='lower', 
               cmap='viridis', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('PINN Solution')
    plt.colorbar()
    
    # Analytical solution
    plt.subplot(1, 3, 2)
    plt.imshow(u_analytical, extent=[0, 1, 0, 1], origin='lower', 
               cmap='viridis', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Analytical Solution')
    plt.colorbar()
    
    # Error
    plt.subplot(1, 3, 3)
    plt.imshow(error, extent=[0, 1, 0, 1], origin='lower', 
               cmap='Reds', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Absolute Error\nMax: {max_error:.2e}')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('pinn_heat_equation.png', dpi=150)
    plt.show()
    
    # Save model
    torch.save(pinn.model.state_dict(), 'pinn_heat_model.pth')
    print("Model saved to pinn_heat_model.pth")