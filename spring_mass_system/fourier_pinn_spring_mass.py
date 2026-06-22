import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

torch.set_default_dtype(torch.float32)
torch.manual_seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FourierFeatures(nn.Module):
    def __init__(self, in_dim=1, mapping_size=32, scale=10.0):
        super().__init__()
        B = torch.randn(in_dim, mapping_size) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierPINN(nn.Module):
    def __init__(self, layers=[64, 64, 64, 64, 1]):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))
        self.act = nn.Tanh()

        for layer in self.net:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.net[:-1]:
            x = self.act(layer(x))
        return self.net[-1](x)


class SpringMassFourierPINN:
    def __init__(self, m=1.0, k=2.5, ff_dim=32, ff_scale=10.0):
        self.m = m
        self.k = k
        self.omega_sq = k / m

        self.fourier = FourierFeatures(in_dim=1, mapping_size=ff_dim, scale=ff_scale).to(device)
        self.model = FourierPINN(layers=[2 * ff_dim, 64, 64, 64, 1]).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def parameters(self):
        return list(self.fourier.parameters()) + list(self.model.parameters())

    def forward(self, t):
        return self.model(self.fourier(t))

    def physics_loss(self, t):
        t = t.clone().detach().requires_grad_(True)
        x = self.forward(t)

        dx_dt = torch.autograd.grad(
            x, t, grad_outputs=torch.ones_like(x), create_graph=True
        )[0]

        d2x_dt2 = torch.autograd.grad(
            dx_dt, t, grad_outputs=torch.ones_like(dx_dt), create_graph=True
        )[0]

        residual = d2x_dt2 + self.omega_sq * x
        return torch.mean(residual**2)

    def ic_loss(self, x0, v0):
        t0 = torch.zeros(1, 1, device=device, requires_grad=True)
        x_pred = self.forward(t0)

        dx_dt = torch.autograd.grad(
            x_pred, t0, grad_outputs=torch.ones_like(x_pred), create_graph=True
        )[0]

        loss_x0 = (x_pred - x0) ** 2
        loss_v0 = (dx_dt - v0) ** 2
        return loss_x0 + loss_v0

    def train(self, epochs=10000, print_freq=1000, x0=1.0, v0=0.0, t_max=10.0, n_colloc=200):
        t_col = torch.linspace(0, t_max, 2000, device=device).view(-1, 1)
        idx = torch.randperm(t_col.shape[0], device=device)[:n_colloc]
        t_col = t_col[idx]

        x0_t = torch.tensor([[x0]], dtype=torch.float32, device=device)
        v0_t = torch.tensor([[v0]], dtype=torch.float32, device=device)

        for epoch in range(epochs):
            p_loss = self.physics_loss(t_col)
            i_loss = self.ic_loss(x0_t, v0_t)
            loss = p_loss + 10.0 * i_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % print_freq == 0:
                print(f"Epoch {epoch:5d} | Total {loss.item():.6e} | Physics {p_loss.item():.6e} | IC {i_loss.item():.6e}")

        print("Training complete.")

    def predict(self, t):
        t = torch.tensor(t, dtype=torch.float32, device=device).view(-1, 1)
        with torch.no_grad():
            x = self.forward(t)
        return x.cpu().numpy().ravel()


def analytical_solution(t, x0=1.0, v0=0.0, m=1.0, k=2.5):
    omega = np.sqrt(k / m)
    return x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)


if __name__ == "__main__":
    m = 1.0
    k = 2.5
    x0 = 1.0
    v0 = 0.0

    pinn = SpringMassFourierPINN(m=m, k=k, ff_dim=32, ff_scale=10.0)
    pinn.train(epochs=10000, print_freq=1000, x0=x0, v0=v0, t_max=10.0, n_colloc=256)

    t_test = np.linspace(0, 10, 500)
    x_pred = pinn.predict(t_test)
    x_true = analytical_solution(t_test, x0=x0, v0=v0, m=m, k=k)

    err = np.abs(x_pred - x_true)
    print(f"Max absolute error: {err.max():.6e}")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(t_test, x_true, "r-", lw=2, label="Analytical")
    plt.plot(t_test, x_pred, "b--", lw=2, label="Fourier PINN")
    plt.xlabel("Time (s)")
    plt.ylabel("x(t)")
    plt.title("Spring-Mass Displacement")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_test, err, "k-", lw=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute Error")
    plt.title("Prediction Error")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("fourier_pinn_spring_mass.png", dpi=150)
    plt.show()

    torch.save(pinn.model.state_dict(), "fourier_pinn_spring_mass.pth")
    print("Saved model to fourier_pinn_spring_mass.pth")