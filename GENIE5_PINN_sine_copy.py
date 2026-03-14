"""
=============================================================
ML4SCI GSoC 2026 — GENIE5 Test Solution
Physics-Informed Neural Network (PINN)
Damped Harmonic Oscillator

Equation:  d²x/dz² + 2ξ·dx/dz + x = 0
Domain:    z ∈ [0, 20]
ICs:       x(0) = 0.7,  dx/dz(0) = 1.2
ξ range:   0.1 to 0.4

This version is built on the code that gave good results
(0.246 for xi=0.1, <0.025 for others) with ONE key change:
  Tanh → Sine activation (SIREN-style)

Why sine works better for this problem:
  The solution x(z) IS a sine/cosine wave times a decaying
  exponential. A network made of sine activations can
  represent this naturally. Tanh has to approximate it
  using S-curves, which is much harder.

The ONLY changes from the working version are:
  1. Sin() class added
  2. nn.Tanh() replaced with Sin()
  3. Weight initialization changed to SIREN init
  4. Input normalization added (helps sine networks)
=============================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================
# SINE ACTIVATION  (the only new class)
# =============================================================
# torch.sin() already exists but nn.Module wraps it so it
# can be used inside nn.Sequential just like nn.Tanh()
# =============================================================

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


# =============================================================
# 1.  NETWORK
# =============================================================
# Identical structure to the working version EXCEPT:
#   - Sin() instead of Tanh()
#   - SIREN weight initialization
#   - Input normalization before entering the network
#
# Input normalization:
#   z  from [0, 20]    → [-1, 1]   via  z/10 - 1
#   ξ  from [0.1, 0.4] → [-1, 1]   via  (ξ-0.25)/0.15
#
# This is important for sine networks because sin(x) for
# large x (like z=15) just gives a random-looking value.
# Normalizing to [-1,1] keeps inputs in the linear region
# of sin where the network starts learning meaningfully.
# =============================================================

class PINN(nn.Module):
    def __init__(self, hidden_layers=4, hidden_neurons=128):
        super(PINN, self).__init__()

        layers = []

        # Input: 2 neurons (z_normalized, xi_normalized)
        layers.append(nn.Linear(2, hidden_neurons))
        layers.append(Sin())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(Sin())

        layers.append(nn.Linear(hidden_neurons, 1))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        SIREN initialization — derived in the original paper.
        Without this, sine networks give garbage output.

        First layer: uniform in [-1/n_in, 1/n_in]
            Keeps first-layer outputs in [-1, 1] so sin
            starts in its approximately-linear regime.

        All other layers: uniform in [-sqrt(6/n_in), sqrt(6/n_in)]
            Ensures the distribution of each layer's output
            stays consistent through the network depth.
        """
        is_first = True
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                n_in = layer.weight.shape[1]
                if is_first:
                    bound = 1.0 / n_in
                    is_first = False
                else:
                    bound = np.sqrt(6.0 / n_in)
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.zeros_(layer.bias)

    def forward(self, z, xi):
        # Normalize inputs before passing to network
        z_norm  = (z  / 10.0) - 1.0          # [0,20]    → [-1,1]
        xi_norm = (xi - 0.25) / 0.15          # [0.1,0.4] → [-1,1]

        inp = torch.cat([z_norm, xi_norm], dim=1)
        return self.network(inp)


# =============================================================
# 2.  GRADIENT HELPER
# =============================================================
# Unchanged from the working version.
# retain_graph=True is essential — without it the second
# gradient call (for d²x/dz²) crashes because the graph
# from the first call has been freed.
# =============================================================

def gradient(output, input_var):
    return torch.autograd.grad(
        outputs=output,
        inputs=input_var,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True
    )[0]


# =============================================================
# 3.  ANALYTICAL SOLUTION
# =============================================================

def analytical_solution(z_np, xi_val):
    x0, v0  = 0.7, 1.2
    omega_d = np.sqrt(1.0 - xi_val ** 2)
    A       = x0
    B       = (v0 + xi_val * x0) / omega_d
    return np.exp(-xi_val * z_np) * (
        A * np.cos(omega_d * z_np) + B * np.sin(omega_d * z_np)
    )


# =============================================================
# 4.  LOSS FUNCTION
# =============================================================
# Unchanged from the working version.
# physics + 10*ic1 + 10*ic2
# 5000 collocation points (more than the original 3000)
# =============================================================

def compute_losses(model, N_physics=5000, N_ic=500):

    # Physics loss
#    z_phys  = (torch.rand(N_physics, 1) * 20.0).to(device)
#    xi_phys = (torch.rand(N_physics, 1) * 0.3 + 0.1).to(device)
#    z_phys.requires_grad_(True)

#    x_pred  = model(z_phys, xi_phys)
#    x_z     = gradient(x_pred, z_phys)
#    x_zz    = gradient(x_z,    z_phys)

#    residual     = x_zz + 2.0 * xi_phys * x_z + x_pred
#    loss_physics = torch.mean(residual ** 2)
    def phys_loss(z_pts, xi_pts):
        z_pts.requires_grad_(True)
        x = model(z_pts, xi_pts)
        x_z = gradient(x, z_pts)
        x_zz = gradient(x_z, z_pts)
        res = x_zz + 2.0 * xi_pts * x_z + x
        return torch.mean(res ** 2)

    # Batch 1: uniform across full domain
    z1  = (torch.rand(N_physics, 1) * 20.0).to(device)
    xi1 = (torch.rand(N_physics, 1) * 0.3 + 0.1).to(device)
    L1  = phys_loss(z1, xi1)

    # Batch 2: late time + low damping  (the hard region)
    z2  = (torch.rand(2000, 1) * 10.0 + 10.0).to(device)  # z∈[10,20]
    xi2 = (torch.rand(2000, 1) * 0.15 + 0.1).to(device)   # ξ∈[0.1,0.25]
    L2  = phys_loss(z2, xi2)

    loss_physics = L1 + 3.0 * L2   # hard region weighted 3x
    # Initial condition losses
    z_ic  = torch.zeros(N_ic, 1, requires_grad=True).to(device)
    xi_ic = (torch.rand(N_ic, 1) * 0.3 + 0.1).to(device)

    x_ic    = model(z_ic, xi_ic)
    x_z_ic  = gradient(x_ic, z_ic)

    loss_ic1 = torch.mean((x_ic   - 0.7) ** 2)
    loss_ic2 = torch.mean((x_z_ic - 1.2) ** 2)

    total = loss_physics + 10.0 * loss_ic1 + 10.0 * loss_ic2
    return total, loss_physics, loss_ic1, loss_ic2


# =============================================================
# 5.  TRAINING
# =============================================================
# Scheduler milestones tuned for 40k epochs.
# Sine networks sometimes need a slightly lower learning rate
# than Tanh networks — using 5e-4 instead of 1e-3 to be safe.
# If you see loss going to NaN in first 500 epochs,
# reduce to 1e-4.
# =============================================================

def train(model, epochs=200000, lr=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30000, 48000], gamma=0.1
    )

    history = {"total": [], "physics": [], "ic1": [], "ic2": []}

    print("Starting training...")
    print(f"{'Epoch':>8} | {'Total':>12} | {'Physics':>12} | "
          f"{'IC1':>10} | {'IC2':>10}")
    print("─" * 62)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        total, l_phys, l_ic1, l_ic2 = compute_losses(model)
        total.backward()

        # Gradient clipping — sine networks can have large
        # gradients early in training
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        history["total"].append(total.item())
        history["physics"].append(l_phys.item())
        history["ic1"].append(l_ic1.item())
        history["ic2"].append(l_ic2.item())

        if epoch % 2000 == 0 or epoch == 1:
            print(f"{epoch:>8} | {total.item():>12.6f} | "
                  f"{l_phys.item():>12.6f} | "
                  f"{l_ic1.item():>10.6f} | "
                  f"{l_ic2.item():>10.6f}")

        # Early warning: if loss explodes, stop and tell user
        if torch.isnan(total) or total.item() > 1e6:
            print(f"\n⚠️  Loss exploded at epoch {epoch}.")
            print("   Restart with lr=1e-4 instead of 5e-4")
            break

    print("\nTraining complete!")
    return history


# =============================================================
# 6.  PLOTS  (unchanged from working version)
# =============================================================

def plot_loss_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Loss History", fontsize=14, fontweight="bold")
    epochs = range(1, len(history["total"]) + 1)

    axes[0].semilogy(epochs, history["total"],
                     color="#2563eb", linewidth=1.5)
    axes[0].set_title("Total Loss (log scale)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    start_val = history["total"][0]
    end_val   = history["total"][-1]
    axes[0].annotate(f"Start\n{start_val:.3f}",
                     xy=(1, start_val),
                     xytext=(2000, start_val * 1.5),
                     fontsize=8, color="#2563eb",
                     arrowprops=dict(arrowstyle="->", color="#2563eb"))
    axes[0].annotate(f"End\n{end_val:.5f}",
                     xy=(len(epochs), end_val),
                     xytext=(len(epochs) - 12000, end_val * 50),
                     fontsize=8, color="#2563eb",
                     arrowprops=dict(arrowstyle="->", color="#2563eb"))

    axes[1].semilogy(epochs, history["physics"],
                     label="Physics (ODE residual)",
                     color="#dc2626", linewidth=1.5)
    axes[1].semilogy(epochs, history["ic1"],
                     label="IC1: x(0) = 0.7",
                     color="#16a34a", linewidth=1.5)
    axes[1].semilogy(epochs, history["ic2"],
                     label="IC2: x'(0) = 1.2",
                     color="#d97706", linewidth=1.5)
    axes[1].set_title("Loss Components (log scale)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_1_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plot_1_loss_curves.png")


def plot_predictions(model):
    xi_values = [0.1, 0.2, 0.3, 0.4]
    colors    = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    z_np     = np.linspace(0, 20, 1000)
    z_tensor = torch.tensor(z_np,
                dtype=torch.float32).reshape(-1, 1).to(device)
    errors = {}

    for idx, (xi_val, color) in enumerate(zip(xi_values, colors)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        x_exact   = analytical_solution(z_np, xi_val)
        xi_tensor = torch.full((1000, 1), xi_val,
                               dtype=torch.float32).to(device)
        with torch.no_grad():
            x_pinn = model(z_tensor, xi_tensor).cpu().numpy().flatten()

        max_err = float(np.max(np.abs(x_pinn - x_exact)))
        mse     = float(np.mean((x_pinn - x_exact) ** 2))
        errors[xi_val] = {"max": max_err, "mse": mse}

        ax.plot(z_np, x_exact, color=color, linewidth=2.5,
                label="Analytical (exact)", linestyle="-")
        ax.plot(z_np, x_pinn,  color="black", linewidth=1.5,
                label="PINN", linestyle="--", alpha=0.85)
        ax.fill_between(z_np, x_exact, x_pinn,
                        alpha=0.15, color=color, label="Error region")

        ax.set_title(f"ξ = {xi_val}   |   Max Error: {max_err:.5f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("z (time)", fontsize=10)
        ax.set_ylabel("x(z) (position)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle=":")

    fig.suptitle(
        "PINN vs Analytical Solution — Damped Harmonic Oscillator\n"
        r"$\frac{d^2x}{dz^2} + 2\xi\frac{dx}{dz} + x = 0$"
        r",  $x(0)=0.7$,  $x'(0)=1.2$",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.savefig("plot_2_pinn_vs_analytical.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plot_2_pinn_vs_analytical.png")
    return errors


def plot_energy_decay(model):
    xi_values = [0.1, 0.2, 0.3, 0.4]
    colors    = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]
    z_np      = np.linspace(0, 20, 1000)
    E0        = 0.5 * 1.2 ** 2 + 0.5 * 0.7 ** 2

    fig, ax = plt.subplots(figsize=(12, 5))

    for xi_val, color in zip(xi_values, colors):
        z_t  = torch.tensor(z_np,
                dtype=torch.float32).reshape(-1, 1).to(device)
        z_t.requires_grad_(True)
        xi_t = torch.full((1000, 1), xi_val,
                          dtype=torch.float32).to(device)

        x_p  = model(z_t, xi_t)
        vel  = gradient(x_p, z_t)
        x_np = x_p.detach().cpu().numpy().flatten()
        v_np = vel.detach().cpu().numpy().flatten()

        energy  = 0.5 * v_np ** 2 + 0.5 * x_np ** 2
        E_exact = E0 * np.exp(-2.0 * xi_val * z_np)

        ax.plot(z_np, energy,  color=color, lw=2.0,
                label=f"ξ = {xi_val} (PINN)")
        ax.plot(z_np, E_exact, color=color, lw=1.0,
                linestyle=":", alpha=0.6)

    ax.plot([], [], color="gray", lw=1.2, linestyle=":",
            label="Theoretical  e^(−2ξz)  (dotted)")
    ax.set_title(
        r"Mechanical Energy  $E(z) = \frac{1}{2}v^2 + \frac{1}{2}x^2$"
        "\nSolid = PINN,   Dotted = Theoretical  "
        r"$E_0\,e^{-2\xi z}$",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("z  (time)", fontsize=11)
    ax.set_ylabel("Energy", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_3_energy_decay.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plot_3_energy_decay.png")


def plot_phase_portrait(model):
    xi_values = [0.1, 0.2, 0.3, 0.4]
    z_np      = np.linspace(0, 20, 2000)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "Phase Portraits  x(z)  vs  dx/dz(z)\n"
        "(Inward spiral = energy dissipation — correct physics)",
        fontsize=13, fontweight="bold"
    )

    from matplotlib.collections import LineCollection
    from matplotlib.colors       import Normalize

    for ax, xi_val in zip(axes, xi_values):
        z_t  = torch.tensor(z_np,
                dtype=torch.float32).reshape(-1, 1).to(device)
        z_t.requires_grad_(True)
        xi_t = torch.full((2000, 1), xi_val,
                          dtype=torch.float32).to(device)

        x_p  = model(z_t, xi_t)
        vel  = gradient(x_p, z_t)
        x_np = x_p.detach().cpu().numpy().flatten()
        v_np = vel.detach().cpu().numpy().flatten()

        points = np.array([x_np, v_np]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        norm   = Normalize(vmin=0, vmax=20)
        lc     = LineCollection(segs, cmap="plasma",
                                norm=norm, lw=1.8, alpha=0.9)
        lc.set_array(z_np[:-1])
        ax.add_collection(lc)

        ax.plot(x_np[0],  v_np[0],  "go", ms=9, label="Start z=0")
        ax.plot(x_np[-1], v_np[-1], "rs", ms=9, label="End  z=20")

        pad = 0.15
        ax.set_xlim(x_np.min() - pad, x_np.max() + pad)
        ax.set_ylim(v_np.min() - pad, v_np.max() + pad)
        ax.axhline(0, color="gray", lw=0.7, linestyle=":")
        ax.axvline(0, color="gray", lw=0.7, linestyle=":")
        ax.set_title(f"ξ = {xi_val}", fontsize=11, fontweight="bold")
        ax.set_xlabel("x  (position)", fontsize=9)
        ax.set_ylabel("dx/dz  (velocity)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("plot_4_phase_portrait.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plot_4_phase_portrait.png")


def print_error_table(errors):
    print("\n" + "=" * 54)
    print("  PINN Accuracy Summary")
    print("=" * 54)
    print(f"  {'ξ':>6} | {'Max Abs Error':>15} | {'MSE':>15}")
    print("-" * 54)
    for xi_val, e in sorted(errors.items()):
        flag = " ✓" if e["max"] < 0.05 else " ← needs attention"
        print(f"  {xi_val:>6.1f} | {e['max']:>15.8f} | "
              f"{e['mse']:>15.8f}{flag}")
    print("=" * 54)


# =============================================================
# 7.  MAIN
# =============================================================

if __name__ == "__main__":

    model    = PINN(hidden_layers=4, hidden_neurons=128).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nModel: SIREN-style PINN")
    print(f"  4 hidden layers × 128 neurons")
    print(f"  Sine activation (SIREN init)")
    print(f"  Input normalization: z→[-1,1], ξ→[-1,1]")
    print(f"  Parameters: {n_params:,}\n")

    history = train(model, epochs=200000, lr=5e-4)

    plot_loss_curves(history)
    errors = plot_predictions(model)
    plot_energy_decay(model)
    plot_phase_portrait(model)
    print_error_table(errors)

    torch.save(model.state_dict(), "pinn_damped_oscillator.pt")

    print("\n✅ Files saved:")
    print("   plot_1_loss_curves.png")
    print("   plot_2_pinn_vs_analytical.png")
    print("   plot_3_energy_decay.png")
    print("   plot_4_phase_portrait.png")
    print("   pinn_damped_oscillator.pt")
    print("\n📬 Submit: https://forms.gle/SPXo8kSwHHptcBmk9")
