"""
Reference implementation for the paper:
"Beyond Local Stability: Finite-Time Error Amplification and
Adversarial Robustness in Learned Dynamical Systems"

Dependencies:
    numpy, torch, matplotlib
Optional:
    pandas, scipy, scikit-learn

The default experiment is intentionally small and reproducible:
Lorenz-96 -> train one-step MLP surrogate -> explicit Jacobian propagator ->
transient-growth direction -> nonlinear PGD adversarial direction ->
epsilon sweep -> rollout-error bound diagnostics.

For publication-scale experiments, increase dataset/model sizes and use the
matrix-free routines rather than forming full propagators.
"""

from __future__ import annotations
import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# -------------------------- reproducibility -------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------- Lorenz-96 ------------------------------ #

def lorenz96_rhs(x: np.ndarray, forcing: float = 8.0) -> np.ndarray:
    """dx_i/dt = (x_{i+1}-x_{i-2}) x_{i-1} - x_i + F, cyclic indices."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + forcing


def rk4_step(x: np.ndarray, dt: float, forcing: float = 8.0) -> np.ndarray:
    k1 = lorenz96_rhs(x, forcing)
    k2 = lorenz96_rhs(x + 0.5 * dt * k1, forcing)
    k3 = lorenz96_rhs(x + 0.5 * dt * k2, forcing)
    k4 = lorenz96_rhs(x + dt * k3, forcing)
    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def simulate_l96(
    n: int = 20,
    forcing: float = 8.0,
    dt: float = 0.01,
    steps: int = 30000,
    burnin: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = forcing * np.ones(n) + 0.01 * rng.standard_normal(n)
    traj = []
    for k in range(steps + burnin):
        x = rk4_step(x, dt, forcing)
        if k >= burnin:
            traj.append(x.copy())
    return np.asarray(traj)


# ------------------------------ model -------------------------------- #

class ResidualMLP(nn.Module):
    """One-step map x_{k+1} = x_k + NN(x_k)."""
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


@dataclass
class Normalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean


def make_dataset(traj: np.ndarray, train_fraction: float = 0.7):
    x = torch.as_tensor(traj[:-1])
    y = torch.as_tensor(traj[1:])
    ntr = int(train_fraction * len(x))
    mean = x[:ntr].mean(0)
    std = x[:ntr].std(0).clamp_min(1e-8)
    norm = Normalizer(mean, std)
    return norm.encode(x[:ntr]), norm.encode(y[:ntr]), \
           norm.encode(x[ntr:]), norm.encode(y[ntr:]), norm


def train_model(
    model: nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(xtr)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            pred = model(xtr[idx])
            loss = ((pred - ytr[idx])**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        print(f"epoch {epoch+1:03d} mse={running/n:.6e}")


# --------------------------- rollout tools ---------------------------- #

def rollout(model: nn.Module, x0: torch.Tensor, steps: int) -> torch.Tensor:
    xs = [x0]
    x = x0
    for _ in range(steps):
        x = model(x)
        xs.append(x)
    return torch.stack(xs)


def true_rollout_normalized(
    x0_norm: torch.Tensor,
    norm: Normalizer,
    steps: int,
    dt: float,
    forcing: float,
) -> torch.Tensor:
    x = norm.decode(x0_norm).detach().cpu().numpy()
    xs = [x.copy()]
    for _ in range(steps):
        x = rk4_step(x, dt, forcing)
        xs.append(x.copy())
    return norm.encode(torch.as_tensor(np.asarray(xs)))


# -------------------- Jacobians and propagators ---------------------- #

def explicit_jacobian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Jacobian df/dx at one state. Suitable for small dimensions."""
    x = x.detach().clone().requires_grad_(True)
    return torch.autograd.functional.jacobian(
        lambda z: model(z), x, create_graph=False, vectorize=True
    ).detach()


def explicit_propagator(
    model: nn.Module, x0: torch.Tensor, horizon: int
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Propagator along the model's nominal trajectory:
    Phi(T,0) = J_{T-1} ... J_0 (column-vector convention).
    """
    xs = rollout(model, x0, horizon).detach()
    dim = x0.numel()
    phi = torch.eye(dim, dtype=x0.dtype)
    jacobians = []
    for k in range(horizon):
        J = explicit_jacobian(model, xs[k])
        jacobians.append(J)
        phi = J @ phi
    return phi, xs, jacobians


def tg_svd(phi: torch.Tensor):
    U, S, Vh = torch.linalg.svd(phi)
    v_tg = Vh[0]  # leading RIGHT singular vector
    return S[0], v_tg / torch.linalg.vector_norm(v_tg), U[:, 0]


def local_metrics(J: torch.Tensor) -> Dict[str, float]:
    eig = torch.linalg.eigvals(J)
    return {
        "spectral_radius": float(eig.abs().max()),
        "sigma1_J": float(torch.linalg.svdvals(J)[0]),
        "fro_J": float(torch.linalg.matrix_norm(J, ord="fro")),
    }


# ------------------------- matrix-free Phi --------------------------- #

def jvp_step(model: nn.Module, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _, jv = torch.func.jvp(model, (x,), (v,))
    return jv


def vjp_step(model: nn.Module, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    _, vjp_fn = torch.func.vjp(model, x)
    return vjp_fn(w)[0]


def phi_mv(
    model: nn.Module, trajectory: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    out = v
    for x in trajectory[:-1]:
        out = jvp_step(model, x, out)
    return out


def phiT_mv(
    model: nn.Module, trajectory: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    out = w
    for x in reversed(trajectory[:-1]):
        out = vjp_step(model, x, out)
    return out


def leading_singular_matrix_free(
    model: nn.Module,
    trajectory: torch.Tensor,
    iters: int = 30,
    tol: float = 1e-8,
):
    v = torch.randn_like(trajectory[0])
    v /= torch.linalg.vector_norm(v)
    last = None
    for _ in range(iters):
        u = phi_mv(model, trajectory, v)
        z = phiT_mv(model, trajectory, u)
        v_new = z / torch.linalg.vector_norm(z).clamp_min(1e-15)
        if last is not None and torch.linalg.vector_norm(v_new - v) < tol:
            v = v_new
            break
        v = v_new
        last = v
    phi_v = phi_mv(model, trajectory, v)
    sigma = torch.linalg.vector_norm(phi_v)
    return sigma.detach(), v.detach()


# ---------------------- nonlinear adversarial PGD ------------------- #

def final_state(model: nn.Module, x: torch.Tensor, horizon: int) -> torch.Tensor:
    for _ in range(horizon):
        x = model(x)
    return x


def pgd_state_separation(
    model: nn.Module,
    x0: torch.Tensor,
    horizon: int,
    epsilon: float,
    steps: int = 100,
    step_size: float | None = None,
    restarts: int = 5,
    init_direction: torch.Tensor | None = None,
):
    """
    Maximize ||f^T(x0+delta)-f^T(x0)||^2 subject to ||delta||_2 <= epsilon.
    """
    if step_size is None:
        step_size = 2.0 * epsilon / max(steps, 1)

    with torch.no_grad():
        target = final_state(model, x0, horizon)

    best_loss = -float("inf")
    best_delta = None

    for r in range(restarts):
        if r == 0 and init_direction is not None:
            d0 = init_direction / torch.linalg.vector_norm(init_direction)
        else:
            d0 = torch.randn_like(x0)
            d0 /= torch.linalg.vector_norm(d0)
        delta = (epsilon * d0).detach().requires_grad_(True)

        for _ in range(steps):
            pert = final_state(model, x0 + delta, horizon)
            loss = ((pert - target)**2).sum()
            grad, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                gnorm = torch.linalg.vector_norm(grad).clamp_min(1e-15)
                delta += step_size * grad / gnorm
                dnorm = torch.linalg.vector_norm(delta)
                if dnorm > epsilon:
                    delta *= epsilon / dnorm
            delta.requires_grad_(True)

        with torch.no_grad():
            loss = ((final_state(model, x0 + delta, horizon) - target)**2).sum()
            if loss.item() > best_loss:
                best_loss = loss.item()
                best_delta = delta.detach().clone()

    v_adv = best_delta / torch.linalg.vector_norm(best_delta)
    return best_delta, v_adv, best_loss


# -------------------- nonlinear/linear comparison ------------------- #

def epsilon_sweep(
    model: nn.Module,
    x0: torch.Tensor,
    horizon: int,
    epsilons: np.ndarray,
    pgd_steps: int = 100,
    restarts: int = 5,
):
    phi, xs, _ = explicit_propagator(model, x0, horizon)
    sigma, v_tg, _ = tg_svd(phi)
    rows = []

    with torch.no_grad():
        nominal_T = final_state(model, x0, horizon)

    for eps in epsilons:
        _, v_adv, adv_loss = pgd_state_separation(
            model, x0, horizon, float(eps),
            steps=pgd_steps, restarts=restarts, init_direction=v_tg
        )
        alignment = torch.abs(torch.dot(v_tg, v_adv)).item()

        with torch.no_grad():
            tg_sep = torch.linalg.vector_norm(
                final_state(model, x0 + float(eps)*v_tg, horizon) - nominal_T
            ).item()
        g_nl_tg = (tg_sep / float(eps))**2

        rows.append({
            "epsilon": float(eps),
            "alignment": alignment,
            "sigma1_sq": float(sigma**2),
            "G_NL_TG": g_nl_tg,
            "G_NL_PGD": adv_loss / float(eps)**2,
        })
    return rows


# ---------------------- rollout error decomposition ----------------- #

def error_propagation_diagnostics(
    model: nn.Module,
    x0_true_norm: torch.Tensor,
    norm: Normalizer,
    horizon: int,
    dt: float,
    forcing: float,
):
    """
    Diagnose rollout error as repeated one-step defects amplified by finite-time
    tail propagators. With delta_k = xhat_k - x_k and

        e_k = f(x_k) - x_{k+1},

    the first-order model is

        delta_T^lin = sum_{k=0}^{T-1} Phi(T,k+1) e_k,

    while the triangle-inequality upper estimate (nonlinear Taylor remainder
    omitted) is

        B_T = sum_k sigma_1(Phi(T,k+1)) ||e_k||.

    The routine returns both quantities so that the paper can distinguish
    worst-case amplification from the amplification of the defects actually
    produced by the learned model.
    """
    true_x = true_rollout_normalized(x0_true_norm, norm, horizon, dt, forcing)
    model_x = rollout(model, x0_true_norm, horizon).detach()
    delta_actual = model_x[-1] - true_x[-1]
    actual = torch.linalg.vector_norm(delta_actual).item()

    with torch.no_grad():
        defects = torch.stack([
            model(true_x[k]) - true_x[k+1] for k in range(horizon)
        ])

    Js = [explicit_jacobian(model, true_x[k]) for k in range(horizon)]
    dim = x0_true_norm.numel()
    P = torch.eye(dim, dtype=x0_true_norm.dtype, device=x0_true_norm.device)
    delta_linear = torch.zeros_like(x0_true_norm)
    bound = 0.0
    weighted_terms = []

    # Moving backward, P is Phi(T,k+1).  At k=T-1, Phi(T,T)=I.
    for k in reversed(range(horizon)):
        propagated_defect = P @ defects[k]
        delta_linear = delta_linear + propagated_defect

        # SVD of the tail propagator Phi(T,k+1).  The leading right singular
        # vector is the worst-case direction for a perturbation injected at k+1.
        U_tail, S_tail, Vh_tail = torch.linalg.svd(P)
        sigma_tail = S_tail[0].item()
        defect_norm = torch.linalg.vector_norm(defects[k]).item()
        propagated_defect_norm = torch.linalg.vector_norm(propagated_defect).item()
        worst_case_term = sigma_tail * defect_norm
        bound += worst_case_term

        if defect_norm > 1e-15:
            defect_gain = propagated_defect_norm / defect_norm
            # P is real in the present experiments, so Vh_tail[0] contains v1.
            defect_unit = defects[k] / defect_norm
            v1 = Vh_tail[0]
            # Phi(T,T)=I has a completely degenerate singular spectrum; its
            # leading singular vector is not unique, so alignment is undefined.
            if P.shape[0] > 1 and (S_tail[0] - S_tail[1]).abs().item() > 1e-10:
                tg_alignment = torch.abs(torch.dot(v1, defect_unit)).item()
            else:
                tg_alignment = float("nan")
            gain_efficiency = defect_gain / max(sigma_tail, 1e-15)
        else:
            defect_gain = float("nan")
            tg_alignment = float("nan")
            gain_efficiency = float("nan")

        # Exact singular-basis decomposition of the realized linear amplification.
        # If e = sum_i c_i v_i, then ||P e||^2 = sum_i sigma_i^2 |c_i|^2.
        # The normalized modal contributions p_i therefore sum to one whenever
        # the propagated defect has nonzero norm.
        spectrum_nontrivial = (S_tail.max() - S_tail.min()).abs().item() > 1e-10
        if defect_norm > 1e-15 and propagated_defect_norm > 1e-15 and spectrum_nontrivial:
            coeffs = Vh_tail @ defects[k]
            modal_energy = (S_tail * coeffs).pow(2)
            modal_total = modal_energy.sum().item()
            modal_fraction = (modal_energy / max(modal_total, 1e-30)).detach().cpu().numpy()
            cumulative_fraction = np.cumsum(modal_fraction)
            m90 = int(np.searchsorted(cumulative_fraction, 0.90) + 1)
            participation = float(1.0 / np.sum(modal_fraction**2))
        else:
            coeffs = torch.zeros_like(S_tail)
            modal_fraction = np.full(len(S_tail), np.nan)
            cumulative_fraction = np.full(len(S_tail), np.nan)
            m90 = -1
            participation = float("nan")

        mode_records = []
        for mode_i in range(len(S_tail)):
            mode_records.append({
                "mode": int(mode_i + 1),
                "sigma": float(S_tail[mode_i].item()),
                "defect_coefficient_abs": float(torch.abs(coeffs[mode_i]).item()),
                "realized_energy_fraction": float(modal_fraction[mode_i]),
                "cumulative_realized_fraction": float(cumulative_fraction[mode_i]),
            })

        weighted_terms.append({
            "k": int(k),
            "sigma_tail": sigma_tail,
            "defect_norm": defect_norm,
            "propagated_defect_norm": propagated_defect_norm,
            "defect_gain": defect_gain,
            "tg_alignment": tg_alignment,
            "gain_efficiency": gain_efficiency,
            "worst_case_term": worst_case_term,
            "realized_fraction_mode1": float(modal_fraction[0]),
            "realized_fraction_top2": float(np.sum(modal_fraction[:min(2, len(modal_fraction))])),
            "realized_fraction_top3": float(np.sum(modal_fraction[:min(3, len(modal_fraction))])),
            "realized_fraction_top5": float(np.sum(modal_fraction[:min(5, len(modal_fraction))])),
            "modes_for_90pct": m90,
            "realized_participation_modes": participation,
            "modes": mode_records,
        })
        P = P @ Js[k]

    weighted_terms.reverse()
    linear_norm = torch.linalg.vector_norm(delta_linear).item()
    linear_vector_error = torch.linalg.vector_norm(delta_linear - delta_actual).item()
    actual_norm_safe = actual + 1e-15

    # Scalar summaries of how the actual defects interact with the propagators.
    propagated_sum = sum(t["propagated_defect_norm"] for t in weighted_terms)
    defect_sum = sum(t["defect_norm"] for t in weighted_terms)
    cancellation_ratio = linear_norm / (propagated_sum + 1e-15)
    effective_defect_gain = propagated_sum / (defect_sum + 1e-15)
    valid_align = [t for t in weighted_terms if np.isfinite(t["tg_alignment"])]
    if valid_align:
        alignment_mean = float(np.mean([t["tg_alignment"] for t in valid_align]))
        alignment_weighted = float(
            sum(t["defect_norm"] * t["tg_alignment"] for t in valid_align)
            / (sum(t["defect_norm"] for t in valid_align) + 1e-15)
        )
        efficiency_mean = float(np.mean([t["gain_efficiency"] for t in valid_align]))
    else:
        alignment_mean = alignment_weighted = efficiency_mean = float("nan")

    return {
        "actual_final_error": actual,
        "linear_predicted_error": linear_norm,
        "linear_to_actual_ratio": linear_norm / actual_norm_safe,
        "linear_vector_error": linear_vector_error,
        "first_order_bound_no_remainder": bound,
        "bound_to_actual_ratio": bound / actual_norm_safe,
        "defect_l2_rms": float(torch.sqrt(torch.mean(torch.sum(defects**2, dim=-1)))),
        "defect_l2_sum": float(torch.linalg.vector_norm(defects, dim=-1).sum()),
        "propagated_defect_norm_sum": propagated_sum,
        "effective_defect_gain": effective_defect_gain,
        "defect_cancellation_ratio": cancellation_ratio,
        "tg_alignment_mean": alignment_mean,
        "tg_alignment_weighted": alignment_weighted,
        "gain_efficiency_mean": efficiency_mean,
        "terms": weighted_terms,
    }


# Backward-compatible name used by the original single-window diagnostic.
def error_bound_diagnostics(*args, **kwargs):
    return error_propagation_diagnostics(*args, **kwargs)


# ---------------- publication-scale diagnostic campaign --------------- #

def relative_final_rollout_error(
    model: nn.Module,
    x0: torch.Tensor,
    norm: Normalizer,
    horizon: int,
    dt: float,
    forcing: float,
) -> Tuple[float, float]:
    """Return absolute and relative final-state rollout errors."""
    true_x = true_rollout_normalized(x0, norm, horizon, dt, forcing)
    pred_x = rollout(model, x0, horizon).detach()
    err = torch.linalg.vector_norm(pred_x[-1] - true_x[-1]).item()
    denom = torch.linalg.vector_norm(true_x[-1]).item() + 1e-12
    return err, err / denom


def diagnostic_campaign(
    model: nn.Module,
    xte: torch.Tensor,
    norm: Normalizer,
    horizons: List[int],
    n_test_states: int,
    dt: float,
    forcing: float,
    seed: int,
) -> List[Dict]:
    """Evaluate TG and local metrics over many held-out states and horizons."""
    max_h = max(horizons)
    max_start = len(xte) - max_h - 1
    if max_start < 0:
        raise ValueError("Test set is shorter than the largest requested horizon.")
    n_states = min(n_test_states, max_start + 1)
    test_indices = np.linspace(0, max_start, n_states, dtype=int)
    rows: List[Dict] = []
    term_rows: List[Dict] = []
    mode_rows: List[Dict] = []

    for j, test_index in enumerate(test_indices):
        x0 = xte[int(test_index)].detach()
        with torch.no_grad():
            one_step_error = torch.linalg.vector_norm(
                model(x0) - xte[int(test_index) + 1]
            ).item()

        for T in horizons:
            phi, _, Js = explicit_propagator(model, x0, int(T))
            sigma, _, _ = tg_svd(phi)
            local = local_metrics(Js[0])
            abs_err, rel_err = relative_final_rollout_error(
                model, x0, norm, int(T), dt, forcing
            )
            propagation = error_propagation_diagnostics(
                model, x0, norm, int(T), dt, forcing
            )
            rows.append({
                "seed": seed,
                "test_index": int(test_index),
                "horizon": int(T),
                "sigma1_phi": float(sigma),
                "Gmax": float(sigma**2),
                "log_Gmax": float(np.log(max(float(sigma**2), 1e-300))),
                "spectral_radius_J": local["spectral_radius"],
                "sigma1_J": local["sigma1_J"],
                "fro_J": local["fro_J"],
                "one_step_error": one_step_error,
                "rollout_error": abs_err,
                "relative_rollout_error": rel_err,
                "linear_predicted_error": propagation["linear_predicted_error"],
                "linear_to_actual_ratio": propagation["linear_to_actual_ratio"],
                "linear_vector_error": propagation["linear_vector_error"],
                "defect_amplification_bound": propagation["first_order_bound_no_remainder"],
                "bound_to_actual_ratio": propagation["bound_to_actual_ratio"],
                "defect_l2_rms": propagation["defect_l2_rms"],
                "defect_l2_sum": propagation["defect_l2_sum"],
                "propagated_defect_norm_sum": propagation["propagated_defect_norm_sum"],
                "effective_defect_gain": propagation["effective_defect_gain"],
                "defect_cancellation_ratio": propagation["defect_cancellation_ratio"],
                "tg_alignment_mean": propagation["tg_alignment_mean"],
                "tg_alignment_weighted": propagation["tg_alignment_weighted"],
                "gain_efficiency_mean": propagation["gain_efficiency_mean"],
            })
            for term in propagation["terms"]:
                scalar_term = {key: value for key, value in term.items() if key != "modes"}
                term_rows.append({
                    "seed": seed,
                    "test_index": int(test_index),
                    "horizon": int(T),
                    **scalar_term,
                })
                for mode in term["modes"]:
                    mode_rows.append({
                        "seed": seed,
                        "test_index": int(test_index),
                        "horizon": int(T),
                        "k": int(term["k"]),
                        **mode,
                    })
        print(f"diagnostic state {j+1:03d}/{len(test_indices):03d}")
    return rows, term_rows, mode_rows


def save_error_propagation_figures(rows: List[Dict], outdir: Path):
    """Figures testing defect injection x finite-time amplification."""
    horizons = sorted({int(r["horizon"]) for r in rows})
    ncols = min(4, len(horizons))
    nrows = int(math.ceil(len(horizons) / ncols))

    # Figure 6a: first-order vector prediction versus actual nonlinear error.
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.3*nrows), squeeze=False)
    for ax, T in zip(axes.ravel(), horizons):
        rr = [r for r in rows if int(r["horizon"]) == T]
        x = np.array([max(r["linear_predicted_error"], 1e-300) for r in rr])
        y = np.array([max(r["rollout_error"], 1e-300) for r in rr])
        rho = spearman_corr(np.log(x), np.log(y))
        ax.scatter(x, y, s=16, alpha=0.65)
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(fr"$T={T}$, Spearman $\rho={rho:.2f}$")
        ax.set_xlabel(r"$\|\delta x_T^{\rm lin}\|_2$")
        ax.set_ylabel(r"$\|\delta x_T^{\rm actual}\|_2$")
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(horizons):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(outdir / "fig6a_linear_error_prediction.pdf"); plt.close(fig)

    # Figure 6b: compare three scalar diagnostics against actual rollout error.
    T = max(horizons)
    rr = [r for r in rows if int(r["horizon"]) == T]
    diagnostics = [
        ("Gmax", r"$G_{\max}$"),
        ("defect_l2_sum", r"$\sum_k\|e_k\|$"),
        ("linear_predicted_error", r"$\|\sum_k\Phi(T,k+1)e_k\|$"),
        ("defect_amplification_bound", r"$\sum_k\sigma_1(\Phi_{T,k+1})\|e_k\|$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    y = np.array([max(r["rollout_error"], 1e-300) for r in rr])
    for ax, (key, label) in zip(axes, diagnostics):
        x = np.array([max(r[key], 1e-300) for r in rr])
        rho = spearman_corr(np.log(x), np.log(y))
        ax.scatter(x, y, s=18, alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(fr"$\rho_s={rho:.2f}$")
        ax.set_xlabel(label); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("actual rollout error")
    fig.suptitle(fr"Error-injection diagnostics, $T={T}$")
    fig.tight_layout(); fig.savefig(outdir / "fig6b_error_injection_diagnostics.pdf"); plt.close(fig)

    # Figure 6c: accuracy of first-order prediction as horizon increases.
    med, q10, q90 = [], [], []
    for T in horizons:
        vals = np.array([r["linear_to_actual_ratio"] for r in rows if int(r["horizon"]) == T])
        med.append(np.median(vals)); q10.append(np.quantile(vals, .1)); q90.append(np.quantile(vals, .9))
    fig = plt.figure(); ax = fig.add_subplot(111)
    ax.plot(horizons, med, marker="o", label="median")
    ax.fill_between(horizons, q10, q90, alpha=.2, label="10--90%")
    ax.axhline(1.0, linestyle="--")
    ax.set_xlabel("horizon T")
    ax.set_ylabel(r"$\|\delta x_T^{\rm lin}\|/\|\delta x_T^{\rm actual}\|$")
    ax.legend(); ax.grid(True, alpha=.3); fig.tight_layout()
    fig.savefig(outdir / "fig6c_linearization_accuracy_vs_horizon.pdf"); plt.close(fig)


def write_rows_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation without a SciPy dependency (ties are rare here)."""
    rx = np.empty_like(np.argsort(x), dtype=float)
    ry = np.empty_like(np.argsort(y), dtype=float)
    rx[np.argsort(x)] = np.arange(len(x), dtype=float)
    ry[np.argsort(y)] = np.arange(len(y), dtype=float)
    if len(x) < 2 or np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def save_error_aligned_figures(rows: List[Dict], term_rows: List[Dict], outdir: Path):
    """Figures diagnosing error-aligned rather than worst-case amplification."""
    horizons = sorted({int(r["horizon"]) for r in rows})
    T = max(horizons)
    rr = [r for r in rows if int(r["horizon"]) == T]
    tt = [r for r in term_rows if int(r["horizon"]) == T and np.isfinite(r["tg_alignment"])]

    # Figure 7a: actual gain experienced by each injected defect versus its
    # alignment with the leading transient-growth direction.
    fig = plt.figure(); ax = fig.add_subplot(111)
    x = np.array([r["tg_alignment"] for r in tt])
    y = np.array([r["gain_efficiency"] for r in tt])
    ax.scatter(x, y, s=14, alpha=.55)
    ax.set_xlabel(r"defect--TG alignment $|v_1^T e_k|/\|e_k\|$")
    ax.set_ylabel(r"gain efficiency $\|\Phi e_k\|/(\sigma_1\|e_k\|)$")
    ax.set_xlim(-.02, 1.02); ax.set_ylim(bottom=0)
    ax.grid(True, alpha=.3); fig.tight_layout()
    fig.savefig(outdir / "fig7a_defect_alignment_vs_gain_efficiency.pdf"); plt.close(fig)

    # Figure 7b: compare scalar summaries against actual rollout error.
    diagnostics = [
        ("defect_l2_sum", r"$\sum_k\|e_k\|$"),
        ("propagated_defect_norm_sum", r"$\sum_k\|\Phi(T,k+1)e_k\|$"),
        ("linear_predicted_error", r"$\|\sum_k\Phi(T,k+1)e_k\|$"),
        ("tg_alignment_weighted", r"weighted defect--TG alignment"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    yerr = np.array([max(r["rollout_error"], 1e-300) for r in rr])
    for ax, (key, label) in zip(axes, diagnostics):
        xval = np.array([r[key] for r in rr])
        mask = np.isfinite(xval)
        rho = spearman_corr(xval[mask], np.log(yerr[mask])) if mask.sum() > 1 else float("nan")
        ax.scatter(xval[mask], yerr[mask], s=18, alpha=.7)
        ax.set_yscale("log")
        if key != "tg_alignment_weighted": ax.set_xscale("log")
        ax.set_title(fr"$\rho_s={rho:.2f}$")
        ax.set_xlabel(label); ax.grid(True, alpha=.3)
    axes[0].set_ylabel("actual rollout error")
    fig.suptitle(fr"Error-aligned amplification diagnostics, $T={T}$")
    fig.tight_layout(); fig.savefig(outdir / "fig7b_error_aligned_diagnostics.pdf"); plt.close(fig)

    # Figure 7c: cancellation of propagated defect vectors with horizon.
    med, q10, q90 = [], [], []
    for h in horizons:
        vals = np.array([r["defect_cancellation_ratio"] for r in rows if int(r["horizon"]) == h])
        med.append(np.median(vals)); q10.append(np.quantile(vals,.1)); q90.append(np.quantile(vals,.9))
    fig = plt.figure(); ax = fig.add_subplot(111)
    ax.plot(horizons, med, marker="o", label="median")
    ax.fill_between(horizons, q10, q90, alpha=.2, label="10--90%")
    ax.set_xlabel("horizon T")
    ax.set_ylabel(r"cancellation ratio $\|\sum_k\Phi e_k\|/\sum_k\|\Phi e_k\|$")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=.3); fig.tight_layout()
    fig.savefig(outdir / "fig7c_defect_cancellation_vs_horizon.pdf"); plt.close(fig)


def save_spectral_realization_figures(term_rows: List[Dict], mode_rows: List[Dict], outdir: Path):
    """Figure 8: how many singular directions realize the propagated defect energy."""
    if not term_rows or not mode_rows:
        return
    T = max(int(r["horizon"]) for r in term_rows)
    # Exclude identity tails and zero/undefined defects.  Each remaining curve is
    # the cumulative fraction P_m = sum_{i<=m} p_i for one injected defect.
    groups = {}
    for r in mode_rows:
        if int(r["horizon"]) != T or not np.isfinite(r["realized_energy_fraction"]):
            continue
        groups.setdefault((int(r["test_index"]), int(r["k"])), []).append(r)
    curves = []
    for recs in groups.values():
        recs = sorted(recs, key=lambda q: int(q["mode"]))
        vals = np.array([q["cumulative_realized_fraction"] for q in recs], dtype=float)
        if np.all(np.isfinite(vals)):
            curves.append(vals)
    if not curves:
        return
    M = min(len(c) for c in curves)
    A = np.stack([c[:M] for c in curves])
    modes = np.arange(1, M + 1)
    med = np.median(A, axis=0); q10 = np.quantile(A,.1,axis=0); q90 = np.quantile(A,.9,axis=0)

    fig = plt.figure(); ax = fig.add_subplot(111)
    ax.plot(modes, med, marker="o", markersize=3, label="median")
    ax.fill_between(modes, q10, q90, alpha=.2, label="10--90%")
    ax.axhline(.9, linestyle="--", linewidth=1, label="90%")
    ax.set_xlabel("number of leading singular directions $m$")
    ax.set_ylabel(r"cumulative realized amplification $P_m$")
    ax.set_xlim(1, M); ax.set_ylim(0, 1.02); ax.grid(True, alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "fig8a_cumulative_realized_singular_amplification.pdf"); plt.close(fig)

    tt = [r for r in term_rows if int(r["horizon"]) == T and int(r["modes_for_90pct"]) > 0]
    m90 = np.array([r["modes_for_90pct"] for r in tt], dtype=float)
    part = np.array([r["realized_participation_modes"] for r in tt], dtype=float)
    fig, axes = plt.subplots(1,2,figsize=(8,3.4))
    bins = np.arange(.5, M+1.5, 1)
    axes[0].hist(m90, bins=bins)
    axes[0].set_xlabel(r"modes required for $P_m\geq0.9$")
    axes[0].set_ylabel("count")
    axes[0].grid(True, alpha=.3)
    axes[1].hist(part[np.isfinite(part)], bins=min(15,max(5,len(part)//5)))
    axes[1].set_xlabel("realized participation number")
    axes[1].set_ylabel("count")
    axes[1].grid(True, alpha=.3)
    fig.suptitle(fr"Spectral dimensionality of realized amplification, $T={T}$")
    fig.tight_layout(); fig.savefig(outdir / "fig8b_realized_spectral_dimensionality.pdf"); plt.close(fig)


def save_diagnostic_figures(rows: List[Dict], outdir: Path):
    """Paper Figures 2--3: diagnostic value and horizon dependence."""
    horizons = sorted({int(r["horizon"]) for r in rows})

    # Figure 2: Gmax versus future relative rollout error, one panel per horizon.
    ncols = min(4, len(horizons))
    nrows = int(math.ceil(len(horizons) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.3*nrows), squeeze=False)
    for ax, T in zip(axes.ravel(), horizons):
        rr = [r for r in rows if int(r["horizon"]) == T]
        x = np.array([max(r["Gmax"], 1e-300) for r in rr])
        y = np.array([max(r["relative_rollout_error"], 1e-300) for r in rr])
        rho = spearman_corr(np.log(x), np.log(y))
        ax.scatter(x, y, s=16, alpha=0.65)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(fr"$T={T}$, Spearman $\rho={rho:.2f}$")
        ax.set_xlabel(r"$G_{\max}(t,T)$")
        ax.set_ylabel("relative rollout error")
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(horizons):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(outdir / "fig2_gmax_vs_rollout_error.pdf")
    plt.close(fig)

    # Companion local-metric comparison at the largest horizon.
    T = horizons[-1]
    rr = [r for r in rows if int(r["horizon"]) == T]
    metrics = [
        ("Gmax", r"$G_{\max}$"),
        ("spectral_radius_J", r"$\rho(J_t)$"),
        ("sigma1_J", r"$\sigma_1(J_t)$"),
        ("fro_J", r"$\|J_t\|_F$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    y = np.array([max(r["relative_rollout_error"], 1e-300) for r in rr])
    for ax, (key, label) in zip(axes, metrics):
        x = np.array([max(r[key], 1e-300) for r in rr])
        rho = spearman_corr(np.log(x), np.log(y))
        ax.scatter(x, y, s=16, alpha=0.65)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(fr"$\rho_s={rho:.2f}$")
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("relative rollout error")
    fig.suptitle(fr"Local versus finite-time diagnostics, $T={T}$")
    fig.tight_layout()
    fig.savefig(outdir / "fig2b_local_vs_finite_time_metrics.pdf")
    plt.close(fig)

    # Figure 3: horizon dependence, individual curves plus median and 10--90% band.
    indices = sorted({int(r["test_index"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for idx in indices[:min(30, len(indices))]:
        rr = sorted([r for r in rows if int(r["test_index"]) == idx], key=lambda z: z["horizon"])
        ax.plot([r["horizon"] for r in rr], [r["Gmax"] for r in rr], alpha=0.16)
    vals = np.array([[next(r["Gmax"] for r in rows if int(r["test_index"]) == idx and int(r["horizon"]) == T)
                      for T in horizons] for idx in indices])
    med = np.median(vals, axis=0)
    lo = np.quantile(vals, 0.10, axis=0)
    hi = np.quantile(vals, 0.90, axis=0)
    ax.plot(horizons, med, marker="o", linewidth=2, label="median")
    ax.fill_between(horizons, lo, hi, alpha=0.22, label="10--90%")
    ax.set_yscale("log")
    ax.set_xlabel("horizon $T$")
    ax.set_ylabel(r"$G_{\max}(t,T)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig3_horizon_dependence.pdf")
    plt.close(fig)


def perturbation_campaign(
    model: nn.Module,
    xte: torch.Tensor,
    horizon: int,
    n_states: int,
    epsilon: float,
    pgd_steps: int,
    restarts: int,
    random_directions: int = 20,
) -> List[Dict]:
    """Paper Figure 4: TG, secondary singular mode, random, and nonlinear PGD."""
    max_start = len(xte) - horizon - 1
    indices = np.linspace(0, max_start, min(n_states, max_start + 1), dtype=int)
    rows: List[Dict] = []
    for j, idx in enumerate(indices):
        x0 = xte[int(idx)].detach()
        phi, _, _ = explicit_propagator(model, x0, horizon)
        U, S, Vh = torch.linalg.svd(phi)
        v1 = Vh[0] / torch.linalg.vector_norm(Vh[0])
        v2 = Vh[1] / torch.linalg.vector_norm(Vh[1]) if len(S) > 1 else v1
        with torch.no_grad():
            target = final_state(model, x0, horizon)

            def gain(v):
                sep = torch.linalg.vector_norm(final_state(model, x0 + epsilon*v, horizon) - target).item()
                return (sep / epsilon)**2

            g_tg = gain(v1)
            g_v2 = gain(v2)
            random_gains = []
            for _ in range(random_directions):
                vr = torch.randn_like(x0)
                vr /= torch.linalg.vector_norm(vr)
                random_gains.append(gain(vr))

        _, v_adv, adv_loss = pgd_state_separation(
            model, x0, horizon, epsilon,
            steps=pgd_steps, restarts=restarts, init_direction=v1
        )
        rows.append({
            "test_index": int(idx),
            "horizon": int(horizon),
            "epsilon": float(epsilon),
            "sigma1_sq": float(S[0]**2),
            "G_TG": g_tg,
            "G_v2": g_v2,
            "G_random_mean": float(np.mean(random_gains)),
            "G_random_max": float(np.max(random_gains)),
            "G_PGD": float(adv_loss / epsilon**2),
            "alignment": float(torch.abs(torch.dot(v1, v_adv))),
        })
        print(f"perturbation state {j+1:03d}/{len(indices):03d}")
    return rows


def save_perturbation_figure(rows: List[Dict], outdir: Path):
    labels = ["TG", "2nd singular", "random mean", "PGD"]
    data = [
        [r["G_TG"] for r in rows],
        [r["G_v2"] for r in rows],
        [r["G_random_mean"] for r in rows],
        [r["G_PGD"] for r in rows],
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel(r"nonlinear gain $G_{\rm NL}$")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_perturbation_comparison.pdf")
    plt.close(fig)


def aggregate_epsilon_campaign(
    model: nn.Module,
    xte: torch.Tensor,
    horizon: int,
    n_states: int,
    epsilons: np.ndarray,
    pgd_steps: int,
    restarts: int,
) -> List[Dict]:
    """Paper Figure 5: aggregate epsilon sweep over several held-out states."""
    max_start = len(xte) - horizon - 1
    indices = np.linspace(0, max_start, min(n_states, max_start + 1), dtype=int)
    all_rows: List[Dict] = []
    for j, idx in enumerate(indices):
        rows = epsilon_sweep(model, xte[int(idx)].detach(), horizon, epsilons,
                             pgd_steps=pgd_steps, restarts=restarts)
        for r in rows:
            r = dict(r)
            r["test_index"] = int(idx)
            all_rows.append(r)
        print(f"epsilon state {j+1:03d}/{len(indices):03d}")
    return all_rows


def save_aggregate_epsilon_figures(rows: List[Dict], outdir: Path):
    eps = sorted({float(r["epsilon"]) for r in rows})

    def summary(key, normalize=False):
        med, lo, hi = [], [], []
        for e in eps:
            rr = [r for r in rows if float(r["epsilon"]) == e]
            vals = np.array([
                (r[key] / r["sigma1_sq"] if normalize else r[key]) for r in rr
            ])
            med.append(np.median(vals)); lo.append(np.quantile(vals, .10)); hi.append(np.quantile(vals, .90))
        return np.array(med), np.array(lo), np.array(hi)

    med, lo, hi = summary("alignment")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogx(eps, med, marker="o")
    ax.fill_between(eps, lo, hi, alpha=0.22)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$|v_{\rm TG}^{T}v_{\rm adv}|$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig5a_alignment_vs_epsilon_aggregate.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for key, label in [("G_NL_TG", "TG direction"), ("G_NL_PGD", "PGD")]:
        med, lo, hi = summary(key, normalize=True)
        ax.semilogx(eps, med, marker="o", label=label)
        ax.fill_between(eps, lo, hi, alpha=0.18)
    ax.axhline(1.0, linestyle="--")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$G_{\rm NL}/\sigma_1^2(\Phi)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig5b_nonlinear_gain_vs_epsilon_aggregate.pdf")
    plt.close(fig)


# ---------------------------- figures -------------------------------- #

def save_epsilon_plot(rows: List[Dict], outdir: Path):
    eps = np.array([r["epsilon"] for r in rows])
    align = np.array([r["alignment"] for r in rows])
    ratio_tg = np.array([r["G_NL_TG"]/r["sigma1_sq"] for r in rows])
    ratio_pgd = np.array([r["G_NL_PGD"]/r["sigma1_sq"] for r in rows])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogx(eps, align, marker="o")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$|v_{\rm TG}^T v_{\rm adv}|$")
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outdir / "alignment_vs_epsilon.pdf")
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogx(eps, ratio_tg, marker="o", label="TG direction")
    ax.semilogx(eps, ratio_pgd, marker="s", label="PGD")
    ax.axhline(1.0, linestyle="--")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$G_{\rm NL}/\sigma_1^2(\Phi)$")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outdir / "nonlinear_gain_vs_epsilon.pdf")
    plt.close(fig)


# ------------------------------ main --------------------------------- #

def main(args):
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating Lorenz-96 data...")
    traj = simulate_l96(
        n=args.dim, forcing=args.forcing, dt=args.dt,
        steps=args.data_steps, burnin=args.burnin, seed=args.seed
    )
    xtr, ytr, xte, yte, norm = make_dataset(traj)

    model = ResidualMLP(args.dim, args.hidden)
    print("Training one-step surrogate...")
    train_model(model, xtr, ytr, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr)
    model.eval()

    with torch.no_grad():
        test_mse = ((model(xte) - yte)**2).mean().item()
    print(f"test one-step MSE = {test_mse:.6e}")

    # Publication-scale diagnostic campaign (Figures 2--3).
    if args.campaign:
        horizons = [int(v) for v in args.horizons.split(",") if v.strip()]
        print("\nRunning multi-state / multi-horizon diagnostic campaign...")
        campaign_rows, defect_term_rows, spectral_mode_rows = diagnostic_campaign(
            model, xte, norm, horizons, args.n_test_states,
            args.dt, args.forcing, args.seed
        )
        write_rows_csv(campaign_rows, outdir / "diagnostic_campaign.csv")
        write_rows_csv(defect_term_rows, outdir / "defect_terms_campaign.csv")
        write_rows_csv(spectral_mode_rows, outdir / "spectral_modes_campaign.csv")
        save_diagnostic_figures(campaign_rows, outdir)
        save_error_propagation_figures(campaign_rows, outdir)
        save_error_aligned_figures(campaign_rows, defect_term_rows, outdir)
        save_spectral_realization_figures(defect_term_rows, spectral_mode_rows, outdir)

        print("\nRunning perturbation comparison campaign...")
        pert_rows = perturbation_campaign(
            model, xte, args.perturb_horizon, args.n_pgd_states,
            args.perturb_epsilon, args.pgd_steps, args.restarts,
            random_directions=args.random_directions
        )
        write_rows_csv(pert_rows, outdir / "perturbation_campaign.csv")
        save_perturbation_figure(pert_rows, outdir)

        print("\nRunning aggregate epsilon campaign...")
        epsilons_campaign = np.logspace(args.eps_min, args.eps_max, args.eps_count)
        eps_rows = aggregate_epsilon_campaign(
            model, xte, args.perturb_horizon, args.n_pgd_states,
            epsilons_campaign, args.pgd_steps, args.restarts
        )
        write_rows_csv(eps_rows, outdir / "epsilon_campaign.csv")
        save_aggregate_epsilon_figures(eps_rows, outdir)

    # Choose held-out state for the original single-window diagnostics.
    x0 = xte[args.test_index].detach()

    print("\nExplicit propagator...")
    phi, xs, Js = explicit_propagator(model, x0, args.horizon)
    sigma, v_tg, _ = tg_svd(phi)
    print(f"sigma1(Phi)={sigma.item():.6e}, Gmax={sigma.item()**2:.6e}")
    print("local metrics at window start:", local_metrics(Js[0]))

    print("\nMatrix-free check...")
    sigma_mf, v_mf = leading_singular_matrix_free(model, xs, iters=50)
    print(f"sigma1 matrix-free={sigma_mf.item():.6e}")
    print(f"|v_explicit dot v_matrixfree|={abs(torch.dot(v_tg,v_mf)).item():.6f}")

    print("\nEpsilon sweep: TG vs nonlinear PGD...")
    epsilons = np.logspace(args.eps_min, args.eps_max, args.eps_count)
    rows = epsilon_sweep(
        model, x0, args.horizon, epsilons,
        pgd_steps=args.pgd_steps, restarts=args.restarts
    )
    for row in rows:
        print(row)
    save_epsilon_plot(rows, outdir)

    print("\nFirst-order rollout-error bound diagnostic...")
    bd = error_bound_diagnostics(
        model, x0, norm, args.horizon, args.dt, args.forcing
    )
    print("actual final error:", bd["actual_final_error"])
    print("linear predicted final error:", bd["linear_predicted_error"])
    print("linear/actual ratio:", bd["linear_to_actual_ratio"])
    print("first-order bound (Taylor remainder omitted):",
          bd["first_order_bound_no_remainder"])
    print("bound/actual ratio:", bd["bound_to_actual_ratio"])

    # Save compact CSV without pandas dependency.
    keys = list(rows[0].keys())
    with open(outdir / "epsilon_sweep.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    torch.save({
        "model_state_dict": model.state_dict(),
        "mean": norm.mean,
        "std": norm.std,
        "args": vars(args),
    }, outdir / "checkpoint.pt")

    print(f"\nOutputs written to {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=20)
    p.add_argument("--forcing", type=float, default=8.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--data-steps", type=int, default=20000)
    p.add_argument("--burnin", type=int, default=2000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test-index", type=int, default=100)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--pgd-steps", type=int, default=100)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--eps-min", type=float, default=-6)
    p.add_argument("--eps-max", type=float, default=-1)
    p.add_argument("--eps-count", type=int, default=10)
    p.add_argument("--campaign", action="store_true",
                   help="run the multi-state publication diagnostic campaign")
    p.add_argument("--horizons", default="1,2,5,10,20,40,80,160",
                   help="comma-separated horizons for Figures 2--3")
    p.add_argument("--n-test-states", type=int, default=100)
    p.add_argument("--n-pgd-states", type=int, default=20)
    p.add_argument("--perturb-horizon", type=int, default=20)
    p.add_argument("--perturb-epsilon", type=float, default=1e-2)
    p.add_argument("--random-directions", type=int, default=20)
    main(p.parse_args())
