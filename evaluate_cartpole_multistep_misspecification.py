"""
Multi-step CartPole trajectory error rate evaluation under dynamics misspecification.

Compare a plain neural transition model (lambda_ODE=0) with the corrected
ODE-informed model (lambda_ODE=1e-5) when predictions are recursively rolled
forward. Models are trained only on nominal CartPole transitions, then evaluated
against nominal and misspecified ground-truth dynamics using identical initial
states and action sequences.

1) Does prediction error accumulate more slowly for the ODE-informed model as
   rollout depth increases?
2) Does that relative advantage persist when the true environment differs from
   the nominal physics assumed during training?

The script reports:
- L1 state error and cumulative L1 trajectory error (to mirror Ramesh &
  Ravindran's imaginary-trajectory analysis), and
- squared-L2/MSE-style metrics (to remain aligned with the dissertation's
  transition-error metrics).

No Gymnasium dependency is required for evaluation: the CartPole equations are
propagated directly with the same explicit-Euler convention used by CartPole-v1.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


#
# 1. Fixed experimental design
#

SEEDS = [42, 123, 456, 789, 1024]
LAMBDA_VALUES = [0.0, 1e-5]
METHOD_LABELS = {0.0: "Plain NN", 1e-5: "ODE-informed NN"}

# Project-definition misspecification settings.
FORCE_CALIBRATION_VALUES = [0.75, 1.0, 1.25]
RAIL_FRICTION_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]

# One 16-step rollout provides all requested horizons.
HORIZONS = [1, 3, 5, 10, 16]
MAX_HORIZON = max(HORIZONS)

BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
DT = 0.02
N_ROLLOUTS = 1000

STATE_DIM = 4
ACTION_DIM = 2
PLUS_MINUS = "\N{PLUS-MINUS SIGN}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#
# 2. Reproducibility and model
#


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(6, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 4),
    ).to(DEVICE)


def make_model_input(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    action_one_hot = F.one_hot(actions.long(), num_classes=ACTION_DIM).float()
    return torch.cat([states, action_one_hot], dim=1)


#
# 3. CartPole dynamics and misspecification
#


def cartpole_derivative(
    states: torch.Tensor,
    actions: torch.Tensor,
    force_calibration: float = 1.0,
) -> torch.Tensor:
    """Continuous derivative for [x, x_dot, theta, theta_dot]."""
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0

    x_dot = states[:, 1]
    theta = states[:, 2]
    theta_dot = states[:, 3]

    force = (
        (2.0 * actions.to(states.dtype) - 1.0)
        * force_mag
        * float(force_calibration)
    )

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    temp = (force + polemass_length * theta_dot.pow(2) * sintheta) / total_mass
    theta_acc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta.pow(2) / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * costheta / total_mass

    return torch.stack([x_dot, x_acc, theta_dot, theta_acc], dim=1)


def true_cartpole_step(
    states_np: np.ndarray,
    actions_np: np.ndarray,
    *,
    force_calibration: float = 1.0,
    beta_x: float = 0.0,
) -> np.ndarray:
    """
    Propagate one deterministic CartPole step.

    Force misspecification:
        F'(a) = c_F F(a)

    Rail-friction misspecification follows the existing project implementation:
        x_dot'_(t+1) = x_dot_(t+1) - beta_x * x_dot_t * dt
    plus the corresponding displacement correction dt * delta_x_dot.
    """
    s = torch.as_tensor(states_np, dtype=torch.float32, device=DEVICE)
    a = torch.as_tensor(actions_np, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        derivative = cartpole_derivative(
            s,
            a,
            force_calibration=force_calibration,
        )
        next_s = s + DT * derivative

        if beta_x != 0.0:
            delta_x_dot = -float(beta_x) * s[:, 1] * DT
            next_s[:, 1] += delta_x_dot
            next_s[:, 0] += DT * delta_x_dot

    return next_s.cpu().numpy().astype(np.float32)


#
# 4. Corrected nominal training objective
#


def dynamics_loss(
    model: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    lambda_ode: float,
):
    pred = model(make_model_input(states, actions))
    transition_loss = F.mse_loss(pred, next_states)

    # Corrected CartPole kinematic residual used in the final lambda sweep:
    # q = [x, theta], q_dot = [x_dot, theta_dot].
    q_current = states[:, [0, 2]]
    q_dot_current = states[:, [1, 3]]
    q_predicted_next = pred[:, [0, 2]]

    ode_residual = (
        (q_predicted_next - q_current) / DT
        - q_dot_current
    )
    ode_loss = torch.mean(torch.sum(ode_residual.pow(2), dim=1))

    total_loss = transition_loss + float(lambda_ode) * ode_loss
    return total_loss, transition_loss, ode_loss


def train_model(
    train_dataset: TensorDataset,
    *,
    seed: int,
    lambda_ode: float,
    epochs: int,
) -> tuple[nn.Module, dict, float]:
    # Same initialization and minibatch order for Plain/ODE within each seed.
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"total": [], "transition": [], "ode": []}
    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_sum = 0.0
        transition_sum = 0.0
        ode_sum = 0.0
        n_seen = 0

        for x_batch, y_batch, state_batch, action_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            state_batch = state_batch.to(DEVICE)
            action_batch = action_batch.to(DEVICE)

            optimizer.zero_grad()
            total, transition, ode = dynamics_loss(
                model,
                state_batch,
                action_batch,
                y_batch,
                lambda_ode,
            )
            total.backward()
            optimizer.step()

            n = len(x_batch)
            n_seen += n
            total_sum += float(total.item()) * n
            transition_sum += float(transition.item()) * n
            ode_sum += float(ode.item()) * n

        history["total"].append(total_sum / n_seen)
        history["transition"].append(transition_sum / n_seen)
        history["ode"].append(ode_sum / n_seen)

        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == epochs:
            print(
                f"  epoch {epoch + 1:3d}/{epochs}: "
                f"total={history['total'][-1]:.8f}, "
                f"transition={history['transition'][-1]:.8f}, "
                f"ODE={history['ode'][-1]:.8f}"
            )

    return model, history, time.time() - start


#
# 5. Recursive model and ground-truth rollouts
#


@torch.no_grad()
def model_step(
    model: nn.Module,
    states_np: np.ndarray,
    actions_np: np.ndarray,
) -> np.ndarray:
    model.eval()
    s = torch.as_tensor(states_np, dtype=torch.float32, device=DEVICE)
    a = torch.as_tensor(actions_np, dtype=torch.long, device=DEVICE)
    pred = model(make_model_input(s, a))
    return pred.cpu().numpy().astype(np.float32)


def rollout_true(
    initial_states: np.ndarray,
    action_sequences: np.ndarray,
    *,
    force_calibration: float,
    beta_x: float,
) -> np.ndarray:
    """Return shape [N, MAX_HORIZON+1, 4]."""
    n = len(initial_states)
    traj = np.zeros((n, MAX_HORIZON + 1, STATE_DIM), dtype=np.float32)
    traj[:, 0] = initial_states

    current = initial_states.copy()
    for h in range(MAX_HORIZON):
        current = true_cartpole_step(
            current,
            action_sequences[:, h],
            force_calibration=force_calibration,
            beta_x=beta_x,
        )
        traj[:, h + 1] = current
    return traj


def rollout_model(
    model: nn.Module,
    initial_states: np.ndarray,
    action_sequences: np.ndarray,
) -> np.ndarray:
    """Return recursively predicted trajectory [N, MAX_HORIZON+1, 4]."""
    n = len(initial_states)
    traj = np.zeros((n, MAX_HORIZON + 1, STATE_DIM), dtype=np.float32)
    traj[:, 0] = initial_states

    current = initial_states.copy()
    for h in range(MAX_HORIZON):
        current = model_step(model, current, action_sequences[:, h])
        traj[:, h + 1] = current
    return traj


#
# 6. Metrics
#


def compute_step_rows(
    *,
    seed: int,
    lambda_ode: float,
    experiment: str,
    condition_value: float,
    condition_label: str,
    pred_traj: np.ndarray,
    true_traj: np.ndarray,
) -> list[dict]:
    rows = []

    for step in range(1, MAX_HORIZON + 1):
        error = pred_traj[:, step] - true_traj[:, step]
        abs_error = np.abs(error)
        sq_error = error**2

        # Ramesh & Ravindran-style state error: L1 norm per state, then average.
        state_l1_each = np.sum(abs_error, axis=1)
        state_l2_sq_each = np.sum(sq_error, axis=1)

        # Physics consistency of the model's own recursively generated step.
        pred_prev = pred_traj[:, step - 1]
        pred_next = pred_traj[:, step]
        kin_residual = (
            (pred_next[:, [0, 2]] - pred_prev[:, [0, 2]]) / DT
            - pred_prev[:, [1, 3]]
        )
        kin_residual_sq_each = np.sum(kin_residual**2, axis=1)

        rows.append(
            {
                "seed": seed,
                "lambda_ODE": lambda_ode,
                "method": METHOD_LABELS[lambda_ode],
                "experiment": experiment,
                "condition_value": condition_value,
                "condition_label": condition_label,
                "rollout_step": step,
                "n_rollouts": len(error),
                "state_l1_mean": float(np.mean(state_l1_each)),
                "state_l1_std_across_rollouts": float(np.std(state_l1_each, ddof=1)),
                "state_l2_sq_mean": float(np.mean(state_l2_sq_each)),
                "component_average_mse": float(np.mean(sq_error)),
                "mse_x": float(np.mean(sq_error[:, 0])),
                "mse_x_dot": float(np.mean(sq_error[:, 1])),
                "mse_theta": float(np.mean(sq_error[:, 2])),
                "mse_theta_dot": float(np.mean(sq_error[:, 3])),
                "model_kinematic_residual_mse": float(np.mean(kin_residual_sq_each)),
            }
        )

    return rows


def compute_horizon_rows(step_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "seed",
        "lambda_ODE",
        "method",
        "experiment",
        "condition_value",
        "condition_label",
    ]

    for keys, group in step_df.groupby(group_cols, sort=False):
        group = group.sort_values("rollout_step")
        base = dict(zip(group_cols, keys))

        for horizon in HORIZONS:
            subset = group[group["rollout_step"] <= horizon]
            endpoint = group[group["rollout_step"] == horizon].iloc[0]
            rows.append(
                {
                    **base,
                    "horizon": horizon,
                    # Sum of mean state errors along the trajectory.
                    "trajectory_l1_error": float(subset["state_l1_mean"].sum()),
                    "trajectory_l2_sq_error": float(subset["state_l2_sq_mean"].sum()),
                    # Average step error controls for the trivial linear growth
                    # of a cumulative metric as H increases.
                    "mean_step_l1_error": float(subset["state_l1_mean"].mean()),
                    "mean_step_l2_sq_error": float(subset["state_l2_sq_mean"].mean()),
                    "endpoint_l1_error": float(endpoint["state_l1_mean"]),
                    "endpoint_l2_sq_error": float(endpoint["state_l2_sq_mean"]),
                    "endpoint_component_average_mse": float(
                        endpoint["component_average_mse"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def aggregate_mean_std(df: pd.DataFrame, group_cols: list[str], metrics: list[str]):
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in agg.columns
    ]
    counts = df.groupby(group_cols, as_index=False).agg(n_seeds=("seed", "nunique"))
    return counts.merge(agg, on=group_cols, how="left")


def build_paired_horizon_table(horizon_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["seed", "experiment", "condition_value", "condition_label", "horizon"]

    for values, group in horizon_df.groupby(keys, sort=False):
        plain = group[np.isclose(group["lambda_ODE"], 0.0)].iloc[0]
        ode = group[np.isclose(group["lambda_ODE"], 1e-5)].iloc[0]
        base = dict(zip(keys, values))

        for metric in [
            "trajectory_l1_error",
            "trajectory_l2_sq_error",
            "mean_step_l1_error",
            "mean_step_l2_sq_error",
            "endpoint_l1_error",
            "endpoint_l2_sq_error",
        ]:
            p = float(plain[metric])
            o = float(ode[metric])
            base[f"plain_{metric}"] = p
            base[f"ode_{metric}"] = o
            base[f"ode_minus_plain_{metric}"] = o - p
            base[f"ode_improvement_pct_{metric}"] = (
                100.0 * (p - o) / p if p > 0.0 else np.nan
            )
        rows.append(base)

    return pd.DataFrame(rows)


#
# 7. Plots
#


def plot_nominal_stepwise(step_agg: pd.DataFrame, output_dir: Path) -> None:
    rows = step_agg[
        (step_agg["experiment"] == "force_calibration")
        & np.isclose(step_agg["condition_value"], 1.0)
    ].sort_values(["method", "rollout_step"])

    plt.figure(figsize=(8.2, 5.2))
    for method in ["Plain NN", "ODE-informed NN"]:
        m = rows[rows["method"] == method]
        x = m["rollout_step"].to_numpy()
        mean = m["state_l1_mean_mean"].to_numpy()
        std = m["state_l1_mean_std"].fillna(0.0).to_numpy()
        line, = plt.plot(x, mean, marker="o", label=method)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15, color=line.get_color())

    plt.xlabel("Rollout step")
    plt.ylabel("Mean L1 state error")
    plt.title("Nominal CartPole: recursive trajectory error")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cartpole_multistep_nominal_state_error.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_relative_improvement(
    paired_agg: pd.DataFrame,
    *,
    experiment: str,
    xlabel_name: str,
    output_path: Path,
) -> None:
    rows = paired_agg[paired_agg["experiment"] == experiment].copy()
    metric = "ode_improvement_pct_trajectory_l1_error_mean"
    metric_std = "ode_improvement_pct_trajectory_l1_error_std"

    plt.figure(figsize=(8.2, 5.2))
    for condition in sorted(rows["condition_value"].unique()):
        c = rows[np.isclose(rows["condition_value"], condition)].sort_values("horizon")
        label = c["condition_label"].iloc[0]
        x = c["horizon"].to_numpy()
        mean = c[metric].to_numpy()
        std = c[metric_std].fillna(0.0).to_numpy()
        line, = plt.plot(x, mean, marker="o", label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.10, color=line.get_color())

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Rollout horizon H")
    plt.ylabel("ODE improvement over Plain in trajectory L1 error (%)")
    plt.title(f"CartPole: horizon sensitivity under {xlabel_name}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


#
# 8. Main experiment
#


def parse_args():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir / "cartpole_transitions.npz",
        help="Path to nominal CartPole transition dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "cartpole_multistep_misspecification_outputs",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one seed, two epochs, 32 rollouts for an end-to-end smoke test.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [SEEDS[0]] if args.smoke else SEEDS
    epochs = 2 if args.smoke else EPOCHS
    n_rollouts = 32 if args.smoke else N_ROLLOUTS
    if args.epochs is not None:
        epochs = args.epochs
    if args.rollouts is not None:
        n_rollouts = args.rollouts

    print("CARTPOLE MULTI-STEP TRAJECTORY ERROR + MISSPECIFICATION")
    print("Device:", DEVICE)
    print("Seeds:", seeds)
    print("Lambdas:", LAMBDA_VALUES)
    print("Horizons:", HORIZONS)
    print("Rollouts per seed:", n_rollouts)
    print("Epochs:", epochs)
    print("Data:", args.data)
    print("Output:", output_dir)

    if not args.data.exists():
        raise FileNotFoundError(
            f"Could not find {args.data}. Put cartpole_transitions.npz beside the script "
            "or pass --data PATH."
        )

    data = np.load(args.data)
    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    next_states = data["next_states"].astype(np.float32)

    if not (len(states) == len(actions) == len(next_states)):
        raise ValueError("states, actions, and next_states must have equal lengths")
    if not np.isin(actions, [0, 1]).all():
        raise ValueError("CartPole actions must be 0/1")
    if not (np.isfinite(states).all() and np.isfinite(next_states).all()):
        raise ValueError("Dataset contains NaN or infinite values")

    actions_one_hot = np.eye(2, dtype=np.float32)[actions]
    inputs = np.concatenate([states, actions_one_hot], axis=1).astype(np.float32)

    all_step_rows: list[dict] = []
    training_rows: list[dict] = []

    for seed in seeds:
        print("\n" + "#" * 80)
        print(f"SEED {seed}")
        print("#" * 80)

        split_rng = np.random.default_rng(seed)
        indices = split_rng.permutation(len(inputs))
        validation_size = int(len(inputs) * VALIDATION_RATIO)
        test_size = int(len(inputs) * TEST_RATIO)
        validation_indices = indices[:validation_size]
        test_indices = indices[validation_size : validation_size + test_size]
        train_indices = indices[validation_size + test_size :]

        x_train = inputs[train_indices]
        y_train = next_states[train_indices]
        state_train = states[train_indices]
        action_train = actions[train_indices]
        state_test = states[test_indices]
        action_test = actions[test_indices]
        y_test_nominal = next_states[test_indices]

        # Sanity check: the direct nominal dynamics used for recursive
        # ground-truth rollouts should reproduce the held-out CartPole-v1
        # transitions generated by the original dataset script.
        generated_nominal = true_cartpole_step(
            state_test,
            action_test,
            force_calibration=1.0,
            beta_x=0.0,
        )
        nominal_generation_max_abs_error = float(
            np.max(np.abs(generated_nominal - y_test_nominal))
        )
        print(
            "Nominal generator max |generated - dataset|: "
            f"{nominal_generation_max_abs_error:.3e}"
        )
        if nominal_generation_max_abs_error > 1e-5:
            print(
                "WARNING: direct CartPole generator differs from the held-out "
                "dataset. Check the dataset generation/integration convention "
                "before interpreting multi-step results."
            )

        train_dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(state_train, dtype=torch.float32),
            torch.tensor(action_train, dtype=torch.long),
        )

        # Common initial states and actions across both models AND every
        # misspecification condition within this seed.
        rollout_rng = np.random.default_rng(seed + 50_000)
        replace = n_rollouts > len(state_test)
        initial_indices = rollout_rng.choice(
            len(state_test), size=n_rollouts, replace=replace
        )
        initial_states = state_test[initial_indices].copy()
        action_sequences = rollout_rng.integers(
            0,
            ACTION_DIM,
            size=(n_rollouts, MAX_HORIZON),
            dtype=np.int64,
        )

        # Precompute ground-truth trajectories once per condition.
        conditions = []
        for c_f in FORCE_CALIBRATION_VALUES:
            conditions.append(
                {
                    "experiment": "force_calibration",
                    "condition_value": float(c_f),
                    "condition_label": f"c_F={c_f}",
                    "force_calibration": float(c_f),
                    "beta_x": 0.0,
                }
            )
        for beta_x in RAIL_FRICTION_VALUES:
            conditions.append(
                {
                    "experiment": "rail_friction",
                    "condition_value": float(beta_x),
                    "condition_label": f"beta_x={beta_x}",
                    "force_calibration": 1.0,
                    "beta_x": float(beta_x),
                }
            )

        true_trajectories = {}
        for condition in conditions:
            key = (condition["experiment"], condition["condition_value"])
            true_trajectories[key] = rollout_true(
                initial_states,
                action_sequences,
                force_calibration=condition["force_calibration"],
                beta_x=condition["beta_x"],
            )

        for lambda_ode in LAMBDA_VALUES:
            print("\n" + "=" * 80)
            print(
                f"Training {METHOD_LABELS[lambda_ode]} | "
                f"seed={seed} | lambda={lambda_ode}"
            )
            print("=" * 80)

            model, history, training_time = train_model(
                train_dataset,
                seed=seed,
                lambda_ode=lambda_ode,
                epochs=epochs,
            )

            pred_traj = rollout_model(model, initial_states, action_sequences)
            if not np.isfinite(pred_traj).all():
                raise RuntimeError(
                    f"Non-finite recursive prediction for seed={seed}, lambda={lambda_ode}"
                )

            training_rows.append(
                {
                    "seed": seed,
                    "lambda_ODE": lambda_ode,
                    "method": METHOD_LABELS[lambda_ode],
                    "epochs": epochs,
                    "train_samples": len(train_indices),
                    "nominal_generation_max_abs_error": nominal_generation_max_abs_error,
                    "training_time_sec": training_time,
                    "final_total_loss": history["total"][-1],
                    "final_transition_loss": history["transition"][-1],
                    "final_ode_loss": history["ode"][-1],
                }
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "lambda_ODE": lambda_ode,
                    "training_history": history,
                },
                output_dir / f"cartpole_multistep_seed_{seed}_lambda_{lambda_ode:g}.pt",
            )

            for condition in conditions:
                key = (condition["experiment"], condition["condition_value"])
                all_step_rows.extend(
                    compute_step_rows(
                        seed=seed,
                        lambda_ode=lambda_ode,
                        experiment=condition["experiment"],
                        condition_value=condition["condition_value"],
                        condition_label=condition["condition_label"],
                        pred_traj=pred_traj,
                        true_traj=true_trajectories[key],
                    )
                )

            pd.DataFrame(all_step_rows).to_csv(
                output_dir / "cartpole_multistep_step_raw_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    step_df = pd.DataFrame(all_step_rows)
    horizon_df = compute_horizon_rows(step_df)
    paired_df = build_paired_horizon_table(horizon_df)
    training_df = pd.DataFrame(training_rows)

    step_metrics = [
        "state_l1_mean",
        "state_l2_sq_mean",
        "component_average_mse",
        "mse_x",
        "mse_x_dot",
        "mse_theta",
        "mse_theta_dot",
        "model_kinematic_residual_mse",
    ]
    step_group_cols = [
        "lambda_ODE",
        "method",
        "experiment",
        "condition_value",
        "condition_label",
        "rollout_step",
    ]
    step_agg = aggregate_mean_std(step_df, step_group_cols, step_metrics)

    horizon_metrics = [
        "trajectory_l1_error",
        "trajectory_l2_sq_error",
        "mean_step_l1_error",
        "mean_step_l2_sq_error",
        "endpoint_l1_error",
        "endpoint_l2_sq_error",
        "endpoint_component_average_mse",
    ]
    horizon_group_cols = [
        "lambda_ODE",
        "method",
        "experiment",
        "condition_value",
        "condition_label",
        "horizon",
    ]
    horizon_agg = aggregate_mean_std(horizon_df, horizon_group_cols, horizon_metrics)

    paired_metrics = [
        c for c in paired_df.columns
        if c.startswith("ode_minus_plain_") or c.startswith("ode_improvement_pct_")
    ]
    paired_group_cols = ["experiment", "condition_value", "condition_label", "horizon"]
    paired_agg = aggregate_mean_std(paired_df, paired_group_cols, paired_metrics)

    # Save all results.
    step_df.to_csv(output_dir / "cartpole_multistep_step_raw.csv", index=False, encoding="utf-8-sig")
    step_agg.to_csv(output_dir / "cartpole_multistep_step_aggregate.csv", index=False, encoding="utf-8-sig")
    horizon_df.to_csv(output_dir / "cartpole_multistep_horizon_raw.csv", index=False, encoding="utf-8-sig")
    horizon_agg.to_csv(output_dir / "cartpole_multistep_horizon_aggregate.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(output_dir / "cartpole_multistep_paired_raw.csv", index=False, encoding="utf-8-sig")
    paired_agg.to_csv(output_dir / "cartpole_multistep_paired_aggregate.csv", index=False, encoding="utf-8-sig")
    training_df.to_csv(output_dir / "cartpole_multistep_training_summary.csv", index=False, encoding="utf-8-sig")

    config = {
        "seeds": seeds,
        "lambda_values": LAMBDA_VALUES,
        "force_calibration_values": FORCE_CALIBRATION_VALUES,
        "rail_friction_values": RAIL_FRICTION_VALUES,
        "horizons": HORIZONS,
        "max_horizon": MAX_HORIZON,
        "n_rollouts_per_seed": n_rollouts,
        "epochs": epochs,
        "dt": DT,
        "device": str(DEVICE),
        "smoke": bool(args.smoke),
    }
    with (output_dir / "cartpole_multistep_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Figures.
    plot_nominal_stepwise(step_agg, output_dir)
    plot_relative_improvement(
        paired_agg,
        experiment="force_calibration",
        xlabel_name="force calibration",
        output_path=output_dir / "cartpole_multistep_force_relative_improvement.png",
    )
    plot_relative_improvement(
        paired_agg,
        experiment="rail_friction",
        xlabel_name="rail friction",
        output_path=output_dir / "cartpole_multistep_friction_relative_improvement.png",
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    # Compact nominal summary at requested horizons.
    nominal = horizon_agg[
        (horizon_agg["experiment"] == "force_calibration")
        & np.isclose(horizon_agg["condition_value"], 1.0)
    ][
        [
            "method",
            "horizon",
            "trajectory_l1_error_mean",
            "trajectory_l1_error_std",
            "mean_step_l1_error_mean",
            "endpoint_l1_error_mean",
        ]
    ]
    print("\nNominal multi-step summary:")
    print(nominal.to_string(index=False))

    paired_nominal = paired_agg[
        (paired_agg["experiment"] == "force_calibration")
        & np.isclose(paired_agg["condition_value"], 1.0)
    ][
        [
            "horizon",
            "ode_improvement_pct_trajectory_l1_error_mean",
            "ode_improvement_pct_trajectory_l1_error_std",
        ]
    ]
    print("\nODE improvement over Plain in cumulative trajectory L1 error:")
    print(paired_nominal.to_string(index=False))
    print("\nSaved outputs to:", output_dir)


if __name__ == "__main__":
    main()
