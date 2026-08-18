import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


#-
# 1. Experiment settings
#-

SEEDS = [42, 123, 456, 789, 1024]
LAMBDA_VALUES = [0.0, 0.00001]
FORCE_CALIBRATION_VALUES = [0.75, 1.0, 1.25]
RAIL_FRICTION_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]

BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
DT = 0.02
PLUS_MINUS = "\N{PLUS-MINUS SIGN}"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "cartpole_transitions.npz"
OUTPUT_DIR = SCRIPT_DIR / "cartpole_physical_misspecification_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


#-
# 2. Reproducibility, model, and nominal dynamics
#-

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def lambda_name(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(6, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 4),
    ).to(device)


def cartpole_derivative(
    states_tensor: torch.Tensor,
    actions_tensor: torch.Tensor,
    force_calibration: float = 1.0,
) -> torch.Tensor:
    """Return ds/dt for [x, x_dot, theta, theta_dot]."""
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0

    x_dot = states_tensor[:, 1]
    theta = states_tensor[:, 2]
    theta_dot = states_tensor[:, 3]
    force = (
        (2.0 * actions_tensor.to(states_tensor.dtype) - 1.0)
        * force_mag
        * force_calibration
    )

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
    theta_acc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * costheta / total_mass
    return torch.stack([x_dot, x_acc, theta_dot, theta_acc], dim=1)


#-
# 3. Misspecified target generation
#-

def generate_force_targets(
    states_np: np.ndarray,
    actions_np: np.ndarray,
    force_calibration: float,
) -> np.ndarray:
    """Explicit-Euler targets under F_mis(a) = c_F * F(a)."""
    states_tensor = torch.tensor(states_np, dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions_np, dtype=torch.long, device=device)
    with torch.no_grad():
        derivative = cartpole_derivative(
            states_tensor,
            actions_tensor,
            force_calibration=force_calibration,
        )
        targets_tensor = states_tensor + DT * derivative
    return targets_tensor.cpu().numpy().astype(np.float32)


def generate_friction_targets(
    states_np: np.ndarray,
    nominal_targets_np: np.ndarray,
    beta_x: float,
) -> np.ndarray:
    """Apply velocity damping while keeping beta=0 exactly nominal."""
    targets_mis = nominal_targets_np.copy()
    delta_x_dot = -beta_x * states_np[:, 1] * DT
    targets_mis[:, 1] += delta_x_dot

    # Apply only the displacement induced by the damping correction. Using
    # x_t + DT*x_dot_next_raw directly would change the dataset's integration
    # convention even at beta=0 and invalidate the nominal control.
    targets_mis[:, 0] += DT * delta_x_dot
    return targets_mis.astype(np.float32)


#-
# 4. Training and evaluation
#-

def train_model(
    train_dataset: TensorDataset,
    seed: int,
    lambda_ode: float,
) -> tuple[nn.Module, dict, float]:
    """Train one fresh model with a reproducible minibatch order."""
    set_seed(seed)
    data_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=data_generator,
    )

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()
    history = {"total": [], "transition": [], "ode": []}
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_transition_loss = 0.0
        total_ode_loss = 0.0

        for x_batch, y_batch, state_batch, action_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)
            transition_loss = loss_function(predictions, y_batch)
            nominal_derivative = cartpole_derivative(state_batch, action_batch)
            ode_residual = (
                (predictions - state_batch) / DT - nominal_derivative
            )
            ode_loss = torch.mean(torch.sum(ode_residual**2, dim=1))
            loss = transition_loss + lambda_ode * ode_loss
            loss.backward()
            optimizer.step()

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_transition_loss += transition_loss.item() * batch_size
            total_ode_loss += ode_loss.item() * batch_size

        average_total = total_loss / len(train_dataset)
        average_transition = total_transition_loss / len(train_dataset)
        average_ode = total_ode_loss / len(train_dataset)
        history["total"].append(average_total)
        history["transition"].append(average_transition)
        history["ode"].append(average_ode)

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{EPOCHS}, "
                f"total={average_total:.8f}, "
                f"transition={average_transition:.8f}, "
                f"ODE={average_ode:.8f}"
            )

    return model, history, time.time() - start_time


def evaluate_model(
    model: nn.Module,
    x_np: np.ndarray,
    y_np: np.ndarray,
    alert_threshold: float | None = None,
    kinematic_alert_threshold: float | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate prediction and target-aware residual-discrepancy metrics."""
    x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_np, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        predictions = model(x_tensor)
        states_tensor = x_tensor[:, :4]
        actions_tensor = torch.argmax(x_tensor[:, 4:6], dim=1)
        error = predictions - y_tensor

        component_average_mse = torch.mean(error**2).item()
        next_state_mse = torch.mean(torch.sum(error**2, dim=1)).item()
        component_mse = torch.mean(error**2, dim=0).cpu().numpy()
        component_rmse = np.sqrt(component_mse)

        nominal_derivative = cartpole_derivative(states_tensor, actions_tensor)
        model_residual = (
            (predictions - states_tensor) / DT - nominal_derivative
        )
        observed_residual = (
            (y_tensor - states_tensor) / DT - nominal_derivative
        )
        residual_discrepancy = model_residual - observed_residual
        discrepancy_d = torch.sum(residual_discrepancy**2, dim=1)

        # Auxiliary model-only kinematic statistic requested in the project
        # notes. It is expected to remain invariant across target-only shifts.
        r_x = (predictions[:, 0] - states_tensor[:, 0]) / DT - states_tensor[:, 1]
        r_theta = (
            (predictions[:, 2] - states_tensor[:, 2]) / DT
            - states_tensor[:, 3]
        )
        kinematic_d = r_x**2 + r_theta**2

        alert_rate = float("nan")
        if alert_threshold is not None:
            alert_rate = torch.mean(
                (discrepancy_d > alert_threshold).to(torch.float32)
            ).item()

        kinematic_alert_rate = float("nan")
        if kinematic_alert_threshold is not None:
            kinematic_alert_rate = torch.mean(
                (kinematic_d > kinematic_alert_threshold).to(torch.float32)
            ).item()

        metrics = {
            "component_average_mse": component_average_mse,
            "component_average_rmse": float(np.sqrt(component_average_mse)),
            "next_state_mse": next_state_mse,
            "next_state_rmse": float(np.sqrt(next_state_mse)),
            "mse_x": float(component_mse[0]),
            "mse_x_dot": float(component_mse[1]),
            "mse_theta": float(component_mse[2]),
            "mse_theta_dot": float(component_mse[3]),
            "rmse_x": float(component_rmse[0]),
            "rmse_x_dot": float(component_rmse[1]),
            "rmse_theta": float(component_rmse[2]),
            "rmse_theta_dot": float(component_rmse[3]),
            "model_ode_residual_mse": torch.mean(
                torch.sum(model_residual**2, dim=1)
            ).item(),
            "observed_ode_residual_mse": torch.mean(
                torch.sum(observed_residual**2, dim=1)
            ).item(),
            "residual_discrepancy_mse": torch.mean(discrepancy_d).item(),
            "mean_D_t": torch.mean(discrepancy_d).item(),
            "p95_D_t": torch.quantile(discrepancy_d, 0.95).item(),
            "p99_D_t": torch.quantile(discrepancy_d, 0.99).item(),
            "alert_rate": alert_rate,
            "mean_kinematic_D_pred": torch.mean(kinematic_d).item(),
            "p95_kinematic_D_pred": torch.quantile(kinematic_d, 0.95).item(),
            "kinematic_alert_rate": kinematic_alert_rate,
        }

    return (
        metrics,
        predictions.cpu().numpy(),
        discrepancy_d.cpu().numpy(),
        kinematic_d.cpu().numpy(),
    )


def aggregate_mean_std(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = [
        "experiment",
        "condition_value",
        "condition_label",
        "lambda_ODE",
    ]
    metrics = [
        "next_state_mse",
        "next_state_rmse",
        "component_average_mse",
        "component_average_rmse",
        "mse_x",
        "mse_x_dot",
        "mse_theta",
        "mse_theta_dot",
        "model_ode_residual_mse",
        "observed_ode_residual_mse",
        "residual_discrepancy_mse",
        "mean_D_t",
        "p95_D_t",
        "alert_rate",
        "training_time_sec",
    ]
    aggregate = raw_df.groupby(group_columns)[metrics].agg(
        ["mean", "std"]
    ).reset_index()
    aggregate.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in aggregate.columns
    ]
    counts = raw_df.groupby(group_columns, as_index=False).agg(
        n_seeds=("seed", "nunique")
    )
    aggregate = counts.merge(aggregate, on=group_columns, how="left")
    aggregate = aggregate.sort_values(
        ["experiment", "condition_value", "lambda_ODE"]
    ).reset_index(drop=True)

    formatted = aggregate[
        group_columns + ["n_seeds"]
    ].copy()
    for metric in metrics:
        formatted[f"{metric}_mean_std"] = aggregate.apply(
            lambda row: (
                f"{row[f'{metric}_mean']:.10g} {PLUS_MINUS} "
                f"{row[f'{metric}_std']:.3g}"
            ),
            axis=1,
        )
    return aggregate, formatted


#-
# 5. Load nominal transition data
#-

if not np.isclose(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO, 1.0):
    raise ValueError("TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO must equal 1.0")
if any(value <= 0.0 for value in FORCE_CALIBRATION_VALUES):
    raise ValueError("FORCE_CALIBRATION_VALUES must be positive")
if any(value < 0.0 for value in RAIL_FRICTION_VALUES):
    raise ValueError("RAIL_FRICTION_VALUES must be non-negative")
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_FILE}. Put cartpole_transitions.npz beside this script."
    )

data = np.load(DATA_FILE)
states = data["states"].astype(np.float32)
actions = data["actions"].astype(np.int64)
next_states = data["next_states"].astype(np.float32)

if not (len(states) == len(actions) == len(next_states)):
    raise ValueError("states, actions, and next_states must have equal lengths")
if not np.isfinite(states).all() or not np.isfinite(next_states).all():
    raise ValueError("The transition dataset contains NaN or infinite values")
if not np.isin(actions, [0, 1]).all():
    raise ValueError("CartPole actions must contain only 0 and 1")

actions_one_hot = np.eye(2, dtype=np.float32)[actions]
inputs = np.concatenate([states, actions_one_hot], axis=1).astype(np.float32)

print("States:", states.shape)
print("Actions:", actions.shape)
print("Next states:", next_states.shape)
print(f"Training runs: {len(SEEDS) * len(LAMBDA_VALUES)}")
print(
    "Evaluation rows: "
    f"{len(SEEDS) * len(LAMBDA_VALUES) * (len(FORCE_CALIBRATION_VALUES) + len(RAIL_FRICTION_VALUES))}"
)


#-
# 6. Train once, evaluate every misspecification
#-

all_results: list[dict] = []

for seed in SEEDS:
    split_generator = np.random.default_rng(seed)
    indices = split_generator.permutation(len(inputs))
    validation_size = int(len(inputs) * VALIDATION_RATIO)
    test_size = int(len(inputs) * TEST_RATIO)
    validation_indices = indices[:validation_size]
    test_indices = indices[validation_size : validation_size + test_size]
    train_indices = indices[validation_size + test_size :]

    x_train = inputs[train_indices]
    y_train = next_states[train_indices]
    state_train = states[train_indices]
    action_train = actions[train_indices]
    x_validation = inputs[validation_indices]
    y_validation = next_states[validation_indices]
    x_test = inputs[test_indices]
    y_test_nominal = next_states[test_indices]
    state_test = states[test_indices]
    action_test = actions[test_indices]

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(state_train, dtype=torch.float32),
        torch.tensor(action_train, dtype=torch.long),
    )

    # Verify that the c_F=1 explicit-Euler generator matches the dataset.
    generated_nominal = generate_force_targets(state_test, action_test, 1.0)
    nominal_generation_error = float(
        np.max(np.abs(generated_nominal - y_test_nominal))
    )
    if nominal_generation_error > 1e-5:
        print(
            "WARNING: force-target generator differs from nominal data; "
            f"max absolute difference={nominal_generation_error:.3e}"
        )

    conditions = []
    for force_calibration in FORCE_CALIBRATION_VALUES:
        force_targets = (
            y_test_nominal.copy()
            if np.isclose(force_calibration, 1.0)
            else generate_force_targets(
                state_test,
                action_test,
                force_calibration,
            )
        )
        conditions.append(
            {
                "experiment": "force_calibration",
                "condition_value": force_calibration,
                "condition_label": f"c_F={force_calibration}",
                "targets": force_targets,
            }
        )
    for beta_x in RAIL_FRICTION_VALUES:
        conditions.append(
            {
                "experiment": "rail_friction",
                "condition_value": beta_x,
                "condition_label": f"beta_x={beta_x}",
                "targets": generate_friction_targets(
                    state_test,
                    y_test_nominal,
                    beta_x,
                ),
            }
        )

    for lambda_ode in LAMBDA_VALUES:
        print("\n" + "=" * 76)
        print(f"Training seed={seed}, lambda_ODE={lambda_ode} on nominal data")
        print("=" * 76)
        model, history, training_time = train_model(
            train_dataset,
            seed,
            lambda_ode,
        )

        validation_metrics, _, validation_d, validation_kinematic_d = evaluate_model(
            model,
            x_validation,
            y_validation,
        )
        alert_threshold = float(np.quantile(validation_d, 0.95))
        kinematic_alert_threshold = float(
            np.quantile(validation_kinematic_d, 0.95)
        )

        run_name = f"seed_{seed}_lambda_{lambda_name(lambda_ode)}"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "seed": seed,
                "lambda_ODE": lambda_ode,
                "alert_threshold": alert_threshold,
                "kinematic_alert_threshold": kinematic_alert_threshold,
                "validation_metrics": validation_metrics,
                "training_time_sec": training_time,
                "training_history": history,
            },
            OUTPUT_DIR / f"cartpole_physical_{run_name}_model.pt",
        )

        plt.figure(figsize=(8, 5))
        plt.plot(history["total"], label="total loss")
        plt.plot(history["transition"], label="transition loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Nominal training: seed={seed}, lambda={lambda_ode}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"cartpole_physical_{run_name}_training_loss.png",
            dpi=200,
        )
        plt.close()

        for condition in conditions:
            metrics, _, _, _ = evaluate_model(
                model,
                x_test,
                condition["targets"],
                alert_threshold=alert_threshold,
                kinematic_alert_threshold=kinematic_alert_threshold,
            )
            if not all(np.isfinite(value) for value in metrics.values()):
                raise RuntimeError(
                    "Non-finite metric for "
                    f"seed={seed}, lambda={lambda_ode}, "
                    f"condition={condition['condition_label']}"
                )

            result = {
                "seed": seed,
                "lambda_ODE": lambda_ode,
                "experiment": condition["experiment"],
                "condition_value": condition["condition_value"],
                "condition_label": condition["condition_label"],
                "train_samples": len(train_indices),
                "test_samples": len(test_indices),
                "alert_threshold": alert_threshold,
                "kinematic_alert_threshold": kinematic_alert_threshold,
                "nominal_generation_max_abs_error": nominal_generation_error,
                **metrics,
                "final_total_loss": history["total"][-1],
                "final_transition_loss": history["transition"][-1],
                "final_ode_loss": history["ode"][-1],
                "training_time_sec": training_time,
            }
            all_results.append(result)

            print(
                f"{condition['condition_label']}: "
                f"MSE={metrics['next_state_mse']:.10f}, "
                f"D={metrics['mean_D_t']:.10f}, "
                f"alert={metrics['alert_rate']:.4f}"
            )

            # Save after every evaluation condition so completed work survives.
            pd.DataFrame(all_results).to_csv(
                OUTPUT_DIR / "cartpole_physical_misspecification_raw_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )


#-
# 7. Raw, aggregate, and paired outputs
#-

raw_df = pd.DataFrame(all_results)
raw_path = OUTPUT_DIR / "cartpole_physical_misspecification_raw.csv"
raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

aggregate_df, mean_std_df = aggregate_mean_std(raw_df)
aggregate_path = OUTPUT_DIR / "cartpole_physical_misspecification_aggregate.csv"
mean_std_path = OUTPUT_DIR / "cartpole_physical_misspecification_mean_std.csv"
aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
mean_std_df.to_csv(mean_std_path, index=False, encoding="utf-8-sig")

paired_rows = []
for experiment in ["force_calibration", "rail_friction"]:
    experiment_rows = raw_df[raw_df["experiment"] == experiment]
    for condition_value in sorted(experiment_rows["condition_value"].unique()):
        condition_rows = experiment_rows[
            np.isclose(experiment_rows["condition_value"], condition_value)
        ]
        for seed in SEEDS:
            seed_rows = condition_rows[condition_rows["seed"] == seed]
            baseline = seed_rows[np.isclose(seed_rows["lambda_ODE"], 0.0)].iloc[0]
            informed = seed_rows[
                np.isclose(seed_rows["lambda_ODE"], 0.00001)
            ].iloc[0]
            paired_rows.append(
                {
                    "seed": seed,
                    "experiment": experiment,
                    "condition_value": condition_value,
                    "condition_label": informed["condition_label"],
                    "baseline_next_state_mse": baseline["next_state_mse"],
                    "informed_next_state_mse": informed["next_state_mse"],
                    "next_state_mse_difference": (
                        informed["next_state_mse"] - baseline["next_state_mse"]
                    ),
                    "next_state_mse_percent_change": 100.0
                    * (
                        informed["next_state_mse"] / baseline["next_state_mse"]
                        - 1.0
                    ),
                    "baseline_residual_discrepancy_mse": baseline[
                        "residual_discrepancy_mse"
                    ],
                    "informed_residual_discrepancy_mse": informed[
                        "residual_discrepancy_mse"
                    ],
                    "residual_discrepancy_difference": (
                        informed["residual_discrepancy_mse"]
                        - baseline["residual_discrepancy_mse"]
                    ),
                    "baseline_alert_rate": baseline["alert_rate"],
                    "informed_alert_rate": informed["alert_rate"],
                    "alert_rate_difference": (
                        informed["alert_rate"] - baseline["alert_rate"]
                    ),
                }
            )

paired_df = pd.DataFrame(paired_rows)
paired_path = OUTPUT_DIR / "cartpole_physical_misspecification_paired_comparisons.csv"
paired_df.to_csv(paired_path, index=False, encoding="utf-8-sig")

paired_metrics = [
    "next_state_mse_difference",
    "next_state_mse_percent_change",
    "residual_discrepancy_difference",
    "alert_rate_difference",
]
paired_aggregate = paired_df.groupby(
    ["experiment", "condition_value", "condition_label"]
)[paired_metrics].agg(["mean", "std"]).reset_index()
paired_aggregate.columns = [
    "_".join(str(part) for part in column if str(part))
    if isinstance(column, tuple)
    else str(column)
    for column in paired_aggregate.columns
]
paired_counts = paired_df.groupby(
    ["experiment", "condition_value", "condition_label"],
    as_index=False,
).agg(n_seeds=("seed", "nunique"))
paired_aggregate = paired_counts.merge(
    paired_aggregate,
    on=["experiment", "condition_value", "condition_label"],
    how="left",
)
paired_aggregate_path = (
    OUTPUT_DIR
    / "cartpole_physical_misspecification_paired_comparisons_aggregate.csv"
)
paired_aggregate.to_csv(
    paired_aggregate_path,
    index=False,
    encoding="utf-8-sig",
)


#-
# 8. Dissertation-ready plots
#-

def save_metric_plot(
    experiment: str,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    rows = aggregate_df[aggregate_df["experiment"] == experiment]
    plt.figure(figsize=(8, 5))
    for lambda_ode, label in [
        (0.0, "Baseline (lambda=0)"),
        (0.00001, "ODE-informed"),
    ]:
        model_rows = rows[
            np.isclose(rows["lambda_ODE"], lambda_ode)
        ].sort_values("condition_value")
        plt.errorbar(
            model_rows["condition_value"],
            model_rows[f"{metric}_mean"],
            yerr=model_rows[f"{metric}_std"],
            marker="o",
            capsize=4,
            label=label,
        )
    plt.xlabel("Force calibration c_F" if experiment == "force_calibration" else "Rail damping beta_x")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


for experiment, prefix, display_name in [
    ("force_calibration", "force_calibration", "Force calibration"),
    ("rail_friction", "rail_friction", "Rail friction"),
]:
    save_metric_plot(
        experiment,
        "next_state_mse",
        "Next-state MSE",
        f"CartPole {display_name}: prediction error",
        f"cartpole_{prefix}_mse.png",
    )
    save_metric_plot(
        experiment,
        "residual_discrepancy_mse",
        "Residual discrepancy MSE",
        f"CartPole {display_name}: residual discrepancy",
        f"cartpole_{prefix}_residual.png",
    )
    save_metric_plot(
        experiment,
        "alert_rate",
        "Alert rate",
        f"CartPole {display_name}: clean-validation P95 alerts",
        f"cartpole_{prefix}_alert_rate.png",
    )

print("CartPole physical misspecification experiment complete")
print(f"Saved raw results: {raw_path}")
print(f"Saved aggregate results: {aggregate_path}")
print(f"Saved mean {PLUS_MINUS} std table: {mean_std_path}")
print(f"Saved paired comparisons: {paired_path}")
print(f"Saved paired aggregate: {paired_aggregate_path}")
