import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# 1. Experiment settings

SEEDS = [42, 123, 456, 789, 1024]
LAMBDA_VALUES = [0.0, 0.00001, 0.0001, 0.001, 0.01]

BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
DT = 0.02  # CartPole-v1 time step
PLUS_MINUS = "\N{PLUS-MINUS SIGN}"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "cartpole_transitions.npz"
OUTPUT_DIR = SCRIPT_DIR / "cartpole_lambda_sweep_kinematic_5seeds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a reproducible run."""
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# 2. CartPole nominal ODE dynamics


def cartpole_nominal_derivative(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Return ds/dt for [x, x_dot, theta, theta_dot]."""
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
    force = (2.0 * actions.to(states.dtype) - 1.0) * force_mag

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
    theta_acc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * costheta / total_mass

    return torch.stack([x_dot, x_acc, theta_dot, theta_acc], dim=1)



# 3. Helpers

def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(6, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 4),
    ).to(device)


def make_evaluation_tensors(x: np.ndarray, y: np.ndarray):
    return (
        torch.tensor(x, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
    )


def evaluate_model(
    model: nn.Module,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
) -> tuple[dict, np.ndarray]:
    model.eval()
    with torch.no_grad():
        predictions = model(x_tensor)
        error = predictions - y_tensor

        component_average_mse = torch.mean(error**2).item()
        component_average_rmse = float(np.sqrt(component_average_mse))
        next_state_mse = torch.mean(torch.sum(error**2, dim=1)).item()
        next_state_rmse = float(np.sqrt(next_state_mse))

        component_mse = torch.mean(error**2, dim=0).cpu().numpy()
        component_rmse = np.sqrt(component_mse)

        states_raw = x_tensor[:, :4]
        actions_raw = torch.argmax(x_tensor[:, 4:6], dim=1)

        q_current = states_raw[:, [0, 2]]
        q_dot_current = states_raw[:, [1, 3]]
        q_predicted_next = predictions[:, [0, 2]]

        model_ode_residual = (
            (q_predicted_next - q_current) / DT
            - q_dot_current
        )
        model_ode_residual_mse = torch.mean(
            torch.sum(model_ode_residual**2, dim=1)).item()
        nominal_derivative = cartpole_nominal_derivative(
            states_raw,
            actions_raw
        )
        observed_ode_residual = (
            (y_tensor - states_raw) / DT - nominal_derivative
        )
        observed_d = torch.sum(observed_ode_residual**2, dim=1)

        metrics = {
            "component_average_mse": component_average_mse,
            "component_average_rmse": component_average_rmse,
            "next_state_mse": next_state_mse,
            "next_state_rmse": next_state_rmse,
            "model_ode_residual_mse": model_ode_residual_mse,
            "mean_D_obs": torch.mean(observed_d).item(),
            "p95_D_obs": torch.quantile(observed_d, 0.95).item(),
            "p99_D_obs": torch.quantile(observed_d, 0.99).item(),
            "mse_x": float(component_mse[0]),
            "mse_x_dot": float(component_mse[1]),
            "mse_theta": float(component_mse[2]),
            "mse_theta_dot": float(component_mse[3]),
            "rmse_x": float(component_rmse[0]),
            "rmse_x_dot": float(component_rmse[1]),
            "rmse_theta": float(component_rmse[2]),
            "rmse_theta_dot": float(component_rmse[3]),
        }

    return metrics, predictions.cpu().numpy()


def print_split_metrics(split: str, metrics: dict) -> None:
    print(f"\n{split.capitalize()} results")
    print("-" * 32)
    print(f"Next-state MSE: {metrics['next_state_mse']:.10f}")
    print(f"Next-state RMSE: {metrics['next_state_rmse']:.10f}")
    print(f"ODE residual MSE: {metrics['model_ode_residual_mse']:.10f}")


def add_mean_std_columns(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Create a human-readable table while retaining split and lambda columns."""
    formatted = aggregate[["split", "lambda_ODE", "n_seeds"]].copy()
    metrics = [
        "next_state_mse",
        "next_state_rmse",
        "component_average_mse",
        "component_average_rmse",
        "model_ode_residual_mse",
        "training_time_sec",
    ]
    for metric in metrics:
        formatted[f"{metric}_mean_std"] = aggregate.apply(
            lambda row: (
                f"{row[f'{metric}_mean']:.10g} {PLUS_MINUS} "
                f"{row[f'{metric}_std']:.3g}"
            ),
            axis=1,
        )
    return formatted


# 4. Load data once and build model inputs

if not np.isclose(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO, 1.0):
    raise ValueError("TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO must equal 1.0")

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

actions_one_hot = np.eye(2, dtype=np.float32)[actions]
inputs = np.concatenate([states, actions_one_hot], axis=1).astype(np.float32)
targets = next_states

print("States:", states.shape)
print("Actions:", actions.shape)
print("Next states:", next_states.shape)
print(f"Runs: {len(SEEDS)} seeds x {len(LAMBDA_VALUES)} lambdas = {len(SEEDS) * len(LAMBDA_VALUES)}")



# 5. Five-seed lambda sweep

all_results: list[dict] = []

for seed in SEEDS:
    set_seed(seed)

    # A distinct but reproducible split for each seed. Every lambda within a
    # seed uses exactly the same train/validation/test observations.
    split_generator = np.random.default_rng(seed)
    indices = split_generator.permutation(len(inputs))
    validation_size = int(len(inputs) * VALIDATION_RATIO)
    test_size = int(len(inputs) * TEST_RATIO)

    validation_indices = indices[:validation_size]
    test_indices = indices[validation_size : validation_size + test_size]
    train_indices = indices[validation_size + test_size :]

    x_train = inputs[train_indices]
    y_train = targets[train_indices]
    state_train = states[train_indices]
    action_train = actions[train_indices]

    x_validation = inputs[validation_indices]
    y_validation = targets[validation_indices]
    x_test = inputs[test_indices]
    y_test = targets[test_indices]

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(state_train, dtype=torch.float32),
        torch.tensor(action_train, dtype=torch.long),
    )
    x_validation_tensor, y_validation_tensor = make_evaluation_tensors(
        x_validation, y_validation
    )
    x_test_tensor, y_test_tensor = make_evaluation_tensors(x_test, y_test)

    print("\n" + "#" * 72)
    print(f"Seed {seed}: train={len(train_indices)}, validation={len(validation_indices)}, test={len(test_indices)}")
    print("#" * 72)

    for lambda_ode in LAMBDA_VALUES:
        print("\n" + "=" * 72)
        print(f"Seed {seed}, lambda_ODE = {lambda_ode}")
        print("=" * 72)

        # Equal initial weights and minibatch order make lambda comparisons
        # paired and fair within each seed.
        set_seed(seed)
        data_generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=data_generator,
        )

        model = build_model()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        training_loss_history = []
        transition_loss_history = []
        ode_loss_history = []
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
                # CartPole kinematic physics residual
                # q = [x, theta]
                # q_dot = [x_dot, theta_dot]
                q_current = state_batch[:, [0, 2]]
                q_dot_current = state_batch[:, [1, 3]]
                q_predicted_next = predictions[:, [0, 2]]

                ode_residual = ((q_predicted_next - q_current) / DT - q_dot_current)
                ode_loss = torch.mean(torch.sum(ode_residual**2, dim=1))
                loss = transition_loss + lambda_ode * ode_loss
                loss.backward()
                optimizer.step()

                batch_size = x_batch.size(0)
                total_loss += loss.item() * batch_size
                total_transition_loss += transition_loss.item() * batch_size
                total_ode_loss += ode_loss.item() * batch_size

            average_training_loss = total_loss / len(train_dataset)
            average_transition_loss = total_transition_loss / len(train_dataset)
            average_ode_loss = total_ode_loss / len(train_dataset)
            training_loss_history.append(average_training_loss)
            transition_loss_history.append(average_transition_loss)
            ode_loss_history.append(average_ode_loss)

            if epoch == 0 or (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{EPOCHS}, "
                    f"total={average_training_loss:.8f}, "
                    f"transition={average_transition_loss:.8f}, "
                    f"ODE={average_ode_loss:.8f}"
                )

        training_time = time.time() - start_time
        validation_metrics, validation_predictions = evaluate_model(
            model, x_validation_tensor, y_validation_tensor
        )
        test_metrics, test_predictions = evaluate_model(
            model, x_test_tensor, y_test_tensor
        )

        if not all(
            np.isfinite(value)
            for metrics in (validation_metrics, test_metrics)
            for value in metrics.values()
        ):
            raise RuntimeError(
                f"Non-finite metric detected for seed={seed}, lambda={lambda_ode}"
            )

        print_split_metrics("validation", validation_metrics)
        print_split_metrics("test", test_metrics)
        print(f"Training time: {training_time:.2f} seconds")

        training_result = {
            "final_total_loss": training_loss_history[-1],
            "final_transition_loss": transition_loss_history[-1],
            "final_ode_loss": ode_loss_history[-1],
            "training_time_sec": training_time,
        }
        all_results.append(
            {
                "seed": seed,
                "lambda_ODE": lambda_ode,
                "split": "validation",
                **validation_metrics,
                **training_result,
            }
        )
        all_results.append(
            {
                "seed": seed,
                "lambda_ODE": lambda_ode,
                "split": "test",
                **test_metrics,
                **training_result,
            }
        )

        # Save raw progress after every completed model (two rows per model).
        pd.DataFrame(all_results).to_csv(
            OUTPUT_DIR / "cartpole_lambda_sweep_raw_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

        run_name = f"seed_{seed}_lambda_{lambda_name(lambda_ode)}"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": 6,
                "output_size": 4,
                "hidden_size": 64,
                "seed": seed,
                "lambda_ODE": lambda_ode,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "training_time_sec": training_time,
            },
            OUTPUT_DIR / f"cartpole_{run_name}_model.pt",
        )
        np.savez(
            OUTPUT_DIR / f"cartpole_{run_name}_results.npz",
            validation_predictions=validation_predictions,
            validation_targets=y_validation,
            validation_inputs=x_validation,
            test_predictions=test_predictions,
            test_targets=y_test,
            test_inputs=x_test,
            training_loss=np.asarray(training_loss_history),
            transition_loss=np.asarray(transition_loss_history),
            ode_loss=np.asarray(ode_loss_history),
            lambda_ODE=lambda_ode,
            training_time_sec=training_time,
            seed=seed,
        )

        plt.figure(figsize=(8, 5))
        plt.plot(training_loss_history, label="total loss")
        plt.plot(transition_loss_history, label="transition loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training loss: seed={seed}, lambda={lambda_ode}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"cartpole_{run_name}_training_loss.png", dpi=200)
        plt.close()


# 6. Save raw and aggregate mean +/- standard deviation

raw_df = pd.DataFrame(all_results)
raw_path = OUTPUT_DIR / "cartpole_lambda_sweep_raw.csv"
raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

aggregate_df = raw_df.groupby(["split", "lambda_ODE"], as_index=False).agg(
    n_seeds=("seed", "nunique"),
    next_state_mse_mean=("next_state_mse", "mean"),
    next_state_mse_std=("next_state_mse", "std"),
    next_state_rmse_mean=("next_state_rmse", "mean"),
    next_state_rmse_std=("next_state_rmse", "std"),
    component_average_mse_mean=("component_average_mse", "mean"),
    component_average_mse_std=("component_average_mse", "std"),
    component_average_rmse_mean=("component_average_rmse", "mean"),
    component_average_rmse_std=("component_average_rmse", "std"),
    model_ode_residual_mse_mean=("model_ode_residual_mse", "mean"),
    model_ode_residual_mse_std=("model_ode_residual_mse", "std"),
    training_time_sec_mean=("training_time_sec", "mean"),
    training_time_sec_std=("training_time_sec", "std"),
)
aggregate_df = aggregate_df.sort_values(["split", "lambda_ODE"]).reset_index(drop=True)
aggregate_path = OUTPUT_DIR / "cartpole_lambda_sweep_aggregate.csv"
aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

mean_std_df = add_mean_std_columns(aggregate_df)
mean_std_path = OUTPUT_DIR / "cartpole_lambda_sweep_mean_std.csv"
mean_std_df.to_csv(mean_std_path, index=False, encoding="utf-8-sig")


# 7. Select lambda using validation means only

validation_summary = aggregate_df[aggregate_df["split"] == "validation"].copy()
baseline = validation_summary[np.isclose(validation_summary["lambda_ODE"], 0.0)].iloc[0]

acceptable = validation_summary[
    (validation_summary["lambda_ODE"] > 0.0)
    & (
        validation_summary["next_state_mse_mean"]
        <= baseline["next_state_mse_mean"] * 1.05
    )
    & (
        validation_summary["model_ode_residual_mse_mean"]
        < baseline["model_ode_residual_mse_mean"]
    )
].sort_values("lambda_ODE")

if acceptable.empty:
    selected_lambda = 0.0
    selection_reason = "No positive lambda met the validation rule; selected baseline."
else:
    selected_lambda = float(acceptable.iloc[0]["lambda_ODE"])
    selection_reason = (
        "Selected the smallest positive lambda whose mean validation ODE residual "
        "was below baseline while mean validation next-state MSE was within 5% of baseline."
    )

selection = {
    "selected_lambda_ODE": selected_lambda,
    "selection_split": "validation",
    "selection_reason": selection_reason,
    "baseline_validation_next_state_mse_mean": float(
        baseline["next_state_mse_mean"]
    ),
    "maximum_allowed_validation_next_state_mse_mean": float(
        baseline["next_state_mse_mean"] * 1.05
    ),
    "baseline_validation_ode_residual_mse_mean": float(
        baseline["model_ode_residual_mse_mean"]
    ),
}
with (OUTPUT_DIR / "cartpole_lambda_selection.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(selection, file, indent=2)

selected_test = aggregate_df[
    (aggregate_df["split"] == "test")
    & np.isclose(aggregate_df["lambda_ODE"], selected_lambda)
].iloc[0]

print("\n" + "=" * 72)
print("Five-seed lambda sweep complete")
print("=" * 72)
print(f"Validation summary (mean {PLUS_MINUS} sample standard deviation):")
print(mean_std_df[mean_std_df["split"] == "validation"].to_string(index=False))
print(f"\nSelected lambda_ODE: {selected_lambda}")
print(selection_reason)
print(
    f"\nSelected-lambda test results "
    f"(mean {PLUS_MINUS} sample standard deviation):"
)
print(
    "Next-state MSE: "
    f"{selected_test['next_state_mse_mean']:.10g} {PLUS_MINUS} "
    f"{selected_test['next_state_mse_std']:.3g}"
)
print(
    "Next-state RMSE: "
    f"{selected_test['next_state_rmse_mean']:.10g} {PLUS_MINUS} "
    f"{selected_test['next_state_rmse_std']:.3g}"
)
print(
    "ODE residual MSE: "
    f"{selected_test['model_ode_residual_mse_mean']:.10g} {PLUS_MINUS} "
    f"{selected_test['model_ode_residual_mse_std']:.3g}"
)
print(f"\nSaved raw results: {raw_path}")
print(f"Saved numeric aggregate: {aggregate_path}")
print(f"Saved mean {PLUS_MINUS} std table: {mean_std_path}")
print(f"Saved selection decision: {OUTPUT_DIR / 'cartpole_lambda_selection.json'}")
