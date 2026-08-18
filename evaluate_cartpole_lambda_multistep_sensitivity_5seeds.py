"""
CartPole multi-step lambda sensitivity analysis.

Test if a stronger physics-informed loss improves recursive trajectory
prediction at longer horizons, without changing the main lambda selected from
the one-step validation sweep.

lambda_ODE in {0, 1e-5, 1e-4}

- Train all models only on the nominal CartPole transition dataset.
- Use the same data split, initial states, and action sequences within each seed.
- Recursively roll each learned model forward for up to H=16.
- Compare against explicit CartPole dynamics under force calibration:
    c_F in {0.75, 1.0, 1.25}
- Report checkpoints H in {1, 5, 16}.
- Positive relative improvement means lower error than the plain NN (lambda=0).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


#-
# 1. SETTINGS
#-

SEEDS = [42, 123, 456, 789, 1024]

LAMBDA_VALUES = [0.0, 1e-5, 1e-4]
METHOD_LABELS = {
    0.0: "Plain NN",
    1e-5: "ODE-informed NN (lambda=1e-5)",
    1e-4: "ODE-informed NN (lambda=1e-4)",
}

FORCE_CALIBRATION_VALUES = [0.75, 1.0, 1.25]

HORIZONS = [1, 5, 16]
MAX_HORIZON = max(HORIZONS)

DT = 0.02
BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 1e-3

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

N_ROLLOUTS_PER_SEED = 1000

STATE_DIM = 4
ACTION_DIM = 2

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "cartpole_transitions.npz"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "cartpole_lambda_multistep_sensitivity_outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


-
# 2. REPRODUCIBILITY

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def lambda_key(value: float) -> str:
    if np.isclose(value, 0.0):
        return "0"
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


# 3. MODEL

class DynamicsModel(nn.Module):
    """6 -> 64 tanh -> 64 tanh -> 4 next-state predictor."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_dynamics_input(
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    action_one_hot = F.one_hot(
        actions.long(),
        num_classes=ACTION_DIM,
    ).float()

    return torch.cat(
        [states, action_one_hot],
        dim=1,
    )


# 4. CORRECTED CARTPOLE PHYSICS-INFORMED LOSS
def dynamics_loss(
    model: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    lambda_ode: float,
):
    model_input = make_dynamics_input(states, actions)
    predictions = model(model_input)

    # Full next-state supervised prediction loss.
    transition_loss = F.mse_loss(
        predictions,
        next_states,
    )

    # Corrected lightweight kinematic residual:
    # q = [x, theta]
    # q_dot = [x_dot, theta_dot]
    q_current = states[:, [0, 2]]
    q_dot_current = states[:, [1, 3]]
    q_predicted_next = predictions[:, [0, 2]]

    ode_residual = (
        (q_predicted_next - q_current) / DT
        - q_dot_current
    )

    ode_loss = torch.mean(
        torch.sum(
            ode_residual.pow(2),
            dim=1,
        )
    )

    total_loss = (
        transition_loss
        + lambda_ode * ode_loss
    )

    return total_loss, transition_loss, ode_loss


@torch.no_grad()
def evaluate_one_step(
    model: nn.Module,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
) -> Dict[str, float]:
    model.eval()

    predictions = model(x_tensor)
    error = predictions - y_tensor

    next_state_mse = torch.mean(
        torch.sum(error.pow(2), dim=1)
    ).item()

    component_average_mse = torch.mean(
        error.pow(2)
    ).item()

    states = x_tensor[:, :4]

    q_current = states[:, [0, 2]]
    q_dot_current = states[:, [1, 3]]
    q_predicted_next = predictions[:, [0, 2]]

    ode_residual = (
        (q_predicted_next - q_current) / DT
        - q_dot_current
    )

    ode_residual_mse = torch.mean(
        torch.sum(
            ode_residual.pow(2),
            dim=1,
        )
    ).item()

    component_mse = torch.mean(
        error.pow(2),
        dim=0,
    ).cpu().numpy()

    return {
        "next_state_mse": float(next_state_mse),
        "component_average_mse": float(component_average_mse),
        "model_ode_residual_mse": float(ode_residual_mse),
        "mse_x": float(component_mse[0]),
        "mse_x_dot": float(component_mse[1]),
        "mse_theta": float(component_mse[2]),
        "mse_theta_dot": float(component_mse[3]),
    }


#-
# 5. EXPLICIT CARTPOLE DYNAMICS
#-

def cartpole_step_numpy(
    states: np.ndarray,
    actions: np.ndarray,
    force_scale: float = 1.0,
) -> np.ndarray:
    """
    One Euler CartPole step matching Gymnasium CartPole-v1 nominal dynamics,
    except that the action force magnitude is multiplied by force_scale.

    states shape:  [N, 4]
    actions shape: [N]
    """

    states = np.asarray(
        states,
        dtype=np.float64,
    )
    actions = np.asarray(
        actions,
        dtype=np.int64,
    )

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0 * float(force_scale)

    x = states[:, 0]
    x_dot = states[:, 1]
    theta = states[:, 2]
    theta_dot = states[:, 3]

    force = np.where(
        actions == 1,
        force_mag,
        -force_mag,
    )

    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    temp = (
        force
        + polemass_length
        * theta_dot**2
        * sintheta
    ) / total_mass

    theta_acc = (
        gravity * sintheta
        - costheta * temp
    ) / (
        length
        * (
            4.0 / 3.0
            - masspole
            * costheta**2
            / total_mass
        )
    )

    x_acc = (
        temp
        - polemass_length
        * theta_acc
        * costheta
        / total_mass
    )

    # Gymnasium's default CartPole kinematics integrator is Euler:
    # positions are advanced using the current velocities.
    x_next = x + DT * x_dot
    x_dot_next = x_dot + DT * x_acc
    theta_next = theta + DT * theta_dot
    theta_dot_next = theta_dot + DT * theta_acc

    return np.stack(
        [
            x_next,
            x_dot_next,
            theta_next,
            theta_dot_next,
        ],
        axis=1,
    ).astype(np.float32)


#-
# 6. DATA
#-

def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. "
            "Put cartpole_transitions.npz beside this script."
        )

    data = np.load(DATA_FILE)

    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    next_states = data["next_states"].astype(np.float32)

    if not (
        len(states)
        == len(actions)
        == len(next_states)
    ):
        raise ValueError(
            "states, actions, and next_states must have equal lengths"
        )

    if not (
        np.isfinite(states).all()
        and np.isfinite(next_states).all()
    ):
        raise ValueError(
            "Dataset contains NaN or infinite values"
        )

    return states, actions, next_states


def build_inputs(
    states: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    action_one_hot = np.eye(
        ACTION_DIM,
        dtype=np.float32,
    )[actions]

    return np.concatenate(
        [states, action_one_hot],
        axis=1,
    ).astype(np.float32)


def split_indices(
    n_samples: int,
    seed: int,
):
    if not np.isclose(
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO,
        1.0,
    ):
        raise ValueError(
            "Train/validation/test ratios must sum to 1"
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    validation_size = int(
        n_samples * VALIDATION_RATIO
    )
    test_size = int(
        n_samples * TEST_RATIO
    )

    validation_indices = indices[
        :validation_size
    ]

    test_indices = indices[
        validation_size:
        validation_size + test_size
    ]

    train_indices = indices[
        validation_size + test_size:
    ]

    return (
        train_indices,
        validation_indices,
        test_indices,
    )


#-
# 7. MODEL TRAINING
#-

def train_model(
    seed: int,
    lambda_ode: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    state_train: np.ndarray,
    action_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
):
    # Reset to the same initialization and minibatch ordering
    # for every lambda within a seed.
    set_seed(seed)

    dataset = TensorDataset(
        torch.tensor(
            x_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            state_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            action_train,
            dtype=torch.long,
        ),
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
    )

    model = DynamicsModel().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    final_total_loss = np.nan
    final_transition_loss = np.nan
    final_ode_loss = np.nan

    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        total_sum = 0.0
        transition_sum = 0.0
        ode_sum = 0.0

        for (
            x_batch,
            y_batch,
            state_batch,
            action_batch,
        ) in loader:

            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            state_batch = state_batch.to(DEVICE)
            action_batch = action_batch.to(DEVICE)

            optimizer.zero_grad()

            total_loss, transition_loss, ode_loss = dynamics_loss(
                model,
                state_batch,
                action_batch,
                y_batch,
                lambda_ode,
            )

            total_loss.backward()
            optimizer.step()

            batch_n = len(x_batch)

            total_sum += (
                float(total_loss.item())
                * batch_n
            )

            transition_sum += (
                float(transition_loss.item())
                * batch_n
            )

            ode_sum += (
                float(ode_loss.item())
                * batch_n
            )

        final_total_loss = (
            total_sum / len(dataset)
        )
        final_transition_loss = (
            transition_sum / len(dataset)
        )
        final_ode_loss = (
            ode_sum / len(dataset)
        )

        if (
            epoch == 0
            or (epoch + 1) % 20 == 0
            or epoch + 1 == epochs
        ):
            print(
                f"seed={seed:4d} | "
                f"lambda={lambda_ode:.0e} | "
                f"epoch={epoch + 1:3d}/{epochs} | "
                f"total={final_total_loss:.8f} | "
                f"transition={final_transition_loss:.8f} | "
                f"ODE={final_ode_loss:.8f}"
            )

    training_time = time.time() - start_time

    x_validation_tensor = torch.tensor(
        x_validation,
        dtype=torch.float32,
        device=DEVICE,
    )
    y_validation_tensor = torch.tensor(
        y_validation,
        dtype=torch.float32,
        device=DEVICE,
    )

    x_test_tensor = torch.tensor(
        x_test,
        dtype=torch.float32,
        device=DEVICE,
    )
    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.float32,
        device=DEVICE,
    )

    validation_metrics = evaluate_one_step(
        model,
        x_validation_tensor,
        y_validation_tensor,
    )

    test_metrics = evaluate_one_step(
        model,
        x_test_tensor,
        y_test_tensor,
    )

    training_summary = {
        "seed": seed,
        "lambda_ODE": lambda_ode,
        "method": METHOD_LABELS[lambda_ode],
        "epochs": epochs,
        "training_time_sec": training_time,
        "final_total_loss": final_total_loss,
        "final_transition_loss": final_transition_loss,
        "final_ode_loss": final_ode_loss,
    }

    for key, value in validation_metrics.items():
        training_summary[
            f"validation_{key}"
        ] = value

    for key, value in test_metrics.items():
        training_summary[
            f"test_{key}"
        ] = value

    return model, training_summary


#-
# 8. RECURSIVE MODEL ROLLOUT
#-

@torch.no_grad()
def model_rollout(
    model: nn.Module,
    initial_states: np.ndarray,
    action_sequences: np.ndarray,
) -> np.ndarray:
    """
    Open-loop recursive model rollout.

    Returns
    -------
    predictions : [N, MAX_HORIZON, 4]
    """

    model.eval()

    current_states = np.asarray(
        initial_states,
        dtype=np.float32,
    ).copy()

    all_predictions = []

    for step in range(
        action_sequences.shape[1]
    ):
        actions = action_sequences[:, step]

        state_tensor = torch.as_tensor(
            current_states,
            dtype=torch.float32,
            device=DEVICE,
        )

        action_tensor = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=DEVICE,
        )

        model_input = make_dynamics_input(
            state_tensor,
            action_tensor,
        )

        predicted_next = model(
            model_input
        ).cpu().numpy().astype(np.float32)

        all_predictions.append(
            predicted_next
        )

        # Recursive rollout:
        # prediction becomes next model input.
        current_states = predicted_next

    return np.stack(
        all_predictions,
        axis=1,
    )


def true_rollout(
    initial_states: np.ndarray,
    action_sequences: np.ndarray,
    force_scale: float,
) -> np.ndarray:
    """
    Explicit open-loop CartPole rollout under the selected
    force-calibration condition.

    Returns
    -------
    states : [N, MAX_HORIZON, 4]
    """

    current_states = np.asarray(
        initial_states,
        dtype=np.float32,
    ).copy()

    trajectory = []

    for step in range(
        action_sequences.shape[1]
    ):
        current_states = cartpole_step_numpy(
            current_states,
            action_sequences[:, step],
            force_scale=force_scale,
        )

        trajectory.append(
            current_states.copy()
        )

    return np.stack(
        trajectory,
        axis=1,
    )


#-
# 9. TRAJECTORY METRICS
#-

def compute_trajectory_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
):
    """
    predicted/truth shape:
        [N, MAX_HORIZON, 4]

    Returns two collections:
    - step metrics at every rollout step
    - horizon metrics at H in HORIZONS
    """

    error = predicted - truth

    # Per rollout, per step.
    step_l1_per_rollout = np.sum(
        np.abs(error),
        axis=2,
    )

    step_l2_sq_per_rollout = np.sum(
        error**2,
        axis=2,
    )

    step_rows = []

    for step_index in range(
        predicted.shape[1]
    ):
        component_mse = np.mean(
            error[:, step_index, :]**2,
            axis=0,
        )

        step_rows.append(
            {
                "rollout_step": step_index + 1,
                "mean_l1_state_error": float(
                    np.mean(
                        step_l1_per_rollout[
                            :,
                            step_index,
                        ]
                    )
                ),
                "mean_l2_sq_state_error": float(
                    np.mean(
                        step_l2_sq_per_rollout[
                            :,
                            step_index,
                        ]
                    )
                ),
                "mse_x": float(
                    component_mse[0]
                ),
                "mse_x_dot": float(
                    component_mse[1]
                ),
                "mse_theta": float(
                    component_mse[2]
                ),
                "mse_theta_dot": float(
                    component_mse[3]
                ),
            }
        )

    horizon_rows = []

    for horizon in HORIZONS:
        h_slice = slice(
            0,
            horizon,
        )

        cumulative_l1_per_rollout = np.sum(
            step_l1_per_rollout[:, h_slice],
            axis=1,
        )

        cumulative_l2_sq_per_rollout = np.sum(
            step_l2_sq_per_rollout[:, h_slice],
            axis=1,
        )

        mean_step_l1_per_rollout = np.mean(
            step_l1_per_rollout[:, h_slice],
            axis=1,
        )

        mean_step_l2_sq_per_rollout = np.mean(
            step_l2_sq_per_rollout[:, h_slice],
            axis=1,
        )

        endpoint_l1_per_rollout = (
            step_l1_per_rollout[
                :,
                horizon - 1,
            ]
        )

        endpoint_l2_sq_per_rollout = (
            step_l2_sq_per_rollout[
                :,
                horizon - 1,
            ]
        )

        component_cumulative_mse = np.mean(
            np.sum(
                error[:, h_slice, :]**2,
                axis=1,
            ),
            axis=0,
        )

        horizon_rows.append(
            {
                "horizon": horizon,
                "trajectory_l1_error": float(
                    np.mean(
                        cumulative_l1_per_rollout
                    )
                ),
                "trajectory_l2_sq_error": float(
                    np.mean(
                        cumulative_l2_sq_per_rollout
                    )
                ),
                "mean_step_l1_error": float(
                    np.mean(
                        mean_step_l1_per_rollout
                    )
                ),
                "mean_step_l2_sq_error": float(
                    np.mean(
                        mean_step_l2_sq_per_rollout
                    )
                ),
                "endpoint_l1_error": float(
                    np.mean(
                        endpoint_l1_per_rollout
                    )
                ),
                "endpoint_l2_sq_error": float(
                    np.mean(
                        endpoint_l2_sq_per_rollout
                    )
                ),
                "cumulative_mse_x": float(
                    component_cumulative_mse[0]
                ),
                "cumulative_mse_x_dot": float(
                    component_cumulative_mse[1]
                ),
                "cumulative_mse_theta": float(
                    component_cumulative_mse[2]
                ),
                "cumulative_mse_theta_dot": float(
                    component_cumulative_mse[3]
                ),
            }
        )

    return step_rows, horizon_rows


#-
# 10. PAIRED COMPARISON AGAINST PLAIN NN
#-

def build_lambda_comparison_table(
    horizon_df: pd.DataFrame,
) -> pd.DataFrame:

    keys = [
        "seed",
        "force_scale",
        "horizon",
    ]

    metrics = [
        "trajectory_l1_error",
        "trajectory_l2_sq_error",
        "mean_step_l1_error",
        "mean_step_l2_sq_error",
        "endpoint_l1_error",
        "endpoint_l2_sq_error",
    ]

    rows = []

    for key_values, group in horizon_df.groupby(
        keys,
        sort=False,
    ):
        if not isinstance(
            key_values,
            tuple,
        ):
            key_values = (
                key_values,
            )

        plain_rows = group[
            np.isclose(
                group["lambda_ODE"],
                0.0,
            )
        ]

        if len(plain_rows) != 1:
            raise RuntimeError(
                "Expected exactly one plain row "
                f"for keys={key_values}"
            )

        plain = plain_rows.iloc[0]

        for lambda_ode in [
            value
            for value in LAMBDA_VALUES
            if value > 0.0
        ]:
            ode_rows = group[
                np.isclose(
                    group["lambda_ODE"],
                    lambda_ode,
                )
            ]

            if len(ode_rows) != 1:
                raise RuntimeError(
                    "Expected exactly one ODE row "
                    f"for lambda={lambda_ode}, "
                    f"keys={key_values}"
                )

            ode = ode_rows.iloc[0]

            row = dict(
                zip(
                    keys,
                    key_values,
                )
            )

            row["lambda_ODE"] = lambda_ode
            row["method"] = METHOD_LABELS[
                lambda_ode
            ]

            for metric in metrics:
                plain_value = float(
                    plain[metric]
                )
                ode_value = float(
                    ode[metric]
                )

                row[
                    f"plain_{metric}"
                ] = plain_value

                row[
                    f"ode_{metric}"
                ] = ode_value

                row[
                    f"ode_minus_plain_{metric}"
                ] = (
                    ode_value
                    - plain_value
                )

                row[
                    f"ode_improvement_pct_{metric}"
                ] = (
                    100.0
                    * (
                        plain_value
                        - ode_value
                    )
                    / plain_value
                    if plain_value > 0.0
                    else np.nan
                )

            rows.append(row)

    return pd.DataFrame(rows)


#-
# 11. AGGREGATION
#-

def aggregate_mean_std(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    rows = []

    for group_values, group in df.groupby(
        group_cols,
        sort=True,
    ):
        if not isinstance(
            group_values,
            tuple,
        ):
            group_values = (
                group_values,
            )

        row = dict(
            zip(
                group_cols,
                group_values,
            )
        )

        row["n_seeds"] = int(
            group["seed"].nunique()
        )

        for metric in metric_cols:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )

            values = values[
                np.isfinite(values)
            ]

            row[
                f"{metric}_mean"
            ] = (
                float(values.mean())
                if len(values)
                else np.nan
            )

            row[
                f"{metric}_std"
            ] = (
                float(
                    values.std(
                        ddof=1
                    )
                )
                if len(values) > 1
                else (
                    0.0
                    if len(values) == 1
                    else np.nan
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


#-
# 12. PLOTS
#-

def plot_nominal_step_error(
    step_aggregate: pd.DataFrame,
    output_dir: Path,
):
    nominal = step_aggregate[
        np.isclose(
            step_aggregate[
                "force_scale"
            ],
            1.0,
        )
    ].copy()

    plt.figure(
        figsize=(8.5, 5.4)
    )

    for lambda_ode in LAMBDA_VALUES:
        rows = nominal[
            np.isclose(
                nominal["lambda_ODE"],
                lambda_ode,
            )
        ].sort_values(
            "rollout_step"
        )

        x = rows[
            "rollout_step"
        ].to_numpy()

        mean = rows[
            "mean_l1_state_error_mean"
        ].to_numpy()

        std = rows[
            "mean_l1_state_error_std"
        ].fillna(
            0.0
        ).to_numpy()

        line, = plt.plot(
            x,
            mean,
            marker="o",
            label=METHOD_LABELS[
                lambda_ode
            ],
        )

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.12,
            color=line.get_color(),
        )

    plt.xlabel(
        "Rollout step"
    )
    plt.ylabel(
        "Mean L1 state error"
    )
    plt.title(
        "Nominal CartPole: recursive trajectory error"
    )
    plt.grid(
        True,
        alpha=0.25,
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir
        / "lambda_sensitivity_nominal_state_error.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_force_improvement_for_lambda(
    paired_aggregate: pd.DataFrame,
    lambda_ode: float,
    output_dir: Path,
):
    rows_lambda = paired_aggregate[
        np.isclose(
            paired_aggregate[
                "lambda_ODE"
            ],
            lambda_ode,
        )
    ].copy()

    plt.figure(
        figsize=(8.5, 5.4)
    )

    for force_scale in FORCE_CALIBRATION_VALUES:
        rows = rows_lambda[
            np.isclose(
                rows_lambda[
                    "force_scale"
                ],
                force_scale,
            )
        ].sort_values(
            "horizon"
        )

        x = rows[
            "horizon"
        ].to_numpy()

        mean = rows[
            "ode_improvement_pct_trajectory_l1_error_mean"
        ].to_numpy()

        std = rows[
            "ode_improvement_pct_trajectory_l1_error_std"
        ].fillna(
            0.0
        ).to_numpy()

        line, = plt.plot(
            x,
            mean,
            marker="o",
            label=f"c_F={force_scale:g}",
        )

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.12,
            color=line.get_color(),
        )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.xlabel(
        "Rollout horizon H"
    )
    plt.ylabel(
        "Improvement over Plain in trajectory L1 error (%)"
    )

    plt.title(
        "CartPole lambda sensitivity: "
        f"lambda={lambda_ode:.0e}"
    )

    plt.grid(
        True,
        alpha=0.25,
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir
        / (
            "lambda_sensitivity_force_"
            f"lambda_{lambda_key(lambda_ode)}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_nominal_lambda_comparison(
    paired_aggregate: pd.DataFrame,
    output_dir: Path,
):
    nominal = paired_aggregate[
        np.isclose(
            paired_aggregate[
                "force_scale"
            ],
            1.0,
        )
    ].copy()

    plt.figure(
        figsize=(8.5, 5.4)
    )

    for lambda_ode in [
        value
        for value in LAMBDA_VALUES
        if value > 0.0
    ]:
        rows = nominal[
            np.isclose(
                nominal[
                    "lambda_ODE"
                ],
                lambda_ode,
            )
        ].sort_values(
            "horizon"
        )

        x = rows[
            "horizon"
        ].to_numpy()

        mean = rows[
            "ode_improvement_pct_trajectory_l1_error_mean"
        ].to_numpy()

        std = rows[
            "ode_improvement_pct_trajectory_l1_error_std"
        ].fillna(
            0.0
        ).to_numpy()

        line, = plt.plot(
            x,
            mean,
            marker="o",
            label=f"lambda={lambda_ode:.0e}",
        )

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.12,
            color=line.get_color(),
        )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.xlabel(
        "Rollout horizon H"
    )
    plt.ylabel(
        "Improvement over Plain in trajectory L1 error (%)"
    )
    plt.title(
        "Nominal CartPole: physics-weight sensitivity"
    )
    plt.grid(
        True,
        alpha=0.25,
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir
        / "lambda_sensitivity_nominal_improvement.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


#-
# 13. MAIN EXPERIMENT
#-

def run_experiment(
    output_dir: Path,
    smoke: bool,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir = (
        output_dir
        / "models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seeds = (
        [SEEDS[0]]
        if smoke
        else SEEDS
    )

    epochs = (
        2
        if smoke
        else EPOCHS
    )

    n_rollouts = (
        32
        if smoke
        else N_ROLLOUTS_PER_SEED
    )

    print(
        "=" * 80
    )
    print(
        "CARTPOLE MULTI-STEP LAMBDA SENSITIVITY"
    )
    print(
        "=" * 80
    )
    print(
        "Device:",
        DEVICE,
    )
    print(
        "Seeds:",
        seeds,
    )
    print(
        "Lambdas:",
        LAMBDA_VALUES,
    )
    print(
        "Force scales:",
        FORCE_CALIBRATION_VALUES,
    )
    print(
        "Horizons:",
        HORIZONS,
    )
    print(
        "Rollouts per seed:",
        n_rollouts,
    )
    print(
        "Epochs:",
        epochs,
    )
    print(
        "Smoke:",
        smoke,
    )

    states, actions, next_states = load_dataset()
    inputs = build_inputs(
        states,
        actions,
    )

    print(
        "Dataset:",
        states.shape,
        actions.shape,
        next_states.shape,
    )

    # Sanity check: explicit nominal generator should reproduce
    # the transition dataset to numerical precision.
    nominal_generated = cartpole_step_numpy(
        states,
        actions,
        force_scale=1.0,
    )

    nominal_max_abs_error = float(
        np.max(
            np.abs(
                nominal_generated
                - next_states
            )
        )
    )

    print(
        "\nNominal generator max |generated - dataset|:",
        f"{nominal_max_abs_error:.3e}",
    )

    if nominal_max_abs_error > 1e-4:
        print(
            "WARNING: nominal generator differs from the dataset "
            "more than expected. Check the environment integration "
            "before interpreting the experiment."
        )

    training_rows = []
    step_rows_all = []
    horizon_rows_all = []

    for seed in seeds:
        print(
            "\n"
            + "#" * 80
        )
        print(
            f"SEED {seed}"
        )
        print(
            "#" * 80
        )

        (
            train_indices,
            validation_indices,
            test_indices,
        ) = split_indices(
            len(states),
            seed,
        )

        x_train = inputs[
            train_indices
        ]
        y_train = next_states[
            train_indices
        ]
        state_train = states[
            train_indices
        ]
        action_train = actions[
            train_indices
        ]

        x_validation = inputs[
            validation_indices
        ]
        y_validation = next_states[
            validation_indices
        ]

        x_test = inputs[
            test_indices
        ]
        y_test = next_states[
            test_indices
        ]

        # Same held-out rollout roots and same action sequences
        # for every lambda and every force condition within a seed.
        rollout_rng = np.random.default_rng(
            seed + 700_000
        )

        replace = (
            len(test_indices)
            < n_rollouts
        )

        rollout_local_indices = rollout_rng.choice(
            len(test_indices),
            size=n_rollouts,
            replace=replace,
        )

        rollout_initial_states = states[
            test_indices[
                rollout_local_indices
            ]
        ].copy()

        action_sequences = rollout_rng.integers(
            0,
            ACTION_DIM,
            size=(
                n_rollouts,
                MAX_HORIZON,
            ),
            dtype=np.int64,
        )

        # Build all ground-truth trajectories once per seed.
        truth_by_force = {}

        for force_scale in FORCE_CALIBRATION_VALUES:
            truth_by_force[
                force_scale
            ] = true_rollout(
                rollout_initial_states,
                action_sequences,
                force_scale=force_scale,
            )

        for lambda_ode in LAMBDA_VALUES:
            print(
                "\n"
                + "=" * 80
            )
            print(
                f"Training seed={seed}, "
                f"lambda_ODE={lambda_ode}"
            )
            print(
                "=" * 80
            )

            model, training_summary = train_model(
                seed=seed,
                lambda_ode=lambda_ode,
                x_train=x_train,
                y_train=y_train,
                state_train=state_train,
                action_train=action_train,
                x_validation=x_validation,
                y_validation=y_validation,
                x_test=x_test,
                y_test=y_test,
                epochs=epochs,
            )

            training_rows.append(
                training_summary
            )

            torch.save(
                {
                    "seed": seed,
                    "lambda_ODE": lambda_ode,
                    "model_state_dict": model.state_dict(),
                    "training_summary": training_summary,
                },
                model_dir
                / (
                    "cartpole_lambda_"
                    f"{lambda_key(lambda_ode)}"
                    f"_seed_{seed}.pt"
                ),
            )

            predicted = model_rollout(
                model,
                rollout_initial_states,
                action_sequences,
            )

            for force_scale in FORCE_CALIBRATION_VALUES:
                truth = truth_by_force[
                    force_scale
                ]

                (
                    step_rows,
                    horizon_rows,
                ) = compute_trajectory_metrics(
                    predicted,
                    truth,
                )

                for row in step_rows:
                    step_rows_all.append(
                        {
                            "seed": seed,
                            "lambda_ODE": lambda_ode,
                            "method": METHOD_LABELS[
                                lambda_ode
                            ],
                            "force_scale": force_scale,
                            **row,
                        }
                    )

                for row in horizon_rows:
                    horizon_rows_all.append(
                        {
                            "seed": seed,
                            "lambda_ODE": lambda_ode,
                            "method": METHOD_LABELS[
                                lambda_ode
                            ],
                            "force_scale": force_scale,
                            **row,
                        }
                    )

            # Partial saves after each trained model.
            pd.DataFrame(
                training_rows
            ).to_csv(
                output_dir
                / "cartpole_lambda_multistep_training_summary_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

            pd.DataFrame(
                step_rows_all
            ).to_csv(
                output_dir
                / "cartpole_lambda_multistep_step_raw_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

            pd.DataFrame(
                horizon_rows_all
            ).to_csv(
                output_dir
                / "cartpole_lambda_multistep_horizon_raw_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    #-
    # Final raw tables
    #-

    training_df = pd.DataFrame(
        training_rows
    )

    step_df = pd.DataFrame(
        step_rows_all
    )

    horizon_df = pd.DataFrame(
        horizon_rows_all
    )

    paired_df = build_lambda_comparison_table(
        horizon_df
    )

    training_df.to_csv(
        output_dir
        / "cartpole_lambda_multistep_training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    step_df.to_csv(
        output_dir
        / "cartpole_lambda_multistep_step_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_df.to_csv(
        output_dir
        / "cartpole_lambda_multistep_horizon_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_df.to_csv(
        output_dir
        / "cartpole_lambda_multistep_paired_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    #-
    # Aggregates across seeds
    #-

    step_metric_cols = [
        "mean_l1_state_error",
        "mean_l2_sq_state_error",
        "mse_x",
        "mse_x_dot",
        "mse_theta",
        "mse_theta_dot",
    ]

    step_aggregate = aggregate_mean_std(
        step_df,
        group_cols=[
            "lambda_ODE",
            "method",
            "force_scale",
            "rollout_step",
        ],
        metric_cols=step_metric_cols,
    )

    horizon_metric_cols = [
        "trajectory_l1_error",
        "trajectory_l2_sq_error",
        "mean_step_l1_error",
        "mean_step_l2_sq_error",
        "endpoint_l1_error",
        "endpoint_l2_sq_error",
        "cumulative_mse_x",
        "cumulative_mse_x_dot",
        "cumulative_mse_theta",
        "cumulative_mse_theta_dot",
    ]

    horizon_aggregate = aggregate_mean_std(
        horizon_df,
        group_cols=[
            "lambda_ODE",
            "method",
            "force_scale",
            "horizon",
        ],
        metric_cols=horizon_metric_cols,
    )

    paired_metric_cols = [
        column
        for column in paired_df.columns
        if (
            column.startswith(
                "ode_minus_plain_"
            )
            or column.startswith(
                "ode_improvement_pct_"
            )
        )
    ]

    paired_aggregate = aggregate_mean_std(
        paired_df,
        group_cols=[
            "lambda_ODE",
            "method",
            "force_scale",
            "horizon",
        ],
        metric_cols=paired_metric_cols,
    )

    step_aggregate.to_csv(
        output_dir
        / "cartpole_lambda_multistep_step_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_aggregate.to_csv(
        output_dir
        / "cartpole_lambda_multistep_horizon_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_aggregate.to_csv(
        output_dir
        / "cartpole_lambda_multistep_paired_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    #-
    # Configuration record
    #-

    config = {
        "seeds": seeds,
        "lambda_values": LAMBDA_VALUES,
        "method_labels": {
            str(key): value
            for key, value in METHOD_LABELS.items()
        },
        "force_calibration_values": FORCE_CALIBRATION_VALUES,
        "horizons": HORIZONS,
        "max_horizon": MAX_HORIZON,
        "n_rollouts_per_seed": n_rollouts,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dt": DT,
        "device": str(DEVICE),
        "smoke": smoke,
        "nominal_generator_max_abs_error": nominal_max_abs_error,
        "main_selected_lambda_remains": 1e-5,
        "experiment_role": "lambda sensitivity only",
    }

    with (
        output_dir
        / "cartpole_lambda_multistep_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    #-
    # Figures
    #-

    plot_nominal_step_error(
        step_aggregate,
        output_dir,
    )

    for lambda_ode in [
        value
        for value in LAMBDA_VALUES
        if value > 0.0
    ]:
        plot_force_improvement_for_lambda(
            paired_aggregate,
            lambda_ode,
            output_dir,
        )

    plot_nominal_lambda_comparison(
        paired_aggregate,
        output_dir,
    )

    #-
    # Console summary
    #-

    print(
        "\n"
        + "=" * 80
    )
    print(
        "FINAL LAMBDA SENSITIVITY SUMMARY"
    )
    print(
        "=" * 80
    )

    display_columns = [
        "lambda_ODE",
        "force_scale",
        "horizon",
        "ode_improvement_pct_trajectory_l1_error_mean",
        "ode_improvement_pct_trajectory_l1_error_std",
        "ode_improvement_pct_trajectory_l2_sq_error_mean",
        "ode_improvement_pct_trajectory_l2_sq_error_std",
    ]

    display_columns = [
        column
        for column in display_columns
        if column
        in paired_aggregate.columns
    ]

    print(
        paired_aggregate[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nPositive improvement means lower error "
        "than the plain lambda=0 model."
    )

    print(
        "\nMain selected lambda remains 1e-5; "
        "this experiment is sensitivity analysis only."
    )

    print(
        "\nSaved outputs to:",
        output_dir,
    )


#-
# 14. CLI
#-

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CartPole multi-step lambda sensitivity analysis"
        )
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run 1 seed, 2 epochs, and 32 rollouts "
            "to check the complete pipeline."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for CSVs, models, config, and figures."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir

    if args.smoke:
        output_dir = (
            output_dir.parent
            / (
                output_dir.name
                + "_smoke"
            )
        )

    run_experiment(
        output_dir=output_dir,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
