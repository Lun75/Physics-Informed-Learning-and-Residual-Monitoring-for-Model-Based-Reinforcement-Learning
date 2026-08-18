"""
CartPole Dyna-DQN pipeline for the MSc physics-informed MBRL project.

Main comparison:
  1) DQN (real replay only)
  2) Plain Dyna-DQN (lambda_ODE = 0)
  3) ODE-informed Dyna-DQN (lambda_ODE = 1e-5)

Optional:
  4) ODE-informed Dyna-DQN + persistent residual-alert gating

The learned transition model matches the existing CartPole experiments:
  input  = [x, x_dot, theta, theta_dot, action_0, action_1]
  model  = 6 -> 64 tanh -> 64 tanh -> 4
  output = next state directly

Dyna loop:
  real environment -> real replay -> train dynamics -> imagine H=1 transitions
  -> DQN planning updates -> updated policy -> real environment
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. settings and hyperparameters

SEEDS = [42, 123, 456, 789, 1024]
DEBUG_ONE_SEED = False   # False for final 5-seed experiment

RUN_DQN_BASELINE = False
RUN_PLAIN_DYNA = True
RUN_ODE_DYNA = True
RUN_ALERT_VARIANT = False

PLAIN_LAMBDA_ODE = 0.0
SELECTED_LAMBDA_ODE = 1e-5

ENV_ID = "CartPole-v1"
STATE_DIM = 4
ACTION_DIM = 2
DT = 0.02

MAX_REAL_STEPS = 20_000
INITIAL_RANDOM_STEPS = 500

EVAL_INTERVAL = 500
EVAL_EPISODES = 10
PERFORMANCE_THRESHOLD = 475.0
PERFORMANCE_CONSECUTIVE_EVALS = 3

REAL_REPLAY_CAPACITY = 100_000
IMAGINED_REPLAY_CAPACITY = 100_000

Q_BATCH_SIZE = 128
Q_LEARNING_RATE = 1e-3
GAMMA = 0.99
TARGET_UPDATE_INTERVAL = 250
GRAD_CLIP_NORM = 10.0

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 10_000
REAL_Q_UPDATES_PER_STEP = 1

DYNAMICS_BATCH_SIZE = 256
DYNAMICS_LEARNING_RATE = 1e-3
DYNAMICS_MIN_SAMPLES = INITIAL_RANDOM_STEPS
DYNAMICS_INITIAL_UPDATES = 500
DYNAMICS_REFRESH_INTERVAL = 250
DYNAMICS_REFRESH_UPDATES = 50

# testing horizon ablation 1, 5
IMAGINATION_HORIZON = 5 
IMAGINATION_BATCH_SIZE = 128
PLANNING_EVERY_REAL_STEPS = 1
PLANNING_Q_UPDATES = 1

MAX_ABS_PREDICTED_STATE = 100.0
CART_POSITION_TERMINATION = 2.4
POLE_ANGLE_TERMINATION = 12.0 * math.pi / 180.0

# Optional monitoring integration.
ALERT_THRESHOLD_CSV = Path("cartpole_final_monitoring_thresholds.csv")
ALERT_THRESHOLD_COLUMN = "alert_threshold"
ALERT_EWMA_ALPHA = 0.10
ALERT_PERSISTENCE_WINDOW = 20
ALERT_PERSISTENCE_MIN_EXCEEDANCES = 10

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "cartpole_dyna_outputs_H5"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. seed setting for reproducibility


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 3. replay buffer for real and imagined transitions

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, state, action, reward, next_state, terminal) -> None:
        i = self.pos
        self.states[i] = np.asarray(state, dtype=np.float32)
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = np.asarray(next_state, dtype=np.float32)
        self.terminals[i] = float(terminal)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states, actions, rewards, next_states, terminals) -> None:
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, terminals):
            self.add(s, a, r, ns, d)

    def sample(self, batch_size: int, rng: np.random.Generator):
        if self.size < batch_size:
            raise ValueError(f"Replay has {self.size}, needs {batch_size}")
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.terminals[idx],
        )

    def sample_states(self, batch_size: int, rng: np.random.Generator):
        if self.size < batch_size:
            raise ValueError(f"Replay has {self.size}, needs {batch_size}")
        idx = rng.integers(0, self.size, size=batch_size)
        return self.states[idx].copy()


# 4. neural network architectures for Q-learning and dynamics model

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, x):
        return self.net(x)


class DynamicsModel(nn.Module):
    """Same 6-64-64-4 transition architecture used in prior experiments."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.net(x)


# 5. CartPole kinematic physics-informed loss and helpers

def cartpole_nominal_derivative(states: torch.Tensor, actions: torch.Tensor):
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

    force = torch.where(
        actions == 1,
        torch.full_like(x_dot, force_mag),
        torch.full_like(x_dot, -force_mag),
    )

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    temp = (force + polemass_length * theta_dot.pow(2) * sintheta) / total_mass
    theta_acc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta.pow(2) / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * costheta / total_mass

    return torch.stack([x_dot, x_acc, theta_dot, theta_acc], dim=1)


def make_dynamics_input(states: torch.Tensor, actions: torch.Tensor):
    a_one_hot = F.one_hot(actions.long(), num_classes=ACTION_DIM).float()
    return torch.cat([states, a_one_hot], dim=1)


def dynamics_loss(model, states, actions, next_states, lambda_ode: float):
    model_input = make_dynamics_input(states, actions)
    pred = model(model_input)

    # Transition prediction loss over the full next state
    transition_loss = F.mse_loss(pred, next_states)

    # Physics-informed CartPole residual
    # q = [x, theta]
    # q_dot = [x_dot, theta_dot]
    q_current = states[:, [0, 2]]
    q_dot_current = states[:, [1, 3]]
    q_pred_next = pred[:, [0, 2]]

    ode_residual = (
        q_pred_next - q_current
    ) / DT - q_dot_current

    ode_loss = torch.mean(
        torch.sum(ode_residual.pow(2), dim=1)
    )

    total_loss = (
        transition_loss
        + lambda_ode * ode_loss
    )

    return total_loss, transition_loss, ode_loss


# 6. DQN HELPERS

def epsilon_by_step(real_step: int) -> float:
    f = min(max(real_step, 0) / float(EPSILON_DECAY_STEPS), 1.0)
    return EPSILON_START + f * (EPSILON_END - EPSILON_START)


@torch.no_grad()
def greedy_actions(q_net: QNetwork, states_np: np.ndarray):
    states_t = torch.as_tensor(states_np, dtype=torch.float32, device=DEVICE)
    return torch.argmax(q_net(states_t), dim=1).cpu().numpy().astype(np.int64)


def epsilon_greedy_action(q_net, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, ACTION_DIM))
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return int(torch.argmax(q_net(s), dim=1).item())


def epsilon_greedy_actions_batch(q_net, states_np, epsilon, rng):
    actions = greedy_actions(q_net, states_np)
    mask = rng.random(len(states_np)) < epsilon
    n = int(mask.sum())
    if n:
        actions[mask] = rng.integers(0, ACTION_DIM, size=n)
    return actions.astype(np.int64)


def q_update(q_net, target_net, optimizer, replay: ReplayBuffer, rng):
    if len(replay) < Q_BATCH_SIZE:
        return None

    states, actions, rewards, next_states, terminals = replay.sample(Q_BATCH_SIZE, rng)
    s = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
    a = torch.as_tensor(actions, dtype=torch.long, device=DEVICE)
    r = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE)
    ns = torch.as_tensor(next_states, dtype=torch.float32, device=DEVICE)
    d = torch.as_tensor(terminals, dtype=torch.float32, device=DEVICE)

    q_selected = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(ns).max(dim=1).values
        target = r + GAMMA * (1.0 - d) * next_q

    loss = F.smooth_l1_loss(q_selected, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    return float(loss.item())


# 7. DYNAMICS MODEL TRAINING AND IMAGINATION

def train_dynamics_updates(model, optimizer, real_replay, rng, lambda_ode, n_updates):
    if len(real_replay) < DYNAMICS_BATCH_SIZE:
        return {"total_loss": np.nan, "transition_loss": np.nan, "ode_loss": np.nan}

    model.train()
    totals, transitions, odes = [], [], []

    for _ in range(n_updates):
        states, actions, _, next_states, _ = real_replay.sample(DYNAMICS_BATCH_SIZE, rng)
        s = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        a = torch.as_tensor(actions, dtype=torch.long, device=DEVICE)
        ns = torch.as_tensor(next_states, dtype=torch.float32, device=DEVICE)

        total, trans, ode = dynamics_loss(model, s, a, ns, lambda_ode)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        totals.append(float(total.item()))
        transitions.append(float(trans.item()))
        odes.append(float(ode.item()))

    return {
        "total_loss": float(np.mean(totals)),
        "transition_loss": float(np.mean(transitions)),
        "ode_loss": float(np.mean(odes)),
    }


@torch.no_grad()
def predict_next_states(model, states_np, actions_np):
    model.eval()
    s = torch.as_tensor(states_np, dtype=torch.float32, device=DEVICE)
    a = torch.as_tensor(actions_np, dtype=torch.long, device=DEVICE)
    return model(make_dynamics_input(s, a)).cpu().numpy().astype(np.float32)


def predicted_cartpole_terminal(next_states_np):
    x = next_states_np[:, 0]
    theta = next_states_np[:, 2]
    return (
        (np.abs(x) > CART_POSITION_TERMINATION)
        | (np.abs(theta) > POLE_ANGLE_TERMINATION)
    )


def generate_imagined_transitions(
    model,
    q_net,
    real_replay,
    imagined_replay,
    rng,
    epsilon,
):
    """
    Generate model rollouts of length IMAGINATION_HORIZON.
    Each rollout begins from a state sampled from real replay.
    Subsequent states are recursively predicted by the learned
    dynamics model, so model error can accumulate during the rollout.
    """

    if len(real_replay) < IMAGINATION_BATCH_SIZE:
        return 0

    # Start every rollout from real observed states.
    current_states = real_replay.sample_states(
        IMAGINATION_BATCH_SIZE,
        rng,
    )

    total_added = 0

    for _ in range(IMAGINATION_HORIZON):

        if len(current_states) == 0:
            break

        # Policy chooses an action at the current imagined state.
        actions = epsilon_greedy_actions_batch(
            q_net,
            current_states,
            epsilon,
            rng,
        )

        # Model predicts the next state.
        next_states = predict_next_states(
            model,
            current_states,
            actions,
        )

        # Reject numerically invalid predictions.
        keep = (
            np.isfinite(next_states).all(axis=1)
            & (
                np.max(
                    np.abs(next_states),
                    axis=1
                ) < MAX_ABS_PREDICTED_STATE
            )
        )

        if not np.any(keep):
            break

        current_states = current_states[keep]
        actions = actions[keep]
        next_states = next_states[keep]

        # Predicted CartPole failure.
        terminals = predicted_cartpole_terminal(
            next_states
        )

        # CartPole gives +1 for each valid step,
        # including the terminating step.
        rewards = np.ones(
            len(current_states),
            dtype=np.float32,
        )

        imagined_replay.add_batch(
            current_states,
            actions,
            rewards,
            next_states,
            terminals,
        )

        total_added += len(current_states)

        # Do not propagate trajectories beyond a predicted terminal state.
        nonterminal = ~terminals

        if not np.any(nonterminal):
            break

        # THIS creates the recursive multi-step rollout:
        # predicted s_(t+1) becomes input for the next prediction.
        current_states = next_states[nonterminal]

    return int(total_added)


# 8. Persistent residual monitoring for alert

def observed_residual_D(state, action, observed_next_state) -> float:
    s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    a = torch.as_tensor([action], dtype=torch.long, device=DEVICE)
    ns = torch.as_tensor(observed_next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        residual = (ns - s) / DT - cartpole_nominal_derivative(s, a)
        return float(torch.sum(residual.pow(2), dim=1).item())


class PersistentResidualMonitor:
    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        self.ewma: Optional[float] = None
        self.history: List[int] = []

    def reset_episode(self):
        self.ewma = None
        self.history = []

    def update(self, d_value: float):
        if self.ewma is None:
            self.ewma = float(d_value)
        else:
            self.ewma = (
                ALERT_EWMA_ALPHA * float(d_value)
                + (1.0 - ALERT_EWMA_ALPHA) * self.ewma
            )

        ordinary = self.ewma > self.threshold
        self.history.append(int(ordinary))
        self.history = self.history[-ALERT_PERSISTENCE_WINDOW:]
        persistent = (
            len(self.history) == ALERT_PERSISTENCE_WINDOW
            and sum(self.history) >= ALERT_PERSISTENCE_MIN_EXCEEDANCES
        )
        return bool(ordinary), bool(persistent), float(self.ewma)


def load_alert_thresholds() -> Dict[int, float]:
    if not ALERT_THRESHOLD_CSV.exists():
        raise FileNotFoundError(
            f"RUN_ALERT_VARIANT=True but {ALERT_THRESHOLD_CSV} was not found"
        )
    df = pd.read_csv(ALERT_THRESHOLD_CSV)
    if "seed" not in df.columns or ALERT_THRESHOLD_COLUMN not in df.columns:
        raise KeyError(
            f"Threshold CSV needs columns 'seed' and '{ALERT_THRESHOLD_COLUMN}'"
        )
    return {
        int(row["seed"]): float(row[ALERT_THRESHOLD_COLUMN])
        for _, row in df.iterrows()
    }

# 9. Policy evaluation

@torch.no_grad()
def evaluate_policy(q_net: QNetwork, seed: int, n_episodes: int = EVAL_EPISODES):
    env = gym.make(ENV_ID)
    returns = []
    q_net.eval()

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + 100_000 + ep)
        ret = 0.0
        while True:
            s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = int(torch.argmax(q_net(s), dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            ret += float(reward)
            if terminated or truncated:
                break
        returns.append(ret)

    env.close()
    return float(np.mean(returns)), float(np.std(returns, ddof=1)), returns


# 10. method configuration dataclass and builder

@dataclass(frozen=True)
class MethodConfig:
    name: str
    label: str
    use_model: bool
    lambda_ode: float = 0.0
    use_alert: bool = False


def build_methods():
    methods = []
    if RUN_DQN_BASELINE:
        methods.append(MethodConfig("dqn", "DQN", False))
    if RUN_PLAIN_DYNA:
        methods.append(MethodConfig("plain_dyna", "Plain Dyna-DQN", True, 0.0))
    if RUN_ODE_DYNA:
        methods.append(MethodConfig("ode_dyna", "ODE-informed Dyna-DQN", True, SELECTED_LAMBDA_ODE))
    if RUN_ALERT_VARIANT:
        methods.append(MethodConfig(
            "ode_dyna_alert", "ODE Dyna-DQN + alert", True,
            SELECTED_LAMBDA_ODE, True,
        ))
    return methods


# 11. Main training loop for one seed and one method

def train_one_run(seed: int, method: MethodConfig, alert_thresholds=None):
    set_seed(seed)
    rng = np.random.default_rng(seed)

    env = gym.make(ENV_ID)
    env.action_space.seed(seed)

    q_net = QNetwork().to(DEVICE)
    target_net = QNetwork().to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    q_optimizer = torch.optim.Adam(q_net.parameters(), lr=Q_LEARNING_RATE)

    real_replay = ReplayBuffer(REAL_REPLAY_CAPACITY)
    imagined_replay = ReplayBuffer(IMAGINED_REPLAY_CAPACITY)

    dynamics_model = None
    dynamics_optimizer = None
    if method.use_model:
        dynamics_model = DynamicsModel().to(DEVICE)
        dynamics_optimizer = torch.optim.Adam(
            dynamics_model.parameters(), lr=DYNAMICS_LEARNING_RATE
        )

    monitor = None
    if method.use_alert:
        if alert_thresholds is None or seed not in alert_thresholds:
            raise ValueError(f"Missing calibrated alert threshold for seed {seed}")
        monitor = PersistentResidualMonitor(alert_thresholds[seed])

    eval_rows, episode_rows = [], []
    q_losses_real, q_losses_imagined = [], []
    imagined_transition_count = 0
    planning_events = 0
    blocked_planning_events = 0
    ordinary_alert_steps = 0
    persistent_alert_steps = 0
    dynamics_refresh_count = 0
    dynamics_ready = False

    last_dyn = {"total_loss": np.nan, "transition_loss": np.nan, "ode_loss": np.nan}

    # Evaluation before real training samples.
    m, s, _ = evaluate_policy(q_net, seed)
    eval_rows.append({
        "seed": seed,
        "method": method.name,
        "method_label": method.label,
        "lambda_ODE": method.lambda_ode if method.use_model else np.nan,
        "real_steps": 0,
        "eval_return_mean": m,
        "eval_return_std_within_seed": s,
    })

    state, _ = env.reset(seed=seed)
    if monitor is not None:
        monitor.reset_episode()

    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    start_time = time.time()

    for real_step in range(1, MAX_REAL_STEPS + 1):
        epsilon = epsilon_by_step(real_step)

        if real_step <= INITIAL_RANDOM_STEPS:
            action = int(rng.integers(0, ACTION_DIM))
        else:
            action = epsilon_greedy_action(q_net, state, epsilon, rng)

        next_state, reward, terminated, truncated, _ = env.step(action)

        # True failure is terminal for Bellman backup; TimeLimit truncation is not.
        real_replay.add(state, action, reward, next_state, bool(terminated))
        episode_return += float(reward)
        episode_length += 1

        persistent_alert_active = False
        if monitor is not None:
            d_value = observed_residual_D(state, action, next_state)
            ordinary, persistent_alert_active, _ = monitor.update(d_value)
            ordinary_alert_steps += int(ordinary)
            persistent_alert_steps += int(persistent_alert_active)

        # Train dynamics ONLY from real replay.
        if (
            method.use_model
            and dynamics_model is not None
            and dynamics_optimizer is not None
            and len(real_replay) >= DYNAMICS_MIN_SAMPLES
        ):
            if not dynamics_ready:
                last_dyn = train_dynamics_updates(
                    dynamics_model, dynamics_optimizer, real_replay, rng,
                    method.lambda_ode, DYNAMICS_INITIAL_UPDATES,
                )
                dynamics_ready = True
                dynamics_refresh_count += 1
            elif real_step % DYNAMICS_REFRESH_INTERVAL == 0:
                last_dyn = train_dynamics_updates(
                    dynamics_model, dynamics_optimizer, real_replay, rng,
                    method.lambda_ode, DYNAMICS_REFRESH_UPDATES,
                )
                dynamics_refresh_count += 1

        # Standard DQN update from real data.
        if real_step > INITIAL_RANDOM_STEPS:
            for _ in range(REAL_Q_UPDATES_PER_STEP):
                loss = q_update(q_net, target_net, q_optimizer, real_replay, rng)
                if loss is not None:
                    q_losses_real.append(loss)

        # Dyna update from H=1 imagined data.
        if (
            method.use_model
            and dynamics_ready
            and dynamics_model is not None
            and real_step > INITIAL_RANDOM_STEPS
            and real_step % PLANNING_EVERY_REAL_STEPS == 0
        ):
            if method.use_alert and persistent_alert_active:
                blocked_planning_events += 1
            else:
                n_added = generate_imagined_transitions(
                    dynamics_model, q_net, real_replay, imagined_replay, rng, epsilon
                )
                imagined_transition_count += n_added
                planning_events += 1

                for _ in range(PLANNING_Q_UPDATES):
                    loss = q_update(
                        q_net, target_net, q_optimizer, imagined_replay, rng
                    )
                    if loss is not None:
                        q_losses_imagined.append(loss)

        if real_step % TARGET_UPDATE_INTERVAL == 0:
            target_net.load_state_dict(q_net.state_dict())

        if terminated or truncated:
            episode_rows.append({
                "seed": seed,
                "method": method.name,
                "method_label": method.label,
                "lambda_ODE": method.lambda_ode if method.use_model else np.nan,
                "episode": episode_index,
                "real_steps": real_step,
                "episode_return": episode_return,
                "episode_length": episode_length,
            })
            episode_index += 1
            state, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
            if monitor is not None:
                monitor.reset_episode()
        else:
            state = next_state

        if real_step % EVAL_INTERVAL == 0:
            m, s, _ = evaluate_policy(q_net, seed)
            eval_rows.append({
                "seed": seed,
                "method": method.name,
                "method_label": method.label,
                "lambda_ODE": method.lambda_ode if method.use_model else np.nan,
                "real_steps": real_step,
                "eval_return_mean": m,
                "eval_return_std_within_seed": s,
            })
            print(
                f"seed={seed:4d} | {method.label:23s} | "
                f"steps={real_step:6d} | return={m:7.2f} | "
                f"eps={epsilon:.3f} | imag={len(imagined_replay):6d}"
            )

    wall_time = time.time() - start_time
    env.close()

    eval_df = pd.DataFrame(eval_rows)
    episodes_df = pd.DataFrame(episode_rows)

    x = eval_df["real_steps"].to_numpy(dtype=np.float64)
    y = eval_df["eval_return_mean"].to_numpy(dtype=np.float64)
    if len(x) >= 2 and x[-1] > x[0]:
        auc = float(np.trapezoid(y, x))
        normalized_auc = auc / float(x[-1] - x[0])
    else:
        auc = np.nan
        normalized_auc = np.nan

    above_threshold = (eval_df["eval_return_mean"].to_numpy() >= PERFORMANCE_THRESHOLD
)

    window = PERFORMANCE_CONSECUTIVE_EVALS

    if len(above_threshold) >= window:
        rolling_hits = np.convolve(
            above_threshold.astype(np.int32),
            np.ones(window, dtype=np.int32),
            mode="valid",
        )

        sustained_starts = np.flatnonzero(
            rolling_hits == window
        )
    else:
        sustained_starts = np.array([], dtype=np.int64)

    if len(sustained_starts):
        sustained_end_index = (
            sustained_starts[0] + window - 1
        )
        steps_to_threshold = float(
            x[sustained_end_index]
        )
    else:
        steps_to_threshold = np.nan

    summary = {
        "seed": seed,
        "method": method.name,
        "method_label": method.label,
        "lambda_ODE": method.lambda_ode if method.use_model else np.nan,
        "final_eval_return": float(y[-1]),
        "best_eval_return": float(np.max(y)),
        "return_auc": auc,
        "normalized_return_auc": normalized_auc,
        "steps_to_475": steps_to_threshold,
        "reached_475": float(np.isfinite(steps_to_threshold)),
        "wall_time_sec": wall_time,
        "n_real_transitions": len(real_replay),
        "n_imagined_transitions_generated": imagined_transition_count,
        "n_planning_events": planning_events,
        "n_blocked_planning_events": blocked_planning_events,
        "ordinary_alert_rate_training_pct": (
            100.0 * ordinary_alert_steps / MAX_REAL_STEPS if monitor else np.nan
        ),
        "persistent_alert_rate_training_pct": (
            100.0 * persistent_alert_steps / MAX_REAL_STEPS if monitor else np.nan
        ),
        "mean_real_q_loss": float(np.mean(q_losses_real)) if q_losses_real else np.nan,
        "mean_imagined_q_loss": (
            float(np.mean(q_losses_imagined)) if q_losses_imagined else np.nan
        ),
        "final_dynamics_total_loss": last_dyn["total_loss"],
        "final_dynamics_transition_loss": last_dyn["transition_loss"],
        "final_dynamics_ode_loss": last_dyn["ode_loss"],
        "dynamics_refresh_count": dynamics_refresh_count,
    }

    checkpoint = {
        "seed": seed,
        "method": method.name,
        "lambda_ODE": method.lambda_ode,
        "q_network_state_dict": q_net.state_dict(),
        "summary": summary,
        "settings": {
            "MAX_REAL_STEPS": MAX_REAL_STEPS,
            "INITIAL_RANDOM_STEPS": INITIAL_RANDOM_STEPS,
            "IMAGINATION_HORIZON": IMAGINATION_HORIZON,
            "SELECTED_LAMBDA_ODE": SELECTED_LAMBDA_ODE,
        },
    }
    if dynamics_model is not None:
        checkpoint["dynamics_model_state_dict"] = dynamics_model.state_dict()

    torch.save(
        checkpoint,
        CHECKPOINT_DIR / f"cartpole_{method.name}_seed_{seed}.pt",
    )

    return eval_df, episodes_df, summary


# 12. Evaluation and summary aggregation

def aggregate_evaluation(eval_raw: pd.DataFrame):
    return (
        eval_raw.groupby(["method", "method_label", "real_steps"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            eval_return_mean=("eval_return_mean", "mean"),
            eval_return_std=("eval_return_mean", "std"),
        )
        .sort_values(["method", "real_steps"])
        .reset_index(drop=True)
    )


def aggregate_summary(summary_df: pd.DataFrame):
    metrics = [
        "final_eval_return",
        "best_eval_return",
        "normalized_return_auc",
        "steps_to_475",
        "reached_475",
        "wall_time_sec",
        "n_imagined_transitions_generated",
        "n_blocked_planning_events",
        "ordinary_alert_rate_training_pct",
        "persistent_alert_rate_training_pct",
    ]

    rows = []
    for (method, label), group in summary_df.groupby(["method", "method_label"]):
        row = {"method": method, "method_label": label, "n_seeds": group["seed"].nunique()}
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce")
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = (
                float(vals.std(ddof=1)) if len(vals) > 1
                else (0.0 if len(vals) == 1 else np.nan)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sample_efficiency(aggregate_df: pd.DataFrame):
    plt.figure(figsize=(8.2, 5.2))

    for label in aggregate_df["method_label"].drop_duplicates():
        rows = aggregate_df[aggregate_df["method_label"] == label].sort_values("real_steps")
        x = rows["real_steps"].to_numpy()
        mean = rows["eval_return_mean"].to_numpy()
        std = rows["eval_return_std"].fillna(0.0).to_numpy()

        line, = plt.plot(x, mean, label=label)
        plt.fill_between(
            x, mean - std, mean + std,
            alpha=0.15,
            color=line.get_color(),
        )

    plt.axhline(PERFORMANCE_THRESHOLD, linestyle="--", linewidth=1.0)
    plt.xlabel("Real environment steps")
    plt.ylabel("Evaluation return")
    plt.ylim(0, 510)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "cartpole_dyna_sample_efficiency.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# 13. Main entry point for running all seeds and methods

def main():
    print("=" * 80)
    print("CARTPOLE DYNA-DQN: PHYSICS-INFORMED MBRL")
    print("=" * 80)
    print("Device:", DEVICE)
    print("Selected lambda_ODE:", SELECTED_LAMBDA_ODE)
    print("Fixed imagination horizon:", IMAGINATION_HORIZON)
    print("Real-step budget per seed/method:", MAX_REAL_STEPS)

    methods = build_methods()
    seeds = [SEEDS[0]] if DEBUG_ONE_SEED else SEEDS
    print("Seeds:", seeds)
    print("Methods:", [m.label for m in methods])

    alert_thresholds = load_alert_thresholds() if RUN_ALERT_VARIANT else None

    all_eval = []
    all_episodes = []
    summaries = []

    for seed in seeds:
        for method in methods:
            print("\n" + "=" * 80)
            print(f"START seed={seed} | {method.label}")
            print("=" * 80)

            eval_df, episodes_df, summary = train_one_run(
                seed, method, alert_thresholds
            )
            all_eval.append(eval_df)
            all_episodes.append(episodes_df)
            summaries.append(summary)

            # Save after every completed run.
            pd.concat(all_eval, ignore_index=True).to_csv(
                OUTPUT_DIR / "cartpole_dyna_eval_raw_partial.csv",
                index=False, encoding="utf-8-sig"
            )
            pd.concat(all_episodes, ignore_index=True).to_csv(
                OUTPUT_DIR / "cartpole_dyna_training_episodes_partial.csv",
                index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(summaries).to_csv(
                OUTPUT_DIR / "cartpole_dyna_run_summary_partial.csv",
                index=False, encoding="utf-8-sig"
            )

    eval_raw = pd.concat(all_eval, ignore_index=True)
    episodes_raw = pd.concat(all_episodes, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    eval_agg = aggregate_evaluation(eval_raw)
    summary_agg = aggregate_summary(summary_df)

    eval_raw.to_csv(
        OUTPUT_DIR / "cartpole_dyna_eval_raw.csv",
        index=False, encoding="utf-8-sig"
    )
    eval_agg.to_csv(
        OUTPUT_DIR / "cartpole_dyna_eval_aggregate.csv",
        index=False, encoding="utf-8-sig"
    )
    episodes_raw.to_csv(
        OUTPUT_DIR / "cartpole_dyna_training_episodes.csv",
        index=False, encoding="utf-8-sig"
    )
    summary_df.to_csv(
        OUTPUT_DIR / "cartpole_dyna_run_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    summary_agg.to_csv(
        OUTPUT_DIR / "cartpole_dyna_summary_aggregate.csv",
        index=False, encoding="utf-8-sig"
    )

    plot_sample_efficiency(eval_agg)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    cols = [
        "method_label", "n_seeds",
        "final_eval_return_mean", "final_eval_return_std",
        "normalized_return_auc_mean", "normalized_return_auc_std",
        "steps_to_475_mean", "steps_to_475_std",
        "reached_475_mean",
    ]
    cols = [c for c in cols if c in summary_agg.columns]
    print(summary_agg[cols].to_string(index=False))
    print("\nSaved outputs to:", OUTPUT_DIR)

    if DEBUG_ONE_SEED:
        print(
            "\nDEBUG_ONE_SEED=True: smoke test only. "
            "Set DEBUG_ONE_SEED=False for the final five-seed run."
        )


if __name__ == "__main__":
    main()
