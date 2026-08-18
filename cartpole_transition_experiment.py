import gymnasium as gym
import torch
import numpy as np
import pandas as pd

# Reproducibility
SEED = 42
NUM_TRANSITIONS = 10_000

env = gym.make("CartPole-v1")

states = []
actions = []
next_states = []

state, info = env.reset(seed=SEED)
env.action_space.seed(SEED)

for _ in range(NUM_TRANSITIONS):
    # Random exploratory action: 0 = push left, 1 = push right
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    states.append(state.copy())
    actions.append(action)
    next_states.append(next_state.copy())

    if terminated or truncated:
        state, info = env.reset()
    else:
        state = next_state

env.close()

states = np.asarray(states, dtype=np.float32)
actions = np.asarray(actions, dtype=np.int64)
next_states = np.asarray(next_states, dtype=np.float32)

print("States shape:", states.shape)
print("Actions shape:", actions.shape)
print("Next states shape:", next_states.shape)

print("\nFirst transition:")
print("State:", states[0])
print("Action:", actions[0])
print("Next state:", next_states[0])

# Save a readable CSV file
dataset = pd.DataFrame({
    "x": states[:, 0],
    "x_dot": states[:, 1],
    "theta": states[:, 2],
    "theta_dot": states[:, 3],
    "action": actions,
    "next_x": next_states[:, 0],
    "next_x_dot": next_states[:, 1],
    "next_theta": next_states[:, 2],
    "next_theta_dot": next_states[:, 3],
})

dataset.to_csv("cartpole_transitions.csv", index=False)

# Save NumPy arrays for model training
np.savez(
    "cartpole_transitions.npz",
    states=states,
    actions=actions,
    next_states=next_states,
)

print("\nDataset saved successfully.")