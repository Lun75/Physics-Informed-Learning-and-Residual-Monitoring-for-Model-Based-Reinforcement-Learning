# Physics-Informed-Learning-and-Residual-Monitoring-for-Model-Based-Reinforcement-Learning

MSc Computer Science Project
Queen Mary University of London
2025/26

## Overview
This repository contains the implementation and supporting material for
the MSc project investigating lightweight physics-informed transition
learning and residual monitoring in model-based reinforcement learning.

## Environment

- Python
- PyTorch
- Gymnasium 1.3.0
- CartPole-v1

CartPole-v1:
https://gymnasium.farama.org/environments/classic_control/cart_pole/

## Repository Structure

data/
    cartpole_transitions.npz

src/
    cartpole_transition_experiment.py
    train_cartpole_lambda.py

## Reproducing the Experiments

1. Install dependencies

pip install -r requirements.txt

2. Generate the CartPole dataset

python src/cartpole_transition_experiment.py

3. Physics-weight sweep

python src/train_cartpole_lambda.py

4. Recursive prediction experiments

python src/evaluate_cartpole_lambda_multistep_sensitivity.py

5. Dynamics misspecification

python src/evaluate_cartpole_physical_misspecification.py
python src/evaluate_cartpole_multistep_misspecification.py

6. Residual monitoring

python src/evaluate_cartpole_final_residual_monitoring.py

7. Dyna-DQN evaluation

python src/train_cartpole_dyna_dqn_5seeds.py

## Experimental Settings

Random seeds:
42, 123, 456, 789, 1024

Selected physics weight:
lambda_ODE = 1e-5

CartPole time step:
0.02 s


## External Software

Gymnasium:
https://github.com/Farama-Foundation/Gymnasium

PyTorch:
https://pytorch.org/

## Licence / Academic Use

This repository accompanies an MSc research project at Queen Mary
University of London.
