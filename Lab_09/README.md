# Lab 09: Reinforcement Learning & Q-Learning

## Problem Statement
The goal was to solve the CartPole-v1 environment to understand how agents learn through trial and error, specifically exploring the bridge between classical Reinforcement Learning (RL) and modern RLHF techniques.

## Approach and Methodology
Implemented a Q-Learning agent from scratch. Tuned the exploration vs. exploitation tradeoff (`epsilon`) and discount factor (`gamma`). The agent learned to balance the pole by updating its Q-table based on environment rewards.

## Results and Evaluation
The agent transitioned from random failures to competent performance, eventually balancing the pole consistently. Careful tuning of the decay rate (0.995) proved critical for successful convergence.

## Your Learning Outcomes
I learned the fundamental RL training loop (State → Action → Reward). I also drew a clear conceptual bridge from this classical approach to the RLHF systems used to train modern Large Language Models (LLMs).

## Requirements or Dependencies
* Ensure `requirements.txt` from the root directory is installed.
* Standard Python 3.8+ environment with PyTorch.

## Sample Data
* Instructions for data access or the necessary subsets are detailed within the respective Jupyter Notebook cells or provided through standard torch/ torchvision datasets.
