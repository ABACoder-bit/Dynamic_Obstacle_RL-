Autonomous Navigation in Dynamic Environments using Q-Learning
Project Overview

This project focuses on the development of an autonomous robot capable of navigating a grid-based environment (12×12) populated with dynamic obstacles.
Using a model-free Reinforcement Learning algorithm (Q-Learning), the agent learns the optimal strategy to move from the starting position (0, 0) to the goal (11, 11) through repeated trial and error.
The project also benchmarks the Q-Learning agent's performance against two other path-planning strategies: a Greedy algorithm and an Artificial Potential Field (APF).

Features
Dynamic Environment: The grid contains 6 moving obstacles that travel at varying speeds towards random targets (70% goal-directed, 30% random).
Q-Learning Implementation: Uses an Epsilon-Greedy strategy to balance exploring new paths and exploiting known safe routes via the Bellman Equation.
Visualizer: Includes Matplotlib-based visualization to watch the robot navigate the grid and avoid obstacles in real-time.
Benchmarking: Scripts to compare Q-Learning success rates, crash frequencies, and step counts against Greedy and APF algorithms.

Dependencies
Python 3.x
NumPy (for Q-table management)
Matplotlib (for visualization and graphing results)
Hyperparameters & Reward Logic
The agent's behavior is shaped by the following hyperparameters and reward structure:
Hyperparameters: Learning Rate (α) = 0.1, Discount Factor (γ) = 0.9, Exploration Rate (ϵ) = 0.3.

Rewards:
Reaching the Goal: +300
Colliding with an Obstacle: -100
Standard Step: -1 (encourages finding the shortest path)
Moving Closer to Goal: +4
Looping/Revisiting previous position: -6

Core Files
Dynamic_qlearning_robot.py: Trains the Q-Learning agent over thousands of episodes and generates the Q-table.
test_agent.py: Loads the trained Q-table (q_table_v2.npy / q_table_v3.npy) and visualizes the agent's performance in a single test run.
test_large.py: Runs a large batch of test episodes (e.g., 100 or 1000) without visualization to statistically evaluate success, crash, and loop rates.
Compare_APF.py: Tests and plots the success rate of the trained Q-Learning agent against an Artificial Potential Field (APF) agent
.
Compare_greedy.py: Tests and plots the success rate of the Q-Learning agent against a Greedy algorithm agent
