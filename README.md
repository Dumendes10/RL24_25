# Reinforcement Learning Algorithm Evaluation

**Group 30**  
*Spring Semester 2024–2025*

## Project Members
- Eduardo Mendes (20240850)
- Jorge Cordeiro (20240594)
- Marta Boavida (20240519)
- Sofia Gomes (20240848)

---

## Overview

This project investigates and compares reinforcement learning algorithms in two environments:

- **ALE/Bowling** (discrete-action, visual input)
- **PandaReach-v3** (continuous-control, robotic manipulation)

The aim is to evaluate algorithm performance, address sparse reward challenges, and analyze generalization across different state-action spaces.

---

## To-Do List

1. **Problem Definition**
   - Specify environment goals and constraints.
2. **Algorithm Implementation**
   - Develop Q-Learning, DQN, PPO for ALE/Bowling.
   - Implement SAC, TD3, and HER variants for PandaReach-v3.
3. **State & Action Representation**
   - Process visual frames for discrete tasks.
   - Encode continuous states and actions for robotics.
4. **Training & Evaluation**
   - Run experiments, tune hyperparameters, and collect metrics.
5. **Results Analysis**
   - Visualize learning curves and compare algorithm performance.
6. **Reporting & Documentation**
   - Prepare detailed report and maintain clear code documentation.

---

## Methodology

| Environment      | Algorithms Tested         | Key Features                        |
|------------------|--------------------------|-------------------------------------|
| ALE/Bowling      | Q-Learning, DQN, PPO     | Discrete actions, CNN processing    |
| PandaReach-v3    | SAC, TD3, HER variants   | Continuous control, goal-based RL   |

- **Metrics:** Success rate, average reward, policy stability
- **Tools:** Python, OpenAI Gym, Stable Baselines3, Matplotlib

---

## Key Results

- **ALE/Bowling:**
  - PPO achieved the highest and most stable rewards (55–60).
  - DQN showed moderate improvement (30–35).
  - Q-Learning struggled with visual complexity (22–30).
- **PandaReach-v3:**
  - All tested algorithms (SAC/TD3 ± HER) had low success rates
