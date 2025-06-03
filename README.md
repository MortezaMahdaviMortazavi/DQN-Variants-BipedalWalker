# Deep Q-Learning Variants on BipedalWalker-v3

This repository contains a series of experiments on the `BipedalWalker-v3` environment using several variants of Deep Q-Learning, including:

- **DQN (Deep Q-Network)**
- **Double DQN**
- **Dueling Double DQN (D3QN)**

The goal of these experiments is to compare the learning dynamics, stability, and final performance of these algorithms under different hyperparameter settings and memory configurations.

---

## 🧪 Experiments Overview

Each folder in the root directory represents a specific experiment group. The naming convention follows this structure:


For example:
- `figures & d3qn`, `figures & d3qn-2`, `figures & d3qn-enhanced`
- `figures & ddqn`, `figures & ddqn2`
- `figures & dqn`

Each folder contains:
- Episode-wise reward plots (e.g., rewards from 0 to 100, 0 to 200, ..., up to 1600)
- Visualization of training stability
- Results from different hyperparameter settings like batch size, replay memory size, learning rate, and priority sampling

---

## 🧠 Algorithms Description

### 🔹 DQN
The basic Deep Q-Network algorithm approximates the Q-value using a single neural network and updates Q-values by minimizing temporal difference (TD) errors. However, it suffers from overestimation of Q-values and unstable learning, especially in dynamic environments like BipedalWalker.

### 🔸 Double DQN
Improves upon DQN by separating the action selection and evaluation steps to reduce overestimation bias. This typically results in more stable training and better long-term performance.

### 🔶 Dueling Double DQN (D3QN)
Introduces separate streams in the network architecture for estimating the state value and the advantage function, then combines them to compute Q-values. This architecture enables the agent to distinguish between valuable and less valuable states even when actions don’t differ much in terms of outcome.

---

## 📊 Results Summary

### ✅ DQN
- **Episodes 0–300:** Rewards remained low (~ -110). Agent was exploring and learning environment dynamics.
- **Episodes 300–1000:** Highly unstable learning with large reward fluctuations due to lack of a target network.
- **Episodes 1000–1600:** Some positive trends emerged, but instability persisted. Occasional severe drops to -200 indicated incomplete learning.

### ✅ Double DQN
- **Episodes 0–700:** Initial exploration phase with relatively negative but stable rewards.
- **Episodes 900–1600:** Steady improvement in performance, with rewards sometimes reaching 200. Reduced variance indicated better learning control.
- **Why better?** It successfully prevented overestimation bias via separate Q estimation and target networks.

### ✅ Dueling Double DQN (D3QN)
- **Episodes 0–200:** Poor performance, as expected in early exploration.
- **Episodes 200–1000:** Rewards began to stabilize with reduced variance. Statistical focus improved.
- **Episodes 1000–1600:** Significant reward improvements occurred. However, a sudden performance collapse was observed — a known issue in RL due to unstable Q-updates or memory over-dependence.
- **Takeaway:** Most consistent and efficient learning overall, with better focus but vulnerability to instability in later phases.

---

## 🧪 Prioritized Replay

Some experiments used **Prioritized Experience Replay**, where transitions are sampled based on TD error. This aimed to speed up learning by focusing on valuable experiences.

However, in this specific implementation, **prioritized replay did not significantly boost performance** — and in some cases introduced instability due to biased sampling. Final conclusion: it was not helpful in this particular task.

---

## 📁 Directory Structure

```text
├── figures & dqn/                # DQN run (baseline)
├── figures & ddqn/               # Double DQN run
├── figures & d3qn/               # Dueling Double DQN
├── figures & [name]-2/           # Second run with changed hyperparameters
├── figures & [name]-enhanced/    # Enhanced version with prioritized memory, or tuned architecture
├── main.py                       # Entry point
├── main.ipynb                    # Interactive notebook (for plotting or analysis)
├── dqn.py, ddqn.py, d3qn.py      # Algorithm implementations
├── RL-HW3.pdf                    # Detailed written report (same as summary above)
├── README.md                     # This file


