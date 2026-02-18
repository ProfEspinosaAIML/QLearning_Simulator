# 🎯 Q-Learning Simulator

Welcome to the **Q-Learning Simulator**.

This project is designed to help you **understand, experiment with, and visualize Q-Learning** as presented in:

> Sutton & Barto, *Reinforcement Learning: An Introduction (2nd Edition)*

The simulator is intentionally simple, tabular, and transparent—so you can see exactly how the algorithm works.

---

## 📚 What You Will Learn

By running and modifying this project, you will learn:

- How the Q-Learning update works:
  ```text
  Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s',a') − Q(s,a) ]
  ```
- The difference between:
  - Exploration (ε-greedy behavior)
  - Exploitation (greedy policy)
  - Training vs. evaluation
  - Sparse vs. dense rewards
- How convergence emerges from repeated TD updates
- Why early success ≠ convergence

---

## 🏗 Project Structure

Single-file implementation:

```text
q_learning_simulator.py
```

Includes:

- Tabular Q-Learning agent
- FrozenLake (Gymnasium) support
- Built-in ASCII GridWorld (no Gym required)
- Early stopping options
- Training logs + success-rate tracking
- Slow-motion visual “play” mode
- Model saving/loading

---

## 🚀 Installation

### 1) Create and activate a virtual environment (recommended)

**Windows (PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install numpy
pip install "gymnasium[toy-text]"
```

`gymnasium[toy-text]` installs FrozenLake and **pygame** (needed for GUI rendering).

---

## 🧪 Quick Start

### 1️⃣ Train Silently (Fast)

```bash
python q_learning_simulator.py --env gym --gym-id FrozenLake-v1 --episodes 5000 --epsilon-decay
```

You will see training logs and final evaluation metrics. No graphics yet—just learning.

---

## ✅ Best Teaching Mode (Recommended)

Train quickly, stop when performance is stable, then **visually demonstrate** the learned greedy policy:

```bash
python q_learning_simulator.py --env gym --gym-id FrozenLake-v1   --episodes 5000   --epsilon-decay   --log-every 10   --stop-avg-reward 0.7   --ma-window 100   --play-after-training 5   --render human   --play-sleep 0.15
```

What happens:

1. Console prints learning progress (including success rate).
2. Training stops when the moving-average reward meets the threshold.
3. Then 5 greedy episodes are played slowly in the GUI window.

> For FrozenLake, “average reward” is effectively the **success probability** (because rewards are sparse).

---

## 🎲 Stop on First Success (Great for a lesson)

```bash
python q_learning_simulator.py --env gym --gym-id FrozenLake-v1   --episodes 5000   --epsilon-decay   --log-every 10   --stop-on-goal   --play-after-training 5   --render human   --play-sleep 0.15
```

This is a good demonstration that **one successful episode does not mean convergence**.

---

## 🕹 Live Training + Graphics (Exploration Demo)

If you want students to watch exploration happening in real time, keep episodes small and add a delay:

```bash
python q_learning_simulator.py --env gym --gym-id FrozenLake-v1   --episodes 300   --epsilon-decay   --log-every 1   --render human   --train-render-sleep 0.05
```

This is typically “messier” (lots of exploration), but it’s excellent for teaching ε-greedy behavior.

---

## 🧱 Built-in GridWorld (No Gym Required)

Train on the included ASCII GridWorld:

```bash
python q_learning_simulator.py --env grid --episodes 2000 --epsilon-decay --log-every 20
```

Live ASCII visualization (slowed):

```bash
python q_learning_simulator.py --env grid   --episodes 300   --epsilon-decay   --log-every 1   --render human   --train-render-sleep 0.08
```

---

## 📊 Understanding the Logs

Example:

```text
[episode    200] avg_reward=+0.320 avg_len=8.4 success_rate=32.00% eps=0.818
```

- `avg_reward`: average return over the last `log-every` episodes  
- `avg_len`: average episode length  
- `success_rate`: fraction of episodes that reached the goal  
- `eps`: current epsilon (exploration rate)

---

## 💾 Saving and Loading Models

Train and save:

```bash
python q_learning_simulator.py --env gym --gym-id FrozenLake-v1   --episodes 5000   --epsilon-decay   --save models/frozenlake_Q.npz
```

Load and play (no retraining):

```bash
python q_learning_simulator.py --env gym --gym-id FrozenLake-v1   --episodes 0   --load models/frozenlake_Q.npz   --play-after-training 10   --render human   --play-sleep 0.2
```

This is great for lectures: you can demo the learned policy instantly.

---

## 🔥 Hyperparameter Challenge

In the simulator source code you will find three key hyperparameters:

```python
gamma: float = 0.99
alpha: float = 0.1
epsilon: float = 0.1
```

Your challenge is to achieve a **more optimal and stable** run by experimenting with γ, α, and ε.

Ideas to try:

- **γ (gamma)**: try 0.8 vs 0.999 — short-sighted vs long-horizon planning  
- **α (alpha)**: try 0.5 vs 0.01 — fast/unstable vs slow/stable learning  
- **ε (epsilon)**: too high → noisy; too low → premature convergence; combine with decay  

🎯 Goal:
- Maximize success rate
- Minimize episode length
- Achieve stable convergence (not just one lucky success)

Write down what you changed and **why** the behavior changed.

---

## 🛠 Troubleshooting

If GUI rendering fails for FrozenLake, reinstall the toy-text extras:

```bash
pip install "gymnasium[toy-text]"
```

If you are running in WSL/SSH/headless mode, GUI windows may not appear without display forwarding.

---

## ✅ Final Advice

Don’t just run the code.

Change it. Break it. Instrument it. Compare runs.

Reinforcement Learning becomes intuitive only when you experiment.

Happy Learning 🚀
