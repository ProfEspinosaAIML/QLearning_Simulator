"""
q_learning_simulator.py (Teaching Edition)

Tabular Q-Learning simulator aligned with Sutton & Barto pseudocode:
  Q(S,A) <- Q(S,A) + alpha * [ R + gamma * max_a' Q(S',a') - Q(S,A) ]

Improvements for teaching:
- Early stopping options (stop on first goal / stop on moving-average reward threshold)
- "Play after training" mode with slow, visual greedy rollouts
- Optional step delay during rendering to avoid flicker
- More frequent / configurable logging

Works with:
  1) Gymnasium discrete environments (e.g., FrozenLake-v1)
  2) Included ASCII GridWorld (no external deps)

INSTALL (FrozenLake GUI):
  pip install "gymnasium[toy-text]"
  (or) pip install pygame

EXAMPLES

A) Recommended classroom flow (train fast, then visually demo policy)
    python q_learning_simulator.py --env gym --gym-id FrozenLake-v1 \
      --episodes 5000 --epsilon-decay --log-every 50 \
      --stop-on-goal \
      --play-after-training 5 --render human --play-sleep 0.15

B) If you want logs every 10 episodes
    python q_learning_simulator.py --env gym --gym-id FrozenLake-v1 \
      --episodes 5000 --epsilon-decay --log-every 10 \
      --play-after-training 5 --render human --play-sleep 0.15

C) Live training + rendering (usually noisy, use fewer episodes + delay)
    python q_learning_simulator.py --env gym --gym-id FrozenLake-v1 \
      --episodes 300 --epsilon-decay --log-every 1 \
      --render human --train-render-sleep 0.05

D) Two-run workflow (no code edits): train once, save; then load and play
    python q_learning_simulator.py --env gym --gym-id FrozenLake-v1 \
      --episodes 5000 --epsilon-decay --save models/frozenlake_Q.npz
    python q_learning_simulator.py --env gym --gym-id FrozenLake-v1 \
      --episodes 0 --load models/frozenlake_Q.npz \
      --play-after-training 5 --render human --play-sleep 0.15

E) Logs print every 10 seconds, training stops as soon as the first successful episode occurs
    python q_learning_simulator.py --env gym --gym-id FrozenLake-v1 \
        --episodes 5000 --epsilon-decay --log-every 10 --stop-on-goal \
            --play-after-training 5 --render human --play-sleep 0.15
    Sample log output:
    [episode     10] avg_reward=+0.000 avg_len=8.5 success_rate=0.00% eps=0.991
    [episode     20] avg_reward=+0.000 avg_len=6.8 success_rate=0.00% eps=0.982
    [episode     30] avg_reward=+0.000 avg_len=8.0 success_rate=0.00% eps=0.973
    [episode     40] avg_reward=+0.000 avg_len=8.6 success_rate=0.00% eps=0.964
    [episode     50] avg_reward=+0.000 avg_len=4.4 success_rate=0.00% eps=0.955
    [episode     60] avg_reward=+0.000 avg_len=6.9 success_rate=0.00% eps=0.946
    [episode     70] avg_reward=+0.000 avg_len=7.2 success_rate=0.00% eps=0.937
    [episode     80] avg_reward=+0.000 avg_len=7.5 success_rate=0.00% eps=0.928
    [episode     90] avg_reward=+0.000 avg_len=7.6 success_rate=0.00% eps=0.919
    [episode    100] avg_reward=+0.000 avg_len=5.7 success_rate=0.00% eps=0.910
    [episode    110] avg_reward=+0.000 avg_len=6.7 success_rate=0.00% eps=0.902
    [episode    120] avg_reward=+0.000 avg_len=4.4 success_rate=0.00% eps=0.893
    [episode    130] avg_reward=+0.000 avg_len=7.9 success_rate=0.00% eps=0.885
    [episode    140] avg_reward=+0.000 avg_len=9.9 success_rate=0.00% eps=0.877
    [episode    150] avg_reward=+0.000 avg_len=7.1 success_rate=0.00% eps=0.868
    [episode    160] avg_reward=+0.000 avg_len=6.3 success_rate=0.00% eps=0.860
    [episode    170] avg_reward=+0.000 avg_len=8.2 success_rate=0.00% eps=0.852
    [episode    180] avg_reward=+0.000 avg_len=6.6 success_rate=0.00% eps=0.844
    [episode    190] avg_reward=+0.000 avg_len=9.4 success_rate=0.00% eps=0.836
    [episode    200] avg_reward=+0.000 avg_len=9.1 success_rate=0.00% eps=0.828
    Early stop: first success at episode 207 (reward=+1.000).

F) For better convergence (better than stop on the first goal)
    python q_learning_simulator.py --env gym --gym-id FrozenLake-v1 \
        --episodes 5000 --epsilon-decay \
        --log-every 20 \
        --stop-avg-reward 0.7 --ma-window 100 \
        --play-after-training 5 --render human --play-sleep 0.2


"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------
# Utilities
# ----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def moving_average(values: List[float], window: int) -> float:
    if window <= 1:
        return float(values[-1]) if values else 0.0
    if len(values) < window:
        return float(np.mean(values)) if values else 0.0
    return float(np.mean(values[-window:]))


# ----------------------------
# Environment abstraction
# ----------------------------

class DiscreteEnv:
    """Minimal interface for discrete tabular RL environments."""
    @property
    def n_states(self) -> int:
        raise NotImplementedError

    @property
    def n_actions(self) -> int:
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> int:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    def render(self) -> None:
        pass


class GymnasiumWrapper(DiscreteEnv):
    """Wrap a Gymnasium env with discrete obs/action spaces."""
    def __init__(self, gym_env: Any, render: str = "none") -> None:
        self.env = gym_env
        self._render = render
        try:
            self._n_states = int(self.env.observation_space.n)
            self._n_actions = int(self.env.action_space.n)
        except Exception as e:
            raise ValueError("Requires discrete observation_space.n and action_space.n") from e

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def reset(self, seed: Optional[int] = None) -> int:
        out = self.env.reset(seed=seed)
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
        else:
            obs = out
        return int(obs)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        out = self.env.step(int(action))
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = out
            done = bool(done)
        return int(obs), float(reward), bool(done), dict(info) if info is not None else {}

    def render(self) -> None:
        if self._render != "none":
            try:
                self.env.render()
            except Exception:
                pass


# ----------------------------
# Built-in GridWorld (ASCII)
# ----------------------------

class GridWorld(DiscreteEnv):
    """
    Deterministic ASCII GridWorld.

    States: s = r*n_cols + c
    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    """
    def __init__(
        self,
        n_rows: int = 4,
        n_cols: int = 4,
        start: Tuple[int, int] = (0, 0),
        goal: Tuple[int, int] = (3, 3),
        pits: Optional[List[Tuple[int, int]]] = None,
        walls: Optional[List[Tuple[int, int]]] = None,
        step_reward: float = -0.01,
        goal_reward: float = 1.0,
        pit_reward: float = -1.0,
        max_steps: int = 200,
    ) -> None:
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.start_rc = start
        self.goal_rc = goal
        self.pits = set(pits or [])
        self.walls = set(walls or [])
        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)
        self.pit_reward = float(pit_reward)
        self.max_steps = int(max_steps)
        self._state_rc = self.start_rc
        self._steps = 0

    @property
    def n_states(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def n_actions(self) -> int:
        return 4

    def _encode(self, rc: Tuple[int, int]) -> int:
        r, c = rc
        return r * self.n_cols + c

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            set_global_seed(seed)
        self._state_rc = self.start_rc
        self._steps = 0
        return self._encode(self._state_rc)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        self._steps += 1
        r, c = self._state_rc
        dr, dc = 0, 0
        if action == 0: dr = -1
        elif action == 1: dc = 1
        elif action == 2: dr = 1
        elif action == 3: dc = -1
        else: raise ValueError("Invalid action (0..3)")

        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols) or (nr, nc) in self.walls:
            nr, nc = r, c

        self._state_rc = (nr, nc)

        done = False
        reward = self.step_reward
        info: Dict[str, Any] = {"position": self._state_rc, "steps": self._steps}

        if self._state_rc == self.goal_rc:
            done, reward = True, self.goal_reward
            info["terminal"] = "goal"
        elif self._state_rc in self.pits:
            done, reward = True, self.pit_reward
            info["terminal"] = "pit"
        elif self._steps >= self.max_steps:
            done = True
            info["terminal"] = "timeout"

        return self._encode(self._state_rc), float(reward), bool(done), info

    def render(self) -> None:
        lines = []
        for rr in range(self.n_rows):
            row = []
            for cc in range(self.n_cols):
                cell = (rr, cc)
                if cell in self.walls: row.append("█")
                elif cell == self.goal_rc: row.append("G")
                elif cell in self.pits: row.append("P")
                elif cell == self._state_rc: row.append("A")
                elif cell == self.start_rc: row.append("S")
                else: row.append("·")
            lines.append(" ".join(row))
        print("\n".join(lines))
        print()


# ----------------------------
# Q-Learning agent
# ----------------------------

@dataclass
class EpsilonSchedule:
    """Exponential decay schedule for epsilon (exploration rate)."""

    start: float = 1.0   # initial epsilon (high exploration)
    end: float = 0.05    # minimum epsilon (some exploration remains)
    decay: float = 0.999 # multiplicative decay per episode

    def value(self, episode_idx: int) -> float:
        eps = self.end + (self.start - self.end) * (self.decay ** episode_idx)
        return float(max(self.end, min(self.start, eps)))



@dataclass
class QLearningConfig:
    # Discount factor (γ)
    # Controls how much the agent values future rewards.
    # γ → 1.0  : long-term planning (future rewards matter a lot)
    # γ → 0.0  : short-sighted (only immediate reward matters)
    gamma: float = 0.99

    # Learning rate (α)
    # Controls how much new information overrides old Q-values.
    # α → 1.0  : aggressive updates (fast but can be unstable)
    # α → 0.0  : very slow learning
    alpha: float = 0.1

    # Exploration rate (ε)
    # Probability of choosing a random action instead of the greedy one.
    # Higher ε → more exploration; lower ε → more exploitation
    epsilon: float = 0.1
    epsilon_schedule: Optional[EpsilonSchedule] = None
    max_steps_per_episode: int = 10_000
    seed: int = 42


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, cfg: QLearningConfig) -> None:
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.cfg = cfg
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        set_global_seed(cfg.seed)

    def get_epsilon(self, episode_idx: int) -> float:
        return self.cfg.epsilon if self.cfg.epsilon_schedule is None else self.cfg.epsilon_schedule.value(episode_idx)

    def act(self, state: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        q = self.Q[state]
        m = float(np.max(q))
        candidates = np.flatnonzero(q == m)
        return int(random.choice(candidates))

    def update(self, s: int, a: int, r: float, sp: int, done: bool) -> float:
        q_sa = self.Q[s, a]
        target = r if done else (r + self.cfg.gamma * float(np.max(self.Q[sp])))
        td_error = target - q_sa
        self.Q[s, a] = q_sa + self.cfg.alpha * td_error
        return float(td_error)

    def save(self, path: str) -> None:
        np.savez_compressed(path, Q=self.Q, cfg=dataclasses.asdict(self.cfg))

    @staticmethod
    def load(path: str) -> "QLearningAgent":
        data = np.load(path, allow_pickle=True)
        Q = data["Q"]
        cfg_dict = data["cfg"].item() if isinstance(data["cfg"], np.ndarray) else data["cfg"]
        eps_sched = None
        if cfg_dict.get("epsilon_schedule") is not None:
            eps_sched = EpsilonSchedule(**cfg_dict["epsilon_schedule"])
        cfg = QLearningConfig(
            gamma=float(cfg_dict["gamma"]),
            alpha=float(cfg_dict["alpha"]),
            epsilon=float(cfg_dict["epsilon"]),
            epsilon_schedule=eps_sched,
            max_steps_per_episode=int(cfg_dict["max_steps_per_episode"]),
            seed=int(cfg_dict["seed"]),
        )
        agent = QLearningAgent(Q.shape[0], Q.shape[1], cfg)
        agent.Q = Q.astype(np.float64)
        return agent


# ----------------------------
# Simulator
# ----------------------------

@dataclass
class EpisodeStats:
    episode: int
    steps: int
    total_reward: float
    epsilon: float
    td_error_mean_abs: float
    success: bool


class QLearningSimulator:
    def __init__(self, env: DiscreteEnv, agent: QLearningAgent) -> None:
        self.env = env
        self.agent = agent
        self.stats: List[EpisodeStats] = []

    def train(
        self,
        episodes: int,
        render: bool = False,
        log_every: int = 50,
        seed: Optional[int] = None,
        stop_on_goal: bool = False,
        stop_avg_reward: Optional[float] = None,
        ma_window: int = 100,
        train_render_sleep: float = 0.0,
    ) -> List[EpisodeStats]:
        if seed is not None:
            set_global_seed(seed)
        self.stats.clear()

        rewards_hist: List[float] = []

        for ep in range(int(episodes)):
            eps = float(self.agent.get_epsilon(ep))
            s = int(self.env.reset(seed=None))
            total_reward = 0.0
            td_abs: List[float] = []
            success = False

            for t in range(self.agent.cfg.max_steps_per_episode):
                if render:
                    self.env.render()
                    if train_render_sleep > 0:
                        time.sleep(train_render_sleep)

                a = self.agent.act(s, eps)
                sp, r, done, info = self.env.step(a)

                td = self.agent.update(s, a, r, sp, done)
                td_abs.append(abs(td))

                total_reward += r
                s = sp

                if done:
                    # In many sparse-reward tasks (e.g., FrozenLake), success corresponds to reward > 0
                    term = str(info.get("terminal", ""))
                    success = (r > 0) or (term == "goal")
                    steps = t + 1
                    break
            else:
                steps = self.agent.cfg.max_steps_per_episode

            rewards_hist.append(float(total_reward))

            self.stats.append(EpisodeStats(
                episode=ep,
                steps=int(steps),
                total_reward=float(total_reward),
                epsilon=float(eps),
                td_error_mean_abs=float(np.mean(td_abs) if td_abs else 0.0),
                success=bool(success),
            ))

            if log_every > 0 and (ep + 1) % log_every == 0:
                recent = self.stats[-log_every:]
                avg_r = float(np.mean([x.total_reward for x in recent]))
                avg_len = float(np.mean([x.steps for x in recent]))
                succ_rate = float(np.mean([1.0 if x.success else 0.0 for x in recent]))
                print(f"[episode {ep+1:>6}] avg_reward={avg_r:+.3f} avg_len={avg_len:.1f} "
                      f"success_rate={succ_rate:.2%} eps={eps:.3f}", flush=True)

            # Early stop options (teaching-friendly)
            if stop_on_goal and success:
                print(f"Early stop: first success at episode {ep+1} (reward={total_reward:+.3f}).", flush=True)
                break

            if stop_avg_reward is not None:
                ma = moving_average(rewards_hist, ma_window)
                if ma >= float(stop_avg_reward):
                    print(f"Early stop: moving-average reward {ma:.3f} >= {float(stop_avg_reward):.3f} "
                          f"(window={ma_window}) at episode {ep+1}.", flush=True)
                    break

        return self.stats

    def play(
        self,
        episodes: int = 5,
        sleep: float = 0.15,
        greedy: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """Visual demonstration rollouts (usually greedy)."""
        return self.evaluate(episodes=episodes, greedy=greedy, render=True, seed=seed, step_sleep=sleep)

    def evaluate(
        self,
        episodes: int = 100,
        greedy: bool = True,
        render: bool = False,
        seed: Optional[int] = None,
        step_sleep: float = 0.0,
    ) -> Dict[str, float]:
        if seed is not None:
            set_global_seed(seed)

        rewards: List[float] = []
        lengths: List[int] = []
        successes: List[float] = []

        for _ in range(int(episodes)):
            s = int(self.env.reset(seed=None))
            total = 0.0
            success = False

            for t in range(self.agent.cfg.max_steps_per_episode):
                if render:
                    self.env.render()
                    if step_sleep > 0:
                        time.sleep(step_sleep)

                if greedy:
                    q = self.agent.Q[s]
                    m = float(np.max(q))
                    candidates = np.flatnonzero(q == m)
                    a = int(random.choice(candidates))
                else:
                    a = self.agent.act(s, self.agent.cfg.epsilon)

                s, r, done, info = self.env.step(a)
                total += r
                if done:
                    term = str(info.get("terminal", ""))
                    success = (r > 0) or (term == "goal")
                    lengths.append(t + 1)
                    break
            else:
                lengths.append(self.agent.cfg.max_steps_per_episode)

            rewards.append(float(total))
            successes.append(1.0 if success else 0.0)

        return {
            "episodes": float(episodes),
            "avg_reward": float(np.mean(rewards) if rewards else 0.0),
            "std_reward": float(np.std(rewards) if rewards else 0.0),
            "avg_length": float(np.mean(lengths) if lengths else 0.0),
            "success_rate": float(np.mean(successes) if successes else 0.0),
        }


# ----------------------------
# CLI
# ----------------------------

def build_env(args: argparse.Namespace) -> DiscreteEnv:
    if args.env == "grid":
        pits = [(1, 1), (2, 3)] if args.grid_pits else []
        walls = [(1, 2)] if args.grid_walls else []
        return GridWorld(
            n_rows=args.grid_rows,
            n_cols=args.grid_cols,
            start=(0, 0),
            goal=(args.grid_rows - 1, args.grid_cols - 1),
            pits=pits,
            walls=walls,
            step_reward=args.grid_step_reward,
            goal_reward=args.grid_goal_reward,
            pit_reward=args.grid_pit_reward,
            max_steps=args.grid_max_steps,
        )

    if args.env == "gym":
        try:
            import gymnasium as gym  # type: ignore
        except Exception as e:
            raise RuntimeError("Gymnasium not installed. Run: pip install gymnasium") from e

        render_mode = None if args.render == "none" else args.render
        gym_env = gym.make(args.gym_id, render_mode=render_mode)
        return GymnasiumWrapper(gym_env, render=args.render)

    raise ValueError("Invalid env.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tabular Q-Learning Simulator (Teaching Edition).")

    p.add_argument("--env", choices=["grid", "gym"], default="grid")
    p.add_argument("--gym-id", default="FrozenLake-v1")
    p.add_argument("--render", choices=["none", "human"], default="none")

    # training
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.2)

    p.add_argument("--epsilon-decay", action="store_true")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=float, default=0.999)

    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)

    # teaching UX knobs
    p.add_argument("--stop-on-goal", action="store_true", help="Stop training immediately on the first successful episode.")
    p.add_argument("--stop-avg-reward", type=float, default=None,
                   help="Stop when moving-average training reward >= this value.")
    p.add_argument("--ma-window", type=int, default=100, help="Window size for moving-average early stop.")
    p.add_argument("--train-render-sleep", type=float, default=0.0, help="Sleep (seconds) after each rendered train step.")
    p.add_argument("--play-after-training", type=int, default=0,
                   help="After training (or loading), visually play this many greedy episodes.")
    p.add_argument("--play-sleep", type=float, default=0.15, help="Sleep (seconds) after each rendered play step.")

    # evaluation / IO
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--save", type=str, default="")
    p.add_argument("--load", type=str, default="")
    p.add_argument("--export-stats", type=str, default="")

    # gridworld
    p.add_argument("--grid-rows", type=int, default=4)
    p.add_argument("--grid-cols", type=int, default=4)
    p.add_argument("--grid-step-reward", type=float, default=-0.01)
    p.add_argument("--grid-goal-reward", type=float, default=1.0)
    p.add_argument("--grid-pit-reward", type=float, default=-1.0)
    p.add_argument("--grid-max-steps", type=int, default=200)
    p.add_argument("--grid-pits", action="store_true")
    p.add_argument("--grid-walls", action="store_true")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_global_seed(args.seed)

    env = build_env(args)

    # agent
    if args.load:
        agent = QLearningAgent.load(args.load)
        if agent.n_states != env.n_states or agent.n_actions != env.n_actions:
            raise ValueError(f"Loaded Q shape {agent.Q.shape} incompatible with env (S={env.n_states}, A={env.n_actions}).")
        agent.cfg.gamma = args.gamma
        agent.cfg.alpha = args.alpha
        agent.cfg.max_steps_per_episode = args.max_steps
    else:
        eps_sched = None
        if args.epsilon_decay:
            eps_sched = EpsilonSchedule(start=args.eps_start, end=args.eps_end, decay=args.eps_decay)

        cfg = QLearningConfig(
            gamma=args.gamma,
            alpha=args.alpha,
            epsilon=args.epsilon,
            epsilon_schedule=eps_sched,
            max_steps_per_episode=args.max_steps,
            seed=args.seed,
        )
        agent = QLearningAgent(env.n_states, env.n_actions, cfg)

    sim = QLearningSimulator(env, agent)

    # TRAIN (rendering during training is optional and usually not recommended for long runs)
    if args.episodes > 0:
        sim.train(
            episodes=args.episodes,
            render=(args.render == "human" and args.train_render_sleep > 0),  # render only if you intentionally slow it
            log_every=args.log_every,
            seed=args.seed,
            stop_on_goal=args.stop_on_goal,
            stop_avg_reward=args.stop_avg_reward,
            ma_window=args.ma_window,
            train_render_sleep=args.train_render_sleep,
        )

    # Always do a quick greedy evaluation summary (no render)
    metrics = sim.evaluate(episodes=args.eval_episodes, greedy=True, render=False, seed=args.seed)
    print("\nEvaluation (greedy):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}, flush=True)

    # Optional: visual "play" episodes after training/loading (this is the best teaching demo)
    if args.play_after_training > 0:
        if args.render != "human":
            print("Note: --play-after-training requested but --render is not 'human'. Use --render human to see a window.", flush=True)
        else:
            print(f"\nPlaying {args.play_after_training} greedy episode(s)...", flush=True)
            play_metrics = sim.play(episodes=args.play_after_training, sleep=args.play_sleep, greedy=True, seed=args.seed)
            print("Play metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in play_metrics.items()}, flush=True)

    if args.export_stats:
        payload = {
            "config": vars(args),
            "train_stats": [dataclasses.asdict(s) for s in sim.stats],
            "eval": metrics,
        }
        os.makedirs(os.path.dirname(args.export_stats) or ".", exist_ok=True)
        with open(args.export_stats, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote stats to {args.export_stats}", flush=True)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        agent.save(args.save)
        print(f"Saved Q-table to {args.save}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
