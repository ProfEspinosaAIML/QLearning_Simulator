"""
q_learning_simulator.py

A single-file, object-oriented Q-Learning simulator aligned with Sutton & Barto (2e) pseudocode:
Q(S,A) <- Q(S,A) + alpha * [ R + gamma * max_a' Q(S',a') - Q(S,A) ]
Behavior: epsilon-greedy (or other) derived from Q; Target: greedy via max.

Designed to run either:
  1) With a Gymnasium environment (recommended), e.g. FrozenLake-v1
  2) With the included simple GridWorld environment (no external deps)

USAGE (Gymnasium):
  pip install gymnasium pygame
  python q_learning_simulator.py --env gym --gym-id FrozenLake-v1 --episodes 5000 --render none

USAGE (Built-in GridWorld):
  python q_learning_simulator.py --env grid --episodes 2000
  python q_learning_simulator.py --env grid --episodes 2000 --epsilon-decay --render human

For classroom robustness, I recommend installing:
    pip install gymnasium[toy-text,classic-control]
    pip install pygame

For a more comprehensive implementation (better teaching User Interface):
    Train without rendering, then evaluate with --render human to watch the learned policy.
    For example:
    python q_learning_simulator.py --env gym --gym-id FrozenLake-v1 --episodes 5000 --epsilon-decay

Train silently but evaluate with rendering:
    python q_learning_simulator.py --env gym --gym-id FrozenLake-v1 --episodes 5000 --epsilon-decay --eval-episodes 5 --render human


NOTES:
- For discrete state/action spaces (tabular Q-learning).
- Supports epsilon decay, training curves export, and model save/load (npz).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------
# Utilities
# ----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
        """Return initial discrete state id."""
        raise NotImplementedError

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Return (next_state, reward, done, info)."""
        raise NotImplementedError

    def render(self) -> None:
        """Optional text render."""
        pass


class GymnasiumWrapper(DiscreteEnv):
    """
    Wrap a Gymnasium environment with Discrete observation and action spaces.
    Supports envs returning either:
      - (obs, reward, done, info)
      - (obs, reward, terminated, truncated, info)
    """
    def __init__(self, gym_env: Any, render: str = "none") -> None:
        self.env = gym_env
        self._render = render
        try:
            self._n_states = int(self.env.observation_space.n)
            self._n_actions = int(self.env.action_space.n)
        except Exception as e:
            raise ValueError("GymnasiumWrapper requires discrete observation_space.n and action_space.n") from e

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
        return int(obs), float(reward), bool(done), dict(info) if info is not None else {}

    def render(self) -> None:
        if self._render != "none":
            try:
                self.env.render()
            except Exception:
                pass


# ----------------------------
# Built-in GridWorld (optional)
# ----------------------------

class GridWorld(DiscreteEnv):
    """
    Simple deterministic GridWorld with terminal goal and optional pits/walls.

    States are encoded as s = r * n_cols + c.
    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.
    """
    ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT")

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
        if action == 0:
            dr = -1
        elif action == 1:
            dc = 1
        elif action == 2:
            dr = 1
        elif action == 3:
            dc = -1
        else:
            raise ValueError("Invalid action (must be 0..3).")

        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols) or (nr, nc) in self.walls:
            nr, nc = r, c  # bounce

        self._state_rc = (nr, nc)

        done = False
        reward = self.step_reward
        info: Dict[str, Any] = {"position": self._state_rc, "steps": self._steps}

        if self._state_rc == self.goal_rc:
            done = True
            reward = self.goal_reward
            info["terminal"] = "goal"
        elif self._state_rc in self.pits:
            done = True
            reward = self.pit_reward
            info["terminal"] = "pit"
        elif self._steps >= self.max_steps:
            done = True
            info["terminal"] = "timeout"

        return self._encode(self._state_rc), float(reward), bool(done), info

    def render(self) -> None:
        grid_lines = []
        for rr in range(self.n_rows):
            row = []
            for cc in range(self.n_cols):
                cell = (rr, cc)
                if cell in self.walls:
                    row.append("█")
                elif cell == self.goal_rc:
                    row.append("G")
                elif cell in self.pits:
                    row.append("P")
                elif cell == self._state_rc:
                    row.append("A")
                elif cell == self.start_rc:
                    row.append("S")
                else:
                    row.append("·")
            grid_lines.append(" ".join(row))
        print("\n".join(grid_lines))
        print()


# ----------------------------
# Q-Learning agent
# ----------------------------

@dataclass
class EpsilonSchedule:
    start: float = 1.0
    end: float = 0.05
    decay: float = 0.999  # per episode

    def value(self, episode_idx: int) -> float:
        eps = self.end + (self.start - self.end) * (self.decay ** episode_idx)
        return float(max(self.end, min(self.start, eps)))


@dataclass
class QLearningConfig:
    gamma: float = 0.99
    alpha: float = 0.1
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
        max_q = float(np.max(q))
        candidates = np.flatnonzero(q == max_q)
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


class QLearningSimulator:
    def __init__(self, env: DiscreteEnv, agent: QLearningAgent) -> None:
        self.env = env
        self.agent = agent
        self.stats: List[EpisodeStats] = []

    def train(self, episodes: int, render: bool = False, log_every: int = 100, seed: Optional[int] = None) -> List[EpisodeStats]:
        if seed is not None:
            set_global_seed(seed)
        self.stats.clear()

        for ep in range(int(episodes)):
            eps = float(self.agent.get_epsilon(ep))
            s = int(self.env.reset(seed=None))
            total_reward = 0.0
            td_abs: List[float] = []

            for t in range(self.agent.cfg.max_steps_per_episode):
                if render:
                    self.env.render()

                a = self.agent.act(s, eps)
                sp, r, done, _ = self.env.step(a)

                td = self.agent.update(s, a, r, sp, done)
                td_abs.append(abs(td))

                total_reward += r
                s = sp
                if done:
                    steps = t + 1
                    break
            else:
                steps = self.agent.cfg.max_steps_per_episode

            self.stats.append(EpisodeStats(
                episode=ep,
                steps=int(steps),
                total_reward=float(total_reward),
                epsilon=float(eps),
                td_error_mean_abs=float(np.mean(td_abs) if td_abs else 0.0),
            ))

            if log_every > 0 and (ep + 1) % log_every == 0:
                recent = self.stats[-log_every:]
                avg_r = float(np.mean([x.total_reward for x in recent]))
                avg_len = float(np.mean([x.steps for x in recent]))
                print(f"[episode {ep+1:>6}] avg_reward={avg_r:+.3f} avg_len={avg_len:.1f} eps={eps:.3f}")

        return self.stats

    def evaluate(self, episodes: int = 100, greedy: bool = True, render: bool = False, seed: Optional[int] = None) -> Dict[str, float]:
        if seed is not None:
            set_global_seed(seed)

        rewards: List[float] = []
        lengths: List[int] = []

        for _ in range(int(episodes)):
            s = int(self.env.reset(seed=None))
            total = 0.0

            for t in range(self.agent.cfg.max_steps_per_episode):
                if render:
                    self.env.render()

                if greedy:
                    q = self.agent.Q[s]
                    max_q = float(np.max(q))
                    candidates = np.flatnonzero(q == max_q)
                    a = int(random.choice(candidates))
                else:
                    a = self.agent.act(s, self.agent.cfg.epsilon)

                s, r, done, _ = self.env.step(a)
                total += r
                if done:
                    lengths.append(t + 1)
                    break
            else:
                lengths.append(self.agent.cfg.max_steps_per_episode)

            rewards.append(total)

        return {
            "episodes": float(episodes),
            "avg_reward": float(np.mean(rewards) if rewards else 0.0),
            "std_reward": float(np.std(rewards) if rewards else 0.0),
            "avg_length": float(np.mean(lengths) if lengths else 0.0),
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
    p = argparse.ArgumentParser(description="Tabular Q-Learning Simulator (single-file, OOP).")

    p.add_argument("--env", choices=["grid", "gym"], default="grid")
    p.add_argument("--gym-id", default="FrozenLake-v1")
    p.add_argument("--render", choices=["none", "human"], default="none")

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
    p.add_argument("--log-every", type=int, default=100)

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

    if args.load:
        agent = QLearningAgent.load(args.load)
        if agent.n_states != env.n_states or agent.n_actions != env.n_actions:
            raise ValueError(f"Loaded Q shape {agent.Q.shape} incompatible with env (S={env.n_states}, A={env.n_actions}).")
        # Allow overriding some runtime knobs
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

    if args.episodes > 0:
        sim.train(
            episodes=args.episodes,
            render=(args.env == "grid" and args.render == "human"),
            log_every=args.log_every,
            seed=args.seed,
        )

#    metrics = sim.evaluate(episodes=args.eval_episodes, greedy=True, render=False, seed=args.seed)
    metrics = sim.evaluate(
    episodes=args.eval_episodes,
    greedy=True,
    render=(args.render == "human"),
    seed=args.seed
)

    print("\nEvaluation (greedy):", {k: round(v, 4) for k, v in metrics.items()})

    if args.export_stats:
        payload = {
            "config": {
                "env": args.env,
                "gym_id": args.gym_id,
                "gamma": args.gamma,
                "alpha": args.alpha,
                "epsilon": args.epsilon,
                "epsilon_decay": bool(args.epsilon_decay),
                "eps_start": args.eps_start,
                "eps_end": args.eps_end,
                "eps_decay": args.eps_decay,
                "max_steps": args.max_steps,
                "seed": args.seed,
                "episodes": args.episodes,
            },
            "train_stats": [dataclasses.asdict(s) for s in sim.stats],
            "eval": metrics,
        }
        os.makedirs(os.path.dirname(args.export_stats) or ".", exist_ok=True)
        with open(args.export_stats, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote stats to {args.export_stats}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        agent.save(args.save)
        print(f"Saved Q-table to {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
