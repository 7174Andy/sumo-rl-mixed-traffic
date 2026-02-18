import pickle
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
import hydra

from rl_mixed_traffic.env.ring_env import RingRoadEnv
from rl_mixed_traffic.env.discretizer import (
    DiscretizeActionWrapper,
    DiscretizerConfig,
    StateDiscretizer,
)
from rl_mixed_traffic.agents.q_agent import QLearningAgent
from rl_mixed_traffic.configs.sumo_config import SumoConfig

from rl_mixed_traffic.utils.plot_utils import plot_returns
from rl_mixed_traffic.utils.sumo_utils import save_returns_csv


def snapshot_q(q_defaultdict):
    """Convert defaultdict to a plain dict with numpy arrays (picklable)."""
    return {k: np.array(v, dtype=np.float32) for k, v in q_defaultdict.items()}


def train(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    sumocfg_path = str(Path(orig_cwd) / cfg.env.sumocfg_path)

    base_env = RingRoadEnv(
        sumo_config=SumoConfig(sumocfg_path=sumocfg_path, use_gui=cfg.gui),
        gui=cfg.gui,
        num_vehicles=cfg.env.num_vehicles,
    )

    env = DiscretizeActionWrapper(base_env, n_bins=cfg.env.n_bins)

    obs_dim = env.observation_space.shape[0]
    state_discretizer = StateDiscretizer(
        obs_dim, DiscretizerConfig(bins_per_dim=cfg.discretizer.bins_per_dim)
    )

    agent = QLearningAgent(
        action_space=env.action_space.n,
        alpha=cfg.agent.alpha,
        gamma=cfg.agent.gamma,
        eps_start=cfg.agent.eps_start,
        eps_end=cfg.agent.eps_end,
        eps_decay_steps=cfg.num_episodes * env.max_steps,
    )

    returns = []
    best_return = -float("inf")
    best_Q = None

    try:
        for episode in range(cfg.num_episodes):
            s, _ = env.reset()
            s = state_discretizer(s)
            done = False
            G = 0.0
            steps = 0

            while not done:
                a = agent.act(s)
                s_next, r, done, _ = env.step(a)
                s_next = state_discretizer(s_next)
                agent.update(s, a, r, s_next, done)
                s = s_next
                G += r
                steps += 1

            returns.append(G)
            print(
                f"Episode {episode + 1}/{cfg.num_episodes}, Return: {G:.2f}, Steps: {steps}, Epsilon: {agent.epsilon():.4f}"
            )

            if G > best_return:
                best_return = G
                best_Q = agent.q_table.copy()
    finally:
        env.close()

    out_dir = Path(orig_cwd) / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    q_table_path = out_dir / "q_table.pkl"
    with open(q_table_path, "wb") as f:
        pickle.dump(snapshot_q(best_Q), f)
    print(
        f"Training completed. Best return: {best_return:.2f}. Q-table saved to {q_table_path}."
    )

    return returns, out_dir


@hydra.main(version_base=None, config_path="conf", config_name="q_train")
def main(cfg: DictConfig):
    returns, out_dir = train(cfg)
    save_returns_csv(returns, out_path=str(out_dir / "returns.csv"))
    plot_returns(
        returns,
        out_path=str(out_dir / "returns.png"),
        smooth_window=10,
        title="Episode Returns",
    )


if __name__ == "__main__":
    main()
