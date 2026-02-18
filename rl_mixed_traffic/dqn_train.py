from pathlib import Path
from omegaconf import DictConfig
import hydra
from tqdm import trange

from rl_mixed_traffic.env.ring_env import RingRoadEnv
from rl_mixed_traffic.env.discretizer import DiscretizeActionWrapper
from rl_mixed_traffic.configs.sumo_config import SumoConfig
from rl_mixed_traffic.configs.dqn_config import DQNConfig
from rl_mixed_traffic.agents.dqn_agent import DQNAgent

from rl_mixed_traffic.utils.plot_utils import plot_returns, plot_losses


def make_env(sumocfg_path: str, gui: bool, num_vehicles: int, num_bins: int):
    sumo_config = SumoConfig(
        sumocfg_path=sumocfg_path,
        use_gui=gui,
    )
    base_env = RingRoadEnv(
        sumo_config=sumo_config,
        gui=gui,
        num_vehicles=num_vehicles,
    )
    env = DiscretizeActionWrapper(base_env, num_bins)
    return env


def train(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    sumocfg_path = str(Path(orig_cwd) / cfg.env.sumocfg_path)

    env = make_env(
        sumocfg_path=sumocfg_path,
        gui=cfg.gui,
        num_vehicles=cfg.env.num_vehicles,
        num_bins=cfg.env.n_bins,
    )

    returns = []

    obs_dim = env.observation_space.shape[0]

    dqn_cfg = DQNConfig(
        gamma=cfg.agent.gamma,
        lr=cfg.agent.lr,
        batch_size=cfg.agent.batch_size,
        buffer_size=cfg.agent.buffer_size,
        start_learning_after=cfg.agent.start_learning_after,
        train_freq=cfg.agent.train_freq,
        target_update_freq=cfg.agent.target_update_freq,
        tau=cfg.agent.tau,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_end=cfg.agent.epsilon_end,
        epsilon_decay_steps=cfg.agent.epsilon_decay_steps,
        max_grad_norm=cfg.agent.max_grad_norm,
    )
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=env.action_space.n,
        config=dqn_cfg,
    )

    s, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    losses = []

    for t in trange(1, cfg.total_steps + 1):
        a = agent.act(state=s)
        s_next, r, done, _ = env.step(a)
        loss = agent.update(state=s, action=a, reward=r, next_state=s_next, done=done)

        s = s_next
        ep_ret += r
        ep_len += 1

        if done:
            print(
                f"Step: {t}, Episode Return: {ep_ret}, Episode Length: {ep_len}, Epsilon: {agent.epsilon():.3f}"
            )
            returns.append(ep_ret)
            if loss is not None:
                losses.append(float(loss))
            s, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

    out_dir = Path(orig_cwd) / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    agent.save(str(out_dir / "dqn_agent.pth"))
    env.close()

    plot_losses(
        losses,
        out_path=str(out_dir / "dqn_training_losses.png"),
        title="DQN Training Losses",
    )
    return returns, out_dir


@hydra.main(version_base=None, config_path="conf", config_name="dqn_train")
def main(cfg: DictConfig):
    returns, out_dir = train(cfg)
    plot_returns(
        returns,
        out_path=str(out_dir / "dqn_training_returns.png"),
        title="DQN Training Returns",
    )


if __name__ == "__main__":
    main()
