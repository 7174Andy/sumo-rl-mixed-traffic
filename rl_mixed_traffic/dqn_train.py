from rl_mixed_traffic.env.ring_env import RingRoadEnv
from rl_mixed_traffic.env.discretizer import DiscretizeActionWrapper
from rl_mixed_traffic.config import SumoConfig
from rl_mixed_traffic.configs.dqn_config import DQNConfig
from rl_mixed_traffic.dqn.dqn_agent import DQNAgent

from rl_mixed_traffic.utils.plot_utils import plot_returns, plot_losses


def make_env(gui: bool=False, num_bins=21):
    sumo_config = SumoConfig(
        sumocfg_path="configs/ring/simulation.sumocfg",
        use_gui=gui,
    )
    base_env = RingRoadEnv(
        sumo_config=sumo_config,
        gui=gui,
        num_vehicles=4,
    )
    env = DiscretizeActionWrapper(base_env, num_bins)
    return env

def train(total_steps: int = 350_000, num_bins=21):
    env = make_env(gui=False, num_bins=num_bins)

    returns = []

    obs_dim = env.observation_space.shape[0]

    cfg = DQNConfig()
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=env.action_space.n,
        config=cfg,
    )

    s, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    losses = []

    for t in range(1, total_steps + 1):
        a  = agent.act(state=s)
        # print(f"Step: {t}, Action: {env.actions[a]}")
        s_next, r, done, _ = env.step(a)
        agent.buffer.append((s, a, r, s_next, done))
        loss = agent.learn()

        s = s_next
        ep_ret += r
        ep_len += 1

        if done:
            print(f"Step: {t}, Episode Return: {ep_ret}, Episode Length: {ep_len}, Epsilon: {agent.epsilon():.3f}")
            returns.append(ep_ret)
            if loss is not None:
                losses.append(float(loss))
            s, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

    agent.save("dqn_results/dqn_agent.pth")
    env.close()

    # Plot training losses
    plot_losses(
        losses,
        out_path="dqn_results/dqn_training_losses.png",
        title="DQN Training Losses"
    )
    return returns

if __name__ == "__main__":
    returns = train()
    plot_returns(
        returns,
        out_path="dqn_results/dqn_training_returns.png",
        title="DQN Training Returns"
    )
