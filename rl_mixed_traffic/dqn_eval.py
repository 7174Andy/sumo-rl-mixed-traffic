from rl_mixed_traffic.dqn.dqn_agent import DQNAgent
from rl_mixed_traffic.env.ring_env import RingRoadEnv
from rl_mixed_traffic.env.discretizer import DiscretizeActionWrapper, StateDiscretizer, DiscretizerConfig
from rl_mixed_traffic.configs.sumo_config import SumoConfig
from rl_mixed_traffic.configs.dqn_config import DQNConfig
import time

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

def evaluate(agent_path: str, gui: bool = True, num_bins: int = 21):
    env = make_env(gui=gui, num_bins=num_bins)
    state_discretizer = StateDiscretizer(env.observation_space.shape[0], DiscretizerConfig(bins_per_dim=num_bins))

    obs_dim = env.observation_space.shape[0]

    cfg = DQNConfig()
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=env.action_space.n,
        config=cfg,
    )
    agent.load(agent_path)

    s, _ = env.reset()
    s = state_discretizer(s)
    done = False
    G = 0.0
    steps = 0

    try:
        while not done:
            a = agent.act(state=s, eval_mode=True)
            # print(f"Step: {steps}, State: {s}, Action: {a}")
            s_next, r, done, _ = env.step(a)
            s = s_next
            G += r
            steps += 1
        print(f"Evaluation Return: {G:.2f}, Steps: {steps}")
    finally:
        env.close()

if __name__ == "__main__":
    evaluate(agent_path="dqn_results/dqn_agent.pth", gui=True, num_bins=21)