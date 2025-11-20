import pickle
import numpy as np
from rl_mixed_traffic.configs.sumo_config import SumoConfig
from rl_mixed_traffic.env.discretizer import DiscretizeActionWrapper, DiscretizerConfig, StateDiscretizer
from rl_mixed_traffic.env.ring_env import RingRoadEnv

CFG = "./configs/ring/simulation.sumocfg"


def greedy_action(q: dict, s, n_actions):
    q = q.get(s, np.zeros(n_actions, dtype=np.float32))
    return int(np.argmax(q))


def eval_policy(q_table_path: str, gui: bool = True):
    base_env = RingRoadEnv(
        sumo_config=SumoConfig(sumocfg_path=CFG, use_gui=gui),
        gui=gui,
        episode_length=500.0,
        num_vehicles=4,
    )

    env = DiscretizeActionWrapper(base_env, n_bins=20)

    obs_dim = env.observation_space.shape[0]
    state_discretizer = StateDiscretizer(obs_dim, DiscretizerConfig(bins_per_dim=20))

    with open(q_table_path, "rb") as f:
        Q = pickle.load(f)

    try:
        s, _ = env.reset()
        done = False
        G = 0.0
        steps = 0

        while not done:
            a = greedy_action(Q, s, env.action_space.n)
            s_next, r, done, _ = env.step(a)
            s_next = state_discretizer(s_next)
            s = s_next
            G += r
            steps += 1
        print(f"Evaluation Return: {G:.2f}, Steps: {steps}")
    finally:
        env.close()


if __name__ == "__main__":
    eval_policy("./output/q_table.pkl")
