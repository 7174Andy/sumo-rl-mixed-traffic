import pickle
import numpy as np
from config import SumoConfig
from env import RingRoadEnv

CFG = "configs/simulation.sumocfg"
AGENT_ID = "car0"

def greedy_action(Q, s, n_actions):
    q = Q.get(s, np.zeros(n_actions, dtype=np.float32))
    return int(np.argmax(q))

def eval_policy(q_table_path: str, gui: bool = True):
    env = RingRoadEnv(
        sumo_config=SumoConfig(sumocfg_path=CFG, use_gui=gui),
        agent_id=AGENT_ID,
        gui=gui,
        episode_length=120.0,
        dv=0.5,
        action_k=2,
    )

    with open(q_table_path, "rb") as f:
        Q = pickle.load(f)

    try:
        s = env.reset()
        done = False
        G = 0.0
        steps = 0

        while not done:
            a = greedy_action(Q, s, len(env.actions))
            s_next, r, done, _ = env.step(a)
            s = s_next
            G += r
            steps += 1

        print(f"Evaluation Return: {G:.2f}, Steps: {steps}")
    finally:
        env.close()

if __name__ == "__main__":
    eval_policy("trained_agents/q_table.pkl")