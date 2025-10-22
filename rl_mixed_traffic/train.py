import pickle
import numpy as np
from pathlib import Path
from env import RingRoadEnv
from agent import QLearningAgent
from config import SumoConfig

from utils.plot_utils import plot_returns
from utils.sumo_utils import save_returns_csv

AGENT_ID = "car0"
CFG = "configs/ring/simulation.sumocfg"

def snapshot_q(q_defaultdict):
    """Convert defaultdict to a plain dict with numpy arrays (picklable)."""
    return {k: np.array(v, dtype=np.float32) for k, v in q_defaultdict.items()}

def train(num_episodes: int = 150, gui: bool = False, out_path: str = "q_table.pkl"):
    env = RingRoadEnv(
        sumo_config=SumoConfig(sumocfg_path=CFG, use_gui=gui),
        agent_id=AGENT_ID,
        gui=gui,
        dv=0.5,
    )

    agent = QLearningAgent(
        action_space=len(env.actions),
        alpha=0.2,
        gamma=0.98,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=num_episodes*env.max_steps,
    )

    returns = []
    best_return = -float('inf')
    best_Q = None

    try:
        for episode in range(num_episodes):
            s = env.reset()
            done = False
            G = 0.0
            steps = 0

            while not done:
                a = agent.act(s)
                s_next, r, done, _ = env.step(a)
                agent.update(s, a, r, s_next, done)
                s = s_next
                G += r
                steps += 1

            returns.append(G)
            print(f"Episode {episode+1}/{num_episodes}, Return: {G:.2f}, Steps: {steps}, Epsilon: {agent.epsilon():.4f}")

            if G > best_return:
                best_return = G
                best_Q = agent.q_table.copy()
    finally:
        env.close()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(snapshot_q(best_Q), f)
    print(f"Training completed. Best return: {best_return:.2f}. Q-table saved to {out_path}.")

    return returns

if __name__ == "__main__":
    returns = train(num_episodes=250, gui=True, out_path="output/q_table.pkl")
    save_returns_csv(returns, out_path="output/returns.csv")
    plot_returns(returns, out_path="output/returns.png", smooth_window=10, title="Episode Returns")
