from rl_mixed_traffic.dqn.dqn_agent import DQNAgent
from rl_mixed_traffic.env.ring_env import RingRoadEnv
from rl_mixed_traffic.env.discretizer import DiscretizeActionWrapper, StateDiscretizer, DiscretizerConfig
from rl_mixed_traffic.configs.sumo_config import SumoConfig
from rl_mixed_traffic.configs.dqn_config import DQNConfig
import time
import os
import shutil
import ffmpeg

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


def evaluate(agent_path: str, gui: bool = True, record: bool = False, num_bins: int = 21):
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

    # Create a directory to store frames if recording
    if record:
        frames_dir = "dqn_eval_frames"
        os.makedirs(frames_dir)

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

            # Take a frame if recording
            if record:
                env.sumo.gui.screenshot("View #0", os.path.join(frames_dir, f"frame_{steps:04d}.png"))

            s = s_next
            G += r
            steps += 1
        print(f"Evaluation Return: {G:.2f}, Steps: {steps}")

        # Compile frames into a video if recording
        if record:
            (
                ffmpeg
                .input(os.path.join(frames_dir, 'frame_%04d.png'), framerate=30)
                .output('dqn_evaluation.mp4', pix_fmt='yuv420p', vcodec='libx264')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            print("Saved evaluation video to dqn_evaluation.mp4")
    finally:
        env.close()
        if os.path.exists("dqn_eval_frames"):
            shutil.rmtree("dqn_eval_frames")

if __name__ == "__main__":
    evaluate(agent_path="dqn_results/dqn_agent.pth", gui=True, num_bins=21)