from rl_mixed_traffic.utils.sumo_utils import run_simulation
from rl_mixed_traffic.utils.sumo_utils import SumoConfig

if __name__ == "__main__":
    # Example usage: run a SUMO simulation for 1000 steps
    sim_config = SumoConfig(
        sumocfg_path="configs/ring/simulation.sumocfg",
        use_gui=True,
    )
    run_simulation(sim_config, num_steps=4000)