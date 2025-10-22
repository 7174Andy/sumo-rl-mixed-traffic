import os
import sys
from pathlib import Path

# If SUMO is installed, SUMO_HOME should point to its root (where tools/ lives).
if "SUMO_HOME" in os.environ:
    tools = Path(os.environ["SUMO_HOME"]) / "share" / "sumo" / "tools"
    sys.path.append(str(tools))
else:
    raise EnvironmentError(
        "Please set the SUMO_HOME environment variable to your SUMO install path."
    )

import traci

from utils.sumo_utils import pretty_time, start_traci
from config import SumoConfig

CFG = "configs/simulation.sumocfg"

def main():
    start_traci(SumoConfig(sumocfg_path=CFG, use_gui=True, delay_ms=100))

    # Target free-flow speeds for each car (m/s). SUMO car-following will override if unsafe.
    desired_speeds = {"car0": 12.0, "car1": 10.0, "car2": 14.0}

    # Keep track of first time we see each vehicle
    seen = set()

    sim_time = 0.0
    MAX_STEPS = int(120 / SumoConfig.step_length)

    for k in range(MAX_STEPS):
        # New departures this step
        for vid in traci.simulation.getDepartedIDList():
            seen.add(vid)
            # Color by id to visualize easily if you run with sumo-gui
            if vid == "car0":
                traci.vehicle.setColor(vid, (255, 0, 0, 255))
            elif vid == "car1":
                traci.vehicle.setColor(vid, (0, 150, 255, 255))
            else:
                traci.vehicle.setColor(vid, (0, 200, 0, 255))

        # Simple controller:
        # - try to hold a desired speed
        # - back off if the leader is too close (time headway control)
        for vid in list(seen):
            if vid not in traci.vehicle.getIDList():
                # has arrived/left network
                continue

            # Try to apply desired speed
            v_des = desired_speeds.get(vid, 12.0)

            # Keep a safe time headway to leader
            # getLeader returns (leaderID, gapMeters) or None
            leader_info = traci.vehicle.getLeader(vid)
            if leader_info is not None:
                leader_id, gap = leader_info
                # Basic time-headway control:
                # v_cmd = min(v_des, gap / tau)
                tau = 1.2  # seconds
                v_cmd = min(v_des, max(0.0, gap / tau))
            else:
                v_cmd = v_des

            # Respect the vehicle's max speed (from vType)
            v_max = traci.vehicle.getMaxSpeed(vid)
            v_cmd = min(v_cmd, v_max)

            traci.vehicle.setSpeed(vid, v_cmd) # setSpeed sets the desired speed for the next step

        # Log status once per second
        if k % int(1.0 / SumoConfig.step_length) == 0:
            ids = traci.vehicle.getIDList()
            prints = []
            for vid in sorted(ids):
                pos = traci.vehicle.getPosition(vid)
                lane = traci.vehicle.getLaneID(vid)
                speed = traci.vehicle.getSpeed(vid)
                prints.append(f"{vid}: lane={lane} x={pos[0]:7.2f} v={speed:5.2f} m/s")
            print(pretty_time(sim_time), " | ".join(prints))

        traci.simulationStep()
        sim_time += SumoConfig.step_length

        # Stop early if everyone arrived
        if traci.simulation.getMinExpectedNumber() == 0:
            break

    traci.close()
    print("Simulation finished.")

# TODO: Simple RL algorithm -> for pipeline
# - learn policy offline in a simplified environment
# - simple algorithm: Q-learning with discretized state/action spaces
# - env: ring road with head vehicle with 1 CAV and 1-2 HDVs