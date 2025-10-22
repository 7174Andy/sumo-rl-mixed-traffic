from config import SumoConfig
import os
import sys
from pathlib import Path

if "SUMO_HOME" in os.environ:
    tools = Path(os.environ["SUMO_HOME"]) / "tools"
    sys.path.append(str(tools))
else:
    raise EnvironmentError(
        "Please set the SUMO_HOME environment variable to your SUMO install path."
    )

import traci
from sumolib import checkBinary  # finds sumo / sumo-gui binary

SUMO_BINARY = checkBinary("sumo-gui") 


def start_traci(sim: SumoConfig):
    traci.start([SUMO_BINARY, "-c", sim.sumocfg_path, "--no-step-log", "true", "--start", "--quit-on-end", "true", "--delay", str(sim.delay_ms)])
    print("Connected to SUMO via TraCI.")

def pretty_time(t: float) -> str:
    return f"t={t:5.1f}s"


def save_returns_csv(returns, out_path: str):
    import csv
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'return'])
        for i, G in enumerate(returns):
            writer.writerow([i + 1, G])
    print(f"Saved returns to {out_path}.")