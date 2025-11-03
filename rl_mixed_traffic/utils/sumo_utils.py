import os
import sys
from pathlib import Path
from config import SumoConfig
import numpy as np
from typing import Tuple

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
    traci.start(
        [
            SUMO_BINARY,
            "-c",
            sim.sumocfg_path,
            "--no-step-log",
            "true",
            "--start",
            "--quit-on-end",
            "true",
            "--delay",
            str(sim.delay_ms),
        ]
    )
    print("Connected to SUMO via TraCI.")


def pretty_time(t: float) -> str:
    return f"t={t:5.1f}s"


def save_returns_csv(returns, out_path: str):
    import csv

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return"])
        for i, G in enumerate(returns):
            writer.writerow([i + 1, G])
    print(f"Saved returns to {out_path}.")


def compute_ring_length(agent_id: str) -> float:
    """Estimate ring length by summing the lengths of edges on the agent's route.
    Falls back to a rough perimeter using all vehicles' current edges if needed.
    """

    def edge_length(edge_id: str) -> float:
        try:
            # use lane 0 length (lanes have same geometry length)
            lane_id = f"{edge_id}_0"
            return float(traci.lane.getLength(lane_id))
        except traci.TraCIException:
            # last resort: try average of all lanes on this edge
            try:
                n = traci.edge.getLaneNumber(edge_id)
                lens = []
                for i in range(n):
                    lens.append(float(traci.lane.getLength(f"{edge_id}_{i}")))
                return float(np.mean(lens)) if lens else 0.0
            except traci.TraCIException:
                return 0.0

    # Try: agent route first
    if agent_id in traci.vehicle.getIDList():
        try:
            route = traci.vehicle.getRoute(agent_id)  # list of edge ids
            if route:
                L = sum(edge_length(e) for e in route)
                if L > 0.0:
                    return L
        except traci.TraCIException:
            pass

    # Fallback: union of all edges vehicles are currently on
    try:
        edges = set()
        for vid in traci.vehicle.getIDList():
            try:
                edges.add(traci.vehicle.getRoadID(vid))
            except traci.TraCIException:
                continue
        L = sum(edge_length(e) for e in edges if e and not e.startswith(":"))
        if L > 0.0:
            return L
    except traci.TraCIException:
        pass

    # Final fallback: safe default (prevents div0; you can adjust)
    return 1e3


def get_vehicles_pos_speed(ring_length: float) -> Tuple[list, list, list]:
    """Return lists (ids_sorted, speeds_mps_sorted, positions_m_sorted_mod) sorted by position along ring.

    Position is based on distance traveled (mod ring length) for ring-road stability.
    """
    ids = [vid for vid in traci.vehicle.getIDList()]
    if not ids:
        return [], [], []

    L = float(max(1e-6, ring_length or 0.0))

    positions = []
    speeds = []

    for vid in ids:
        try:
            s_abs = traci.vehicle.getDistance(vid)  # meters traveled since depart
            v = traci.vehicle.getSpeed(vid)  # m/s
        except traci.TraCIException:
            continue

        s_wrapped = float(s_abs % L)  # wrap around ring
        positions.append(s_wrapped)
        speeds.append(float(v))

    order = np.argsort(positions)
    ids_sorted = [ids[i] for i in order]
    speeds_sorted = [speeds[i] for i in order]
    positions_sorted = [positions[i] for i in order]
    return ids_sorted, speeds_sorted, positions_sorted

