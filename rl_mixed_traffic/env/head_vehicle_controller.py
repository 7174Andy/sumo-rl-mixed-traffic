import os
import sys
from pathlib import Path

if "SUMO_HOME" in os.environ:
    tools = Path(os.environ["SUMO_HOME"]) / "share" / "sumo" / "tools"
    sys.path.append(str(tools))
else:
    raise EnvironmentError(
        "Please set the SUMO_HOME environment variable to your SUMO install path."
    )

import traci

class HeadVehicleController:
    def __init__(self, head_id: str, head_speed_min: int, head_speed_max: int):
        self.head_id = head_id
        self.head_speed_min = head_speed_min
        self.head_speed_max = head_speed_max

    def set_head_speed(self, speed: float):
        """Sets the speed of the head vehicle within defined limits."""
        clamped_speed = max(self.head_speed_min, min(speed, self.head_speed_max))
        traci.vehicle.setSpeed(self.head_id, clamped_speed)
    
    def set_random_head_speed(self):
        """Sets a random speed for the head vehicle within defined limits."""
        if self.head_id not in traci.vehicle.getIDList():
            return
        
        import random
        
        random_speed = random.uniform(self.head_speed_min, self.head_speed_max)
        traci.vehicle.setSpeed(self.head_id, random_speed)