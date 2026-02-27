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

    @property
    def update_every_step(self) -> bool:
        return False

    def reset(self):
        pass

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


class EmergencyBrakingController(HeadVehicleController):
    """Four-phase emergency braking profile for the head vehicle.

    Phase 1 (cruise):   t < t_brake             → v_cruise
    Phase 2 (brake):    t_brake ≤ t < t_brake + brake_duration → linear decel to v_low
    Phase 3 (hold):     hold v_low for hold_duration seconds
    Phase 4 (recover):  linear accel back to v_cruise over recover_duration seconds
    After recovery:     cruise at v_cruise indefinitely
    """

    def __init__(
        self,
        head_id: str = "car0",
        step_length: float = 0.1,
        v_cruise: float = 15.0,
        v_low: float = 2.0,
        t_brake: float = 50.0,
        brake_duration: float = 3.0,
        hold_duration: float = 10.0,
        recover_duration: float = 5.0,
    ):
        super().__init__(
            head_id=head_id,
            head_speed_min=0,
            head_speed_max=int(v_cruise + 1),
        )
        self.step_length = step_length
        self.v_cruise = v_cruise
        self.v_low = v_low
        self.t_brake = t_brake
        self.brake_duration = brake_duration
        self.hold_duration = hold_duration
        self.recover_duration = recover_duration
        self._sim_time = 0.0

    @property
    def update_every_step(self) -> bool:
        return True

    def reset(self):
        self._sim_time = 0.0

    def get_target_speed(self, t: float) -> float:
        """Return the target speed at simulation time *t*."""
        t1 = self.t_brake
        t2 = t1 + self.brake_duration
        t3 = t2 + self.hold_duration
        t4 = t3 + self.recover_duration

        if t < t1:
            return self.v_cruise
        elif t < t2:
            frac = (t - t1) / self.brake_duration
            return self.v_cruise + frac * (self.v_low - self.v_cruise)
        elif t < t3:
            return self.v_low
        elif t < t4:
            frac = (t - t3) / self.recover_duration
            return self.v_low + frac * (self.v_cruise - self.v_low)
        else:
            return self.v_cruise

    def set_random_head_speed(self):
        """Override: apply the deterministic braking profile instead of random."""
        if self.head_id not in traci.vehicle.getIDList():
            self._sim_time += self.step_length
            return

        speed = self.get_target_speed(self._sim_time)
        traci.vehicle.setSpeed(self.head_id, speed)
        self._sim_time += self.step_length