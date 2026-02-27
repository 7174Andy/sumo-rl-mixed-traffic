"""Factory for creating head-vehicle controllers based on scenario config."""

from rl_mixed_traffic.env.head_vehicle_controller import EmergencyBrakingController


def make_head_controller(scenario_cfg, head_id: str = "car0", step_length: float = 0.1):
    """Return a head-vehicle controller matching *scenario_cfg.type*.

    Args:
        scenario_cfg: Config object with at least a ``type`` attribute.
            For ``emergency_braking``, also accepts v_cruise, v_low, t_brake,
            brake_duration, hold_duration, recover_duration.
        head_id: SUMO vehicle ID for the head vehicle.
        step_length: Simulation step length in seconds.

    Returns:
        A ``HeadVehicleController`` subclass instance, or ``None`` when the
        default random controller should be used (type == "random").
    """
    scenario_type = getattr(scenario_cfg, "type", "random")

    if scenario_type == "random":
        return None

    if scenario_type == "emergency_braking":
        return EmergencyBrakingController(
            head_id=head_id,
            step_length=step_length,
            v_cruise=getattr(scenario_cfg, "v_cruise", 15.0),
            v_low=getattr(scenario_cfg, "v_low", 2.0),
            t_brake=getattr(scenario_cfg, "t_brake", 50.0),
            brake_duration=getattr(scenario_cfg, "brake_duration", 3.0),
            hold_duration=getattr(scenario_cfg, "hold_duration", 10.0),
            recover_duration=getattr(scenario_cfg, "recover_duration", 5.0),
        )

    raise ValueError(f"Unknown scenario type: {scenario_type!r}")
