from dataclasses import dataclass

@dataclass(frozen=True)
class SumoConfig:
    sumocfg_path: str = "configs/simulation.sumocfg"
    use_gui: bool = False
    delay_ms: int = 0