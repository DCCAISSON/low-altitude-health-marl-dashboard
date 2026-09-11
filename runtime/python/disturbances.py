from __future__ import annotations

import random

from .config import DisturbanceConfig
from .state import DisturbanceState


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class DisturbanceProcess:
    def __init__(self, config: DisturbanceConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng

    def reset(self) -> DisturbanceState:
        return DisturbanceState(
            weather=0.0,
            medical_surge=0.0,
            demand_shock=0.0,
            airspace_control=0.0,
            device_failure=0.0,
        )

    def step(self, previous: DisturbanceState) -> DisturbanceState:
        weather = _clip(
            previous.weather * self.config.weather_persistence + self.rng.gauss(0.0, self.config.weather_std),
            -0.95,
            0.95,
        )
        medical_surge = _clip(
            previous.medical_surge * self.config.medical_surge_persistence
            + self.rng.gauss(0.0, self.config.medical_surge_std),
            -0.95,
            0.95,
        )
        demand_shock = _clip(
            previous.demand_shock * self.config.demand_persistence + self.rng.gauss(0.0, self.config.demand_std),
            -0.95,
            0.95,
        )
        airspace_control = _clip(
            previous.airspace_control * self.config.airspace_persistence + self.rng.gauss(0.0, self.config.airspace_std),
            -0.95,
            0.95,
        )
        device_failure = _clip(
            previous.device_failure * self.config.device_persistence + self.rng.gauss(0.0, self.config.device_std),
            -0.95,
            0.95,
        )
        return DisturbanceState(
            weather=weather,
            medical_surge=medical_surge,
            demand_shock=demand_shock,
            airspace_control=airspace_control,
            device_failure=device_failure,
        )
