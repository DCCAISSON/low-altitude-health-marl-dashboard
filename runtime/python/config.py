from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(slots=True)
class SimulationConfig:
    seed: int
    periods: int
    population: int


@dataclass(slots=True)
class EnvironmentConfig:
    reimbursement_rate: float
    fixed_subsidy: float
    safety_floor: float
    initial_coverage: float
    initial_health: float
    initial_response_time: float
    base_traditional_time: float
    base_demand: float
    health_background_decay: float
    service_health_gain: float
    unmet_need_penalty: float
    trust_update_speed: float


@dataclass(slots=True)
class DisturbanceConfig:
    weather_persistence: float
    medical_surge_persistence: float
    demand_persistence: float
    weather_std: float
    medical_surge_std: float
    demand_std: float
    airspace_persistence: float = 0.4
    airspace_std: float = 0.1
    device_persistence: float = 0.35
    device_std: float = 0.08


@dataclass(slots=True)
class EnterpriseConfig:
    base_capacity: float
    fixed_cost: float
    variable_cost: float
    interruption_cost: float
    price_min: float
    price_max: float
    price_default: float


@dataclass(slots=True)
class MedicalConfig:
    base_capacity: float
    value_per_case: float
    idle_penalty: float
    overflow_penalty: float
    sinking_cost_weight: float


@dataclass(slots=True)
class FarmerConfig:
    price_sensitivity: float
    time_sensitivity: float
    trust_sensitivity: float
    health_need_sensitivity: float
    learning_cost: float
    time_value: float


@dataclass(slots=True)
class ModelConfig:
    simulation: SimulationConfig
    environment: EnvironmentConfig
    disturbances: DisturbanceConfig
    enterprise: EnterpriseConfig
    medical: MedicalConfig
    farmer: FarmerConfig


def _read_section(data: dict, key: str) -> dict:
    if key not in data:
        raise KeyError(f"Missing config section: {key}")
    return data[key]


def load_config(path: str | Path) -> ModelConfig:
    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8-sig")
    raw = tomllib.loads(raw_text)

    simulation = SimulationConfig(**_read_section(raw, "simulation"))
    environment = EnvironmentConfig(**_read_section(raw, "environment"))
    disturbances = DisturbanceConfig(**_read_section(raw, "disturbances"))
    enterprise = EnterpriseConfig(**_read_section(raw, "enterprise"))
    medical = MedicalConfig(**_read_section(raw, "medical"))
    farmer = FarmerConfig(**_read_section(raw, "farmer"))

    return ModelConfig(
        simulation=simulation,
        environment=environment,
        disturbances=disturbances,
        enterprise=enterprise,
        medical=medical,
        farmer=farmer,
    )
