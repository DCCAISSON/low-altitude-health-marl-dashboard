from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DisturbanceState:
    weather: float
    medical_surge: float
    demand_shock: float
    airspace_control: float = 0.0
    device_failure: float = 0.0


@dataclass(slots=True)
class MarketState:
    t: int
    coverage_rate: float
    average_health: float
    average_response_time: float
    trust_level: float
    queue_pressure: float
    enterprise_cash: float
    active_capacity: float
    last_demand: float
    last_served: float
    disturbances: DisturbanceState


@dataclass(slots=True)
class EnterpriseAction:
    capacity_scale: float
    price: float
    reliability_investment: float


@dataclass(slots=True)
class MedicalAction:
    sinking_intensity: float
    referral_acceptance: float
    integration_effort: float


@dataclass(slots=True)
class StepMetrics:
    adoption_rate: float
    demand: float
    supply_capacity: float
    medical_capacity: float
    served: float
    unmet: float
    effective_price: float
    response_time: float


@dataclass(slots=True)
class StepResult:
    state: MarketState
    enterprise_reward: float
    medical_reward: float
    farmer_reward: float
    metrics: StepMetrics
