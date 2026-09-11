from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from math import exp
from pathlib import Path
import random
import tomllib

from .config import DisturbanceConfig
from .disturbances import DisturbanceProcess
from .health_markov import HealthMarkovModel, MarkovConfig
from .state import DisturbanceState


EPS = 1e-9


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _sigmoid(value: float) -> float:
    value = _clip(value, -20.0, 20.0)
    return 1.0 / (1.0 + exp(-value))


def _normalize(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    cleaned = [max(float(value), 0.0) for value in values]
    total = sum(cleaned)
    if total <= EPS:
        size = len(cleaned)
        return tuple(1.0 / size for _ in range(size))
    return tuple(value / total for value in cleaned)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_rate(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= EPS else numerator / denominator


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= EPS:
        return _mean(values)
    return sum(value * weight for value, weight in zip(values, weights)) / total


def _weighted_gini(values: list[float], weights: list[float]) -> float:
    filtered = [(max(float(value), 0.0), max(float(weight), 0.0)) for value, weight in zip(values, weights) if weight > EPS]
    if not filtered:
        return 0.0
    filtered.sort(key=lambda item: item[0])
    total_weight = sum(weight for _, weight in filtered)
    total_value = sum(value * weight for value, weight in filtered)
    if total_weight <= EPS or total_value <= EPS:
        return 0.0
    cumulative_weight = 0.0
    cumulative_value = 0.0
    area = 0.0
    for value, weight in filtered:
        prev_weight = cumulative_weight / total_weight
        prev_value = cumulative_value / total_value
        cumulative_weight += weight
        cumulative_value += value * weight
        next_weight = cumulative_weight / total_weight
        next_value = cumulative_value / total_value
        area += (prev_value + next_value) * (next_weight - prev_weight) / 2.0
    return _clip(1.0 - 2.0 * area, 0.0, 1.0)


def _local_service_multiplier(task_id: str) -> float:
    return {
        'chronic_followup': 1.0,
        'sample_transport': 1.15,
        'medicine_delivery': 0.25,
        'acute_urgent': 0.12,
    }.get(task_id, 0.0)


@dataclass(slots=True)
class VillageCfg:
    id: str
    name: str
    population: int
    base_health: float
    trust: float
    demand_weight: float
    traditional_time: float
    risk_weight: float
    state_distribution: tuple[float, ...] | None = None


@dataclass(slots=True)
class HubCfg:
    id: str
    name: str
    base_capacity: float
    reliability: float


@dataclass(slots=True)
class HospitalCfg:
    id: str
    name: str
    base_capacity: float
    integration_base: float


@dataclass(slots=True)
class Edge:
    origin: str
    destination: str
    travel_time: float
    weather_exposure: float


@dataclass(slots=True)
class TaskTypeCfg:
    id: str
    name: str
    demand_share: float
    priority: float
    time_sensitivity_multiplier: float
    health_benefit_multiplier: float
    price_sensitivity_multiplier: float
    failure_penalty_multiplier: float
    medical_value_multiplier: float
    enterprise_cost_multiplier: float
    reimbursement_bonus: float = 0.0
    trust_bonus: float = 0.0
    demand_shock_multiplier: float = 1.0
    risk_weight_multiplier: float = 1.0


@dataclass(slots=True)
class AdvancedSpatialConfig:
    simulation: dict
    environment: dict
    policy: dict
    constraints: dict
    markov: MarkovConfig
    disturbances: DisturbanceConfig
    enterprise: dict
    medical: dict
    farmer: dict
    task_types: tuple[TaskTypeCfg, ...]
    villages: tuple[VillageCfg, ...]
    hubs: tuple[HubCfg, ...]
    hospitals: tuple[HospitalCfg, ...]
    edges: tuple[Edge, ...]


@dataclass(slots=True)
class Route:
    village_id: str
    hub_id: str
    hospital_id: str
    total_time: float
    weather_exposure: float


@dataclass(slots=True)
class VillageState:
    id: str
    name: str
    population: int
    health: float
    trust: float
    demand_weight: float
    traditional_time: float
    risk_weight: float
    health_distribution: tuple[float, ...]
    last_demand: float
    last_served: float
    last_time: float


@dataclass(slots=True)
class AdvancedSpatialState:
    t: int
    average_health: float
    network_coverage: float
    average_response_time: float
    queue_pressure: float
    enterprise_cash: float
    disturbances: DisturbanceState
    average_state_distribution: tuple[float, ...]
    villages: tuple[VillageState, ...]
    fairness_score: float = 1.0
    coverage_gap: float = 0.0
    enterprise_loss_streak: int = 0
    enterprise_active: bool = True
    chronic_pressure: float = 0.0
    urgent_pressure: float = 0.0


@dataclass(slots=True)
class EnterpriseAction:
    capacity_scale: float
    hub_allocations: tuple[float, ...]
    price: float
    reliability_investment: float
    reserve_capacity_ratio: float = 0.0
    route_activation: tuple[float, ...] = ()


@dataclass(slots=True)
class MedicalAction:
    referral_acceptance: float
    integration_effort: float
    hospital_allocations: tuple[float, ...]


@dataclass(slots=True)
class AdvancedStepMetrics:
    total_demand: float
    total_served: float
    total_unmet: float
    average_adoption: float
    network_coverage: float
    average_response_time: float
    fairness_score: float = 1.0
    minimum_coverage_gap: float = 0.0
    underserved_share: float = 0.0
    remote_service_rate: float = 0.0
    hospital_redline_over: float = 0.0
    enterprise_active: bool = True
    average_effective_price: float = 0.0
    load_factor: float = 0.0
    chronic_management_rate: float = 0.0
    severe_outflow_rate: float = 0.0
    followup_completion_rate: float = 0.0
    avoidable_hospitalization_rate: float = 0.0
    county_retention_rate: float = 0.0
    specialist_bypass_rate: float = 0.0
    bp_glucose_control_rate: float = 0.0
    route_profitability: float = 0.0
    reserve_capacity_ratio: float = 0.0
    battery_or_charge_constraint: float = 0.0
    weather_robust_dispatch_score: float = 0.0
    hub_open_ratio: float = 0.0
    route_activation_share: float = 0.0
    post_exam_management_rate: float = 0.0
    screening_completion_rate: float = 0.0
    township_trust_index: float = 0.0
    latent_demand_release_share: float = 0.0
    big_hospital_diversion_reduction: float = 0.0
    risk_stratification_coverage: float = 0.0
    caregiver_time_release_hours: float = 0.0
    task_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    policy_metrics: dict[str, float] = field(default_factory=dict)
    constraint_flags: dict[str, bool] = field(default_factory=dict)
    interruptions: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class AdvancedStepResult:
    state: AdvancedSpatialState
    enterprise_reward: float
    medical_reward: float
    farmer_reward: float
    metrics: AdvancedStepMetrics


@dataclass(slots=True)
class AdvancedEpisodeSummary:
    periods: int
    final_average_health: float
    final_network_coverage: float
    final_average_response_time: float
    total_enterprise_reward: float
    total_medical_reward: float
    total_farmer_reward: float
    average_adoption: float
    average_service_rate: float
    average_network_coverage: float
    state_names_cn: tuple[str, ...]
    final_state_distribution: tuple[float, ...]
    average_fairness: float = 1.0
    average_coverage_gap: float = 0.0
    average_remote_service_rate: float = 0.0
    policy_cost: float = 0.0
    constraint_breach_count: int = 0
    enterprise_exit: bool = False
    average_load_factor: float = 0.0
    average_chronic_management_rate: float = 0.0
    average_severe_outflow_rate: float = 0.0
    average_followup_completion_rate: float = 0.0
    average_avoidable_hospitalization_rate: float = 0.0
    average_county_retention_rate: float = 0.0
    average_specialist_bypass_rate: float = 0.0
    average_bp_glucose_control_rate: float = 0.0
    average_route_profitability: float = 0.0
    average_reserve_capacity_ratio: float = 0.0
    average_battery_or_charge_constraint: float = 0.0
    average_weather_robust_dispatch_score: float = 0.0
    average_hub_open_ratio: float = 0.0
    average_route_activation_share: float = 0.0
    average_post_exam_management_rate: float = 0.0
    average_screening_completion_rate: float = 0.0
    average_township_trust_index: float = 0.0
    average_latent_demand_release_share: float = 0.0
    average_big_hospital_diversion_reduction: float = 0.0
    average_risk_stratification_coverage: float = 0.0
    average_caregiver_time_release_hours: float = 0.0
    task_service_rates: dict[str, float] = field(default_factory=dict)


def _default_environment() -> dict:
    return {
        'reimbursement_rate': 0.2,
        'fixed_subsidy': 0.0,
        'service_health_gain': 0.08,
        'unmet_need_penalty': 0.0544,
        'background_health_decay': 0.012,
        'trust_update_speed': 0.12,
        'weather_service_penalty': 0.55,
        'medical_surge_penalty': 0.6,
        'airspace_service_penalty': 0.38,
        'device_failure_penalty': 0.25,
    }


def _default_policy() -> dict:
    return {
        'fixed_subsidy': 0.0,
        'per_service_subsidy': 0.0,
        'performance_subsidy_per_coverage': 0.0,
        'fairness_subsidy_per_point': 0.0,
        'remote_service_subsidy': 0.0,
        'chronic_management_subsidy': 0.0,
        'reimbursement_rate_floor': 0.0,
        'patient_uav_delivery_fee': 0.0,
        'local_sampling_subsidy': 0.0,
        'urgent_task_subsidy': 0.0,
        'public_service_procurement_ratio': 0.0,
    }


def _default_enterprise() -> dict:
    return {
        'contract_fixed_revenue': 0.0,
        'contract_stability_bonus': 18.0,
        'empty_flight_cost': 0.35,
        'loss_streak_penalty': 25.0,
        'reliability_unit_cost': 12.0,
        'route_open_threshold': 0.45,
        'route_time_cost_weight': 0.22,
        'reserve_cost_weight': 8.0,
        'battery_time_limit_hours': 3.2,
        'battery_penalty_weight': 55.0,
        'weather_robust_bonus_weight': 20.0,
        'local_contract_service_bonus': 10.0,
        'sample_turnaround_penalty': 6.0,
        'urgent_task_premium_multiplier': 1.0,
        'operator_count': 1.0,
        'lines_per_operator_limit': 12.0,
        'dispatch_scale_efficiency': 0.0,
        'traceability_bonus': 0.0,
        'data_upload_penalty': 0.0,
        'airspace_compliance_cost': 0.0,
        'annual_line_contract_value': 0.0,
        'per_km_price_reference': 0.0,
        'population_density_route_scale': 0.0,
        'village_sink_bonus': 0.0,
        'township_sink_bonus': 0.0,
        'connectivity_penalty_weight': 0.0,
    }


def _default_medical() -> dict:
    return {
        'queue_penalty_weight': 0.70,
        'idle_penalty_weight': 5.0,
        'chronic_management_value': 80.0,
        'severe_outflow_penalty': 140.0,
        'followup_completion_value': 38.0,
        'county_retention_value': 28.0,
        'bp_glucose_control_value': 24.0,
        'specialist_bypass_penalty': 85.0,
        'avoidable_hospitalization_penalty': 95.0,
        'post_exam_management_value': 26.0,
        'screening_completion_value': 18.0,
        'township_trust_value': 16.0,
        'staff_shortage_penalty': 22.0,
        'risk_stratification_value': 12.0,
    }


def _default_farmer() -> dict:
    return {
        'health_gain_weight': 70.0,
        'unmet_penalty_weight': 12.0,
        'severe_outflow_loss': 180.0,
        'travel_time_to_town_minutes': 30.0,
        'travel_roundtrip_minutes': 60.0,
        'roundtrip_transport_cost_yuan': 100.0,
        'transport_cost_reference_yuan': 100.0,
        'uav_delivery_out_of_pocket_yuan': 0.0,
        'sample_test_out_of_pocket_yuan': 20.0,
        'sample_cost_reference_yuan': 20.0,
        'time_cost_weight': 1.25,
        'out_of_pocket_weight': 0.45,
        'local_sample_point_bonus': 0.35,
        'latent_demand_release_rate': 0.30,
        'local_service_trust_gain': 0.06,
        'big_hospital_avoidance_weight': 0.22,
        'caregiver_time_share': 0.35,
        'caregiver_time_weight': 0.55,
        'caregiver_time_value_multiplier': 0.8,
    }


def _default_constraints() -> dict:
    return {
        'minimum_coverage_ratio': 0.5,
        'fairness_floor': 0.62,
        'coverage_shortfall_penalty': 120.0,
        'fairness_penalty': 85.0,
        'underserved_penalty': 60.0,
        'underserved_threshold': 0.4,
        'hospital_capacity_redline': 0.92,
        'hospital_redline_penalty': 90.0,
        'enterprise_exit_cash_threshold': -420.0,
        'enterprise_exit_periods': 3,
        'weather_interrupt_cost_weight': 1.0,
        'airspace_interrupt_cost_weight': 0.7,
        'device_interrupt_cost_weight': 0.6,
        'remote_service_floor': 0.35,
        'remote_floor_penalty': 75.0,
    }


def _default_task_types() -> tuple[dict[str, float | str], ...]:
    return (
        {
            'id': 'chronic_followup',
            'name': '慢病复诊/随访',
            'demand_share': 0.33,
            'priority': 0.70,
            'time_sensitivity_multiplier': 1.05,
            'health_benefit_multiplier': 1.20,
            'price_sensitivity_multiplier': 0.90,
            'failure_penalty_multiplier': 1.15,
            'medical_value_multiplier': 1.10,
            'enterprise_cost_multiplier': 1.00,
            'reimbursement_bonus': 0.06,
            'trust_bonus': 0.04,
            'demand_shock_multiplier': 0.80,
            'risk_weight_multiplier': 1.00,
        },
        {
            'id': 'sample_transport',
            'name': '检验样本送检',
            'demand_share': 0.25,
            'priority': 0.88,
            'time_sensitivity_multiplier': 1.28,
            'health_benefit_multiplier': 1.00,
            'price_sensitivity_multiplier': 0.95,
            'failure_penalty_multiplier': 1.35,
            'medical_value_multiplier': 1.26,
            'enterprise_cost_multiplier': 1.06,
            'reimbursement_bonus': 0.08,
            'trust_bonus': 0.03,
            'demand_shock_multiplier': 1.05,
            'risk_weight_multiplier': 1.08,
        },
        {
            'id': 'medicine_delivery',
            'name': '常用药配送',
            'demand_share': 0.24,
            'priority': 0.56,
            'time_sensitivity_multiplier': 0.86,
            'health_benefit_multiplier': 0.92,
            'price_sensitivity_multiplier': 1.08,
            'failure_penalty_multiplier': 0.82,
            'medical_value_multiplier': 0.94,
            'enterprise_cost_multiplier': 0.90,
            'reimbursement_bonus': 0.03,
            'trust_bonus': 0.02,
            'demand_shock_multiplier': 0.95,
            'risk_weight_multiplier': 0.92,
        },
        {
            'id': 'acute_urgent',
            'name': '急性高优先级需求',
            'demand_share': 0.18,
            'priority': 1.00,
            'time_sensitivity_multiplier': 1.65,
            'health_benefit_multiplier': 1.55,
            'price_sensitivity_multiplier': 0.55,
            'failure_penalty_multiplier': 1.80,
            'medical_value_multiplier': 1.52,
            'enterprise_cost_multiplier': 1.22,
            'reimbursement_bonus': 0.12,
            'trust_bonus': 0.08,
            'demand_shock_multiplier': 1.28,
            'risk_weight_multiplier': 1.20,
        },
    )


def _default_disturbances() -> dict:
    return {
        'weather_persistence': 0.6,
        'medical_surge_persistence': 0.45,
        'demand_persistence': 0.35,
        'weather_std': 0.18,
        'medical_surge_std': 0.12,
        'demand_std': 0.1,
        'airspace_persistence': 0.4,
        'airspace_std': 0.1,
        'device_persistence': 0.35,
        'device_std': 0.08,
    }


def _parse_markov(raw: dict) -> MarkovConfig:
    return MarkovConfig(
        state_names=tuple(raw['state_names']),
        state_names_cn=tuple(raw.get('state_names_cn', raw['state_names'])),
        state_weights=tuple(float(value) for value in raw['state_weights']),
        base_transition=tuple(tuple(float(item) for item in row) for row in raw['base_transition']),
        service_effect=float(raw['service_effect']),
        unmet_need_effect=float(raw['unmet_need_effect']),
        shock_effect=float(raw['shock_effect']),
        trust_effect=float(raw.get('trust_effect', 0.0)),
    )


def _parse_village(raw: dict) -> VillageCfg:
    state_distribution = raw.get('state_distribution')
    return VillageCfg(
        id=raw['id'],
        name=raw['name'],
        population=int(raw['population']),
        base_health=float(raw['base_health']),
        trust=float(raw['trust']),
        demand_weight=float(raw['demand_weight']),
        traditional_time=float(raw['traditional_time']),
        risk_weight=float(raw['risk_weight']),
        state_distribution=None if state_distribution is None else tuple(float(value) for value in state_distribution),
    )


def _parse_task_type(raw: dict) -> TaskTypeCfg:
    return TaskTypeCfg(
        id=raw['id'],
        name=raw.get('name', raw['id']),
        demand_share=float(raw['demand_share']),
        priority=float(raw['priority']),
        time_sensitivity_multiplier=float(raw['time_sensitivity_multiplier']),
        health_benefit_multiplier=float(raw['health_benefit_multiplier']),
        price_sensitivity_multiplier=float(raw['price_sensitivity_multiplier']),
        failure_penalty_multiplier=float(raw['failure_penalty_multiplier']),
        medical_value_multiplier=float(raw['medical_value_multiplier']),
        enterprise_cost_multiplier=float(raw['enterprise_cost_multiplier']),
        reimbursement_bonus=float(raw.get('reimbursement_bonus', 0.0)),
        trust_bonus=float(raw.get('trust_bonus', 0.0)),
        demand_shock_multiplier=float(raw.get('demand_shock_multiplier', 1.0)),
        risk_weight_multiplier=float(raw.get('risk_weight_multiplier', 1.0)),
    )


def load_advanced_spatial_config(path: str | Path) -> AdvancedSpatialConfig:
    raw = tomllib.loads(Path(path).read_text(encoding='utf-8-sig'))
    environment = _default_environment()
    environment.update(raw.get('environment', {}))
    policy = _default_policy()
    policy.update(raw.get('policy', {}))
    constraints = _default_constraints()
    constraints.update(raw.get('constraints', {}))
    disturbances = _default_disturbances()
    disturbances.update(raw.get('disturbances', {}))
    enterprise = _default_enterprise()
    enterprise.update(raw.get('enterprise', {}))
    medical = _default_medical()
    medical.update(raw.get('medical', {}))
    farmer = _default_farmer()
    farmer.update(raw.get('farmer', {}))
    task_types_raw = raw.get('task_types', list(_default_task_types()))
    task_types = tuple(_parse_task_type(item) for item in task_types_raw)
    total_share = sum(item.demand_share for item in task_types)
    if total_share <= EPS:
        raise ValueError('task_types demand_share must sum to a positive value')
    normalized_tasks = tuple(
        TaskTypeCfg(
            id=item.id,
            name=item.name,
            demand_share=item.demand_share / total_share,
            priority=item.priority,
            time_sensitivity_multiplier=item.time_sensitivity_multiplier,
            health_benefit_multiplier=item.health_benefit_multiplier,
            price_sensitivity_multiplier=item.price_sensitivity_multiplier,
            failure_penalty_multiplier=item.failure_penalty_multiplier,
            medical_value_multiplier=item.medical_value_multiplier,
            enterprise_cost_multiplier=item.enterprise_cost_multiplier,
            reimbursement_bonus=item.reimbursement_bonus,
            trust_bonus=item.trust_bonus,
            demand_shock_multiplier=item.demand_shock_multiplier,
            risk_weight_multiplier=item.risk_weight_multiplier,
        )
        for item in task_types
    )
    return AdvancedSpatialConfig(
        simulation=raw['simulation'],
        environment=environment,
        policy=policy,
        constraints=constraints,
        markov=_parse_markov(raw['markov']),
        disturbances=DisturbanceConfig(**disturbances),
        enterprise=enterprise,
        medical=medical,
        farmer=farmer,
        task_types=normalized_tasks,
        villages=tuple(_parse_village(village) for village in raw['villages']),
        hubs=tuple(HubCfg(**hub) for hub in raw['hubs']),
        hospitals=tuple(HospitalCfg(**hospital) for hospital in raw['hospitals']),
        edges=tuple(Edge(**edge) for edge in raw['edges']),
    )


def clone_advanced_spatial_config(cfg: AdvancedSpatialConfig) -> AdvancedSpatialConfig:
    return deepcopy(cfg)


def available_advanced_spatial_scenarios() -> tuple[str, ...]:
    return ('baseline', 'harsh_weather', 'low_trust', 'sparse_demand', 'chronic_burden', 'policy_push')


def config_snapshot(cfg: AdvancedSpatialConfig) -> dict[str, object]:
    return {
        'simulation': dict(cfg.simulation),
        'environment': dict(cfg.environment),
        'policy': dict(cfg.policy),
        'constraints': dict(cfg.constraints),
        'enterprise': dict(cfg.enterprise),
        'medical': dict(cfg.medical),
        'farmer': dict(cfg.farmer),
        'disturbances': {
            'weather_persistence': cfg.disturbances.weather_persistence,
            'medical_surge_persistence': cfg.disturbances.medical_surge_persistence,
            'demand_persistence': cfg.disturbances.demand_persistence,
            'weather_std': cfg.disturbances.weather_std,
            'medical_surge_std': cfg.disturbances.medical_surge_std,
            'demand_std': cfg.disturbances.demand_std,
            'airspace_persistence': cfg.disturbances.airspace_persistence,
            'airspace_std': cfg.disturbances.airspace_std,
            'device_persistence': cfg.disturbances.device_persistence,
            'device_std': cfg.disturbances.device_std,
        },
        'task_types': [
            {
                'id': task.id,
                'name': task.name,
                'demand_share': task.demand_share,
                'priority': task.priority,
                'time_sensitivity_multiplier': task.time_sensitivity_multiplier,
                'health_benefit_multiplier': task.health_benefit_multiplier,
                'price_sensitivity_multiplier': task.price_sensitivity_multiplier,
                'failure_penalty_multiplier': task.failure_penalty_multiplier,
                'medical_value_multiplier': task.medical_value_multiplier,
                'enterprise_cost_multiplier': task.enterprise_cost_multiplier,
                'reimbursement_bonus': task.reimbursement_bonus,
                'trust_bonus': task.trust_bonus,
                'demand_shock_multiplier': task.demand_shock_multiplier,
                'risk_weight_multiplier': task.risk_weight_multiplier,
            }
            for task in cfg.task_types
        ],
    }


def apply_runtime_overrides(cfg: AdvancedSpatialConfig, overrides: dict[str, object] | None) -> AdvancedSpatialConfig:
    cloned = clone_advanced_spatial_config(cfg)
    if not overrides:
        return cloned
    for section_name in ('simulation', 'environment', 'policy', 'constraints', 'enterprise', 'medical', 'farmer'):
        section_override = overrides.get(section_name)
        if isinstance(section_override, dict):
            section = dict(getattr(cloned, section_name))
            for key, value in section_override.items():
                if value is not None:
                    section[key] = value
            setattr(cloned, section_name, section)
    disturbance_override = overrides.get('disturbances')
    if isinstance(disturbance_override, dict):
        for key, value in disturbance_override.items():
            if value is not None and hasattr(cloned.disturbances, key):
                setattr(cloned.disturbances, key, value)
    task_override = overrides.get('task_types')
    if isinstance(task_override, list):
        task_map = {task.id: task for task in cloned.task_types}
        updated_tasks: list[TaskTypeCfg] = []
        for raw_item in task_override:
            if not isinstance(raw_item, dict):
                continue
            task_id = str(raw_item.get('id', '')).strip()
            if not task_id or task_id not in task_map:
                continue
            source = task_map[task_id]
            updated_tasks.append(
                TaskTypeCfg(
                    id=source.id,
                    name=str(raw_item.get('name', source.name)),
                    demand_share=float(raw_item.get('demand_share', source.demand_share)),
                    priority=float(raw_item.get('priority', source.priority)),
                    time_sensitivity_multiplier=float(raw_item.get('time_sensitivity_multiplier', source.time_sensitivity_multiplier)),
                    health_benefit_multiplier=float(raw_item.get('health_benefit_multiplier', source.health_benefit_multiplier)),
                    price_sensitivity_multiplier=float(raw_item.get('price_sensitivity_multiplier', source.price_sensitivity_multiplier)),
                    failure_penalty_multiplier=float(raw_item.get('failure_penalty_multiplier', source.failure_penalty_multiplier)),
                    medical_value_multiplier=float(raw_item.get('medical_value_multiplier', source.medical_value_multiplier)),
                    enterprise_cost_multiplier=float(raw_item.get('enterprise_cost_multiplier', source.enterprise_cost_multiplier)),
                    reimbursement_bonus=float(raw_item.get('reimbursement_bonus', source.reimbursement_bonus)),
                    trust_bonus=float(raw_item.get('trust_bonus', source.trust_bonus)),
                    demand_shock_multiplier=float(raw_item.get('demand_shock_multiplier', source.demand_shock_multiplier)),
                    risk_weight_multiplier=float(raw_item.get('risk_weight_multiplier', source.risk_weight_multiplier)),
                )
            )
        if updated_tasks:
            merged = {task.id: task for task in cloned.task_types}
            for task in updated_tasks:
                merged[task.id] = task
            total_share = sum(task.demand_share for task in merged.values())
            if total_share > EPS:
                cloned.task_types = tuple(
                    TaskTypeCfg(
                        id=task.id,
                        name=task.name,
                        demand_share=task.demand_share / total_share,
                        priority=task.priority,
                        time_sensitivity_multiplier=task.time_sensitivity_multiplier,
                        health_benefit_multiplier=task.health_benefit_multiplier,
                        price_sensitivity_multiplier=task.price_sensitivity_multiplier,
                        failure_penalty_multiplier=task.failure_penalty_multiplier,
                        medical_value_multiplier=task.medical_value_multiplier,
                        enterprise_cost_multiplier=task.enterprise_cost_multiplier,
                        reimbursement_bonus=task.reimbursement_bonus,
                        trust_bonus=task.trust_bonus,
                        demand_shock_multiplier=task.demand_shock_multiplier,
                        risk_weight_multiplier=task.risk_weight_multiplier,
                    )
                    for task in merged.values()
                )
    return cloned


def apply_advanced_spatial_scenario(cfg: AdvancedSpatialConfig, scenario_name: str | None) -> AdvancedSpatialConfig:
    scenario = (scenario_name or 'baseline').strip().lower()
    cloned = clone_advanced_spatial_config(cfg)
    if scenario == 'baseline':
        return cloned
    if scenario == 'harsh_weather':
        cloned.disturbances.weather_std *= 1.8
        cloned.disturbances.weather_persistence = min(0.9, cloned.disturbances.weather_persistence + 0.15)
        cloned.disturbances.airspace_std *= 1.6
        cloned.environment['weather_service_penalty'] *= 1.18
        cloned.environment['airspace_service_penalty'] *= 1.15
        cloned.constraints['minimum_coverage_ratio'] = min(0.7, cloned.constraints['minimum_coverage_ratio'] + 0.05)
        return cloned
    if scenario == 'low_trust':
        cloned.villages = tuple(
            VillageCfg(
                id=v.id,
                name=v.name,
                population=v.population,
                base_health=v.base_health,
                trust=max(0.15, v.trust - 0.18),
                demand_weight=v.demand_weight,
                traditional_time=v.traditional_time,
                risk_weight=v.risk_weight,
                state_distribution=v.state_distribution,
            )
            for v in cloned.villages
        )
        return cloned
    if scenario == 'sparse_demand':
        cloned.simulation = dict(cloned.simulation)
        cloned.simulation['base_need_rate'] *= 0.90
        cloned.enterprise = dict(cloned.enterprise)
        cloned.enterprise['base_total_capacity'] *= 0.90
        cloned.enterprise['route_open_threshold'] = 0.68
        cloned.enterprise['route_time_cost_weight'] *= 1.30
        cloned.enterprise['operator_count'] *= 0.82
        cloned.enterprise['lines_per_operator_limit'] *= 0.90
        cloned.enterprise['sample_turnaround_penalty'] *= 1.35
        cloned.enterprise['local_contract_service_bonus'] *= 0.82
        cloned.farmer = dict(cloned.farmer)
        cloned.farmer['latent_demand_release_rate'] *= 0.32
        cloned.farmer['local_sample_point_bonus'] *= 0.32
        cloned.farmer['local_service_trust_gain'] *= 0.42
        cloned.farmer['big_hospital_avoidance_weight'] *= 0.72
        cloned.farmer['learning_cost'] *= 1.18
        cloned.environment = dict(cloned.environment)
        cloned.environment['trust_update_speed'] *= 0.74
        cloned.environment['service_health_gain'] *= 0.86
        cloned.environment['unmet_need_penalty'] *= 1.15
        cloned.environment['airspace_service_penalty'] *= 1.12
        cloned.environment['device_failure_penalty'] *= 1.10
        cloned.disturbances.airspace_std *= 1.22
        cloned.disturbances.device_std *= 1.18
        sparse_villages = []
        for village in cloned.villages:
            remote = village.traditional_time >= 9.0 or village.risk_weight >= 0.65
            sparse_villages.append(
                VillageCfg(
                    id=village.id,
                    name=village.name,
                    population=village.population,
                    base_health=max(0.1, village.base_health - (0.045 if remote else 0.025)),
                    trust=max(0.12, village.trust - (0.12 if remote else 0.08)),
                    demand_weight=village.demand_weight * (0.95 if remote else 0.98),
                    traditional_time=village.traditional_time * (1.28 if remote else 1.16),
                    risk_weight=min(1.0, village.risk_weight + (0.04 if remote else 0.02)),
                    state_distribution=village.state_distribution,
                )
            )
        cloned.villages = tuple(sparse_villages)
        sparse_edges = []
        for edge in cloned.edges:
            if edge.origin.startswith('V'):
                time_factor = 1.32
                exposure_bump = 0.09
            else:
                time_factor = 1.12
                exposure_bump = 0.03
            sparse_edges.append(
                Edge(
                    origin=edge.origin,
                    destination=edge.destination,
                    travel_time=edge.travel_time * time_factor,
                    weather_exposure=min(0.95, edge.weather_exposure + exposure_bump),
                )
            )
        cloned.edges = tuple(sparse_edges)
        task_scale = {
            'chronic_followup': 0.78,
            'sample_transport': 0.82,
            'medicine_delivery': 0.92,
            'acute_urgent': 1.55,
        }
        sparse_tasks = [
            TaskTypeCfg(
                id=task.id,
                name=task.name,
                demand_share=task.demand_share * task_scale.get(task.id, 1.0),
                priority=task.priority,
                time_sensitivity_multiplier=task.time_sensitivity_multiplier,
                health_benefit_multiplier=task.health_benefit_multiplier,
                price_sensitivity_multiplier=task.price_sensitivity_multiplier,
                failure_penalty_multiplier=task.failure_penalty_multiplier,
                medical_value_multiplier=task.medical_value_multiplier,
                enterprise_cost_multiplier=task.enterprise_cost_multiplier,
                reimbursement_bonus=task.reimbursement_bonus,
                trust_bonus=task.trust_bonus,
                demand_shock_multiplier=task.demand_shock_multiplier,
                risk_weight_multiplier=task.risk_weight_multiplier,
            )
            for task in cloned.task_types
        ]
        total_share = sum(task.demand_share for task in sparse_tasks)
        cloned.task_types = tuple(
            TaskTypeCfg(
                id=task.id,
                name=task.name,
                demand_share=task.demand_share / total_share,
                priority=task.priority,
                time_sensitivity_multiplier=task.time_sensitivity_multiplier,
                health_benefit_multiplier=task.health_benefit_multiplier,
                price_sensitivity_multiplier=task.price_sensitivity_multiplier,
                failure_penalty_multiplier=task.failure_penalty_multiplier,
                medical_value_multiplier=task.medical_value_multiplier,
                enterprise_cost_multiplier=task.enterprise_cost_multiplier,
                reimbursement_bonus=task.reimbursement_bonus,
                trust_bonus=task.trust_bonus,
                demand_shock_multiplier=task.demand_shock_multiplier,
                risk_weight_multiplier=task.risk_weight_multiplier,
            )
            for task in sparse_tasks
        )
        return cloned
    if scenario == 'chronic_burden':
        shifted = []
        for village in cloned.villages:
            distribution = village.state_distribution or cloned.markov.base_transition[0]
            dist = list(_normalize(distribution))
            if len(dist) >= 5:
                transfer = min(dist[0] * 0.18, 0.08)
                dist[0] -= transfer
                dist[2] += transfer * 0.55
                dist[3] += transfer * 0.45
            shifted.append(
                VillageCfg(
                    id=village.id,
                    name=village.name,
                    population=village.population,
                    base_health=max(cloned.markov.state_weights[-1], village.base_health - 0.04),
                    trust=village.trust,
                    demand_weight=village.demand_weight * 1.08,
                    traditional_time=village.traditional_time,
                    risk_weight=min(1.0, village.risk_weight + 0.06),
                    state_distribution=tuple(dist),
                )
            )
        cloned.villages = tuple(shifted)
        cloned.simulation = dict(cloned.simulation)
        cloned.simulation['base_need_rate'] *= 1.08
        return cloned
    if scenario == 'policy_push':
        cloned.policy = dict(cloned.policy)
        cloned.policy['per_service_subsidy'] += 0.4
        cloned.policy['performance_subsidy_per_coverage'] += 135.0
        cloned.policy['fairness_subsidy_per_point'] += 60.0
        cloned.policy['remote_service_subsidy'] += 0.50
        cloned.policy['chronic_management_subsidy'] += 30.0
        cloned.policy['local_sampling_subsidy'] += 0.20
        cloned.policy['urgent_task_subsidy'] += 0.22
        cloned.policy['reimbursement_rate_floor'] = max(cloned.policy.get('reimbursement_rate_floor', 0.0), 0.22)
        cloned.environment = dict(cloned.environment)
        cloned.environment['reimbursement_rate'] = min(0.6, cloned.environment['reimbursement_rate'] + 0.15)
        cloned.environment['trust_update_speed'] *= 1.05
        cloned.environment['service_health_gain'] *= 1.03
        cloned.enterprise = dict(cloned.enterprise)
        cloned.enterprise['base_total_capacity'] *= 1.16
        cloned.enterprise['route_open_threshold'] = 0.32
        cloned.enterprise['route_time_cost_weight'] *= 0.88
        cloned.enterprise['operator_count'] *= 1.18
        cloned.enterprise['lines_per_operator_limit'] *= 1.10
        cloned.enterprise['sample_turnaround_penalty'] *= 0.84
        cloned.enterprise['local_contract_service_bonus'] *= 1.18
        cloned.enterprise['population_density_route_scale'] *= 0.35
        cloned.farmer = dict(cloned.farmer)
        cloned.farmer['local_sample_point_bonus'] *= 1.10
        cloned.farmer['local_service_trust_gain'] *= 1.08
        cloned.constraints = dict(cloned.constraints)
        cloned.constraints['minimum_coverage_ratio'] = 0.85
        cloned.constraints['coverage_shortfall_penalty'] *= 2.00
        cloned.constraints['underserved_penalty'] *= 1.30
        cloned.constraints['fairness_floor'] = min(0.80, cloned.constraints['fairness_floor'] + 0.05)
        cloned.constraints['fairness_penalty'] *= 1.25
        cloned.constraints['remote_service_floor'] = min(0.58, cloned.constraints['remote_service_floor'] + 0.10)
        cloned.constraints['remote_floor_penalty'] *= 1.45
        cloned.hubs = tuple(
            HubCfg(
                id=hub.id,
                name=hub.name,
                base_capacity=hub.base_capacity * 1.07,
                reliability=min(0.98, hub.reliability + 0.03),
            )
            for hub in cloned.hubs
        )
        cloned.hospitals = tuple(
            HospitalCfg(
                id=hospital.id,
                name=hospital.name,
                base_capacity=hospital.base_capacity * 1.04,
                integration_base=min(0.98, hospital.integration_base + 0.02),
            )
            for hospital in cloned.hospitals
        )
        cloned.villages = tuple(
            VillageCfg(
                id=village.id,
                name=village.name,
                population=village.population,
                base_health=village.base_health,
                trust=min(0.90, village.trust + (0.05 if (village.traditional_time >= 9.0 or village.risk_weight >= 0.65) else 0.03)),
                demand_weight=village.demand_weight,
                traditional_time=village.traditional_time * (0.95 if (village.traditional_time >= 9.0 or village.risk_weight >= 0.65) else 0.98),
                risk_weight=village.risk_weight,
                state_distribution=village.state_distribution,
            )
            for village in cloned.villages
        )
        return cloned
    raise ValueError(f'Unsupported spatial scenario: {scenario_name}')


class AdvancedSpatialNetwork:
    def __init__(self, cfg: AdvancedSpatialConfig) -> None:
        self.hub_ids = tuple(item.id for item in cfg.hubs)
        self.hospital_ids = tuple(item.id for item in cfg.hospitals)
        self.route_catalog: list[Route] = []
        self.route_index: dict[tuple[str, str, str], int] = {}
        village_edges: dict[str, list[Edge]] = defaultdict(list)
        hub_edges: dict[str, list[Edge]] = defaultdict(list)
        village_ids = {item.id for item in cfg.villages}
        hub_ids = set(self.hub_ids)
        hospital_ids = set(self.hospital_ids)
        for edge in cfg.edges:
            if edge.origin in village_ids and edge.destination in hub_ids:
                village_edges[edge.origin].append(edge)
            elif edge.origin in hub_ids and edge.destination in hospital_ids:
                hub_edges[edge.origin].append(edge)
        self.routes: dict[str, list[Route]] = {}
        for village in cfg.villages:
            routes = []
            for village_edge in village_edges[village.id]:
                for hub_edge in hub_edges[village_edge.destination]:
                    routes.append(
                        Route(
                            village_id=village.id,
                            hub_id=village_edge.destination,
                            hospital_id=hub_edge.destination,
                            total_time=village_edge.travel_time + hub_edge.travel_time,
                            weather_exposure=(village_edge.weather_exposure + hub_edge.weather_exposure) / 2.0,
                        )
                    )
            if not routes:
                raise ValueError(f'No route for village {village.id}')
            ordered_routes = sorted(routes, key=lambda item: item.total_time)
            self.routes[village.id] = ordered_routes
            for route in ordered_routes:
                key = (route.village_id, route.hub_id, route.hospital_id)
                if key not in self.route_index:
                    self.route_index[key] = len(self.route_catalog)
                    self.route_catalog.append(route)


class AdvancedSpatialMarketEnvironment:
    def __init__(self, cfg: AdvancedSpatialConfig) -> None:
        self.cfg = cfg
        self.network = AdvancedSpatialNetwork(cfg)
        self.markov = HealthMarkovModel(cfg.markov)
        self.rng = random.Random(cfg.simulation['seed'])
        self.disturbance_process = DisturbanceProcess(cfg.disturbances, self.rng)
        self.state: AdvancedSpatialState | None = None

    def _average_distribution(self, villages: tuple[VillageState, ...]) -> tuple[float, ...]:
        return self.markov.average_distribution(
            distributions=tuple(village.health_distribution for village in villages),
            weights=tuple(village.population for village in villages),
        )

    def _remote_threshold(self) -> float:
        times = sorted(village.traditional_time for village in self.cfg.villages)
        return times[len(times) // 2] if times else 0.0

    def _chronic_indices(self) -> tuple[int, ...]:
        if self.markov.n_states >= 5:
            return (2, 3)
        if self.markov.n_states >= 4:
            return (2,)
        return (max(self.markov.n_states - 2, 0),)

    def reset(self) -> AdvancedSpatialState:
        villages = []
        for village in self.cfg.villages:
            distribution = _normalize(village.state_distribution) if village.state_distribution is not None else self.markov.distribution_from_health(village.base_health)
            health = self.markov.health_from_distribution(distribution)
            villages.append(
                VillageState(
                    id=village.id,
                    name=village.name,
                    population=village.population,
                    health=health,
                    trust=village.trust,
                    demand_weight=village.demand_weight,
                    traditional_time=village.traditional_time,
                    risk_weight=village.risk_weight,
                    health_distribution=distribution,
                    last_demand=0.0,
                    last_served=0.0,
                    last_time=village.traditional_time,
                )
            )
        village_states = tuple(villages)
        avg_distribution = self._average_distribution(village_states)
        chronic_indices = self._chronic_indices()
        chronic_pressure = _weighted_mean([sum(v.health_distribution[idx] for idx in chronic_indices if idx < len(v.health_distribution)) for v in village_states], [v.population for v in village_states])
        urgent_pressure = _weighted_mean([v.health_distribution[-1] for v in village_states], [v.population for v in village_states])
        self.state = AdvancedSpatialState(
            t=0,
            average_health=self.markov.health_from_distribution(avg_distribution),
            network_coverage=0.0,
            average_response_time=sum(v.traditional_time for v in village_states) / len(village_states),
            queue_pressure=0.0,
            enterprise_cash=0.0,
            disturbances=self.disturbance_process.reset(),
            average_state_distribution=avg_distribution,
            villages=village_states,
            fairness_score=1.0,
            coverage_gap=max(self.cfg.constraints['minimum_coverage_ratio'], 0.0),
            enterprise_loss_streak=0,
            enterprise_active=True,
            chronic_pressure=chronic_pressure,
            urgent_pressure=urgent_pressure,
        )
        return self.state

    def _pick_route(
        self,
        village_id: str,
        enterprise: EnterpriseAction,
        medical: MedicalAction,
        hub_remaining: dict[str, float],
        hospital_remaining: dict[str, float],
        weather: float,
        airspace: float,
        device: float,
    ) -> tuple[Route | None, float]:
        best = None
        best_score = None
        best_time = None
        threshold = float(self.cfg.enterprise.get('route_open_threshold', 0.45))
        for route in self.network.routes[village_id]:
            hub_index = self.network.hub_ids.index(route.hub_id)
            hospital_index = self.network.hospital_ids.index(route.hospital_id)
            route_idx = self.network.route_index[(route.village_id, route.hub_id, route.hospital_id)]
            activation = enterprise.route_activation[route_idx] if route_idx < len(enterprise.route_activation) else 1.0
            is_open = activation >= threshold
            route_time = route.total_time * (1.0 + weather * route.weather_exposure) * (1.0 + 0.28 * airspace) * (1.0 + 0.18 * device) * (1.0 - 0.15 * enterprise.reliability_investment) * (1.0 - 0.10 * medical.integration_effort)
            feasible = is_open and hub_remaining[route.hub_id] > EPS and hospital_remaining[route.hospital_id] > EPS
            score = 1.20 * enterprise.hub_allocations[hub_index] + 0.95 * medical.hospital_allocations[hospital_index] + 0.35 * activation - 0.18 * route_time
            if feasible and (best_score is None or score > best_score):
                best = route
                best_score = score
                best_time = route_time
        if best is not None:
            return best, best_time if best_time is not None else best.total_time
        fallback = self.network.routes[village_id][0]
        fallback_time = fallback.total_time * (1.0 + weather * fallback.weather_exposure) * (1.0 + 0.28 * airspace) * (1.0 + 0.18 * device)
        return None, fallback_time

    def step(self, enterprise_action: EnterpriseAction, medical_action: MedicalAction) -> AdvancedStepResult:
        if self.state is None:
            raise RuntimeError('reset first')
        state = self.state
        default_route_activation = tuple(1.0 for _ in self.network.route_catalog)
        route_activation_raw = tuple(float(value) for value in getattr(enterprise_action, 'route_activation', ()) or ())
        if len(route_activation_raw) != len(default_route_activation):
            route_activation = default_route_activation
        else:
            route_activation = tuple(_clip(value, 0.0, 1.0) for value in route_activation_raw)
        enterprise = EnterpriseAction(
            capacity_scale=_clip(enterprise_action.capacity_scale, 0.0, 1.8),
            hub_allocations=_normalize(enterprise_action.hub_allocations),
            price=_clip(enterprise_action.price, self.cfg.enterprise['price_min'], self.cfg.enterprise['price_max']),
            reliability_investment=_clip(enterprise_action.reliability_investment, 0.0, 1.0),
            reserve_capacity_ratio=_clip(getattr(enterprise_action, 'reserve_capacity_ratio', 0.0), 0.0, 0.45),
            route_activation=route_activation,
        )
        medical = MedicalAction(
            referral_acceptance=_clip(medical_action.referral_acceptance, 0.2, 1.0),
            integration_effort=_clip(medical_action.integration_effort, 0.0, 1.0),
            hospital_allocations=_normalize(medical_action.hospital_allocations),
        )
        weather = max(state.disturbances.weather, 0.0)
        surge = max(state.disturbances.medical_surge, 0.0)
        airspace = max(state.disturbances.airspace_control, 0.0)
        device = max(state.disturbances.device_failure, 0.0)
        demand_shock = state.disturbances.demand_shock
        active_enterprise = state.enterprise_active
        if not active_enterprise:
            enterprise = EnterpriseAction(capacity_scale=0.0, hub_allocations=_normalize([1.0 for _ in self.cfg.hubs]), price=enterprise.price, reliability_investment=0.0, reserve_capacity_ratio=0.0, route_activation=tuple(0.0 for _ in self.network.route_catalog))
        weather_factor = 1.0 - self.cfg.environment['weather_service_penalty'] * weather
        airspace_factor = 1.0 - self.cfg.environment['airspace_service_penalty'] * airspace
        device_factor = 1.0 - self.cfg.environment['device_failure_penalty'] * device
        hub_routine_remaining: dict[str, float] = {}
        hub_emergency_remaining: dict[str, float] = {}
        hub_total_available: dict[str, float] = {}
        for share, hub in zip(enterprise.hub_allocations, self.cfg.hubs):
            deployed = self.cfg.enterprise['base_total_capacity'] * enterprise.capacity_scale * share
            reliability_factor = 0.80 + 0.20 * (hub.reliability + enterprise.reliability_investment) / 2.0
            active_factor = 1.0 if active_enterprise else 0.0
            available = max(0.0, min(hub.base_capacity, deployed) * max(weather_factor, 0.0) * max(airspace_factor, 0.0) * max(device_factor + 0.10 * enterprise.reliability_investment, 0.0) * reliability_factor * active_factor)
            reserve = available * enterprise.reserve_capacity_ratio
            hub_total_available[hub.id] = available
            hub_routine_remaining[hub.id] = max(0.0, available - reserve)
            hub_emergency_remaining[hub.id] = max(0.0, reserve)
        hospital_remaining: dict[str, float] = {}
        for share, hospital in zip(medical.hospital_allocations, self.cfg.hospitals):
            integration_factor = 0.70 + 0.30 * (hospital.integration_base + medical.integration_effort) / 2.0
            surge_factor = 1.0 - self.cfg.environment['medical_surge_penalty'] * surge
            device_med_factor = 1.0 - 0.15 * device
            hospital_remaining[hospital.id] = max(0.0, hospital.base_capacity * medical.referral_acceptance * integration_factor * max(surge_factor, 0.0) * max(device_med_factor, 0.0) * (0.85 + 0.15 * share))
        remote_threshold = self._remote_threshold()
        chronic_indices = self._chronic_indices()
        villages_next = []
        total_demand = total_served = total_unmet = adoption_weight = weighted_time = total_time_saving = 0.0
        total_caregiver_time_saved = 0.0
        total_effective_price = 0.0
        total_farmer_extra_cost = 0.0
        observed_demand_total = 0.0
        released_demand_total = 0.0
        local_service_trust_gain_total = 0.0
        served_villages = 0
        total_pop = 0.0
        remote_demand = remote_served = 0.0
        service_rates: list[float] = []
        service_weights: list[float] = []
        task_totals = {task.id: {'demand': 0.0, 'served': 0.0, 'unmet': 0.0, 'value': 0.0, 'cost': 0.0} for task in self.cfg.task_types}
        ordered_tasks = sorted(self.cfg.task_types, key=lambda item: item.priority, reverse=True)
        chronic_pressure_values = []
        urgent_pressure_values = []
        weighted_pops = []
        trust_values = []
        trust_weights = []
        route_stats = {
            key: {'demand': 0.0, 'served': 0.0, 'revenue': 0.0, 'cost': 0.0, 'time_total': 0.0, 'robustness_total': 0.0, 'battery_pressure_total': 0.0}
            for key in self.network.route_index
        }
        for village in state.villages:
            total_pop += village.population
            health_need = max(0.0, 1.0 - village.health)
            chronic_load = sum(village.health_distribution[idx] for idx in chronic_indices if idx < len(village.health_distribution))
            acute_load = village.health_distribution[-1]
            chronic_pressure_values.append(chronic_load)
            urgent_pressure_values.append(acute_load)
            weighted_pops.append(village.population)
            village_demand = village_served = village_unmet = village_adoption_mass = 0.0
            village_time_total = 0.0
            village_time_saving = 0.0
            village_caregiver_time_saved = 0.0
            village_service_signal = 0.0
            village_unmet_signal = 0.0
            village_route_exposure = self.network.routes[village.id][0].weather_exposure
            village_effective_price_paid = 0.0
            village_extra_cost_paid = 0.0
            village_observed_demand = 0.0
            village_released_demand = 0.0
            local_service_demand = 0.0
            local_service_served = 0.0
            is_remote = village.traditional_time >= remote_threshold or village.risk_weight >= 0.65
            for task in ordered_tasks:
                reserve_access = task.id == 'acute_urgent'
                hub_available = {
                    hub_id: hub_routine_remaining.get(hub_id, 0.0) + (hub_emergency_remaining.get(hub_id, 0.0) if reserve_access else 0.0)
                    for hub_id in hub_total_available
                }
                route, route_time = self._pick_route(village.id, enterprise, medical, hub_available, hospital_remaining, weather, airspace, device)
                village_route_exposure = route.weather_exposure if route is not None else self.network.routes[village.id][0].weather_exposure
                time_saving = max(village.traditional_time - route_time, 0.0)
                effective_reimbursement = _clip(self.cfg.environment['reimbursement_rate'] + task.reimbursement_bonus, self.cfg.policy['reimbursement_rate_floor'], 0.95)
                effective_price = enterprise.price * (1.0 - effective_reimbursement)
                local_service_weight = _local_service_multiplier(task.id)
                transport_cost_burden = (
                    local_service_weight
                    * self.cfg.farmer.get('roundtrip_transport_cost_yuan', 100.0)
                    / max(self.cfg.farmer.get('transport_cost_reference_yuan', 100.0), 1.0)
                )
                sample_cost_burden = (
                    local_service_weight
                    * self.cfg.farmer.get('sample_test_out_of_pocket_yuan', 20.0)
                    / max(self.cfg.farmer.get('sample_cost_reference_yuan', 20.0), 1.0)
                )
                time_saving_signal = _clip(
                    time_saving / max(self.cfg.farmer.get('travel_roundtrip_minutes', 60.0) / 60.0, 1e-6),
                    0.0,
                    1.0,
                )
                local_service_bonus = self.cfg.farmer.get('local_sample_point_bonus', 0.35) * local_service_weight
                caregiver_support_signal = _clip(
                    0.50 * chronic_load * task.health_benefit_multiplier
                    + 0.30 * health_need
                    + 0.20 * village.risk_weight * task.risk_weight_multiplier,
                    0.0,
                    1.5,
                )
                caregiver_time_saved = (
                    time_saving
                    * local_service_weight
                    * self.cfg.farmer.get('caregiver_time_share', 0.0)
                    * caregiver_support_signal
                )
                caregiver_time_signal = _clip(
                    caregiver_time_saved / max(self.cfg.farmer.get('travel_roundtrip_minutes', 60.0) / 60.0, 1e-6),
                    0.0,
                    1.0,
                )
                need = village.population * self.cfg.simulation['base_need_rate'] * village.demand_weight * task.demand_share * (0.42 + 0.52 * health_need + 0.28 * chronic_load * task.health_benefit_multiplier + 0.32 * acute_load * task.priority) * (1.0 + demand_shock * task.demand_shock_multiplier) * (1.0 + 0.18 * village.risk_weight * task.risk_weight_multiplier)
                latent_release_share = self.cfg.farmer.get('latent_demand_release_rate', 0.0) * local_service_weight * _clip(
                    0.35 * time_saving_signal
                    + 0.30 * _clip(transport_cost_burden, 0.0, 1.5)
                    + 0.20 * min(village.trust + task.trust_bonus, 1.0)
                    + 0.15 * health_need,
                    0.0,
                    1.0,
                )
                released_need = need * latent_release_share
                latent = (
                    -0.90
                    + self.cfg.farmer['health_need_sensitivity'] * health_need * task.health_benefit_multiplier
                    + self.cfg.farmer['trust_sensitivity'] * min(village.trust + task.trust_bonus, 1.0)
                    + self.cfg.farmer['time_sensitivity'] * time_saving * task.time_sensitivity_multiplier
                    + self.cfg.farmer.get('time_cost_weight', 1.25) * time_saving_signal
                    + self.cfg.farmer.get('caregiver_time_weight', 0.0) * caregiver_time_signal
                    + local_service_bonus
                    + 0.60 * latent_release_share
                    - self.cfg.farmer['price_sensitivity'] * effective_price * task.price_sensitivity_multiplier
                    - self.cfg.farmer.get('out_of_pocket_weight', 0.45) * sample_cost_burden
                    - 0.18 * transport_cost_burden
                    - self.cfg.farmer['learning_cost'] * (1.08 - 0.25 * task.trust_bonus)
                    + 0.35 * village.risk_weight * task.risk_weight_multiplier
                )
                adoption = _clip(_sigmoid(latent), 0.01, 0.995)
                demand = max(0.0, (need + released_need) * adoption)
                served = 0.0
                if route is not None:
                    served = min(demand, hub_available[route.hub_id], hospital_remaining[route.hospital_id])
                    routine_use = min(served, hub_routine_remaining[route.hub_id])
                    reserve_use = max(0.0, served - routine_use)
                    hub_routine_remaining[route.hub_id] -= routine_use
                    hub_emergency_remaining[route.hub_id] -= reserve_use
                    hospital_remaining[route.hospital_id] -= served
                unmet = max(0.0, demand - served)
                if route is not None:
                    route_key = (route.village_id, route.hub_id, route.hospital_id)
                    route_bucket = route_stats[route_key]
                    route_bucket['demand'] += demand
                    route_bucket['served'] += served
                    route_bucket['revenue'] += served * enterprise.price
                    route_bucket['cost'] += served * (
                        self.cfg.enterprise['variable_cost'] * task.enterprise_cost_multiplier
                        + self.cfg.enterprise.get('route_time_cost_weight', 0.22) * route_time
                    )
                    route_bucket['time_total'] += route_time * served
                    robustness = _clip(
                        (1.0 - weather * route.weather_exposure)
                        * (1.0 - 0.35 * airspace)
                        * (1.0 - 0.22 * device)
                        * (0.78 + 0.22 * enterprise.reliability_investment + 0.12 * enterprise.reserve_capacity_ratio),
                        0.0,
                        1.0,
                    )
                    battery_pressure = max(route_time - self.cfg.enterprise.get('battery_time_limit_hours', 3.2), 0.0) / max(self.cfg.enterprise.get('battery_time_limit_hours', 3.2), 1e-6)
                    route_bucket['robustness_total'] += robustness * served
                    route_bucket['battery_pressure_total'] += battery_pressure * served
                task_totals[task.id]['demand'] += demand
                task_totals[task.id]['served'] += served
                task_totals[task.id]['unmet'] += unmet
                task_totals[task.id]['value'] += served * task.medical_value_multiplier
                task_totals[task.id]['cost'] += served * task.enterprise_cost_multiplier
                village_observed_demand += need * adoption
                village_released_demand += released_need * adoption
                village_demand += demand
                village_served += served
                village_unmet += unmet
                village_adoption_mass += adoption * village.population * task.demand_share
                village_time_total += route_time * served
                village_time_saving += time_saving * served * task.time_sensitivity_multiplier
                village_caregiver_time_saved += caregiver_time_saved * served * task.time_sensitivity_multiplier
                village_service_signal += (served / max(village.population, 1)) * task.health_benefit_multiplier
                village_unmet_signal += (unmet / max(village.population, 1)) * task.failure_penalty_multiplier
                village_effective_price_paid += effective_price * served
                village_extra_cost_paid += (
                    self.cfg.farmer.get('out_of_pocket_weight', 0.45) * sample_cost_burden
                    + 0.10 * self.cfg.policy.get('patient_uav_delivery_fee', self.cfg.farmer.get('uav_delivery_out_of_pocket_yuan', 0.0))
                ) * served
                if local_service_weight > EPS:
                    local_service_demand += demand
                    local_service_served += served
                if is_remote:
                    remote_demand += demand
                    remote_served += served
            success = 1.0 if village_demand <= EPS else village_served / village_demand
            if village_demand > EPS:
                service_rates.append(success)
                service_weights.append(max(village_demand, 1.0))
            route_time_avg = village_time_total / village_served if village_served > EPS else self.network.routes[village.id][0].total_time * (1.0 + weather * village_route_exposure)
            local_service_success = _safe_rate(local_service_served, local_service_demand)
            service_signal = _clip(self.cfg.environment['service_health_gain'] * 8.0 * village_service_signal + 0.24 * success, 0.0, 1.8)
            unmet_signal = _clip(self.cfg.environment['unmet_need_penalty'] * 10.0 * village_unmet_signal + 0.28 * (1.0 - success if village_demand > EPS else 0.0), 0.0, 1.8)
            shock_signal = _clip(self.cfg.environment.get('background_health_decay', 0.0) + 0.42 * weather * (1.0 + village_route_exposure) + 0.24 * airspace + 0.18 * device + 0.45 * surge, 0.0, 1.5)
            trust_signal = _clip(village.trust * success + 0.10 * acute_load, 0.0, 1.0)
            next_distribution = self.markov.transition_distribution(current_distribution=village.health_distribution, service_signal=service_signal, unmet_signal=unmet_signal, shock_signal=shock_signal, trust_signal=trust_signal)
            next_health = self.markov.health_from_distribution(next_distribution)
            time_satisfaction = 1.0 - _clip(route_time_avg / max(village.traditional_time, 1.0), 0.0, 1.0)
            local_trust_gain = self.cfg.farmer.get('local_service_trust_gain', 0.06) * _clip(
                0.55 * local_service_success
                + 0.25 * time_satisfaction
                + 0.20 * success,
                0.0,
                1.0,
            )
            retention_trust_gain = self.cfg.farmer.get('big_hospital_avoidance_weight', 0.22) * 0.08 * local_service_success
            next_trust = _clip(village.trust + self.cfg.environment['trust_update_speed'] * (success - 0.45) + local_trust_gain + retention_trust_gain - 0.04 * weather - 0.03 * airspace - 0.02 * device + 0.03 * (village_served / max(village.population, 1)), 0.0, 1.0)
            if village_served > EPS:
                served_villages += 1
                weighted_time += route_time_avg * village_served
                total_time_saving += village_time_saving
                total_caregiver_time_saved += village_caregiver_time_saved
            total_effective_price += village_effective_price_paid
            total_farmer_extra_cost += village_extra_cost_paid
            observed_demand_total += village_observed_demand
            released_demand_total += village_released_demand
            local_service_trust_gain_total += (local_trust_gain + retention_trust_gain) * village.population
            total_demand += village_demand
            total_served += village_served
            total_unmet += village_unmet
            adoption_weight += village_adoption_mass
            trust_values.append(next_trust)
            trust_weights.append(village.population)
            villages_next.append(VillageState(id=village.id, name=village.name, population=village.population, health=next_health, trust=next_trust, demand_weight=village.demand_weight, traditional_time=village.traditional_time, risk_weight=village.risk_weight, health_distribution=next_distribution, last_demand=village_demand, last_served=village_served, last_time=route_time_avg))
        village_states = tuple(villages_next)
        avg_distribution = self._average_distribution(village_states)
        avg_health = self.markov.health_from_distribution(avg_distribution)
        coverage = served_villages / max(len(village_states), 1)
        avg_time = weighted_time / total_served if total_served > EPS else sum(v.traditional_time for v in village_states) / len(village_states)
        queue = 0.0
        hospital_redline_over = 0.0
        if hospital_remaining:
            for hospital_id, remain in hospital_remaining.items():
                base = next(item.base_capacity for item in self.cfg.hospitals if item.id == hospital_id)
                utilization = 1.0 - min(remain / max(base, EPS), 1.0)
                queue += utilization
                hospital_redline_over += max(utilization - self.cfg.constraints['hospital_capacity_redline'], 0.0)
            queue /= len(hospital_remaining)
            hospital_redline_over /= len(hospital_remaining)
        fairness_score = _clip(1.0 - _weighted_gini(service_rates, service_weights), 0.0, 1.0)
        coverage_gap = max(self.cfg.constraints['minimum_coverage_ratio'] - coverage, 0.0)
        underserved_share = sum(1 for rate in service_rates if rate < self.cfg.constraints['underserved_threshold']) / max(len(village_states), 1)
        remote_service_rate = 1.0 if remote_demand <= EPS else remote_served / remote_demand
        chronic_management_rate = _safe_rate(task_totals['chronic_followup']['served'], task_totals['chronic_followup']['demand'])
        followup_completion_rate = chronic_management_rate
        screening_completion_rate = _clip(
            _safe_rate(task_totals['sample_transport']['served'], task_totals['sample_transport']['demand'])
            * (0.72 + 0.28 * medical.referral_acceptance)
            * (0.74 + 0.26 * medical.integration_effort),
            0.0,
            1.0,
        )
        township_trust_index = _clip(_weighted_mean(trust_values, trust_weights), 0.0, 1.0)
        fixed_subsidy = self.cfg.environment['fixed_subsidy'] + self.cfg.policy['fixed_subsidy']
        per_service_subsidy = self.cfg.policy['per_service_subsidy'] * total_served
        performance_subsidy = self.cfg.policy['performance_subsidy_per_coverage'] * coverage + self.cfg.policy['fairness_subsidy_per_point'] * fairness_score + self.cfg.policy['chronic_management_subsidy'] * chronic_management_rate
        remote_service_subsidy = self.cfg.policy['remote_service_subsidy'] * remote_served + self.cfg.policy.get('local_sampling_subsidy', 0.0) * task_totals['sample_transport']['served']
        total_policy_cost_base = fixed_subsidy + per_service_subsidy + performance_subsidy + remote_service_subsidy
        coverage_penalty = self.cfg.constraints['coverage_shortfall_penalty'] * coverage_gap
        fairness_penalty = self.cfg.constraints['fairness_penalty'] * max(self.cfg.constraints['fairness_floor'] - fairness_score, 0.0)
        underserved_penalty = self.cfg.constraints['underserved_penalty'] * underserved_share
        remote_penalty = self.cfg.constraints['remote_floor_penalty'] * max(self.cfg.constraints['remote_service_floor'] - remote_service_rate, 0.0)
        hospital_penalty = self.cfg.constraints['hospital_redline_penalty'] * hospital_redline_over
        average_cost_multiplier = sum(values['cost'] for values in task_totals.values()) / max(total_served, EPS) if total_served > EPS else 1.0
        average_value_multiplier = sum(values['value'] for values in task_totals.values()) / max(total_served, EPS) if total_served > EPS else 1.0
        severe_outflow_rate = float(avg_distribution[-1]) if avg_distribution else 0.0
        managed_chronic_share = float(avg_distribution[2]) if len(avg_distribution) > 2 else 0.0
        unmanaged_chronic_share = float(avg_distribution[3]) if len(avg_distribution) > 3 else 0.0
        acute_urgent_demand = task_totals['acute_urgent']['demand']
        acute_urgent_served = task_totals['acute_urgent']['served']
        urgent_task_subsidy = self.cfg.policy.get('urgent_task_subsidy', 0.0) * acute_urgent_served
        total_policy_cost = total_policy_cost_base + urgent_task_subsidy
        specialist_bypass_rate = 1.0 - _safe_rate(acute_urgent_served, acute_urgent_demand) if acute_urgent_demand > EPS else severe_outflow_rate
        bp_glucose_control_rate = _clip(
            _safe_rate(managed_chronic_share, managed_chronic_share + unmanaged_chronic_share)
            * (0.70 + 0.30 * followup_completion_rate)
            * (0.78 + 0.22 * medical.integration_effort),
            0.0,
            1.0,
        )
        avoidable_hospitalization_rate = _clip(
            0.45 * unmanaged_chronic_share
            + 0.30 * severe_outflow_rate
            + 0.15 * (1.0 - followup_completion_rate)
            + 0.10 * specialist_bypass_rate,
            0.0,
            1.0,
        )
        severe_outflow_equivalent = task_totals['acute_urgent']['unmet'] + total_pop * severe_outflow_rate * 0.25
        county_retention_rate = _safe_rate(total_served, total_served + severe_outflow_equivalent)
        post_exam_management_rate = _clip(
            screening_completion_rate
            * (0.62 + 0.38 * followup_completion_rate)
            * (0.70 + 0.30 * county_retention_rate)
            * (0.82 + 0.18 * township_trust_index),
            0.0,
            1.0,
        )
        latent_demand_release_share = _safe_rate(released_demand_total, observed_demand_total + released_demand_total)
        big_hospital_diversion_reduction = _clip(
            county_retention_rate
            * (0.58 + 0.24 * township_trust_index + 0.18 * screening_completion_rate)
            * (1.0 - 0.65 * specialist_bypass_rate)
            + self.cfg.farmer.get('big_hospital_avoidance_weight', 0.22) * 0.20,
            0.0,
            1.0,
        )
        risk_stratification_coverage = _clip(
            0.48 * screening_completion_rate
            + 0.32 * followup_completion_rate
            + 0.20 * medical.integration_effort,
            0.0,
            1.0,
        )
        deployed_capacity = self.cfg.enterprise['base_total_capacity'] * enterprise.capacity_scale
        load_factor = _clip(0.0 if deployed_capacity <= EPS else total_served / deployed_capacity, 0.0, 1.0)
        route_profitability = 0.0 if total_served <= EPS else sum(bucket['revenue'] - bucket['cost'] for bucket in route_stats.values()) / total_served
        battery_or_charge_constraint = 0.0 if total_served <= EPS else sum(bucket['battery_pressure_total'] for bucket in route_stats.values()) / total_served
        weather_robust_dispatch_score = 0.0 if total_served <= EPS else sum(bucket['robustness_total'] for bucket in route_stats.values()) / total_served
        hub_open_count = sum(1 for hub_id in hub_total_available if hub_total_available[hub_id] > 0.05 * max(next(item.base_capacity for item in self.cfg.hubs if item.id == hub_id), 1.0))
        hub_open_ratio = _safe_rate(hub_open_count, len(self.cfg.hubs))
        active_route_count = float(sum(1 for value in enterprise.route_activation if value >= self.cfg.enterprise.get('route_open_threshold', 0.45)))
        route_activation_share = _safe_rate(active_route_count, len(self.network.route_catalog))
        empty_flight_cost = self.cfg.enterprise['empty_flight_cost'] * deployed_capacity * max(1.0 - load_factor, 0.0)
        contract_revenue = self.cfg.enterprise['contract_fixed_revenue']
        stability_scale = max(1.0 - state.enterprise_loss_streak / max(float(self.cfg.constraints['enterprise_exit_periods']), 1.0), 0.0)
        contract_stability_bonus = self.cfg.enterprise['contract_stability_bonus'] * stability_scale
        reliability_cost = self.cfg.enterprise['reliability_unit_cost'] * enterprise.reliability_investment
        reserve_cost = self.cfg.enterprise.get('reserve_cost_weight', 8.0) * enterprise.reserve_capacity_ratio
        battery_penalty = self.cfg.enterprise.get('battery_penalty_weight', 55.0) * battery_or_charge_constraint
        weather_robust_bonus = self.cfg.enterprise.get('weather_robust_bonus_weight', 20.0) * weather_robust_dispatch_score
        local_contract_service_bonus = self.cfg.enterprise.get('local_contract_service_bonus', 10.0) * _clip(
            0.55 * screening_completion_rate + 0.25 * remote_service_rate + 0.20 * chronic_management_rate,
            0.0,
            1.0,
        )
        sample_turnaround_penalty = self.cfg.enterprise.get('sample_turnaround_penalty', 6.0) * _clip(
            (1.0 - screening_completion_rate)
            + 0.35 * battery_or_charge_constraint
            + 0.15 * weather
            + 0.10 * airspace,
            0.0,
            2.0,
        )
        operator_capacity = max(
            self.cfg.enterprise.get('operator_count', 1.0) * self.cfg.enterprise.get('lines_per_operator_limit', max(len(self.network.route_catalog), 1.0)),
            1.0,
        )
        operator_load_ratio = active_route_count / operator_capacity
        dispatch_scale_bonus = self.cfg.enterprise.get('dispatch_scale_efficiency', 0.0) * min(operator_load_ratio, 1.0)
        dispatch_scale_penalty = self.cfg.enterprise.get('dispatch_scale_efficiency', 0.0) * 2.0 * max(operator_load_ratio - 1.0, 0.0)
        traceability_score = _clip(0.55 * screening_completion_rate + 0.45 * post_exam_management_rate, 0.0, 1.0)
        traceability_bonus = self.cfg.enterprise.get('traceability_bonus', 0.0) * traceability_score
        sample_transport_share = _safe_rate(task_totals['sample_transport']['served'], total_served)
        data_upload_penalty = self.cfg.enterprise.get('data_upload_penalty', 0.0) * (1.0 - traceability_score) * (0.55 + 0.45 * sample_transport_share)
        urgent_task_premium = enterprise.price * acute_urgent_served * max(self.cfg.enterprise.get('urgent_task_premium_multiplier', 1.0) - 1.0, 0.0)
        airspace_compliance_cost = self.cfg.enterprise.get('airspace_compliance_cost', 0.0) * airspace * max(active_route_count, 1.0)
        loss_streak_penalty = self.cfg.enterprise['loss_streak_penalty'] * max(state.enterprise_loss_streak - 1, 0)
        queue_penalty = self.cfg.medical['queue_penalty_weight'] * total_unmet
        idle_penalty = self.cfg.medical['idle_penalty_weight'] * max(0.15 - queue, 0.0)
        chronic_management_gain = self.cfg.medical['chronic_management_value'] * chronic_management_rate
        severe_outflow_penalty = self.cfg.medical['severe_outflow_penalty'] * severe_outflow_rate
        followup_completion_gain = self.cfg.medical.get('followup_completion_value', 38.0) * followup_completion_rate
        county_retention_gain = self.cfg.medical.get('county_retention_value', 28.0) * county_retention_rate
        bp_glucose_control_gain = self.cfg.medical.get('bp_glucose_control_value', 24.0) * bp_glucose_control_rate
        post_exam_management_gain = self.cfg.medical.get('post_exam_management_value', 26.0) * post_exam_management_rate
        screening_completion_gain = self.cfg.medical.get('screening_completion_value', 18.0) * screening_completion_rate
        township_trust_gain = self.cfg.medical.get('township_trust_value', 16.0) * township_trust_index
        risk_stratification_gain = self.cfg.medical.get('risk_stratification_value', 12.0) * risk_stratification_coverage
        staff_shortage_penalty = self.cfg.medical.get('staff_shortage_penalty', 22.0) * max(queue - 0.55, 0.0) * (0.72 + 0.28 * screening_completion_rate)
        specialist_bypass_penalty = self.cfg.medical.get('specialist_bypass_penalty', 85.0) * specialist_bypass_rate
        avoidable_hospitalization_penalty = self.cfg.medical.get('avoidable_hospitalization_penalty', 95.0) * avoidable_hospitalization_rate
        severe_outflow_loss = self.cfg.farmer['severe_outflow_loss'] * severe_outflow_rate
        caregiver_time_benefit = self.cfg.farmer['time_value'] * self.cfg.farmer.get('caregiver_time_value_multiplier', 0.0) * total_caregiver_time_saved
        weather_interrupt_cost = self.cfg.enterprise['interruption_cost'] * self.cfg.constraints['weather_interrupt_cost_weight'] * weather * total_demand
        airspace_interrupt_cost = self.cfg.enterprise['interruption_cost'] * self.cfg.constraints['airspace_interrupt_cost_weight'] * airspace * total_demand
        device_interrupt_cost = self.cfg.enterprise['interruption_cost'] * self.cfg.constraints['device_interrupt_cost_weight'] * device * total_demand
        enterprise_reward = total_served * enterprise.price + contract_revenue + total_policy_cost + contract_stability_bonus + weather_robust_bonus + local_contract_service_bonus + urgent_task_premium + dispatch_scale_bonus + traceability_bonus - total_served * self.cfg.enterprise['variable_cost'] * average_cost_multiplier - len(self.cfg.hubs) * self.cfg.enterprise['fixed_hub_cost'] * (0.4 + 0.6 * enterprise.capacity_scale) - empty_flight_cost - weather_interrupt_cost - airspace_interrupt_cost - device_interrupt_cost - reliability_cost - reserve_cost - battery_penalty - sample_turnaround_penalty - data_upload_penalty - dispatch_scale_penalty - airspace_compliance_cost - loss_streak_penalty - 0.20 * remote_penalty
        medical_reward = self.cfg.medical['value_per_case'] * total_served * average_value_multiplier + chronic_management_gain + followup_completion_gain + county_retention_gain + bp_glucose_control_gain + post_exam_management_gain + screening_completion_gain + township_trust_gain + risk_stratification_gain - self.cfg.medical['coordination_cost'] * medical.integration_effort - queue_penalty - idle_penalty - staff_shortage_penalty - 0.35 * enterprise.price * total_served - severe_outflow_penalty - specialist_bypass_penalty - avoidable_hospitalization_penalty - 0.15 * hospital_penalty - 0.10 * remote_penalty
        farmer_reward = self.cfg.farmer['time_value'] * total_time_saving + caregiver_time_benefit + self.cfg.farmer['health_gain_weight'] * (avg_health - state.average_health) + 12.0 * township_trust_index + 8.0 * latent_demand_release_share + 10.0 * big_hospital_diversion_reduction - total_effective_price - total_farmer_extra_cost - self.cfg.farmer['unmet_penalty_weight'] * (total_unmet / max(total_pop, 1)) - severe_outflow_loss - 0.10 * remote_penalty
        next_enterprise_cash = state.enterprise_cash + enterprise_reward
        next_loss_streak = state.enterprise_loss_streak + 1 if enterprise_reward < 0 else 0
        enterprise_exit_triggered = state.enterprise_active and next_enterprise_cash < self.cfg.constraints['enterprise_exit_cash_threshold'] and next_loss_streak >= self.cfg.constraints['enterprise_exit_periods']
        next_enterprise_active = state.enterprise_active and not enterprise_exit_triggered
        task_metrics = {task_id: {'demand': values['demand'], 'served': values['served'], 'unmet': values['unmet'], 'service_rate': 0.0 if values['demand'] <= EPS else values['served'] / values['demand']} for task_id, values in task_totals.items()}
        next_state = AdvancedSpatialState(
            t=state.t + 1,
            average_health=avg_health,
            network_coverage=coverage,
            average_response_time=avg_time,
            queue_pressure=queue,
            enterprise_cash=next_enterprise_cash,
            disturbances=self.disturbance_process.step(state.disturbances),
            average_state_distribution=avg_distribution,
            villages=village_states,
            fairness_score=fairness_score,
            coverage_gap=coverage_gap,
            enterprise_loss_streak=next_loss_streak,
            enterprise_active=next_enterprise_active,
            chronic_pressure=_weighted_mean(chronic_pressure_values, weighted_pops),
            urgent_pressure=_weighted_mean(urgent_pressure_values, weighted_pops),
        )
        self.state = next_state
        return AdvancedStepResult(
            state=next_state,
            enterprise_reward=enterprise_reward,
            medical_reward=medical_reward,
            farmer_reward=farmer_reward,
            metrics=AdvancedStepMetrics(
                total_demand=total_demand,
                total_served=total_served,
                total_unmet=total_unmet,
                average_adoption=adoption_weight / max(total_pop, 1),
                network_coverage=coverage,
                average_response_time=avg_time,
                fairness_score=fairness_score,
                minimum_coverage_gap=coverage_gap,
                underserved_share=underserved_share,
                remote_service_rate=remote_service_rate,
                hospital_redline_over=hospital_redline_over,
                enterprise_active=next_enterprise_active,
                average_effective_price=total_effective_price / max(total_served, 1.0),
                load_factor=load_factor,
                chronic_management_rate=chronic_management_rate,
                severe_outflow_rate=severe_outflow_rate,
                followup_completion_rate=followup_completion_rate,
                avoidable_hospitalization_rate=avoidable_hospitalization_rate,
                county_retention_rate=county_retention_rate,
                specialist_bypass_rate=specialist_bypass_rate,
                bp_glucose_control_rate=bp_glucose_control_rate,
                route_profitability=route_profitability,
                reserve_capacity_ratio=enterprise.reserve_capacity_ratio,
                battery_or_charge_constraint=battery_or_charge_constraint,
                weather_robust_dispatch_score=weather_robust_dispatch_score,
                hub_open_ratio=hub_open_ratio,
                route_activation_share=route_activation_share,
                post_exam_management_rate=post_exam_management_rate,
                screening_completion_rate=screening_completion_rate,
                township_trust_index=township_trust_index,
                latent_demand_release_share=latent_demand_release_share,
                big_hospital_diversion_reduction=big_hospital_diversion_reduction,
                risk_stratification_coverage=risk_stratification_coverage,
                caregiver_time_release_hours=total_caregiver_time_saved,
                task_metrics=task_metrics,
                policy_metrics={
                    'fixed_subsidy': fixed_subsidy,
                    'per_service_subsidy': per_service_subsidy,
                    'performance_subsidy': performance_subsidy,
                    'remote_service_subsidy': remote_service_subsidy,
                    'policy_cost_total': total_policy_cost,
                    'contract_revenue': contract_revenue,
                    'contract_stability_bonus': contract_stability_bonus,
                    'empty_flight_cost': empty_flight_cost,
                    'loss_streak_penalty': loss_streak_penalty,
                    'chronic_management_gain': chronic_management_gain,
                    'severe_outflow_penalty': severe_outflow_penalty,
                    'followup_completion_gain': followup_completion_gain,
                    'county_retention_gain': county_retention_gain,
                    'bp_glucose_control_gain': bp_glucose_control_gain,
                    'post_exam_management_gain': post_exam_management_gain,
                    'screening_completion_gain': screening_completion_gain,
                    'township_trust_gain': township_trust_gain,
                    'risk_stratification_gain': risk_stratification_gain,
                    'specialist_bypass_penalty': specialist_bypass_penalty,
                    'avoidable_hospitalization_penalty': avoidable_hospitalization_penalty,
                    'staff_shortage_penalty': staff_shortage_penalty,
                    'reserve_cost': reserve_cost,
                    'battery_penalty': battery_penalty,
                    'weather_robust_bonus': weather_robust_bonus,
                    'local_contract_service_bonus': local_contract_service_bonus,
                    'sample_turnaround_penalty': sample_turnaround_penalty,
                    'urgent_task_subsidy': urgent_task_subsidy,
                    'urgent_task_premium': urgent_task_premium,
                    'dispatch_scale_bonus': dispatch_scale_bonus,
                    'dispatch_scale_penalty': dispatch_scale_penalty,
                    'traceability_score': traceability_score,
                    'traceability_bonus': traceability_bonus,
                    'data_upload_penalty': data_upload_penalty,
                    'operator_load_ratio': operator_load_ratio,
                    'airspace_compliance_cost': airspace_compliance_cost,
                    'local_service_trust_gain': _safe_rate(local_service_trust_gain_total, max(total_pop, 1.0)),
                    'caregiver_time_release_hours': total_caregiver_time_saved,
                    'caregiver_time_benefit': caregiver_time_benefit,
                    'hub_open_decision': {
                        hub_id: 1.0 if hub_total_available[hub_id] > 0.05 * max(next(item.base_capacity for item in self.cfg.hubs if item.id == hub_id), 1.0) else 0.0
                        for hub_id in hub_total_available
                    },
                    'active_route_count': active_route_count,
                },
                constraint_flags={'coverage_floor_breached': coverage_gap > EPS, 'fairness_floor_breached': fairness_score + EPS < self.cfg.constraints['fairness_floor'], 'hospital_redline_breached': hospital_redline_over > EPS, 'remote_floor_breached': remote_service_rate + EPS < self.cfg.constraints['remote_service_floor'], 'enterprise_exit_triggered': enterprise_exit_triggered},
                interruptions={'weather': weather, 'medical_surge': surge, 'demand_shock': demand_shock, 'airspace_control': airspace, 'device_failure': device},
            ),
        )


class HeuristicEnterprisePolicy:
    def __init__(self, cfg: AdvancedSpatialConfig) -> None:
        self.cfg = cfg
        self.network = AdvancedSpatialNetwork(cfg)

    def act(self, state: AdvancedSpatialState) -> EnterpriseAction:
        village_map = {v.id: v for v in state.villages}
        midpoint = len(state.villages) // 2
        left = sum(v.population * (0.45 + v.demand_weight) * (1.0 - v.health + 0.18 * v.risk_weight) for v in state.villages[: midpoint + 1])
        right = sum(v.population * (0.45 + v.demand_weight) * (1.0 - v.health + 0.18 * v.risk_weight) for v in state.villages[midpoint:])
        total = left + right
        allocations = (left / total, right / total) if total > EPS else tuple(1.0 / len(self.cfg.hubs) for _ in self.cfg.hubs)
        capacity_scale = _clip(0.82 + 0.32 * state.queue_pressure + 0.22 * state.urgent_pressure + 0.18 * state.coverage_gap - 0.08 * max(state.disturbances.weather, 0.0) - 0.06 * max(state.disturbances.airspace_control, 0.0), 0.30, 1.65)
        price = _clip(self.cfg.enterprise['price_default'] + 0.30 * max(state.disturbances.weather, 0.0) + 0.18 * max(state.disturbances.airspace_control, 0.0) + 0.20 * state.queue_pressure - 0.22 * state.fairness_score - 0.15 * state.network_coverage, self.cfg.enterprise['price_min'], self.cfg.enterprise['price_max'])
        reliability = _clip(self.cfg.enterprise['reliability_default'] + 0.32 * max(state.disturbances.weather, 0.0) + 0.20 * max(state.disturbances.device_failure, 0.0), 0.05, 1.0)
        reserve_capacity_ratio = _clip(0.08 + 0.22 * max(state.disturbances.weather, 0.0) + 0.18 * state.urgent_pressure + 0.10 * state.queue_pressure, 0.04, 0.35)
        average_population = sum(v.population for v in state.villages) / max(len(state.villages), 1)
        route_activation = []
        for route in self.network.route_catalog:
            village = village_map[route.village_id]
            population_density_signal = _clip(village.population / max(average_population, 1.0) - 1.0, -1.0, 2.0)
            activation = _clip(
                0.62
                + 0.22 * village.risk_weight
                + 0.16 * (1.0 - village.health)
                + 0.08 * state.urgent_pressure
                + self.cfg.enterprise.get('population_density_route_scale', 0.0) * population_density_signal
                - 0.10 * route.total_time / max(village.traditional_time, 1.0)
                - 0.08 * max(state.disturbances.weather, 0.0) * route.weather_exposure,
                0.05,
                1.0,
            )
            route_activation.append(activation)
        if not state.enterprise_active:
            capacity_scale = 0.0
            reserve_capacity_ratio = 0.0
            route_activation = [0.0 for _ in route_activation]
        return EnterpriseAction(capacity_scale=capacity_scale, hub_allocations=allocations, price=price, reliability_investment=reliability, reserve_capacity_ratio=reserve_capacity_ratio, route_activation=tuple(route_activation))


class HeuristicMedicalPolicy:
    def __init__(self, cfg: AdvancedSpatialConfig) -> None:
        self.cfg = cfg

    def act(self, state: AdvancedSpatialState) -> MedicalAction:
        capacities = [hospital.base_capacity for hospital in self.cfg.hospitals]
        total = sum(capacities)
        allocations = tuple(cap / total for cap in capacities)
        referral = _clip(self.cfg.medical['referral_default'] + 0.20 * state.network_coverage + 0.12 * state.urgent_pressure - 0.18 * max(state.disturbances.medical_surge, 0.0), 0.20, 1.0)
        integration = _clip(self.cfg.medical['integration_default'] + 0.28 * state.queue_pressure + 0.16 * state.chronic_pressure + 0.10 * state.coverage_gap + 0.08 * max(state.disturbances.medical_surge, 0.0), 0.0, 1.0)
        return MedicalAction(referral_acceptance=referral, integration_effort=integration, hospital_allocations=allocations)


def run_advanced_spatial_episode(config_path: str | Path, scenario_name: str | None = None) -> AdvancedEpisodeSummary:
    cfg = apply_advanced_spatial_scenario(load_advanced_spatial_config(config_path), scenario_name)
    env = AdvancedSpatialMarketEnvironment(cfg)
    enterprise_policy = HeuristicEnterprisePolicy(cfg)
    medical_policy = HeuristicMedicalPolicy(cfg)
    state = env.reset()
    total_enterprise_reward = total_medical_reward = total_farmer_reward = 0.0
    adoption_series = []
    service_series = []
    coverage_series = []
    fairness_series = []
    coverage_gap_series = []
    remote_service_series = []
    load_factor_series = []
    chronic_management_series = []
    severe_outflow_series = []
    followup_completion_series = []
    avoidable_hospitalization_series = []
    county_retention_series = []
    specialist_bypass_series = []
    bp_glucose_control_series = []
    route_profitability_series = []
    reserve_capacity_series = []
    battery_constraint_series = []
    weather_robust_series = []
    hub_open_ratio_series = []
    route_activation_share_series = []
    post_exam_management_series = []
    screening_completion_series = []
    township_trust_series = []
    latent_release_series = []
    big_hospital_diversion_series = []
    risk_stratification_series = []
    total_policy_cost = 0.0
    breach_count = 0
    task_demand_totals = {task.id: 0.0 for task in cfg.task_types}
    task_served_totals = {task.id: 0.0 for task in cfg.task_types}
    enterprise_exit = False
    for _ in range(cfg.simulation['periods']):
        result = env.step(enterprise_policy.act(state), medical_policy.act(state))
        state = result.state
        total_enterprise_reward += result.enterprise_reward
        total_medical_reward += result.medical_reward
        total_farmer_reward += result.farmer_reward
        adoption_series.append(result.metrics.average_adoption)
        service_series.append(0.0 if result.metrics.total_demand <= EPS else result.metrics.total_served / result.metrics.total_demand)
        coverage_series.append(state.network_coverage)
        fairness_series.append(result.metrics.fairness_score)
        coverage_gap_series.append(result.metrics.minimum_coverage_gap)
        remote_service_series.append(result.metrics.remote_service_rate)
        load_factor_series.append(result.metrics.load_factor)
        chronic_management_series.append(result.metrics.chronic_management_rate)
        severe_outflow_series.append(result.metrics.severe_outflow_rate)
        followup_completion_series.append(result.metrics.followup_completion_rate)
        avoidable_hospitalization_series.append(result.metrics.avoidable_hospitalization_rate)
        county_retention_series.append(result.metrics.county_retention_rate)
        specialist_bypass_series.append(result.metrics.specialist_bypass_rate)
        bp_glucose_control_series.append(result.metrics.bp_glucose_control_rate)
        route_profitability_series.append(result.metrics.route_profitability)
        reserve_capacity_series.append(result.metrics.reserve_capacity_ratio)
        battery_constraint_series.append(result.metrics.battery_or_charge_constraint)
        weather_robust_series.append(result.metrics.weather_robust_dispatch_score)
        hub_open_ratio_series.append(result.metrics.hub_open_ratio)
        route_activation_share_series.append(result.metrics.route_activation_share)
        post_exam_management_series.append(result.metrics.post_exam_management_rate)
        screening_completion_series.append(result.metrics.screening_completion_rate)
        township_trust_series.append(result.metrics.township_trust_index)
        latent_release_series.append(result.metrics.latent_demand_release_share)
        big_hospital_diversion_series.append(result.metrics.big_hospital_diversion_reduction)
        risk_stratification_series.append(result.metrics.risk_stratification_coverage)
        total_policy_cost += result.metrics.policy_metrics.get('policy_cost_total', 0.0)
        breach_count += sum(1 for flag in result.metrics.constraint_flags.values() if flag)
        for task_id, values in result.metrics.task_metrics.items():
            task_demand_totals[task_id] = task_demand_totals.get(task_id, 0.0) + values['demand']
            task_served_totals[task_id] = task_served_totals.get(task_id, 0.0) + values['served']
        enterprise_exit = enterprise_exit or (not result.metrics.enterprise_active)
    return AdvancedEpisodeSummary(
        periods=state.t,
        final_average_health=state.average_health,
        final_network_coverage=state.network_coverage,
        final_average_response_time=state.average_response_time,
        total_enterprise_reward=total_enterprise_reward,
        total_medical_reward=total_medical_reward,
        total_farmer_reward=total_farmer_reward,
        average_adoption=_mean(adoption_series),
        average_service_rate=_mean(service_series),
        average_network_coverage=_mean(coverage_series),
        state_names_cn=cfg.markov.state_names_cn,
        final_state_distribution=state.average_state_distribution,
        average_fairness=_mean(fairness_series),
        average_coverage_gap=_mean(coverage_gap_series),
        average_remote_service_rate=_mean(remote_service_series),
        policy_cost=total_policy_cost,
        constraint_breach_count=breach_count,
        enterprise_exit=enterprise_exit,
        average_load_factor=_mean(load_factor_series),
        average_chronic_management_rate=_mean(chronic_management_series),
        average_severe_outflow_rate=_mean(severe_outflow_series),
        average_followup_completion_rate=_mean(followup_completion_series),
        average_avoidable_hospitalization_rate=_mean(avoidable_hospitalization_series),
        average_county_retention_rate=_mean(county_retention_series),
        average_specialist_bypass_rate=_mean(specialist_bypass_series),
        average_bp_glucose_control_rate=_mean(bp_glucose_control_series),
        average_route_profitability=_mean(route_profitability_series),
        average_reserve_capacity_ratio=_mean(reserve_capacity_series),
        average_battery_or_charge_constraint=_mean(battery_constraint_series),
        average_weather_robust_dispatch_score=_mean(weather_robust_series),
        average_hub_open_ratio=_mean(hub_open_ratio_series),
        average_route_activation_share=_mean(route_activation_share_series),
        average_post_exam_management_rate=_mean(post_exam_management_series),
        average_screening_completion_rate=_mean(screening_completion_series),
        average_township_trust_index=_mean(township_trust_series),
        average_latent_demand_release_share=_mean(latent_release_series),
        average_big_hospital_diversion_reduction=_mean(big_hospital_diversion_series),
        average_risk_stratification_coverage=_mean(risk_stratification_series),
        task_service_rates={task_id: 0.0 if task_demand_totals.get(task_id, 0.0) <= EPS else task_served_totals.get(task_id, 0.0) / task_demand_totals[task_id] for task_id in task_demand_totals},
    )
