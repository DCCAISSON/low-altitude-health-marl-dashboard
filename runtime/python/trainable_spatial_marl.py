from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

from .advanced_spatial import (
    EPS,
    AdvancedSpatialNetwork,
    EnterpriseAction,
    HeuristicEnterprisePolicy,
    HeuristicMedicalPolicy,
    MedicalAction,
    AdvancedSpatialConfig as SpatialConfig,
    AdvancedEpisodeSummary as SpatialEpisodeSummary,
    AdvancedSpatialMarketEnvironment as SpatialMarketEnvironment,
    AdvancedSpatialState as SpatialState,
    apply_advanced_spatial_scenario,
    apply_runtime_overrides,
    clone_advanced_spatial_config,
    config_snapshot,
    load_advanced_spatial_config,
    _sigmoid,
)
from .outcome_mapping import estimate_long_horizon_outcomes


def _softmax(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    bounded = [max(min(float(value), 20.0), -20.0) for value in values]
    anchor = max(bounded)
    exps = [math.exp(value - anchor) for value in bounded]
    total = sum(exps)
    if total <= EPS:
        size = len(exps)
        return tuple(1.0 / size for _ in range(size))
    return tuple(value / total for value in exps)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    center = _mean(values)
    variance = sum((value - center) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 1e-8))


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= EPS:
        return _mean(values)
    return sum(value * weight for value, weight in zip(values, weights)) / total


def _normalize_advantages(values: list[float]) -> list[float]:
    center = _mean(values)
    scale = max(_std(values), 1e-6)
    return [max(min((value - center) / scale, 6.0), -6.0) for value in values]


def _gae_advantages(rewards: list[float], values: list[float], gamma: float, gae_lambda: float) -> tuple[list[float], list[float]]:
    advantages = [0.0 for _ in rewards]
    returns = [0.0 for _ in rewards]
    gae = 0.0
    next_value = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        delta = rewards[idx] + gamma * next_value - values[idx]
        gae = delta + gamma * gae_lambda * gae
        advantages[idx] = gae
        returns[idx] = gae + values[idx]
        next_value = values[idx]
    return advantages, returns


def feature_names(n_states: int) -> list[str]:
    return [
        'bias',
        'avg_health',
        'coverage',
        'response_time_norm',
        'queue_pressure',
        'weather',
        'medical_surge',
        'demand_shock',
        'airspace_control',
        'device_failure',
        'avg_trust',
        'avg_risk',
        'avg_demand_weight_norm',
        'avg_traditional_time_norm',
        'demand_pressure',
        'service_pressure',
        'fairness_score',
        'coverage_gap',
        'enterprise_active',
        'loss_streak_norm',
        'chronic_pressure',
        'urgent_pressure',
    ] + [f'markov_share_s{i + 1}' for i in range(n_states)]


def extract_features(state: SpatialState) -> list[float]:
    populations = [float(v.population) for v in state.villages]
    total_pop = max(sum(populations), 1.0)
    avg_trust = _weighted_mean([v.trust for v in state.villages], populations)
    avg_risk = _weighted_mean([v.risk_weight for v in state.villages], populations)
    avg_demand_weight = _weighted_mean([v.demand_weight for v in state.villages], populations)
    avg_traditional_time = _weighted_mean([v.traditional_time for v in state.villages], populations)
    total_last_demand = sum(v.last_demand for v in state.villages)
    total_last_served = sum(v.last_served for v in state.villages)
    features = [
        1.0,
        state.average_health,
        state.network_coverage,
        state.average_response_time / max(avg_traditional_time, 1.0),
        state.queue_pressure,
        max(state.disturbances.weather, 0.0),
        max(state.disturbances.medical_surge, 0.0),
        state.disturbances.demand_shock,
        max(state.disturbances.airspace_control, 0.0),
        max(state.disturbances.device_failure, 0.0),
        avg_trust,
        avg_risk,
        avg_demand_weight / 1.5,
        avg_traditional_time / 10.0,
        total_last_demand / total_pop,
        total_last_served / total_pop,
        state.fairness_score,
        state.coverage_gap,
        1.0 if state.enterprise_active else 0.0,
        min(state.enterprise_loss_streak / 5.0, 1.0),
        state.chronic_pressure,
        state.urgent_pressure,
    ]
    features.extend(float(item) for item in state.average_state_distribution)
    return features


@dataclass(slots=True)
class PolicyStep:
    features: list[float]
    mean: list[float]
    raw_action: list[float]


@dataclass(slots=True)
class ReplayStep:
    t: int
    average_health: float
    network_coverage: float
    average_response_time: float
    queue_pressure: float
    disturbances: dict[str, float]
    enterprise_action: dict[str, Any]
    medical_action: dict[str, Any]
    enterprise_reward: float
    medical_reward: float
    farmer_reward: float
    average_adoption: float
    total_demand: float
    total_served: float
    total_unmet: float
    load_factor: float
    chronic_management_rate: float
    severe_outflow_rate: float
    followup_completion_rate: float
    avoidable_hospitalization_rate: float
    county_retention_rate: float
    specialist_bypass_rate: float
    bp_glucose_control_rate: float
    route_profitability: float
    reserve_capacity_ratio: float
    battery_or_charge_constraint: float
    weather_robust_dispatch_score: float
    hub_open_ratio: float
    route_activation_share: float
    post_exam_management_rate: float
    screening_completion_rate: float
    township_trust_index: float
    latent_demand_release_share: float
    big_hospital_diversion_reduction: float
    risk_stratification_coverage: float
    caregiver_time_release_hours: float
    fairness_score: float
    coverage_gap: float
    remote_service_rate: float
    hospital_redline_over: float
    task_metrics: dict[str, Any]
    policy_metrics: dict[str, Any]
    constraint_flags: dict[str, Any]
    villages: list[dict[str, Any]]


@dataclass(slots=True)
class EpisodeRollout:
    summary: SpatialEpisodeSummary
    trace: list[ReplayStep]
    enterprise_steps: list[PolicyStep]
    medical_steps: list[PolicyStep]
    enterprise_rewards: list[float]
    medical_rewards: list[float]


@dataclass(slots=True)
class TrainingPoint:
    episode: int
    enterprise_return: float
    medical_return: float
    farmer_return: float
    welfare_return: float
    final_average_health: float
    final_network_coverage: float
    final_average_response_time: float
    average_adoption: float
    average_service_rate: float
    average_network_coverage: float
    average_fairness: float
    average_coverage_gap: float
    average_remote_service_rate: float
    average_load_factor: float
    average_chronic_management_rate: float
    average_severe_outflow_rate: float
    average_followup_completion_rate: float
    average_avoidable_hospitalization_rate: float
    average_county_retention_rate: float
    average_specialist_bypass_rate: float
    average_bp_glucose_control_rate: float
    average_route_profitability: float
    average_reserve_capacity_ratio: float
    average_battery_or_charge_constraint: float
    average_weather_robust_dispatch_score: float
    average_hub_open_ratio: float
    average_route_activation_share: float
    average_post_exam_management_rate: float
    average_screening_completion_rate: float
    average_township_trust_index: float
    average_latent_demand_release_share: float
    average_big_hospital_diversion_reduction: float
    average_risk_stratification_coverage: float
    average_caregiver_time_release_hours: float
    population_life_year_gain: float
    chronic_life_year_gain: float
    economic_gain_total: float
    fiscal_saving_vs_baseline: float
    farmer_welfare_gain_total: float
    constraint_breach_count: int
    enterprise_critic_loss: float
    medical_critic_loss: float
    enterprise_advantage_mean: float
    enterprise_advantage_std: float
    medical_advantage_mean: float
    medical_advantage_std: float
    enterprise_exploration_std: float
    medical_exploration_std: float
    moving_score_mean: float
    moving_score_std: float
    score: float
    policy_degenerate: bool
    convergence_ready: bool
    no_improvement_evals: int
    degenerate_streak: int


@dataclass(slots=True)
class MARLTrainingConfig:
    episodes: int = 160
    gamma: float = 0.96
    gae_lambda: float = 0.92
    enterprise_learning_rate: float = 0.010
    medical_learning_rate: float = 0.009
    critic_learning_rate: float = 0.018
    enterprise_exploration_std: float = 0.30
    medical_exploration_std: float = 0.28
    enterprise_exploration_decay: float = 0.995
    medical_exploration_decay: float = 0.995
    min_exploration_std: float = 0.06
    eval_every: int = 10
    weight_decay: float = 0.0005
    critic_weight_decay: float = 0.0002
    evaluation_window: int = 5
    early_stop_patience: int = 6
    improvement_tolerance: float = 8.0
    convergence_std_tol: float = 6.0
    degeneration_patience: int = 3
    min_service_rate: float = 0.52
    seed: int = 42


@dataclass(slots=True)
class MARLTrainingArtifacts:
    history_json: Path
    latest_trace_json: Path
    best_trace_json: Path
    checkpoint_json: Path
    summary_md: Path


@dataclass(slots=True)
class MARLTrainingResult:
    best_episode: int
    best_score: float
    best_summary: SpatialEpisodeSummary
    history: list[TrainingPoint]
    artifacts: MARLTrainingArtifacts
    stopped_early: bool
    stop_reason: str


class LinearGaussianPolicy:
    def __init__(self, action_dim: int, feature_dim: int, seed: int, init_scale: float, exploration_std: float) -> None:
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.rng = random.Random(seed)
        self.weights = [[self.rng.uniform(-init_scale, init_scale) for _ in range(feature_dim)] for _ in range(action_dim)]
        self.exploration_std = exploration_std

    def mean(self, features: list[float]) -> list[float]:
        return [sum(weight * feature for weight, feature in zip(row, features)) for row in self.weights]

    def sample(self, features: list[float], explore: bool) -> tuple[list[float], list[float]]:
        mean = self.mean(features)
        if not explore:
            return mean, mean
        raw = [mu + self.rng.gauss(0.0, self.exploration_std) for mu in mean]
        return raw, mean

    def update(self, steps: list[PolicyStep], advantages: list[float], learning_rate: float, weight_decay: float) -> None:
        if not steps:
            return
        normalized_advantages = _normalize_advantages(advantages)
        variance = max(self.exploration_std ** 2, 1e-6)
        for step, advantage in zip(steps, normalized_advantages):
            for action_idx in range(self.action_dim):
                score = (step.raw_action[action_idx] - step.mean[action_idx]) / variance
                for feature_idx in range(self.feature_dim):
                    self.weights[action_idx][feature_idx] *= 1.0 - learning_rate * weight_decay
                    self.weights[action_idx][feature_idx] += learning_rate * advantage * score * step.features[feature_idx]

    def decay(self, decay: float, minimum: float) -> None:
        self.exploration_std = max(minimum, self.exploration_std * decay)

    def to_dict(self) -> dict[str, Any]:
        return {
            'action_dim': self.action_dim,
            'feature_dim': self.feature_dim,
            'exploration_std': self.exploration_std,
            'weights': self.weights,
        }


class LinearValueCritic:
    def __init__(self, feature_dim: int, seed: int, init_scale: float = 0.03) -> None:
        rng = random.Random(seed)
        self.weights = [rng.uniform(-init_scale, init_scale) for _ in range(feature_dim)]

    def predict(self, features: list[float]) -> float:
        return sum(weight * feature for weight, feature in zip(self.weights, features))

    def update(self, features_batch: list[list[float]], target_values: list[float], learning_rate: float, weight_decay: float, epochs: int = 2) -> float:
        if not features_batch:
            return 0.0
        losses = []
        for _ in range(max(1, epochs)):
            for features, target in zip(features_batch, target_values):
                prediction = self.predict(features)
                error = prediction - target
                losses.append(error * error)
                for idx, feature in enumerate(features):
                    self.weights[idx] *= 1.0 - learning_rate * weight_decay
                    self.weights[idx] -= learning_rate * 2.0 * error * feature
        return _mean(losses)

    def to_dict(self) -> dict[str, Any]:
        return {'weights': self.weights}


class EnterpriseAgent:
    def __init__(self, cfg: SpatialConfig, feature_count: int, seed: int, exploration_std: float) -> None:
        self.cfg = cfg
        self.route_count = len(AdvancedSpatialNetwork(cfg).route_catalog)
        self.policy = LinearGaussianPolicy(1 + len(cfg.hubs) + 3 + self.route_count, feature_count, seed, 0.05, exploration_std)

    def act(self, state: SpatialState, explore: bool) -> tuple[EnterpriseAction, PolicyStep]:
        features = extract_features(state)
        raw, mean = self.policy.sample(features, explore=explore)
        idx = 0
        capacity_scale = 0.20 + 1.60 * _sigmoid(raw[idx])
        idx += 1
        hub_allocations = _softmax(raw[idx: idx + len(self.cfg.hubs)])
        idx += len(self.cfg.hubs)
        price = self.cfg.enterprise['price_min'] + (self.cfg.enterprise['price_max'] - self.cfg.enterprise['price_min']) * _sigmoid(raw[idx])
        idx += 1
        reliability = _sigmoid(raw[idx])
        idx += 1
        reserve_capacity_ratio = 0.45 * _sigmoid(raw[idx])
        idx += 1
        route_activation = tuple(_sigmoid(value) for value in raw[idx: idx + self.route_count])
        action = EnterpriseAction(capacity_scale=capacity_scale, hub_allocations=hub_allocations, price=price, reliability_investment=reliability, reserve_capacity_ratio=reserve_capacity_ratio, route_activation=route_activation)
        return action, PolicyStep(features=features, mean=mean, raw_action=raw)


class MedicalAgent:
    def __init__(self, cfg: SpatialConfig, feature_count: int, seed: int, exploration_std: float) -> None:
        self.cfg = cfg
        self.policy = LinearGaussianPolicy(2 + len(cfg.hospitals), feature_count, seed, 0.05, exploration_std)

    def act(self, state: SpatialState, explore: bool) -> tuple[MedicalAction, PolicyStep]:
        features = extract_features(state)
        raw, mean = self.policy.sample(features, explore=explore)
        referral = 0.20 + 0.80 * _sigmoid(raw[0])
        integration = _sigmoid(raw[1])
        hospital_allocations = _softmax(raw[2: 2 + len(self.cfg.hospitals)])
        action = MedicalAction(referral_acceptance=referral, integration_effort=integration, hospital_allocations=hospital_allocations)
        return action, PolicyStep(features=features, mean=mean, raw_action=raw)


def _clone_config_with_seed(cfg: SpatialConfig, seed: int) -> SpatialConfig:
    cloned = clone_advanced_spatial_config(cfg)
    cloned.simulation = dict(cfg.simulation)
    cloned.simulation['seed'] = seed
    cloned.environment = dict(cfg.environment)
    cloned.policy = dict(cfg.policy)
    cloned.constraints = dict(cfg.constraints)
    cloned.enterprise = dict(cfg.enterprise)
    cloned.medical = dict(cfg.medical)
    cloned.farmer = dict(cfg.farmer)
    return cloned


def static_network_payload(cfg: SpatialConfig) -> dict[str, Any]:
    network = AdvancedSpatialNetwork(cfg)
    routes = [
        {
            'village_id': route.village_id,
            'hub_id': route.hub_id,
            'hospital_id': route.hospital_id,
            'total_time': route.total_time,
            'weather_exposure': route.weather_exposure,
        }
        for route in network.route_catalog
    ]
    return {
        'villages': [
            {
                'id': v.id,
                'name': v.name,
                'population': v.population,
                'traditional_time': v.traditional_time,
                'risk_weight': v.risk_weight,
                'demand_weight': v.demand_weight,
            }
            for v in cfg.villages
        ],
        'hubs': [{'id': h.id, 'name': h.name, 'base_capacity': h.base_capacity, 'reliability': h.reliability} for h in cfg.hubs],
        'hospitals': [
            {'id': h.id, 'name': h.name, 'base_capacity': h.base_capacity, 'integration_base': h.integration_base}
            for h in cfg.hospitals
        ],
        'edges': [
            {'origin': e.origin, 'destination': e.destination, 'travel_time': e.travel_time, 'weather_exposure': e.weather_exposure}
            for e in cfg.edges
        ],
        'routes': routes,
        'task_types': [{'id': t.id, 'name': t.name, 'priority': t.priority} for t in cfg.task_types],
    }


def run_heuristic_baseline_from_cfg(cfg: SpatialConfig, seed: int) -> SpatialEpisodeSummary:
    seeded_cfg = _clone_config_with_seed(cfg, seed)
    env = SpatialMarketEnvironment(seeded_cfg)
    enterprise_policy = HeuristicEnterprisePolicy(seeded_cfg)
    medical_policy = HeuristicMedicalPolicy(seeded_cfg)
    state = env.reset()
    total_enterprise_reward = 0.0
    total_medical_reward = 0.0
    total_farmer_reward = 0.0
    adoption_series: list[float] = []
    service_series: list[float] = []
    coverage_series: list[float] = []
    fairness_series: list[float] = []
    coverage_gap_series: list[float] = []
    remote_service_series: list[float] = []
    load_factor_series: list[float] = []
    chronic_management_series: list[float] = []
    severe_outflow_series: list[float] = []
    followup_completion_series: list[float] = []
    avoidable_hospitalization_series: list[float] = []
    county_retention_series: list[float] = []
    specialist_bypass_series: list[float] = []
    bp_glucose_control_series: list[float] = []
    route_profitability_series: list[float] = []
    reserve_capacity_series: list[float] = []
    battery_constraint_series: list[float] = []
    weather_robust_series: list[float] = []
    hub_open_ratio_series: list[float] = []
    route_activation_share_series: list[float] = []
    post_exam_management_series: list[float] = []
    screening_completion_series: list[float] = []
    township_trust_series: list[float] = []
    latent_release_series: list[float] = []
    big_hospital_diversion_series: list[float] = []
    risk_stratification_series: list[float] = []
    caregiver_time_release_series: list[float] = []
    total_policy_cost = 0.0
    breach_count = 0
    task_demand_totals = {task.id: 0.0 for task in seeded_cfg.task_types}
    task_served_totals = {task.id: 0.0 for task in seeded_cfg.task_types}
    enterprise_exit = False
    for _ in range(seeded_cfg.simulation['periods']):
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
        caregiver_time_release_series.append(result.metrics.caregiver_time_release_hours)
        total_policy_cost += result.metrics.policy_metrics.get('policy_cost_total', 0.0)
        breach_count += sum(1 for flag in result.metrics.constraint_flags.values() if flag)
        for task_id, values in result.metrics.task_metrics.items():
            task_demand_totals[task_id] = task_demand_totals.get(task_id, 0.0) + values['demand']
            task_served_totals[task_id] = task_served_totals.get(task_id, 0.0) + values['served']
        enterprise_exit = enterprise_exit or (not result.metrics.enterprise_active)
    return SpatialEpisodeSummary(
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
        state_names_cn=seeded_cfg.markov.state_names_cn,
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
        average_caregiver_time_release_hours=_mean(caregiver_time_release_series),
        task_service_rates={task_id: 0.0 if task_demand_totals.get(task_id, 0.0) <= EPS else task_served_totals.get(task_id, 0.0) / task_demand_totals[task_id] for task_id in task_demand_totals},
    )


def trace_step_from_state(state: SpatialState, result, enterprise_action: EnterpriseAction, medical_action: MedicalAction) -> ReplayStep:
    return ReplayStep(
        t=state.t,
        average_health=state.average_health,
        network_coverage=state.network_coverage,
        average_response_time=state.average_response_time,
        queue_pressure=state.queue_pressure,
        disturbances={
            'weather': state.disturbances.weather,
            'medical_surge': state.disturbances.medical_surge,
            'demand_shock': state.disturbances.demand_shock,
            'airspace_control': state.disturbances.airspace_control,
            'device_failure': state.disturbances.device_failure,
        },
        enterprise_action={
            'capacity_scale': enterprise_action.capacity_scale,
            'hub_allocations': list(enterprise_action.hub_allocations),
            'price': enterprise_action.price,
            'reliability_investment': enterprise_action.reliability_investment,
            'reserve_capacity_ratio': enterprise_action.reserve_capacity_ratio,
            'route_activation': list(enterprise_action.route_activation),
        },
        medical_action={
            'referral_acceptance': medical_action.referral_acceptance,
            'integration_effort': medical_action.integration_effort,
            'hospital_allocations': list(medical_action.hospital_allocations),
        },
        enterprise_reward=result.enterprise_reward,
        medical_reward=result.medical_reward,
        farmer_reward=result.farmer_reward,
        average_adoption=result.metrics.average_adoption,
        total_demand=result.metrics.total_demand,
        total_served=result.metrics.total_served,
        total_unmet=result.metrics.total_unmet,
        load_factor=result.metrics.load_factor,
        chronic_management_rate=result.metrics.chronic_management_rate,
        severe_outflow_rate=result.metrics.severe_outflow_rate,
        followup_completion_rate=result.metrics.followup_completion_rate,
        avoidable_hospitalization_rate=result.metrics.avoidable_hospitalization_rate,
        county_retention_rate=result.metrics.county_retention_rate,
        specialist_bypass_rate=result.metrics.specialist_bypass_rate,
        bp_glucose_control_rate=result.metrics.bp_glucose_control_rate,
        route_profitability=result.metrics.route_profitability,
        reserve_capacity_ratio=result.metrics.reserve_capacity_ratio,
        battery_or_charge_constraint=result.metrics.battery_or_charge_constraint,
        weather_robust_dispatch_score=result.metrics.weather_robust_dispatch_score,
        hub_open_ratio=result.metrics.hub_open_ratio,
        route_activation_share=result.metrics.route_activation_share,
        post_exam_management_rate=result.metrics.post_exam_management_rate,
        screening_completion_rate=result.metrics.screening_completion_rate,
        township_trust_index=result.metrics.township_trust_index,
        latent_demand_release_share=result.metrics.latent_demand_release_share,
        big_hospital_diversion_reduction=result.metrics.big_hospital_diversion_reduction,
        risk_stratification_coverage=result.metrics.risk_stratification_coverage,
        caregiver_time_release_hours=result.metrics.caregiver_time_release_hours,
        fairness_score=result.metrics.fairness_score,
        coverage_gap=result.metrics.minimum_coverage_gap,
        remote_service_rate=result.metrics.remote_service_rate,
        hospital_redline_over=result.metrics.hospital_redline_over,
        task_metrics=result.metrics.task_metrics,
        policy_metrics=result.metrics.policy_metrics,
        constraint_flags=result.metrics.constraint_flags,
        villages=[
            {
                'id': village.id,
                'name': village.name,
                'health': village.health,
                'trust': village.trust,
                'last_demand': village.last_demand,
                'last_served': village.last_served,
                'last_time': village.last_time,
                'population': village.population,
                'traditional_time': village.traditional_time,
                'risk_weight': village.risk_weight,
                'health_distribution': list(village.health_distribution),
            }
            for village in state.villages
        ],
    )


def _build_stream_payload(
    stream_context: dict[str, Any] | None,
    cfg: SpatialConfig,
    seed: int,
    explore: bool,
) -> dict[str, Any]:
    payload = dict(stream_context or {})
    payload.setdefault('seed', seed)
    payload.setdefault('explore', explore)
    payload.setdefault('periods_total', int(cfg.simulation['periods']))
    return payload


def rollout_episode(
    cfg: SpatialConfig,
    enterprise_agent: EnterpriseAgent,
    medical_agent: MedicalAgent,
    seed: int,
    explore: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    stream_context: dict[str, Any] | None = None,
) -> EpisodeRollout:
    env = SpatialMarketEnvironment(_clone_config_with_seed(cfg, seed))
    state = env.reset()
    trace: list[ReplayStep] = []
    enterprise_steps: list[PolicyStep] = []
    medical_steps: list[PolicyStep] = []
    enterprise_rewards: list[float] = []
    medical_rewards: list[float] = []
    total_enterprise_reward = 0.0
    total_medical_reward = 0.0
    total_farmer_reward = 0.0
    adoption_series: list[float] = []
    service_series: list[float] = []
    coverage_series: list[float] = []
    fairness_series: list[float] = []
    coverage_gap_series: list[float] = []
    remote_service_series: list[float] = []
    load_factor_series: list[float] = []
    chronic_management_series: list[float] = []
    severe_outflow_series: list[float] = []
    followup_completion_series: list[float] = []
    avoidable_hospitalization_series: list[float] = []
    county_retention_series: list[float] = []
    specialist_bypass_series: list[float] = []
    bp_glucose_control_series: list[float] = []
    route_profitability_series: list[float] = []
    reserve_capacity_series: list[float] = []
    battery_constraint_series: list[float] = []
    weather_robust_series: list[float] = []
    hub_open_ratio_series: list[float] = []
    route_activation_share_series: list[float] = []
    post_exam_management_series: list[float] = []
    screening_completion_series: list[float] = []
    township_trust_series: list[float] = []
    latent_release_series: list[float] = []
    big_hospital_diversion_series: list[float] = []
    risk_stratification_series: list[float] = []
    total_policy_cost = 0.0
    breach_count = 0
    task_demand_totals = {task.id: 0.0 for task in cfg.task_types}
    task_served_totals = {task.id: 0.0 for task in cfg.task_types}
    enterprise_exit = False
    stream_payload = _build_stream_payload(stream_context, cfg, seed, explore) if stream_context is not None else None
    if stream_payload is not None:
        _emit_progress(progress_callback, {
            'type': 'episode_stream_started',
            'payload': {
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'stream': stream_payload,
                'network': static_network_payload(cfg),
            },
        })
    for _ in range(cfg.simulation['periods']):
        enterprise_action, enterprise_step = enterprise_agent.act(state, explore=explore)
        medical_action, medical_step = medical_agent.act(state, explore=explore)
        result = env.step(enterprise_action, medical_action)
        state = result.state
        replay_step = trace_step_from_state(state, result, enterprise_action, medical_action)
        trace.append(replay_step)
        if stream_payload is not None:
            _emit_progress(progress_callback, {
                'type': 'episode_step',
                'payload': {
                    'generated_at': datetime.now().isoformat(timespec='seconds'),
                    'stream': stream_payload,
                    'step': asdict(replay_step),
                    'step_index': replay_step.t,
                },
            })
        if explore:
            enterprise_steps.append(enterprise_step)
            medical_steps.append(medical_step)
        enterprise_rewards.append(result.enterprise_reward)
        medical_rewards.append(result.medical_reward)
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
    summary = SpatialEpisodeSummary(
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
    if stream_payload is not None:
        _emit_progress(progress_callback, {
            'type': 'episode_stream_completed',
            'payload': {
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'stream': stream_payload,
                'summary': asdict(summary),
            },
        })
    return EpisodeRollout(summary=summary, trace=trace, enterprise_steps=enterprise_steps, medical_steps=medical_steps, enterprise_rewards=enterprise_rewards, medical_rewards=medical_rewards)


def summary_score(summary: SpatialEpisodeSummary) -> float:
    welfare = summary.total_enterprise_reward + summary.total_medical_reward + summary.total_farmer_reward
    task_bonus = 80.0 * _mean(list(summary.task_service_rates.values())) if summary.task_service_rates else 0.0
    exit_penalty = 120.0 if summary.enterprise_exit else 0.0
    return welfare + 300.0 * summary.final_average_health + 120.0 * summary.average_service_rate + 60.0 * summary.average_fairness + 70.0 * summary.average_chronic_management_rate + 45.0 * summary.average_followup_completion_rate + 40.0 * summary.average_county_retention_rate + 36.0 * summary.average_bp_glucose_control_rate + 32.0 * summary.average_post_exam_management_rate + 26.0 * summary.average_screening_completion_rate + 24.0 * summary.average_township_trust_index + 18.0 * summary.average_big_hospital_diversion_reduction + 14.0 * summary.average_risk_stratification_coverage + 40.0 * summary.average_load_factor + 18.0 * summary.average_weather_robust_dispatch_score - 120.0 * summary.average_coverage_gap - 160.0 * summary.average_severe_outflow_rate - 48.0 * summary.average_avoidable_hospitalization_rate - 35.0 * summary.average_specialist_bypass_rate - 20.0 * summary.average_battery_or_charge_constraint - 6.0 * summary.constraint_breach_count + task_bonus - exit_penalty


def _policy_diagnostics(trace: list[ReplayStep], cfg: SpatialConfig) -> dict[str, float | bool]:
    if not trace:
        return {'price_near_upper_share': 0.0, 'capacity_floor_share': 0.0, 'hub_concentration': 0.0, 'degenerate': False}
    price_span = max(cfg.enterprise['price_max'] - cfg.enterprise['price_min'], 1e-6)
    upper_hits = 0
    capacity_floor_hits = 0
    hub_concentration = []
    service_rates = []
    for step in trace:
        price = float(step.enterprise_action['price'])
        if (price - cfg.enterprise['price_min']) / price_span >= 0.90:
            upper_hits += 1
        if float(step.enterprise_action['capacity_scale']) <= 0.35:
            capacity_floor_hits += 1
        hub_concentration.append(max(step.enterprise_action['hub_allocations']) if step.enterprise_action['hub_allocations'] else 0.0)
        service_rates.append(0.0 if step.total_demand <= EPS else step.total_served / step.total_demand)
    price_upper_share = upper_hits / len(trace)
    capacity_floor_share = capacity_floor_hits / len(trace)
    concentration = _mean(hub_concentration)
    average_service = _mean(service_rates)
    degenerate = average_service < 0.50 and (price_upper_share > 0.65 or capacity_floor_share > 0.65 or concentration > 0.96)
    return {
        'price_near_upper_share': price_upper_share,
        'capacity_floor_share': capacity_floor_share,
        'hub_concentration': concentration,
        'average_service_rate': average_service,
        'degenerate': degenerate,
    }


def _diagnostics_summary(history: list[TrainingPoint]) -> dict[str, Any]:
    if not history:
        return {}
    latest = history[-1]
    best = max(history, key=lambda item: item.score)
    return {
        'latest_episode': latest.episode,
        'latest_score': latest.score,
        'latest_convergence_ready': latest.convergence_ready,
        'latest_policy_degenerate': latest.policy_degenerate,
        'latest_no_improvement_evals': latest.no_improvement_evals,
        'latest_degenerate_streak': latest.degenerate_streak,
        'latest_enterprise_exploration_std': latest.enterprise_exploration_std,
        'latest_medical_exploration_std': latest.medical_exploration_std,
        'best_episode': best.episode,
        'best_score': best.score,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _write_summary_markdown(path: Path, result: MARLTrainingResult, latest_point: TrainingPoint | None) -> None:
    lines = [
        '# Spatial MARL Training Summary',
        '',
        f'- Best episode: `{result.best_episode}`',
        f'- Best score: `{result.best_score:.3f}`',
        f'- Stopped early: `{result.stopped_early}`',
        f'- Stop reason: `{result.stop_reason}`',
        f'- Final health(best): `{result.best_summary.final_average_health:.3f}`',
        f'- Average fairness(best): `{result.best_summary.average_fairness:.3f}`',
        f'- Average coverage gap(best): `{result.best_summary.average_coverage_gap:.3f}`',
        f'- Enterprise exit(best): `{result.best_summary.enterprise_exit}`',
    ]
    if latest_point is not None:
        lines.extend([
            '',
            '## Latest Eval',
            '',
            f'- Episode: `{latest_point.episode}`',
            f'- Score: `{latest_point.score:.3f}`',
            f'- Moving score mean: `{latest_point.moving_score_mean:.3f}`',
            f'- Moving score std: `{latest_point.moving_score_std:.3f}`',
            f'- Policy degenerate: `{latest_point.policy_degenerate}`',
            f'- Enterprise exploration std: `{latest_point.enterprise_exploration_std:.3f}`',
            f'- Medical exploration std: `{latest_point.medical_exploration_std:.3f}`',
            f'- Average caregiver time release: `{latest_point.average_caregiver_time_release_hours:.2f}` hours/period',
            f'- Farmer welfare gain vs heuristic: `{latest_point.farmer_welfare_gain_total / 10000.0:.2f}` wan yuan/year',
            f'- Convergence ready: `{latest_point.convergence_ready}`',
            f'- No improvement evals: `{latest_point.no_improvement_evals}`',
            f'- Degenerate streak: `{latest_point.degenerate_streak}`',
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def save_checkpoint(path: Path, config_path: str | Path, feature_labels: list[str], enterprise_agent: EnterpriseAgent, medical_agent: MedicalAgent, enterprise_critic: LinearValueCritic, medical_critic: LinearValueCritic, training_cfg: MARLTrainingConfig, best_episode: int, best_score: float, stopped_early: bool) -> None:
    _write_json(path, {
        'schema_version': 2,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'config_path': str(config_path),
        'training_config': asdict(training_cfg),
        'feature_names': feature_labels,
        'best_episode': best_episode,
        'best_score': best_score,
        'stopped_early': stopped_early,
        'enterprise_policy': enterprise_agent.policy.to_dict(),
        'medical_policy': medical_agent.policy.to_dict(),
        'enterprise_critic': enterprise_critic.to_dict(),
        'medical_critic': medical_critic.to_dict(),
    })


def _emit_progress(progress_callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
    if progress_callback is None:
        return
    progress_callback(payload)


def train_spatial_marl(
    config_path: str | Path,
    training_cfg: MARLTrainingConfig | None = None,
    output_root: str | Path | None = None,
    scenario_name: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    preloaded_config: SpatialConfig | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> MARLTrainingResult:
    cfg = clone_advanced_spatial_config(preloaded_config) if preloaded_config is not None else load_advanced_spatial_config(config_path)
    cfg = apply_advanced_spatial_scenario(cfg, scenario_name)
    cfg = apply_runtime_overrides(cfg, runtime_overrides)
    training_cfg = training_cfg or MARLTrainingConfig()
    baseline_summary = run_heuristic_baseline_from_cfg(cfg, training_cfg.seed)
    labels = feature_names(len(cfg.markov.state_weights))
    enterprise_agent = EnterpriseAgent(cfg, len(labels), training_cfg.seed, training_cfg.enterprise_exploration_std)
    medical_agent = MedicalAgent(cfg, len(labels), training_cfg.seed + 101, training_cfg.medical_exploration_std)
    enterprise_critic = LinearValueCritic(len(labels), training_cfg.seed + 201)
    medical_critic = LinearValueCritic(len(labels), training_cfg.seed + 301)
    root = Path(config_path).resolve().parents[1] / 'outputs' if output_root is None else Path(output_root)
    live_dir = root / 'live'
    checkpoints_dir = root / 'checkpoints'
    logs_dir = root / 'logs'
    live_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    history: list[TrainingPoint] = []
    best_episode = 0
    best_score = float('-inf')
    best_summary: SpatialEpisodeSummary | None = None
    best_outcome_mapping: dict[str, Any] | None = None
    best_trace: list[ReplayStep] = []
    stopped_early = False
    stop_reason = 'completed'
    no_improvement_evals = 0
    degenerate_streak = 0
    latest_point: TrainingPoint | None = None
    _emit_progress(progress_callback, {
        'type': 'training_started',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'config_path': str(config_path),
        'scenario_name': scenario_name or 'baseline',
        'training_config': asdict(training_cfg),
        'config_snapshot': config_snapshot(cfg),
    })
    for episode in range(1, training_cfg.episodes + 1):
        rollout = rollout_episode(cfg, enterprise_agent, medical_agent, training_cfg.seed + episode, True)
        enterprise_values = [enterprise_critic.predict(step.features) for step in rollout.enterprise_steps]
        medical_values = [medical_critic.predict(step.features) for step in rollout.medical_steps]
        enterprise_advantages, enterprise_returns = _gae_advantages(rollout.enterprise_rewards, enterprise_values, training_cfg.gamma, training_cfg.gae_lambda)
        medical_advantages, medical_returns = _gae_advantages(rollout.medical_rewards, medical_values, training_cfg.gamma, training_cfg.gae_lambda)
        enterprise_critic_loss = enterprise_critic.update([step.features for step in rollout.enterprise_steps], enterprise_returns, training_cfg.critic_learning_rate, training_cfg.critic_weight_decay)
        medical_critic_loss = medical_critic.update([step.features for step in rollout.medical_steps], medical_returns, training_cfg.critic_learning_rate, training_cfg.critic_weight_decay)
        enterprise_agent.policy.update(rollout.enterprise_steps, enterprise_advantages, training_cfg.enterprise_learning_rate, training_cfg.weight_decay)
        medical_agent.policy.update(rollout.medical_steps, medical_advantages, training_cfg.medical_learning_rate, training_cfg.weight_decay)
        enterprise_agent.policy.decay(training_cfg.enterprise_exploration_decay, training_cfg.min_exploration_std)
        medical_agent.policy.decay(training_cfg.medical_exploration_decay, training_cfg.min_exploration_std)
        if episode % training_cfg.eval_every == 0 or episode == 1 or episode == training_cfg.episodes:
            evaluation = rollout_episode(
                cfg,
                enterprise_agent,
                medical_agent,
                training_cfg.seed + 10000 + episode,
                False,
                progress_callback=progress_callback,
                stream_context={
                    'phase': 'evaluation',
                    'episode': episode,
                    'trace_mode': 'latest',
                },
            )
            diagnostics = _policy_diagnostics(evaluation.trace, cfg)
            score = summary_score(evaluation.summary)
            latest_outcome_mapping = asdict(
                estimate_long_horizon_outcomes(
                    evaluation.summary,
                    baseline_summary=baseline_summary,
                    cfg=cfg,
                )
            )
            recent_scores = [item.score for item in history[-(training_cfg.evaluation_window - 1):]] + [score]
            moving_mean = _mean(recent_scores)
            moving_std = _std(recent_scores)
            point = TrainingPoint(
                episode=episode,
                enterprise_return=evaluation.summary.total_enterprise_reward,
                medical_return=evaluation.summary.total_medical_reward,
                farmer_return=evaluation.summary.total_farmer_reward,
                welfare_return=evaluation.summary.total_enterprise_reward + evaluation.summary.total_medical_reward + evaluation.summary.total_farmer_reward,
                final_average_health=evaluation.summary.final_average_health,
                final_network_coverage=evaluation.summary.final_network_coverage,
                final_average_response_time=evaluation.summary.final_average_response_time,
                average_adoption=evaluation.summary.average_adoption,
                average_service_rate=evaluation.summary.average_service_rate,
                average_network_coverage=evaluation.summary.average_network_coverage,
                average_fairness=evaluation.summary.average_fairness,
                average_coverage_gap=evaluation.summary.average_coverage_gap,
                average_remote_service_rate=evaluation.summary.average_remote_service_rate,
                average_load_factor=evaluation.summary.average_load_factor,
                average_chronic_management_rate=evaluation.summary.average_chronic_management_rate,
                average_severe_outflow_rate=evaluation.summary.average_severe_outflow_rate,
                average_followup_completion_rate=evaluation.summary.average_followup_completion_rate,
                average_avoidable_hospitalization_rate=evaluation.summary.average_avoidable_hospitalization_rate,
                average_county_retention_rate=evaluation.summary.average_county_retention_rate,
                average_specialist_bypass_rate=evaluation.summary.average_specialist_bypass_rate,
                average_bp_glucose_control_rate=evaluation.summary.average_bp_glucose_control_rate,
                average_route_profitability=evaluation.summary.average_route_profitability,
                average_reserve_capacity_ratio=evaluation.summary.average_reserve_capacity_ratio,
                average_battery_or_charge_constraint=evaluation.summary.average_battery_or_charge_constraint,
                average_weather_robust_dispatch_score=evaluation.summary.average_weather_robust_dispatch_score,
                average_hub_open_ratio=evaluation.summary.average_hub_open_ratio,
                average_route_activation_share=evaluation.summary.average_route_activation_share,
                average_post_exam_management_rate=evaluation.summary.average_post_exam_management_rate,
                average_screening_completion_rate=evaluation.summary.average_screening_completion_rate,
                average_township_trust_index=evaluation.summary.average_township_trust_index,
                average_latent_demand_release_share=evaluation.summary.average_latent_demand_release_share,
                average_big_hospital_diversion_reduction=evaluation.summary.average_big_hospital_diversion_reduction,
                average_risk_stratification_coverage=evaluation.summary.average_risk_stratification_coverage,
                average_caregiver_time_release_hours=evaluation.summary.average_caregiver_time_release_hours,
                population_life_year_gain=float(latest_outcome_mapping.get('population_life_year_gain', 0.0)),
                chronic_life_year_gain=float(latest_outcome_mapping.get('chronic_life_year_gain', 0.0)),
                economic_gain_total=float(latest_outcome_mapping.get('economic_gain_total', 0.0)),
                fiscal_saving_vs_baseline=float(latest_outcome_mapping.get('fiscal_saving_vs_baseline', 0.0)),
                farmer_welfare_gain_total=float(latest_outcome_mapping.get('farmer_welfare_gain_total', 0.0)),
                constraint_breach_count=evaluation.summary.constraint_breach_count,
                enterprise_critic_loss=enterprise_critic_loss,
                medical_critic_loss=medical_critic_loss,
                enterprise_advantage_mean=_mean(enterprise_advantages),
                enterprise_advantage_std=_std(enterprise_advantages),
                medical_advantage_mean=_mean(medical_advantages),
                medical_advantage_std=_std(medical_advantages),
                enterprise_exploration_std=enterprise_agent.policy.exploration_std,
                medical_exploration_std=medical_agent.policy.exploration_std,
                moving_score_mean=moving_mean,
                moving_score_std=moving_std,
                score=score,
                policy_degenerate=bool(diagnostics['degenerate']),
                convergence_ready=False,
                no_improvement_evals=no_improvement_evals,
                degenerate_streak=degenerate_streak,
            )
            history.append(point)
            latest_point = point
            if not point.policy_degenerate and score > best_score:
                best_score = score
                best_episode = episode
                best_summary = evaluation.summary
                best_outcome_mapping = latest_outcome_mapping
                best_trace = evaluation.trace
                no_improvement_evals = 0
            else:
                no_improvement_evals += 1
            degenerate_streak = degenerate_streak + 1 if point.policy_degenerate else 0
            early_stop_ready = no_improvement_evals >= training_cfg.early_stop_patience and moving_std <= training_cfg.convergence_std_tol
            point.convergence_ready = early_stop_ready
            point.no_improvement_evals = no_improvement_evals
            point.degenerate_streak = degenerate_streak
            history_json = live_dir / 'spatial_marl_live_metrics.json'
            latest_trace_json = live_dir / 'spatial_marl_latest_trace.json'
            best_trace_json = live_dir / 'spatial_marl_best_trace.json'
            checkpoint_json = checkpoints_dir / 'spatial_marl_policy_checkpoint.json'
            summary_md = logs_dir / 'spatial_marl_training_summary.md'
            _write_json(history_json, {
                'schema_version': 2,
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'history': [asdict(item) for item in history],
                'feature_names': labels,
                'best_episode': best_episode,
                'best_score': best_score,
                'stopped_early': stopped_early,
                'stop_reason': stop_reason,
                'training_config': asdict(training_cfg),
                'latest_diagnostics': diagnostics,
                'diagnostics_summary': _diagnostics_summary(history),
                'baseline_summary': asdict(baseline_summary),
                'latest_outcome_mapping': latest_outcome_mapping,
                'best_outcome_mapping': best_outcome_mapping if best_outcome_mapping is not None else latest_outcome_mapping,
            })
            _write_json(latest_trace_json, {
                'schema_version': 2,
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'mode': 'latest',
                'episode': episode,
                'summary': asdict(evaluation.summary),
                'outcome_mapping': latest_outcome_mapping,
                'network': static_network_payload(cfg),
                'trace': [asdict(step) for step in evaluation.trace],
            })
            _write_json(best_trace_json, {
                'schema_version': 2,
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'mode': 'best',
                'episode': best_episode,
                'summary': asdict(best_summary if best_summary is not None else evaluation.summary),
                'outcome_mapping': best_outcome_mapping if best_outcome_mapping is not None else latest_outcome_mapping,
                'network': static_network_payload(cfg),
                'trace': [asdict(step) for step in best_trace],
            })
            current_result = MARLTrainingResult(
                best_episode=best_episode,
                best_score=best_score,
                best_summary=best_summary if best_summary is not None else evaluation.summary,
                history=history,
                artifacts=MARLTrainingArtifacts(history_json=history_json, latest_trace_json=latest_trace_json, best_trace_json=best_trace_json, checkpoint_json=checkpoint_json, summary_md=summary_md),
                stopped_early=stopped_early,
                stop_reason=stop_reason,
            )
            save_checkpoint(checkpoint_json, config_path, labels, enterprise_agent, medical_agent, enterprise_critic, medical_critic, training_cfg, best_episode, best_score, stopped_early)
            _write_summary_markdown(summary_md, current_result, latest_point)
            _emit_progress(progress_callback, {
                'type': 'training_progress',
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'episode': episode,
                'point': asdict(point),
                'best_episode': best_episode,
                'best_score': best_score,
                'latest_diagnostics': diagnostics,
                'artifacts': {
                    'history_json': str(history_json),
                    'latest_trace_json': str(latest_trace_json),
                    'best_trace_json': str(best_trace_json),
                    'checkpoint_json': str(checkpoint_json),
                    'summary_md': str(summary_md),
                },
            })
            degenerate_stop = degenerate_streak >= training_cfg.degeneration_patience
            if early_stop_ready or degenerate_stop:
                stopped_early = True
                stop_reason = 'degenerate_policy' if degenerate_stop else 'early_convergence'
                break
    if best_summary is None:
        raise RuntimeError('training did not produce any evaluation summary')
    artifacts = MARLTrainingArtifacts(
        history_json=live_dir / 'spatial_marl_live_metrics.json',
        latest_trace_json=live_dir / 'spatial_marl_latest_trace.json',
        best_trace_json=live_dir / 'spatial_marl_best_trace.json',
        checkpoint_json=checkpoints_dir / 'spatial_marl_policy_checkpoint.json',
        summary_md=logs_dir / 'spatial_marl_training_summary.md',
    )
    result = MARLTrainingResult(best_episode=best_episode, best_score=best_score, best_summary=best_summary, history=history, artifacts=artifacts, stopped_early=stopped_early, stop_reason=stop_reason)
    _write_json(artifacts.history_json, {
        'schema_version': 2,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'history': [asdict(item) for item in history],
        'feature_names': labels,
        'best_episode': best_episode,
        'best_score': best_score,
        'stopped_early': stopped_early,
        'stop_reason': stop_reason,
        'training_config': asdict(training_cfg),
        'latest_diagnostics': {} if latest_point is None else {'policy_degenerate': latest_point.policy_degenerate},
        'diagnostics_summary': _diagnostics_summary(history),
        'baseline_summary': asdict(baseline_summary),
        'latest_outcome_mapping': None if latest_point is None else {
            'population_life_year_gain': latest_point.population_life_year_gain,
            'chronic_life_year_gain': latest_point.chronic_life_year_gain,
            'economic_gain_total': latest_point.economic_gain_total,
            'fiscal_saving_vs_baseline': latest_point.fiscal_saving_vs_baseline,
            'farmer_welfare_gain_total': latest_point.farmer_welfare_gain_total,
        },
        'best_outcome_mapping': best_outcome_mapping,
    })
    save_checkpoint(artifacts.checkpoint_json, config_path, labels, enterprise_agent, medical_agent, enterprise_critic, medical_critic, training_cfg, best_episode, best_score, stopped_early)
    _write_summary_markdown(artifacts.summary_md, result, latest_point)
    _emit_progress(progress_callback, {
        'type': 'training_completed',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'best_episode': best_episode,
        'best_score': best_score,
        'stopped_early': stopped_early,
        'stop_reason': stop_reason,
        'best_summary': asdict(best_summary),
        'artifacts': {
            'history_json': str(artifacts.history_json),
            'latest_trace_json': str(artifacts.latest_trace_json),
            'best_trace_json': str(artifacts.best_trace_json),
            'checkpoint_json': str(artifacts.checkpoint_json),
            'summary_md': str(artifacts.summary_md),
        },
    })
    return result
