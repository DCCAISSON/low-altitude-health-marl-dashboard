from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


EPS = 1e-9


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    cleaned = [max(float(item), 0.0) for item in values]
    total = sum(cleaned)
    if total <= EPS:
        size = len(cleaned)
        return tuple(1.0 / size for _ in range(size))
    return tuple(item / total for item in cleaned)


@dataclass(slots=True)
class ParameterEstimate:
    value: float
    source: str
    unit: str
    note: str = ""


@dataclass(slots=True)
class OutcomeMappingParameters:
    healthy_annual_mortality: ParameterEstimate
    atrisk_annual_mortality: ParameterEstimate
    managed_chronic_annual_mortality: ParameterEstimate
    unmanaged_chronic_annual_mortality: ParameterEstimate
    acute_severe_annual_mortality: ParameterEstimate
    population_management_hazard_relief: ParameterEstimate
    chronic_management_hazard_relief: ParameterEstimate
    severe_outflow_hazard_penalty: ParameterEstimate
    service_events_per_person_year: ParameterEstimate
    value_of_time_per_hour: ParameterEstimate
    productivity_days_recovered_per_managed_person: ParameterEstimate
    daily_income_per_person: ParameterEstimate
    reference_severe_outflow_rate: ParameterEstimate
    severe_case_economic_cost: ParameterEstimate
    average_reimbursement_public_cost_per_service: ParameterEstimate
    unmanaged_chronic_public_cost_per_person: ParameterEstimate
    severe_case_public_cost: ParameterEstimate
    caregiver_value_of_time_per_hour: ParameterEstimate
    household_severe_case_cost: ParameterEstimate

    def catalog(self) -> dict[str, dict[str, Any]]:
        return asdict(self)


@dataclass(slots=True)
class OutcomeMappingContext:
    population_total: int
    average_traditional_time: float
    average_risk_weight: float
    periods: int
    sample_test_out_of_pocket_yuan: float = 20.0
    uav_delivery_out_of_pocket_yuan: float = 0.0
    caregiver_time_share: float = 0.0
    caregiver_time_value_multiplier: float = 0.0


@dataclass(slots=True)
class OutcomeMappingResult:
    population_life_expectancy_years_est: float
    chronic_life_expectancy_years_est: float
    economic_value_total: float
    net_fiscal_cost: float
    farmer_welfare_total: float = 0.0
    population_life_year_gain: float = 0.0
    chronic_life_year_gain: float = 0.0
    economic_gain_total: float = 0.0
    fiscal_saving_vs_baseline: float = 0.0
    farmer_welfare_gain_total: float = 0.0
    annual_population_hazard: float = 0.0
    annual_chronic_hazard: float = 0.0
    chronic_population_estimate: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    parameter_catalog: dict[str, dict[str, Any]] = field(default_factory=dict)


def default_outcome_mapping_parameters() -> OutcomeMappingParameters:
    return OutcomeMappingParameters(
        healthy_annual_mortality=ParameterEstimate(0.006, "literature_placeholder", "probability/year", "Low annual mortality for healthy rural adults."),
        atrisk_annual_mortality=ParameterEstimate(0.012, "literature_placeholder", "probability/year", "Elevated mortality for at-risk but non-chronic population."),
        managed_chronic_annual_mortality=ParameterEstimate(0.024, "literature_placeholder", "probability/year", "Managed chronic condition annual mortality."),
        unmanaged_chronic_annual_mortality=ParameterEstimate(0.050, "literature_placeholder", "probability/year", "Unmanaged chronic condition annual mortality."),
        acute_severe_annual_mortality=ParameterEstimate(0.120, "literature_placeholder", "probability/year", "Annual mortality under acute severe state."),
        population_management_hazard_relief=ParameterEstimate(0.12, "default", "hazard multiplier", "Full chronic management lowers overall hazard by up to 12%."),
        chronic_management_hazard_relief=ParameterEstimate(0.25, "literature_placeholder", "hazard multiplier", "Full chronic management lowers chronic cohort hazard by up to 25%."),
        severe_outflow_hazard_penalty=ParameterEstimate(0.55, "literature_placeholder", "hazard multiplier", "Severe outflow raises mortality hazard."),
        service_events_per_person_year=ParameterEstimate(1.15, "fieldwork_pending", "events/person/year", "Average annual low-altitude health service opportunities per resident."),
        value_of_time_per_hour=ParameterEstimate(18.0, "fieldwork_pending", "RMB/hour", "Monetized rural time value."),
        productivity_days_recovered_per_managed_person=ParameterEstimate(4.0, "literature_placeholder", "days/person/year", "Managed chronic condition recovers productive days."),
        daily_income_per_person=ParameterEstimate(120.0, "fieldwork_pending", "RMB/day", "Average effective income per productive day."),
        reference_severe_outflow_rate=ParameterEstimate(0.18, "default", "share", "Reference severe outflow rate under weak management."),
        severe_case_economic_cost=ParameterEstimate(16000.0, "literature_placeholder", "RMB/case", "Broad socioeconomic burden per severe case."),
        average_reimbursement_public_cost_per_service=ParameterEstimate(14.0, "fieldwork_pending", "RMB/service", "Average public reimbursement burden per service."),
        unmanaged_chronic_public_cost_per_person=ParameterEstimate(850.0, "literature_placeholder", "RMB/person/year", "Public spending linked to unmanaged chronic patient."),
        severe_case_public_cost=ParameterEstimate(6200.0, "literature_placeholder", "RMB/case", "Public fiscal burden per severe case."),
        caregiver_value_of_time_per_hour=ParameterEstimate(20.0, "fieldwork_pending", "RMB/hour", "Adult children or caregiver time value released by local sampling and reduced accompaniment."),
        household_severe_case_cost=ParameterEstimate(9000.0, "literature_placeholder", "RMB/case", "Household-side economic loss avoided when severe deterioration is reduced."),
    )


def build_outcome_mapping_context(cfg: Any) -> OutcomeMappingContext:
    villages = tuple(getattr(cfg, "villages", ()))
    population_total = int(sum(int(getattr(village, "population", 0)) for village in villages))
    if population_total <= 0:
        population_total = len(villages) or 1
    avg_traditional_time = sum(float(village.population) * float(village.traditional_time) for village in villages) / max(population_total, 1)
    avg_risk_weight = sum(float(village.population) * float(village.risk_weight) for village in villages) / max(population_total, 1)
    simulation = getattr(cfg, "simulation", {}) if cfg is not None else {}
    farmer = getattr(cfg, "farmer", {}) if cfg is not None else {}
    periods = int(simulation.get("periods", 24)) if isinstance(simulation, dict) else 24
    return OutcomeMappingContext(
        population_total=population_total,
        average_traditional_time=avg_traditional_time,
        average_risk_weight=avg_risk_weight,
        periods=periods,
        sample_test_out_of_pocket_yuan=float(farmer.get("sample_test_out_of_pocket_yuan", 20.0)) if isinstance(farmer, dict) else 20.0,
        uav_delivery_out_of_pocket_yuan=float(farmer.get("uav_delivery_out_of_pocket_yuan", 0.0)) if isinstance(farmer, dict) else 0.0,
        caregiver_time_share=float(farmer.get("caregiver_time_share", 0.0)) if isinstance(farmer, dict) else 0.0,
        caregiver_time_value_multiplier=float(farmer.get("caregiver_time_value_multiplier", 0.0)) if isinstance(farmer, dict) else 0.0,
    )


def _resolve_state_positions(state_names: Sequence[str] | None, state_count: int) -> dict[str, int]:
    names = [str(name).lower().replace("_", "") for name in (state_names or ())]
    mapping: dict[str, int] = {}
    for idx, name in enumerate(names):
        if "healthy" in name and "healthy" not in mapping:
            mapping["healthy"] = idx
        if "risk" in name and "atrisk" not in mapping:
            mapping["atrisk"] = idx
    for idx, name in enumerate(names):
        if "unmanaged" in name and "chronic" in name and "unmanaged_chronic" not in mapping:
            mapping["unmanaged_chronic"] = idx
        if "managed" in name and "chronic" in name and "unmanaged" not in name and "managed_chronic" not in mapping:
            mapping["managed_chronic"] = idx
        if ("acute" in name or "severe" in name) and "acute_severe" not in mapping:
            mapping["acute_severe"] = idx
    mapping.setdefault("healthy", 0)
    mapping.setdefault("atrisk", 1 if state_count > 1 else 0)
    mapping.setdefault("managed_chronic", 2 if state_count > 2 else state_count - 1)
    mapping.setdefault("unmanaged_chronic", 3 if state_count > 3 else state_count - 1)
    mapping.setdefault("acute_severe", state_count - 1)
    return mapping


def _hazard_vector(state_count: int, params: OutcomeMappingParameters) -> list[float]:
    anchors = [
        params.healthy_annual_mortality.value,
        params.atrisk_annual_mortality.value,
        params.managed_chronic_annual_mortality.value,
        params.unmanaged_chronic_annual_mortality.value,
        params.acute_severe_annual_mortality.value,
    ]
    if state_count == len(anchors):
        return list(anchors)
    if state_count <= 1:
        return [anchors[0]]
    start = anchors[0]
    end = anchors[-1]
    step = (end - start) / max(state_count - 1, 1)
    return [start + step * idx for idx in range(state_count)]


def _estimate_absolute_outcomes(
    summary: Any,
    context: OutcomeMappingContext,
    params: OutcomeMappingParameters,
    state_names: Sequence[str] | None = None,
) -> OutcomeMappingResult:
    distribution = _normalize(tuple(getattr(summary, "final_state_distribution", ()) or (1.0,)))
    state_count = len(distribution)
    positions = _resolve_state_positions(state_names, state_count)
    hazards = _hazard_vector(state_count, params)
    weighted_hazard = sum(share * hazards[idx] for idx, share in enumerate(distribution))

    chronic_indices = sorted(set(idx for idx in [
        positions["managed_chronic"],
        positions["unmanaged_chronic"],
        positions["acute_severe"],
    ] if 0 <= idx < state_count))
    chronic_share = sum(distribution[idx] for idx in chronic_indices)
    chronic_population_estimate = context.population_total * chronic_share

    if chronic_share > EPS:
        chronic_distribution = [distribution[idx] / chronic_share for idx in chronic_indices]
        chronic_hazard_base = sum(share * hazards[idx] for share, idx in zip(chronic_distribution, chronic_indices))
    else:
        chronic_hazard_base = weighted_hazard

    risk_adjustment = 1.0 + 0.08 * max(context.average_risk_weight - 0.5, 0.0)
    population_hazard = weighted_hazard * risk_adjustment
    population_hazard *= 1.0 - params.population_management_hazard_relief.value * float(getattr(summary, "average_chronic_management_rate", 0.0))
    population_hazard *= 1.0 + params.severe_outflow_hazard_penalty.value * float(getattr(summary, "average_severe_outflow_rate", 0.0))
    population_hazard = max(population_hazard, EPS)

    chronic_hazard = chronic_hazard_base * risk_adjustment
    chronic_hazard *= 1.0 - params.chronic_management_hazard_relief.value * float(getattr(summary, "average_chronic_management_rate", 0.0))
    chronic_hazard *= 1.0 + params.severe_outflow_hazard_penalty.value * float(getattr(summary, "average_severe_outflow_rate", 0.0))
    chronic_hazard = max(chronic_hazard, EPS)

    population_life_expectancy = 1.0 / population_hazard
    chronic_life_expectancy = 1.0 / chronic_hazard

    annual_service_events = context.population_total * params.service_events_per_person_year.value
    avg_response_time = float(getattr(summary, "final_average_response_time", 0.0))
    service_rate = float(getattr(summary, "average_service_rate", 0.0))
    response_time_saved = max(context.average_traditional_time - avg_response_time, 0.0)
    annual_served_events = annual_service_events * service_rate
    time_access_value = annual_served_events * response_time_saved * params.value_of_time_per_hour.value
    caregiver_time_hours = annual_served_events * response_time_saved * context.caregiver_time_share
    caregiver_time_value = caregiver_time_hours * params.caregiver_value_of_time_per_hour.value * context.caregiver_time_value_multiplier

    chronic_productivity_value = chronic_population_estimate * float(getattr(summary, "average_chronic_management_rate", 0.0))
    chronic_productivity_value *= params.productivity_days_recovered_per_managed_person.value * params.daily_income_per_person.value

    severe_gap = max(params.reference_severe_outflow_rate.value - float(getattr(summary, "average_severe_outflow_rate", 0.0)), 0.0)
    severe_avoidance_value = chronic_population_estimate * severe_gap * params.severe_case_economic_cost.value
    severe_household_avoidance_value = chronic_population_estimate * severe_gap * params.household_severe_case_cost.value
    service_out_of_pocket_cost = annual_served_events * (context.sample_test_out_of_pocket_yuan + context.uav_delivery_out_of_pocket_yuan)
    farmer_welfare_total = time_access_value + caregiver_time_value + severe_household_avoidance_value - service_out_of_pocket_cost

    economic_value_total = time_access_value + chronic_productivity_value + severe_avoidance_value

    reimbursement_cost = annual_service_events * service_rate * params.average_reimbursement_public_cost_per_service.value
    unmanaged_chronic_cost = chronic_population_estimate * max(1.0 - float(getattr(summary, "average_chronic_management_rate", 0.0)), 0.0) * params.unmanaged_chronic_public_cost_per_person.value
    severe_case_public_cost = chronic_population_estimate * float(getattr(summary, "average_severe_outflow_rate", 0.0)) * params.severe_case_public_cost.value
    net_fiscal_cost = float(getattr(summary, "policy_cost", 0.0)) + reimbursement_cost + unmanaged_chronic_cost + severe_case_public_cost

    return OutcomeMappingResult(
        population_life_expectancy_years_est=population_life_expectancy,
        chronic_life_expectancy_years_est=chronic_life_expectancy,
        economic_value_total=economic_value_total,
        net_fiscal_cost=net_fiscal_cost,
        farmer_welfare_total=farmer_welfare_total,
        annual_population_hazard=population_hazard,
        annual_chronic_hazard=chronic_hazard,
        chronic_population_estimate=chronic_population_estimate,
        components={
            "time_access_value": time_access_value,
            "caregiver_time_value": caregiver_time_value,
            "service_out_of_pocket_cost": service_out_of_pocket_cost,
            "severe_household_avoidance_value": severe_household_avoidance_value,
            "farmer_welfare_total": farmer_welfare_total,
            "chronic_productivity_value": chronic_productivity_value,
            "severe_avoidance_value": severe_avoidance_value,
            "policy_cost": float(getattr(summary, "policy_cost", 0.0)),
            "reimbursement_cost": reimbursement_cost,
            "unmanaged_chronic_cost": unmanaged_chronic_cost,
            "severe_case_public_cost": severe_case_public_cost,
        },
        parameter_catalog=params.catalog(),
    )


def estimate_long_horizon_outcomes(
    summary: Any,
    *,
    baseline_summary: Any | None = None,
    cfg: Any | None = None,
    params: OutcomeMappingParameters | None = None,
    context: OutcomeMappingContext | None = None,
    state_names: Sequence[str] | None = None,
) -> OutcomeMappingResult:
    resolved_params = params or default_outcome_mapping_parameters()
    resolved_context = context or build_outcome_mapping_context(cfg)
    resolved_state_names = state_names
    if resolved_state_names is None and cfg is not None:
        resolved_state_names = tuple(getattr(getattr(cfg, "markov", None), "state_names", ()) or ())
    current = _estimate_absolute_outcomes(summary, resolved_context, resolved_params, state_names=resolved_state_names)
    if baseline_summary is None:
        return current

    baseline = _estimate_absolute_outcomes(baseline_summary, resolved_context, resolved_params, state_names=resolved_state_names)
    current.population_life_year_gain = current.population_life_expectancy_years_est - baseline.population_life_expectancy_years_est
    current.chronic_life_year_gain = current.chronic_life_expectancy_years_est - baseline.chronic_life_expectancy_years_est
    current.economic_gain_total = current.economic_value_total - baseline.economic_value_total
    current.fiscal_saving_vs_baseline = baseline.net_fiscal_cost - current.net_fiscal_cost
    current.farmer_welfare_gain_total = current.farmer_welfare_total - baseline.farmer_welfare_total
    current.components["baseline_economic_value_total"] = baseline.economic_value_total
    current.components["baseline_net_fiscal_cost"] = baseline.net_fiscal_cost
    current.components["baseline_population_life_expectancy_years_est"] = baseline.population_life_expectancy_years_est
    current.components["baseline_chronic_life_expectancy_years_est"] = baseline.chronic_life_expectancy_years_est
    current.components["baseline_farmer_welfare_total"] = baseline.farmer_welfare_total
    return current


__all__ = [
    "OutcomeMappingContext",
    "OutcomeMappingParameters",
    "OutcomeMappingResult",
    "ParameterEstimate",
    "build_outcome_mapping_context",
    "default_outcome_mapping_parameters",
    "estimate_long_horizon_outcomes",
]
