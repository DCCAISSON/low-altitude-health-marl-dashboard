from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .advanced_spatial import (
    AdvancedSpatialConfig,
    apply_advanced_spatial_scenario,
    apply_runtime_overrides,
    available_advanced_spatial_scenarios,
    config_snapshot,
    load_advanced_spatial_config,
)


@dataclass(slots=True)
class PlatformRunRequest:
    config_name: str
    config_path: Path
    scenario_name: str
    episodes: int
    eval_every: int
    seed: int
    run_name: str
    report_name: str
    overrides: dict[str, Any]


_PREVIEW_SECTION_ORDER = (
    'simulation',
    'environment',
    'policy',
    'constraints',
    'enterprise',
    'medical',
    'farmer',
    'disturbances',
    'task_types',
    'network',
)
_REMOTE_TRADITIONAL_TIME_THRESHOLD = 9.0
_REMOTE_RISK_THRESHOLD = 0.65


def list_platform_configs(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root)
    catalog: dict[str, str] = {}
    for rel in [
        'configs/spatial_network_recommended.toml',
        'configs/spatial_network_from_master.toml',
        'configs/spatial_network_baseline.toml',
    ]:
        path = root / rel
        if path.exists():
            catalog[path.stem] = str(path.resolve())
    return catalog


def build_default_editor_values() -> dict[str, Any]:
    return {
        'simulation': {'base_need_rate': 0.18},
        'environment': {'reimbursement_rate': 0.20},
        'policy': {
            'per_service_subsidy': 0.15,
            'performance_subsidy_per_coverage': 25.0,
            'fairness_subsidy_per_point': 18.0,
        },
        'constraints': {
            'minimum_coverage_ratio': 0.55,
            'fairness_floor': 0.65,
            'remote_service_floor': 0.38,
        },
        'disturbances': {
            'weather_std': 0.18,
            'airspace_std': 0.10,
            'device_std': 0.08,
        },
    }


def normalize_run_request(payload: dict[str, Any], project_root: str | Path) -> PlatformRunRequest:
    project_root = Path(project_root)
    configs = list_platform_configs(project_root)
    config_name = str(payload.get('config_name', 'spatial_network_recommended')).strip()
    config_path_raw = str(payload.get('config_path') or configs.get(config_name) or configs.get('spatial_network_recommended') or '')
    if not config_path_raw:
        raise FileNotFoundError('No available spatial config')
    config_path = Path(config_path_raw)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')
    scenario_name = str(payload.get('scenario_name', 'baseline')).strip() or 'baseline'
    if scenario_name not in available_advanced_spatial_scenarios():
        scenario_name = 'baseline'
    episodes = max(1, int(payload.get('episodes', 60)))
    eval_every = max(1, int(payload.get('eval_every', 5)))
    seed = int(payload.get('seed', 42))
    run_name = str(payload.get('run_name', f'platform_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')).strip()
    report_name = str(payload.get('report_name', run_name)).strip() or run_name
    overrides = payload.get('overrides') if isinstance(payload.get('overrides'), dict) else {}
    return PlatformRunRequest(
        config_name=config_name,
        config_path=config_path,
        scenario_name=scenario_name,
        episodes=episodes,
        eval_every=eval_every,
        seed=seed,
        run_name=run_name,
        report_name=report_name,
        overrides=overrides,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _state_share(cfg: AdvancedSpatialConfig, state_distribution: tuple[float, ...] | None, index: int) -> float:
    source = tuple(float(value) for value in (state_distribution or cfg.markov.base_transition[0]))
    if index >= len(source):
        return 0.0
    return source[index]


def _preview_network_snapshot(cfg: AdvancedSpatialConfig) -> dict[str, Any]:
    remote_villages = [
        village
        for village in cfg.villages
        if village.traditional_time >= _REMOTE_TRADITIONAL_TIME_THRESHOLD or village.risk_weight >= _REMOTE_RISK_THRESHOLD
    ]
    village_edges = [edge for edge in cfg.edges if edge.origin.startswith('V')]
    return {
        'village_count': len(cfg.villages),
        'hub_count': len(cfg.hubs),
        'hospital_count': len(cfg.hospitals),
        'edge_count': len(cfg.edges),
        'village_avg_base_health': _mean([float(village.base_health) for village in cfg.villages]),
        'village_avg_trust': _mean([float(village.trust) for village in cfg.villages]),
        'village_avg_demand_weight': _mean([float(village.demand_weight) for village in cfg.villages]),
        'village_avg_traditional_time': _mean([float(village.traditional_time) for village in cfg.villages]),
        'village_avg_risk_weight': _mean([float(village.risk_weight) for village in cfg.villages]),
        'village_avg_healthy_share': _mean([_state_share(cfg, village.state_distribution, 0) for village in cfg.villages]),
        'village_avg_managed_chronic_share': _mean([_state_share(cfg, village.state_distribution, 2) for village in cfg.villages]),
        'village_avg_unmanaged_chronic_share': _mean([_state_share(cfg, village.state_distribution, 3) for village in cfg.villages]),
        'village_avg_acute_severe_share': _mean([_state_share(cfg, village.state_distribution, 4) for village in cfg.villages]),
        'remote_village_share': len(remote_villages) / max(len(cfg.villages), 1),
        'remote_village_avg_trust': _mean([float(village.trust) for village in remote_villages]),
        'remote_village_avg_traditional_time': _mean([float(village.traditional_time) for village in remote_villages]),
        'hub_avg_capacity': _mean([float(hub.base_capacity) for hub in cfg.hubs]),
        'hub_avg_reliability': _mean([float(hub.reliability) for hub in cfg.hubs]),
        'hospital_avg_capacity': _mean([float(hospital.base_capacity) for hospital in cfg.hospitals]),
        'hospital_avg_integration_base': _mean([float(hospital.integration_base) for hospital in cfg.hospitals]),
        'edge_avg_travel_time': _mean([float(edge.travel_time) for edge in cfg.edges]),
        'edge_avg_weather_exposure': _mean([float(edge.weather_exposure) for edge in cfg.edges]),
        'village_edge_avg_travel_time': _mean([float(edge.travel_time) for edge in village_edges]),
    }


def _preview_snapshot(cfg: AdvancedSpatialConfig) -> dict[str, Any]:
    snapshot = config_snapshot(cfg)
    snapshot['network'] = _preview_network_snapshot(cfg)
    return snapshot


def _flatten_preview_snapshot(value: Any, prefix: str = '') -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            next_prefix = f'{prefix}.{key}' if prefix else str(key)
            flattened.update(_flatten_preview_snapshot(value[key], next_prefix))
        return flattened
    if isinstance(value, list):
        keyed_items = [
            item
            for item in value
            if isinstance(item, dict) and isinstance(item.get('id'), str) and str(item.get('id')).strip()
        ]
        if len(keyed_items) == len(value):
            for item in keyed_items:
                item_prefix = f"{prefix}.{str(item['id']).strip()}" if prefix else str(item['id']).strip()
                child = {key: item[key] for key in item if key != 'id'}
                flattened.update(_flatten_preview_snapshot(child, item_prefix))
            return flattened
        for index, item in enumerate(value):
            next_prefix = f'{prefix}.{index}' if prefix else str(index)
            flattened.update(_flatten_preview_snapshot(item, next_prefix))
        return flattened
    if prefix:
        flattened[prefix] = value
    return flattened


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return abs(float(left) - float(right)) <= 1e-12
    return left == right


def _preview_sort_key(path: str) -> tuple[int, str]:
    section = path.split('.', 1)[0]
    try:
        section_index = _PREVIEW_SECTION_ORDER.index(section)
    except ValueError:
        section_index = len(_PREVIEW_SECTION_ORDER)
    return section_index, path


def _build_preview_changes(before_snapshot: dict[str, Any], after_snapshot: dict[str, Any], source: str) -> list[dict[str, Any]]:
    before_flat = _flatten_preview_snapshot(before_snapshot)
    after_flat = _flatten_preview_snapshot(after_snapshot)
    changes: list[dict[str, Any]] = []
    for path in sorted(set(before_flat) | set(after_flat), key=_preview_sort_key):
        before = before_flat.get(path)
        after = after_flat.get(path)
        if _values_equal(before, after):
            continue
        changes.append(
            {
                'path': path,
                'before': before,
                'after': after,
                'source': source,
            }
        )
    return changes


def _prepare_platform_layers(request: PlatformRunRequest) -> tuple[AdvancedSpatialConfig, AdvancedSpatialConfig, AdvancedSpatialConfig]:
    base_cfg = load_advanced_spatial_config(request.config_path)
    scenario_cfg = apply_advanced_spatial_scenario(base_cfg, request.scenario_name)
    final_cfg = apply_runtime_overrides(scenario_cfg, request.overrides)
    return base_cfg, scenario_cfg, final_cfg


def prepare_platform_config(request: PlatformRunRequest) -> tuple[AdvancedSpatialConfig, dict[str, Any]]:
    _, _, cfg = _prepare_platform_layers(request)
    snapshot = config_snapshot(cfg)
    return cfg, snapshot


def prepare_platform_preview(request: PlatformRunRequest) -> tuple[AdvancedSpatialConfig, dict[str, Any]]:
    base_cfg, scenario_cfg, final_cfg = _prepare_platform_layers(request)
    base_preview = _preview_snapshot(base_cfg)
    scenario_preview = _preview_snapshot(scenario_cfg)
    final_preview = _preview_snapshot(final_cfg)
    final_snapshot = config_snapshot(final_cfg)
    scenario_changes = _build_preview_changes(base_preview, scenario_preview, source='scenario')
    override_changes = _build_preview_changes(scenario_preview, final_preview, source='override')
    return final_cfg, {
        'snapshot': final_snapshot,
        'base_snapshot': base_preview,
        'scenario_snapshot': scenario_preview,
        'scenario_changes': scenario_changes,
        'override_changes': override_changes,
        'change_counts': {
            'scenario': len(scenario_changes),
            'override': len(override_changes),
        },
    }


def write_platform_state(output_root: str | Path, payload: dict[str, Any]) -> Path:
    output_root = Path(output_root)
    state_dir = output_root / 'runtime'
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / 'platform_state.json'
    try:
        state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    except PermissionError:
        print(f'Warning: cannot overwrite runtime state file, reusing existing file: {state_path}')
    return state_path


def write_platform_request(output_root: str | Path, request: PlatformRunRequest, snapshot: dict[str, Any]) -> Path:
    output_root = Path(output_root)
    state_dir = output_root / 'runtime'
    state_dir.mkdir(parents=True, exist_ok=True)
    request_path = state_dir / 'last_run_request.json'
    try:
        request_path.write_text(
            json.dumps(
                {
                    'generated_at': datetime.now().isoformat(timespec='seconds'),
                    'request': {
                        'config_name': request.config_name,
                        'config_path': str(request.config_path),
                        'scenario_name': request.scenario_name,
                        'episodes': request.episodes,
                        'eval_every': request.eval_every,
                        'seed': request.seed,
                        'run_name': request.run_name,
                        'report_name': request.report_name,
                        'overrides': request.overrides,
                    },
                    'config_snapshot': snapshot,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
    except PermissionError:
        print(f'Warning: cannot overwrite runtime request file, reusing existing file: {request_path}')
    return request_path
