from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
from typing import Any


SCENARIO_NAME_MAP = {
    "baseline": "基准情景",
    "harsh_weather": "恶劣天气情景",
    "low_trust": "低信任情景",
    "sparse_demand": "稀疏需求情景",
    "chronic_burden": "慢病负担高情景",
    "policy_push": "政策推动情景",
    "from_master": "主表编译情景",
    "recommended": "推荐配置情景",
}

TASK_NAME_MAP = {
    "chronic_followup": "慢病复诊/随访",
    "sample_transport": "检验样本送检",
    "medicine_delivery": "常用药配送",
    "acute_urgent": "急性高优需求",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sum_from_steps(trace: list[dict[str, Any]], section: str, key: str) -> float:
    return sum(float(step.get(section, {}).get(key, 0.0)) for step in trace)


def _mean_from_steps(trace: list[dict[str, Any]], section: str, key: str) -> float:
    return _mean([float(step.get(section, {}).get(key, 0.0)) for step in trace])


def _mean_simple(trace: list[dict[str, Any]], key: str) -> float:
    return _mean([float(step.get(key, 0.0)) for step in trace])


def _flag_counts(trace: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in trace:
        for name, active in step.get("constraint_flags", {}).items():
            counts[name] = counts.get(name, 0) + int(bool(active))
    return counts


def _task_rollup(trace: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for step in trace:
        for task_id, metrics in step.get("task_metrics", {}).items():
            bucket = totals.setdefault(task_id, {"demand": 0.0, "served": 0.0, "unmet": 0.0})
            bucket["demand"] += float(metrics.get("demand", 0.0))
            bucket["served"] += float(metrics.get("served", 0.0))
            bucket["unmet"] += float(metrics.get("unmet", 0.0))
    for bucket in totals.values():
        demand = bucket["demand"]
        bucket["service_rate"] = 0.0 if demand <= 1e-9 else bucket["served"] / demand
    return totals


def _trace_snapshot(trace_payload: dict[str, Any]) -> dict[str, Any]:
    trace = trace_payload.get("trace", [])
    summary = trace_payload.get("summary", {})
    last_step = trace[-1] if trace else {}
    villages = last_step.get("villages", [])
    return {
        "periods": len(trace),
        "summary": summary,
        "last_step": last_step,
        "outcome_mapping": trace_payload.get("outcome_mapping", {}),
        "strategy_averages": {
            "enterprise_price_mean": _mean_from_steps(trace, "enterprise_action", "price"),
            "enterprise_capacity_mean": _mean_from_steps(trace, "enterprise_action", "capacity_scale"),
            "enterprise_reliability_mean": _mean_from_steps(trace, "enterprise_action", "reliability_investment"),
            "medical_acceptance_mean": _mean_from_steps(trace, "medical_action", "referral_acceptance"),
            "medical_integration_mean": _mean_from_steps(trace, "medical_action", "integration_effort"),
        },
        "system_averages": {
            "health_mean": _mean_simple(trace, "average_health"),
            "service_rate_mean": _mean_simple(trace, "average_service_rate"),
            "adoption_mean": _mean_simple(trace, "average_adoption"),
            "fairness_mean": _mean_simple(trace, "fairness_score"),
            "coverage_gap_mean": _mean_simple(trace, "coverage_gap"),
            "remote_service_rate_mean": _mean_simple(trace, "remote_service_rate"),
            "response_time_mean": _mean_simple(trace, "average_response_time"),
            "load_factor_mean": _mean_simple(trace, "load_factor"),
            "chronic_management_rate_mean": _mean_simple(trace, "chronic_management_rate"),
            "severe_outflow_rate_mean": _mean_simple(trace, "severe_outflow_rate"),
            "followup_completion_rate_mean": _mean_simple(trace, "followup_completion_rate"),
            "avoidable_hospitalization_rate_mean": _mean_simple(trace, "avoidable_hospitalization_rate"),
            "county_retention_rate_mean": _mean_simple(trace, "county_retention_rate"),
            "specialist_bypass_rate_mean": _mean_simple(trace, "specialist_bypass_rate"),
            "bp_glucose_control_rate_mean": _mean_simple(trace, "bp_glucose_control_rate"),
            "route_profitability_mean": _mean_simple(trace, "route_profitability"),
            "reserve_capacity_ratio_mean": _mean_simple(trace, "reserve_capacity_ratio"),
            "battery_or_charge_constraint_mean": _mean_simple(trace, "battery_or_charge_constraint"),
            "weather_robust_dispatch_score_mean": _mean_simple(trace, "weather_robust_dispatch_score"),
            "hub_open_ratio_mean": _mean_simple(trace, "hub_open_ratio"),
            "route_activation_share_mean": _mean_simple(trace, "route_activation_share"),
        },
        "policy_costs": {
            "policy_cost_total": _sum_from_steps(trace, "policy_metrics", "policy_cost_total"),
            "per_service_subsidy_total": _sum_from_steps(trace, "policy_metrics", "per_service_subsidy"),
            "performance_subsidy_total": _sum_from_steps(trace, "policy_metrics", "performance_subsidy"),
            "remote_service_subsidy_total": _sum_from_steps(trace, "policy_metrics", "remote_service_subsidy"),
        },
        "disturbance_means": {
            "weather": _mean_from_steps(trace, "disturbances", "weather"),
            "medical_surge": _mean_from_steps(trace, "disturbances", "medical_surge"),
            "demand_shock": _mean_from_steps(trace, "disturbances", "demand_shock"),
            "airspace_control": _mean_from_steps(trace, "disturbances", "airspace_control"),
            "device_failure": _mean_from_steps(trace, "disturbances", "device_failure"),
        },
        "constraint_flag_counts": _flag_counts(trace),
        "task_rollup": _task_rollup(trace),
        "village_snapshot": [
            {
                "id": village.get("id"),
                "name": village.get("name"),
                "health": float(village.get("health", 0.0)),
                "trust": float(village.get("trust", 0.0)),
                "last_demand": float(village.get("last_demand", 0.0)),
                "last_served": float(village.get("last_served", 0.0)),
                "last_time": float(village.get("last_time", 0.0)),
                "population": float(village.get("population", 0.0)),
                "risk_weight": float(village.get("risk_weight", 0.0)),
            }
            for village in villages
        ],
    }


def _network_snapshot(network: dict[str, Any]) -> dict[str, Any]:
    villages = network.get("villages", [])
    hubs = network.get("hubs", [])
    hospitals = network.get("hospitals", [])
    return {
        "village_count": len(villages),
        "hub_count": len(hubs),
        "hospital_count": len(hospitals),
        "edge_count": len(network.get("edges", [])),
        "task_type_count": len(network.get("task_types", [])),
        "total_population": sum(float(village.get("population", 0.0)) for village in villages),
        "average_traditional_time": _mean([float(village.get("traditional_time", 0.0)) for village in villages]),
        "average_risk_weight": _mean([float(village.get("risk_weight", 0.0)) for village in villages]),
        "villages": villages,
        "hubs": hubs,
        "hospitals": hospitals,
        "task_types": network.get("task_types", []),
    }


def _scenario_comparison(experiment_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not experiment_summary:
        return None
    return {
        "experiment_name": experiment_summary.get("experiment_name"),
        "generated_at": experiment_summary.get("generated_at"),
        "comparison_type": experiment_summary.get("comparison_type"),
        "scenario_summary": experiment_summary.get("scenario_summary", []),
        "records": experiment_summary.get("records", []),
        "outcome_mapping_parameter_catalog": experiment_summary.get("outcome_mapping_parameter_catalog", {}),
    }


def _resolve_runtime_request(output_root: Path, runtime_request: dict[str, Any] | None) -> dict[str, Any] | None:
    if runtime_request:
        return runtime_request
    payload = _load_json_if_exists(output_root / "runtime" / "last_run_request.json")
    if isinstance(payload, dict) and isinstance(payload.get("request"), dict):
        return payload["request"]
    return None


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return '<p class="muted">暂无数据</p>'
    thead = "".join(f"<th>{html.escape(str(head))}</th>" for head in headers)
    tbody = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"


def _wan_yuan(value: float) -> str:
    return f"{float(value) / 10000.0:.1f}万元"


def _years(value: float) -> str:
    return f"{float(value):.2f}年"


def _scenario_label(code: str) -> str:
    return SCENARIO_NAME_MAP.get(code, code)


def _task_label(task_id: str) -> str:
    return TASK_NAME_MAP.get(task_id, task_id)


def _metric_cards(report: dict[str, Any]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    best = report["best_run"]
    summary = best["summary"]
    outcome = best.get("outcome_mapping", {})
    exit_flag = bool(summary.get("enterprise_exit", False))

    level_cards = [
        ("系统综合福利", f"{float(summary.get('welfare_return', 0.0)):.1f}", "反映企业、医院与农户在最优轮次下的综合收益。"),
        ("平均健康水平", f"{float(summary.get('final_average_health', 0.0)):.3f}", "用于刻画慢病管理与健康改善的整体水平。"),
        ("平均服务率", f"{float(summary.get('average_service_rate', 0.0)):.3f}", "表示需求被满足的比例，衡量网络运行效率。"),
        ("子女照料减负", f"{float(summary.get('average_caregiver_time_release_hours', 0.0)):.1f} 小时/期", "反映本地血样送检与慢病管理对陪诊、接送和照料时间的释放。"),
        ("企业负载率", f"{float(summary.get('average_load_factor', 0.0)):.3f}", "用于判断运力利用是否饱满，支撑企业调度与盈利分析。"),
        ("慢病管理率", f"{float(summary.get('average_chronic_management_rate', 0.0)):.3f}", "衡量慢病复诊与随访需求被稳定承接的程度。"),
        ("检后管理率", f"{float(summary.get('average_post_exam_management_rate', 0.0)):.3f}", "衡量样本送检后结果回流与后续管理是否形成闭环。"),
        ("筛查闭环率", f"{float(summary.get('average_screening_completion_rate', 0.0)):.3f}", "反映采样、送检、结果返回与处置是否完整完成。"),
        ("乡镇信任指数", f"{float(summary.get('average_township_trust_index', 0.0)):.3f}", "反映村民对乡镇医疗机构的动态信任与再就诊意愿。"),
        ("重症外流率", f"{float(summary.get('average_severe_outflow_rate', 0.0)):.3f}", "反映健康管理失效后向高层级医疗外流的压力，越低越好。"),
        ("潜在需求释放率", f"{float(summary.get('average_latent_demand_release_share', 0.0)):.3f}", "反映村级送检点激活原本被交通与时间成本压制的检查需求比例。"),
        ("大医院分流改善", f"{float(summary.get('average_big_hospital_diversion_reduction', 0.0)):.3f}", "反映基层信任与留诊能力提升后，对上级医院虹吸的缓解程度。"),
        ("公平性指数", f"{float(summary.get('average_fairness', 0.0)):.3f}", "反映不同村庄之间服务获得的均衡程度。"),
        ("网络覆盖率", f"{float(summary.get('final_network_coverage', 0.0)):.3f}", "表征低空健康服务网络的可达范围与覆盖深度。"),
        ("企业存活状态", "退出触发" if exit_flag else "稳定运行", "结合退出规则与训练结果，判断供给端持续运营能力。"),
    ]
    counterfactual_cards = [
        ("村民寿命提升", _years(float(outcome.get("population_life_year_gain", 0.0))), "相对于同情景启发式基线的总体预期寿命增量估算。"),
        ("慢病寿命提升", _years(float(outcome.get("chronic_life_year_gain", 0.0))), "相对于同情景启发式基线的慢病人群预期寿命增量估算。"),
        ("社会经济收益增加", _wan_yuan(float(outcome.get("economic_gain_total", 0.0))), "基于时间、生产连续性和重症避免的年度货币化估算。"),
        ("政府财政节约", _wan_yuan(float(outcome.get("fiscal_saving_vs_baseline", 0.0))), "相对于同情景启发式基线的净财政节约估算。"),
        ("农户福利改善", _wan_yuan(float(outcome.get("farmer_welfare_gain_total", 0.0))), "基于患者时间节约、子女照料减负、重症风险避免与自付成本后的年度净改善估算。"),
    ]
    return level_cards, counterfactual_cards


def _scenario_relative_value(row: dict[str, Any], delta_key: str, marl_key: str | None = None, baseline_key: str | None = None) -> float:
    if delta_key in row:
        return float(row.get(delta_key, 0.0))
    if marl_key and baseline_key and marl_key in row and baseline_key in row:
        return float(row.get(marl_key, 0.0)) - float(row.get(baseline_key, 0.0))
    if marl_key and marl_key in row:
        return float(row.get(marl_key, 0.0))
    return 0.0


def _scenario_rows(report: dict[str, Any]) -> list[list[Any]]:
    scenario = report.get("scenario_comparison") or {}
    rows = []
    for item in scenario.get("scenario_summary", []):
        rows.append(
            [
                _scenario_label(str(item.get("scenario", ""))),
                item.get("runs", ""),
                f"{_scenario_relative_value(item, 'delta_final_network_coverage_mean', 'marl_final_network_coverage_mean', 'baseline_final_network_coverage_mean'):.3f}",
                f"{_scenario_relative_value(item, 'delta_final_average_health_mean', 'marl_final_average_health_mean', 'baseline_final_average_health_mean'):.3f}",
                f"{_scenario_relative_value(item, 'delta_average_service_rate_mean', 'marl_average_service_rate_mean', 'baseline_average_service_rate_mean'):.3f}",
                f"{_scenario_relative_value(item, 'delta_average_load_factor_mean', 'marl_average_load_factor_mean', 'baseline_average_load_factor_mean'):.3f}",
                f"{_scenario_relative_value(item, 'delta_average_chronic_management_rate_mean', 'marl_average_chronic_management_rate_mean', 'baseline_average_chronic_management_rate_mean'):.3f}",
                f"{_scenario_relative_value(item, 'delta_average_post_exam_management_rate_mean', 'marl_average_post_exam_management_rate_mean', 'baseline_average_post_exam_management_rate_mean'):.3f}",
                f"{_scenario_relative_value(item, 'delta_average_screening_completion_rate_mean', 'marl_average_screening_completion_rate_mean', 'baseline_average_screening_completion_rate_mean'):.3f}",
                f"{_scenario_relative_value(item, 'delta_average_township_trust_index_mean', 'marl_average_township_trust_index_mean', 'baseline_average_township_trust_index_mean'):.3f}",
                f"{_scenario_relative_value(item, 'delta_average_severe_outflow_rate_mean', 'marl_average_severe_outflow_rate_mean', 'baseline_average_severe_outflow_rate_mean'):.3f}",
                f"{float(item.get('population_life_year_gain_mean', 0.0)):.2f}",
                f"{float(item.get('chronic_life_year_gain_mean', 0.0)):.2f}",
                f"{float(item.get('economic_gain_total_mean', 0.0)) / 10000.0:.1f}",
                f"{float(item.get('fiscal_saving_vs_baseline_mean', 0.0)) / 10000.0:.1f}",
                f"{float(item.get('farmer_welfare_gain_total_mean', 0.0)) / 10000.0:.1f}",
                f"{_scenario_relative_value(item, 'delta_welfare_mean', 'marl_welfare_mean', 'baseline_welfare_mean'):.1f}",
                f"{_scenario_relative_value(item, 'delta_average_fairness_mean', 'marl_average_fairness_mean', 'baseline_average_fairness_mean'):.3f}",
                f"{_scenario_relative_value(item, 'delta_enterprise_survival_rate_mean', 'marl_enterprise_survival_rate', 'baseline_enterprise_survival_rate'):.3f}",
            ]
        )
    return rows


def _render_markdown(report: dict[str, Any]) -> str:
    training = report["training_overview"]
    best = report["best_run"]
    latest = report["latest_run"]
    network = report["network"]
    level_cards, counterfactual_cards = _metric_cards(report)

    task_rows = [
        [_task_label(task_id), f"{metrics['demand']:.1f}", f"{metrics['served']:.1f}", f"{metrics['unmet']:.1f}", f"{metrics['service_rate']:.3f}"]
        for task_id, metrics in best["task_rollup"].items()
    ]
    village_rows = [
        [item["name"], f"{item['health']:.3f}", f"{item['trust']:.3f}", f"{item['last_demand']:.1f}", f"{item['last_served']:.1f}", f"{item['last_time']:.2f}"]
        for item in latest["village_snapshot"]
    ]
    constraint_rows = [[name, count] for name, count in best["constraint_flag_counts"].items()]
    scenario_rows = _scenario_rows(report)

    lines = [
        "# 低空健康服务 MARL 平台报告",
        "",
        f"- 生成时间：`{report['generated_at']}`",
        f"- 报告名称：`{report['report_name']}`",
        f"- 情景：`{_scenario_label(str(training.get('scenario_name', 'baseline')))} `".replace(" `", "`"),
        f"- 配置：`{training.get('config_name', '')}`",
        f"- 最优轮次：`{training['best_episode']}`",
        f"- 最优得分：`{training['best_score']:.3f}`",
        "",
        "## 当前水平值",
        "",
        "- 以下指标展示当前训练最优轮次下的系统实际水平值。",
        "",
    ]
    lines.extend([f"- {label}：`{value}`；{note}" for label, value, note in level_cards])
    lines.extend(
        [
            "",
            "## 反事实差值",
            "",
            "- 以下五项展示相对于同情景启发式基线的估算差值。",
            "",
        ]
    )
    lines.extend([f"- {label}：`{value}`；{note}" for label, value, note in counterfactual_cards])
    lines.extend(
        [
            "",
            "## 网络概况",
            "",
            f"- 村庄数：`{network['village_count']}`；枢纽数：`{network['hub_count']}`；医院数：`{network['hospital_count']}`",
            f"- 总人口：`{network['total_population']:.0f}`；平均传统就医时间：`{network['average_traditional_time']:.2f}`",
            f"- 平均风险权重：`{network['average_risk_weight']:.3f}`",
            "",
            "## 最优回合任务表现",
            "",
            _markdown_table(["任务", "需求", "服务", "未满足", "服务率"], task_rows) or "暂无任务数据",
            "",
            "## 最新期村庄快照",
            "",
            _markdown_table(["村庄", "健康", "信任", "需求", "服务", "响应时间"], village_rows) or "暂无村庄数据",
            "",
            "## 约束触发统计",
            "",
            _markdown_table(["约束", "触发次数"], constraint_rows) or "暂无约束触发记录",
            "",
            "## 策略与运行均值",
            "",
            f"- 企业价格均值：`{best['strategy_averages']['enterprise_price_mean']:.3f}`",
            f"- 企业产能均值：`{best['strategy_averages']['enterprise_capacity_mean']:.3f}`",
            f"- 企业可靠性投入均值：`{best['strategy_averages']['enterprise_reliability_mean']:.3f}`",
            f"- 医疗承接均值：`{best['strategy_averages']['medical_acceptance_mean']:.3f}`",
            f"- 医疗协同均值：`{best['strategy_averages']['medical_integration_mean']:.3f}`",
            f"- 政策总成本：`{float(best['policy_costs']['policy_cost_total']):.2f}`",
            "",
            "## 最新运行末期",
            "",
            f"- 最新健康：`{float(latest['last_step'].get('average_health', 0.0)):.3f}`",
            f"- 最新总需求 / 总服务：`{float(latest['last_step'].get('total_demand', 0.0)):.1f}` / `{float(latest['last_step'].get('total_served', 0.0)):.1f}`",
            f"- 最新公平性：`{float(latest['last_step'].get('fairness_score', 0.0)):.3f}`",
            f"- 最新覆盖缺口：`{float(latest['last_step'].get('coverage_gap', 0.0)):.3f}`",
            f"- 最新偏远村服务率：`{float(latest['last_step'].get('remote_service_rate', 0.0)):.3f}`",
            "",
            "## 情景对照",
            "",
            "- 下表为各情景相对基准情景的最终结果差值；寿命、经济、财政与农户福利四项同样为反事实差值。",
            "",
            _markdown_table(
                ["情景", "运行数", "覆盖", "健康", "服务率", "企业负载率", "慢病管理率", "检后管理率", "筛查闭环率", "乡镇信任指数", "重症外流率", "村民寿命提升", "慢病寿命提升", "社会经济收益增加(万元)", "政府财政节约(万元)", "农户福利改善(万元)", "福利", "公平性", "企业存活率"],
                scenario_rows,
            )
            or "本次未附带批量实验情景对照结果。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_html(report: dict[str, Any]) -> str:
    training = report["training_overview"]
    best = report["best_run"]
    latest = report["latest_run"]
    level_cards, counterfactual_cards = _metric_cards(report)
    task_rows = [
        [_task_label(task_id), f"{metrics['demand']:.1f}", f"{metrics['served']:.1f}", f"{metrics['unmet']:.1f}", f"{metrics['service_rate']:.3f}"]
        for task_id, metrics in best["task_rollup"].items()
    ]
    village_rows = [
        [item["name"], f"{item['health']:.3f}", f"{item['trust']:.3f}", f"{item['last_demand']:.1f}", f"{item['last_served']:.1f}", f"{item['last_time']:.2f}"]
        for item in latest["village_snapshot"]
    ]
    constraint_rows = [[name, count] for name, count in best["constraint_flag_counts"].items()]
    scenario_rows = _scenario_rows(report)

    def render_card_group(title: str, note: str, items: list[tuple[str, str, str]], tone: str) -> str:
        cards = "".join(
            f"<div class='metric-card {tone}'><div class='metric-label'>{html.escape(label)}</div><div class='metric-value'>{html.escape(value)}</div><div class='metric-note'>{html.escape(note_text)}</div></div>"
            for label, value, note_text in items
        )
        return (
            "<div class='metric-section'>"
            f"<div class='section-head'><h2>{html.escape(title)}</h2><div class='section-note'>{html.escape(note)}</div></div>"
            f"<div class='cards'>{cards}</div>"
            "</div>"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>低空健康服务 MARL 平台报告</title>
  <style>
    body {{ font-family: Arial, 'Microsoft YaHei', sans-serif; background: #f5f8fc; color: #0f172a; margin: 0; }}
    .wrap {{ max-width: 1440px; margin: 0 auto; padding: 28px 24px 40px; }}
    .meta {{ color: #475569; margin-bottom: 18px; }}
    .metric-section {{ background: #ffffff; border: 1px solid #d9e3ef; border-radius: 18px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05); padding: 18px 18px 20px; margin-top: 16px; }}
    .section-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 14px; }}
    .section-note {{ color: #64748b; font-size: 13px; max-width: 760px; line-height: 1.55; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .metric-card {{ border-radius: 14px; padding: 14px 16px; border: 1px solid #dbe7f4; background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); }}
    .metric-card.counterfactual {{ border-color: #cfe0f3; background: linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%); }}
    .metric-label {{ color: #5b708b; font-size: 12px; font-weight: 700; }}
    .metric-value {{ font-size: 26px; font-weight: 800; margin-top: 8px; }}
    .metric-note {{ color: #64748b; font-size: 13px; margin-top: 8px; line-height: 1.45; }}
    .panel {{ background: #ffffff; border: 1px solid #d9e3ef; border-radius: 18px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05); padding: 18px; margin-top: 16px; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
    .pill {{ display: inline-block; background: #edf4ff; border: 1px solid #d9e8ff; border-radius: 999px; padding: 4px 10px; margin-right: 8px; margin-bottom: 8px; font-size: 12px; color: #183150; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 9px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #f8fafc; }}
    .muted {{ color: #64748b; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0; font-size: 20px; }}
    ul {{ margin: 0; padding-left: 18px; }}
    @media (max-width: 1100px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>低空健康服务 MARL 平台报告</h1>
  <div class="meta">生成时间：{html.escape(report['generated_at'])} | 情景：{html.escape(_scenario_label(str(training.get('scenario_name', 'baseline'))))} | 报告名称：{html.escape(report['report_name'])}</div>
  {render_card_group('当前水平值', '以下指标展示当前训练最优轮次下的系统实际水平值。', level_cards, 'level')}
  {render_card_group('反事实差值', '以下五项展示相对于同情景启发式基线的估算差值。', counterfactual_cards, 'counterfactual')}
  <div class="panel">
    <h2>训练与网络概况</h2>
    <div style="margin-top:12px;">
      <span class="pill">配置：{html.escape(str(training.get('config_name', '')))}</span>
      <span class="pill">最优轮次：{training.get('best_episode', 0)}</span>
      <span class="pill">最优得分：{float(training.get('best_score', 0.0)):.2f}</span>
      <span class="pill">评估点数：{training.get('evaluation_points', 0)}</span>
      <span class="pill">是否早停：{training.get('stopped_early', False)}</span>
    </div>
    <ul style="margin-top:12px;">
      <li>村庄数：{report['network']['village_count']}；枢纽数：{report['network']['hub_count']}；医院数：{report['network']['hospital_count']}</li>
      <li>总人口：{report['network']['total_population']:.0f}；平均传统就医时间：{report['network']['average_traditional_time']:.2f}</li>
      <li>平均风险权重：{report['network']['average_risk_weight']:.3f}</li>
    </ul>
  </div>
  <div class="grid-2">
    <div class="panel">
      <h2>策略与运行均值</h2>
      <ul style="margin-top:12px;">
        <li>企业价格均值：{best['strategy_averages']['enterprise_price_mean']:.3f}</li>
        <li>企业产能均值：{best['strategy_averages']['enterprise_capacity_mean']:.3f}</li>
        <li>企业可靠性投入均值：{best['strategy_averages']['enterprise_reliability_mean']:.3f}</li>
        <li>医疗承接均值：{best['strategy_averages']['medical_acceptance_mean']:.3f}</li>
        <li>医疗协同均值：{best['strategy_averages']['medical_integration_mean']:.3f}</li>
        <li>政策总成本：{float(best['policy_costs']['policy_cost_total']):.2f}</li>
      </ul>
    </div>
    <div class="panel">
      <h2>最新运行末期</h2>
      <ul style="margin-top:12px;">
        <li>平均健康：{float(latest['last_step'].get('average_health', 0.0)):.3f}</li>
        <li>总需求 / 总服务：{float(latest['last_step'].get('total_demand', 0.0)):.1f} / {float(latest['last_step'].get('total_served', 0.0)):.1f}</li>
        <li>公平性：{float(latest['last_step'].get('fairness_score', 0.0)):.3f}</li>
        <li>覆盖缺口：{float(latest['last_step'].get('coverage_gap', 0.0)):.3f}</li>
        <li>偏远村服务率：{float(latest['last_step'].get('remote_service_rate', 0.0)):.3f}</li>
      </ul>
      <div style="margin-top:12px;">{_html_table(['约束', '触发次数'], constraint_rows)}</div>
    </div>
  </div>
  <div class="grid-2">
    <div class="panel">
      <h2>最优回合任务表现</h2>
      <div style="margin-top:12px;">{_html_table(['任务', '需求', '服务', '未满足', '服务率'], task_rows)}</div>
    </div>
    <div class="panel">
      <h2>最新期村庄快照</h2>
      <div style="margin-top:12px;">{_html_table(['村庄', '健康', '信任', '需求', '服务', '响应时间'], village_rows)}</div>
    </div>
  </div>
  <div class="panel">
    <h2>情景对照</h2>
    <p class="muted" style="margin:12px 0 0;">下表为各情景相对基准情景的最终结果差值；寿命、经济、财政与农户福利四项同样为反事实差值。</p>
    <div style="margin-top:12px;">{_html_table(['情景', '运行数', '覆盖', '健康', '服务率', '企业负载率', '慢病管理率', '检后管理率', '筛查闭环率', '乡镇信任指数', '重症外流率', '村民寿命提升', '慢病寿命提升', '社会经济收益增加(万元)', '政府财政节约(万元)', '农户福利改善(万元)', '福利', '公平性', '企业存活率'], scenario_rows)}</div>
  </div>
</div>
</body>
</html>"""


def export_platform_report(
    output_root: str | Path,
    report_name: str,
    experiment_summary_path: str | Path | None = None,
    runtime_request: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_root = Path(output_root)
    live_dir = output_root / "live"
    report_dir = output_root / "reports" / report_name
    report_dir.mkdir(parents=True, exist_ok=True)

    history_path = live_dir / "spatial_marl_live_metrics.json"
    latest_trace_path = live_dir / "spatial_marl_latest_trace.json"
    best_trace_path = live_dir / "spatial_marl_best_trace.json"
    checkpoint_path = output_root / "checkpoints" / "spatial_marl_policy_checkpoint.json"

    if not history_path.exists() or not latest_trace_path.exists() or not best_trace_path.exists():
        raise FileNotFoundError("未找到可导出的训练产物，请先完成至少一次训练。")

    history_payload = _load_json(history_path)
    latest_trace_payload = _load_json(latest_trace_path)
    best_trace_payload = _load_json(best_trace_path)
    checkpoint_payload = _load_json_if_exists(checkpoint_path)
    experiment_summary = _load_json(Path(experiment_summary_path)) if experiment_summary_path else None

    history = history_payload.get("history", [])
    latest_point = history[-1] if history else {}
    resolved_request = _resolve_runtime_request(output_root, runtime_request)
    training_overview = {
        "config_name": resolved_request.get("config_name") if resolved_request else "",
        "config_path": resolved_request.get("config_path") if resolved_request else "",
        "scenario_name": resolved_request.get("scenario_name") if resolved_request else "baseline",
        "run_name": resolved_request.get("run_name") if resolved_request else report_name,
        "seed": resolved_request.get("seed") if resolved_request else history_payload.get("training_config", {}).get("seed"),
        "episodes": resolved_request.get("episodes") if resolved_request else history_payload.get("training_config", {}).get("episodes"),
        "eval_every": resolved_request.get("eval_every") if resolved_request else history_payload.get("training_config", {}).get("eval_every"),
        "best_episode": history_payload.get("best_episode", 0),
        "best_score": float(history_payload.get("best_score", 0.0)),
        "stopped_early": bool(history_payload.get("stopped_early", False)),
        "evaluation_points": len(history),
        "latest_point": latest_point,
        "checkpoint_available": checkpoint_payload is not None,
        "baseline_summary": history_payload.get("baseline_summary", {}),
        "latest_outcome_mapping": history_payload.get("latest_outcome_mapping", {}),
        "best_outcome_mapping": history_payload.get("best_outcome_mapping", {}),
    }
    network = _network_snapshot(best_trace_payload.get("network", latest_trace_payload.get("network", {})))
    best_run = _trace_snapshot(best_trace_payload)
    latest_run = _trace_snapshot(latest_trace_payload)

    report_payload = {
        "schema_version": 2,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_name": report_name,
        "runtime_request": resolved_request,
        "training_overview": training_overview,
        "network": network,
        "best_run": best_run,
        "latest_run": latest_run,
        "scenario_comparison": _scenario_comparison(experiment_summary),
        "source_files": {
            "history_json": str(history_path),
            "latest_trace_json": str(latest_trace_path),
            "best_trace_json": str(best_trace_path),
            "checkpoint_json": str(checkpoint_path),
            "experiment_summary_json": "" if experiment_summary_path is None else str(Path(experiment_summary_path)),
        },
    }

    report_json = report_dir / "report.json"
    report_md = report_dir / "report.md"
    report_html = report_dir / "report.html"
    report_json.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(_render_markdown(report_payload), encoding="utf-8")
    report_html.write_text(_render_html(report_payload), encoding="utf-8")
    return {
        "generated_at": report_payload["generated_at"],
        "report_name": report_name,
        "report_dir": str(report_dir),
        "report_json": str(report_json),
        "report_md": str(report_md),
        "report_html": str(report_html),
    }


__all__ = ["export_platform_report"]
