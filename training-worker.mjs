import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.6/full/pyodide.mjs";

const PROJECT_ROOT = "/app";
const OUTPUT_ROOT = "/outputs";
const PACKAGE_ROOT = `${PROJECT_ROOT}/rc5_marl_market_model`;
const PYTHON_FILES = [
  "advanced_spatial.py",
  "config.py",
  "disturbances.py",
  "health_markov.py",
  "outcome_mapping.py",
  "platform_runtime.py",
  "report_export.py",
  "state.py",
  "trainable_spatial_marl.py",
];
const CONFIG_FILES = [
  "spatial_network_recommended.toml",
  "spatial_network_from_master.toml",
  "spatial_network_baseline.toml",
];
const DATA_FILES = [
  ["live/spatial_marl_live_metrics.json", `${OUTPUT_ROOT}/live/spatial_marl_live_metrics.json`],
  ["live/spatial_marl_latest_trace.json", `${OUTPUT_ROOT}/live/spatial_marl_latest_trace.json`],
  ["live/spatial_marl_best_trace.json", `${OUTPUT_ROOT}/live/spatial_marl_best_trace.json`],
  ["experiments/spatial_marl_batch_p1/summary.json", `${PROJECT_ROOT}/experiments/spatial_marl_batch_p1/summary.json`],
  ["experiments/smoke_farmer_mapping/summary.json", `${PROJECT_ROOT}/experiments/smoke_farmer_mapping/summary.json`],
  ["experiments/scenario_batch/summary.json", `${PROJECT_ROOT}/experiments/scenario_batch/summary.json`],
];

let pyodide;
let currentRequest = null;
let training = false;

function send(type, payload = {}) {
  self.postMessage({ type, payload });
}

function statePayload() {
  return {
    generated_at: new Date().toISOString(),
    status: training ? "running" : "idle",
    available_configs: Object.fromEntries(
      CONFIG_FILES.map((name) => [name.replace(".toml", ""), `${PROJECT_ROOT}/configs/${name}`]),
    ),
    available_scenarios: [
      "baseline",
      "harsh_weather",
      "low_trust",
      "sparse_demand",
      "chronic_burden",
      "policy_push",
    ],
    editor_defaults: {
      simulation: { base_need_rate: 0.18 },
      environment: { reimbursement_rate: 0.2 },
      policy: {
        per_service_subsidy: 0.15,
        performance_subsidy_per_coverage: 25,
        fairness_subsidy_per_point: 18,
      },
      constraints: {
        minimum_coverage_ratio: 0.55,
        fairness_floor: 0.65,
        remote_service_floor: 0.38,
      },
      disturbances: { weather_std: 0.18, airspace_std: 0.1, device_std: 0.08 },
    },
    current_request: currentRequest,
    master_compile_summary: null,
  };
}

async function fetchText(path) {
  const response = await fetch(new URL(path, self.location.href));
  if (!response.ok) {
    throw new Error(`无法加载运行文件：${path}`);
  }
  return response.text();
}

async function installRuntimeFiles() {
  pyodide.FS.mkdirTree(PACKAGE_ROOT);
  pyodide.FS.mkdirTree(`${PROJECT_ROOT}/configs`);
  pyodide.FS.writeFile(`${PACKAGE_ROOT}/__init__.py`, "", { encoding: "utf8" });

  const pythonSources = await Promise.all(
    PYTHON_FILES.map(async (name) => [name, await fetchText(`runtime/python/${name}`)]),
  );
  for (const [name, source] of pythonSources) {
    pyodide.FS.writeFile(`${PACKAGE_ROOT}/${name}`, source, { encoding: "utf8" });
  }

  const configSources = await Promise.all(
    CONFIG_FILES.map(async (name) => [name, await fetchText(`runtime/configs/${name}`)]),
  );
  for (const [name, source] of configSources) {
    pyodide.FS.writeFile(`${PROJECT_ROOT}/configs/${name}`, source, { encoding: "utf8" });
  }

  const dataSources = await Promise.all(
    DATA_FILES.map(async ([source, target]) => [target, await fetchText(source)]),
  );
  for (const [target, source] of dataSources) {
    pyodide.FS.mkdirTree(target.slice(0, target.lastIndexOf("/")));
    pyodide.FS.writeFile(target, source, { encoding: "utf8" });
  }
}

async function runPythonJson(source, payload) {
  pyodide.globals.set("request_json", JSON.stringify(payload));
  const result = await pyodide.runPythonAsync(source);
  return JSON.parse(result);
}

async function previewScenario(payload) {
  const result = await runPythonJson(`
import json
from datetime import datetime
from rc5_marl_market_model.platform_runtime import normalize_run_request, prepare_platform_preview

request = normalize_run_request(json.loads(request_json), "${PROJECT_ROOT}")
cfg, preview = prepare_platform_preview(request)
json.dumps({
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "request": {
        "config_name": request.config_name,
        "scenario_name": request.scenario_name,
        "episodes": request.episodes,
        "eval_every": request.eval_every,
        "seed": request.seed,
        "run_name": request.run_name,
        "report_name": request.report_name,
    },
    "snapshot": preview["snapshot"],
    "base_snapshot": preview["base_snapshot"],
    "scenario_snapshot": preview["scenario_snapshot"],
    "scenario_changes": preview["scenario_changes"],
    "override_changes": preview["override_changes"],
    "change_counts": preview["change_counts"],
    "task_types": preview["snapshot"]["task_types"],
    "villages": len(cfg.villages),
    "hubs": len(cfg.hubs),
    "hospitals": len(cfg.hospitals),
}, ensure_ascii=False)
`, payload);
  send("scenario_preview", result);
}

async function runTraining(payload) {
  if (training) {
    send("error", { message: "当前已有训练任务在运行" });
    return;
  }
  training = true;
  currentRequest = { ...payload };
  send("server_state", statePayload());
  try {
    const result = await runPythonJson(`
import json
from dataclasses import asdict
from pathlib import Path
from js import postMessage
from rc5_marl_market_model.platform_runtime import normalize_run_request, prepare_platform_config
from rc5_marl_market_model.trainable_spatial_marl import MARLTrainingConfig, train_spatial_marl

request = normalize_run_request(json.loads(request_json), "${PROJECT_ROOT}")
cfg, _snapshot = prepare_platform_config(request)

def emit(event):
    postMessage(json.dumps(event, ensure_ascii=False))

result = train_spatial_marl(
    config_path=request.config_path,
    training_cfg=MARLTrainingConfig(
        episodes=request.episodes,
        eval_every=request.eval_every,
        seed=request.seed,
    ),
    output_root="${OUTPUT_ROOT}",
    scenario_name=None,
    progress_callback=emit,
    preloaded_config=cfg,
)
live_dir = Path("${OUTPUT_ROOT}") / "live"
json.dumps({
    "metrics": json.loads((live_dir / "spatial_marl_live_metrics.json").read_text(encoding="utf-8")),
    "latest": json.loads((live_dir / "spatial_marl_latest_trace.json").read_text(encoding="utf-8")),
    "best": json.loads((live_dir / "spatial_marl_best_trace.json").read_text(encoding="utf-8")),
    "done": {
        "generated_at": result.artifacts.history_json.stat().st_mtime,
        "best_episode": result.best_episode,
        "best_score": result.best_score,
        "best_summary": asdict(result.best_summary),
    },
}, ensure_ascii=False)
`, payload);
    send("browser_training_result", result);
    send("platform_training_done", result.done);
  } finally {
    training = false;
    send("server_state", statePayload());
  }
}

async function exportReport(payload) {
  pyodide.globals.set("runtime_request_json", JSON.stringify(currentRequest));
  const result = await runPythonJson(`
import json
from pathlib import Path
from rc5_marl_market_model.report_export import export_platform_report

payload = json.loads(request_json)
report_name = str(payload.get("report_name") or "platform_report").strip() or "platform_report"
experiment = str(payload.get("experiment_summary") or "").strip()
experiment_path = Path("${PROJECT_ROOT}") / experiment if experiment else None
report = export_platform_report(
    output_root="${OUTPUT_ROOT}",
    report_name=report_name,
    experiment_summary_path=experiment_path,
    runtime_request=json.loads(runtime_request_json) if runtime_request_json != "null" else None,
)
json.dumps({
    "report_name": report_name,
    "files": {
        "html": Path(report["report_html"]).read_text(encoding="utf-8"),
        "md": Path(report["report_md"]).read_text(encoding="utf-8"),
        "json": Path(report["report_json"]).read_text(encoding="utf-8"),
    },
}, ensure_ascii=False)
`, payload);
  send("browser_report_files", result);
}

async function handleMessage(message) {
  try {
    switch (message.type) {
      case "get_state":
        send("server_state", statePayload());
        break;
      case "preview_scenario":
        await previewScenario(message.payload || {});
        break;
      case "run_training":
        await runTraining(message.payload || {});
        break;
      case "export_report":
        await exportReport(message.payload || {});
        break;
      default:
        send("error", { message: `不支持的消息类型：${message.type}` });
    }
  } catch (error) {
    training = false;
    send("error", { message: error instanceof Error ? error.message : String(error) });
  }
}

self.onmessage = (event) => {
  void handleMessage(event.data);
};

try {
  pyodide = await loadPyodide();
  await installRuntimeFiles();
  await pyodide.runPythonAsync(`import sys; sys.path.insert(0, "${PROJECT_ROOT}")`);
  send("runtime_ready");
} catch (error) {
  send("error", { message: error instanceof Error ? error.message : String(error) });
}
