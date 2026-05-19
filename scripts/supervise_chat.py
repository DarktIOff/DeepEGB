#!/usr/bin/env python3
from __future__ import annotations

"""Supervise a one-shot `deepegb chat --plain` run.

This helper launches the real CLI, polls the emitted log file, records
lightweight checkpoints, and writes a markdown report summarizing runtime
behaviour, failures, and a few project-specific integrity checks.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ErrorSpec:
    severity: str
    pattern: str
    likely_source: str


ERROR_SPECS: dict[str, ErrorSpec] = {
    "mcp_init_failure": ErrorSpec(
        severity="medium",
        pattern=r"Failed to initialize MCP toolkit|MCP not connected|arXiv MCP not connected",
        likely_source="src/deepegb/agent/runtime.py:296-307",
    ),
    "llm_connection_failure": ErrorSpec(
        severity="critical",
        pattern=r"Connection refused|APIConnectionError|timed out|ReadTimeout|ConnectError",
        likely_source="src/deepegb/agent/llm.py:46-122",
    ),
    "llm_tool_call_failure": ErrorSpec(
        severity="critical",
        pattern=r"tool call.*failed|function call.*error|malformed JSON|No tool call",
        likely_source="src/deepegb/agent/runtime.py:389-445",
    ),
    "pysr_api_mismatch": ErrorSpec(
        severity="high",
        pattern=r"equation_file|not a valid keyword|PySR rejected a constructor argument|unexpected keyword argument",
        likely_source="src/deepegb/agent/tools.py:27-147",
    ),
    "julia_empty_collection": ErrorSpec(
        severity="high",
        pattern=r"UNHANDLED TASK ERROR|ArgumentError: reducing over an empty collection",
        likely_source="src/deepegb/search/pysr_search.py:1164-1246",
    ),
    "julia_other_crash": ErrorSpec(
        severity="high",
        pattern=r"TaskFailedException|MethodError|BoundsError|LoadError|Segmentation fault|Julia .* failed",
        likely_source="src/deepegb/search/pysr_search.py:1164-1246",
    ),
    "pysr_nondeterminism_warning": ErrorSpec(
        severity="low",
        pattern=r"UserWarning: Note: Setting random_state",
        likely_source="src/deepegb/search/pysr_search.py:905-1048",
    ),
    "numerical_warning": ErrorSpec(
        severity="medium",
        pattern=r"RuntimeWarning: divide by zero|RuntimeWarning: invalid value|overflow encountered|invalid value encountered",
        likely_source="src/deepegb/physics/egb_perturbations.py:905-963",
    ),
    "nan_observables": ErrorSpec(
        severity="high",
        pattern=r'"n_s":\\s*NaN|"r":\\s*NaN|"delta1":\\s*NaN|observables NaN',
        likely_source="src/deepegb/physics/diagnostics.py:89-187",
    ),
    "tool_error": ErrorSpec(
        severity="high",
        pattern=r"TOOL_ERROR:",
        likely_source="src/deepegb/agent/tools.py:27-147",
    ),
    "julia_fallback": ErrorSpec(
        severity="medium",
        pattern=r"falling back to multi-family MSE|Julia-loss path failed|Julia ξ-loss path failed|using multi-family MSE only",
        likely_source="src/deepegb/search/pysr_search.py:905-1048",
    ),
    "agent_crash": ErrorSpec(
        severity="critical",
        pattern=r"Traceback \(most recent call last\):|Exception:",
        likely_source="src/deepegb/agent/runtime.py:355-451",
    ),
    "async_mismatch": ErrorSpec(
        severity="critical",
        pattern=r"can't be used with synchronous|asyncio\\.run\\(\\) cannot be called|event loop|coroutine",
        likely_source="src/deepegb/agent/runtime.py:328-338",
    ),
    "process_hang": ErrorSpec(
        severity="critical",
        pattern=r"",
        likely_source="src/deepegb/search/pysr_search.py:1193-1246",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervise `deepegb chat --plain`.")
    parser.add_argument("--message", "-m", required=True, help="One-shot chat prompt.")
    parser.add_argument("--provider", default=None, help="Optional LLM provider override.")
    parser.add_argument("--no-arxiv", action="store_true", help="Disable arXiv MCP tools.")
    parser.add_argument("--no-rag", action="store_true", help="Disable local RAG tool.")
    parser.add_argument("--poll-interval", type=float, default=15.0, help="Polling interval in seconds.")
    parser.add_argument("--stall-threshold", type=float, default=120.0, help="No-log-output threshold in seconds.")
    parser.add_argument("--out-dir", default="runs/supervised", help="Parent directory for supervised runs.")
    parser.add_argument("--deepegb-bin", default=None, help="Explicit `deepegb` executable path.")
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolve_deepegb_command(deepegb_bin: str | None) -> tuple[list[str], dict[str, str]]:
    if deepegb_bin:
        return [deepegb_bin], {}
    discovered = shutil.which("deepegb")
    if discovered:
        return [discovered], {}
    env_updates: dict[str, str] = {}
    src_path = str(ROOT / "src")
    existing = os.environ.get("PYTHONPATH")
    env_updates["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing}" if existing else src_path
    return [sys.executable, "-m", "deepegb.cli"], env_updates


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    base_cmd, env_updates = resolve_deepegb_command(args.deepegb_bin)
    command = [*base_cmd, "chat", "--plain", "-m", args.message]
    if args.provider:
        command.extend(["--provider", args.provider])
    if args.no_arxiv:
        command.append("--no-arxiv")
    if args.no_rag:
        command.append("--no-rag")
    return command, env_updates


def create_run_dir(out_dir: str) -> Path:
    run_dir = ROOT / out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def read_new_log_text(log_path: Path, offset: int) -> tuple[str, int]:
    if not log_path.exists():
        return "", offset
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        text = handle.read()
        return text, handle.tell()


def detect_categories(text: str) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    for category, spec in ERROR_SPECS.items():
        if category == "process_hang":
            continue
        if re.search(spec.pattern, text, flags=re.IGNORECASE):
            found.append(category)
    return found


def run_command(command: list[str]) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    return proc.stdout


def get_child_processes(pid: int) -> list[dict[str, str]]:
    output = run_command(["ps", "-o", "pid=,ppid=,stat=,etime=,args=", "--ppid", str(pid)])
    children: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=4)
        if len(parts) != 5:
            continue
        children.append(
            {
                "pid": parts[0],
                "ppid": parts[1],
                "stat": parts[2],
                "etime": parts[3],
                "args": parts[4],
            }
        )
    return children


def child_processes_look_active(children: list[dict[str, str]]) -> bool:
    return any(str(child.get("stat", "")).startswith(("R", "D")) for child in children)


def classify_stall(children: list[dict[str, str]], stalled_for_threshold: bool) -> bool:
    if not stalled_for_threshold:
        return False
    return not child_processes_look_active(children)


def local_llm_connection_visible() -> bool:
    output = run_command(["ss", "-tpn"])
    return "127.0.0.1:8001" in output


def append_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_checkpoints(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    checkpoints: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                checkpoints.append(json.loads(line))
    return checkpoints


def count_errors(full_log: str, checkpoints: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for category, spec in ERROR_SPECS.items():
        if category == "process_hang":
            stalled = sum(1 for cp in checkpoints if cp.get("stalled"))
            if stalled:
                counts[category] = stalled
            continue
        matches = re.findall(spec.pattern, full_log, flags=re.IGNORECASE)
        if matches:
            counts[category] = len(matches)
    return counts


def extract_delta1_values(full_log: str) -> list[float]:
    values: list[float] = []
    for match in re.findall(r'"delta1"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)', full_log):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def integrity_checks(full_log: str, counts: dict[str, int]) -> list[str]:
    lines: list[str] = []

    if re.search(r'"xi_expr"\s*:\s*"0"', full_log):
        lines.append("- `nontrivial_xi`: FAIL - found `\"xi_expr\": \"0\"` in the log.")
    else:
        lines.append("- `nontrivial_xi`: PASS - no explicit GR-limit `xi_expr = 0` found.")

    delta1_values = extract_delta1_values(full_log)
    if any(abs(value) > 1.0e-4 for value in delta1_values):
        lines.append("- `delta1_not_gr_limit`: PASS - at least one parsed `delta1` exceeds `1e-4` in magnitude.")
    else:
        lines.append("- `delta1_not_gr_limit`: FAIL - no parsed `delta1` exceeds `1e-4`.")

    used_v = "[V search] Julia loss used: True" in full_log
    used_xi = "[ξ search] Julia: True" in full_log
    fallback_seen = bool(re.search(ERROR_SPECS["julia_fallback"].pattern, full_log, flags=re.IGNORECASE))
    if used_v and used_xi and not fallback_seen:
        lines.append("- `julia_physics_loss_used`: PASS - V and ξ searches both report Julia physics loss usage.")
    elif fallback_seen:
        lines.append("- `julia_physics_loss_used`: FAIL - Julia fallback markers were detected in the log.")
    else:
        lines.append("- `julia_physics_loss_used`: FAIL - did not find both Julia-loss success markers.")

    tool_hits = [
        tool for tool in (
            "search_egb_potentials",
            "analyze_egb_model_tool",
            "plot_egb_model_tool",
        )
        if tool in full_log
    ]
    if tool_hits:
        lines.append(f"- `tool_chain_complete`: observed tool names: {', '.join(tool_hits)}.")
    else:
        lines.append("- `tool_chain_complete`: FAIL - none of `search_egb_potentials`, `analyze_egb_model_tool`, `plot_egb_model_tool` appeared in the log.")

    explicit_failure = "TOOL_ERROR:" in full_log or "Traceback" in full_log
    surfaced_categories = {
        category for category, count in counts.items()
        if count and ERROR_SPECS[category].severity in {"high", "critical"}
        and category not in {"process_hang"}
    }
    if not surfaced_categories:
        lines.append("- `honest_failure_reporting`: PASS - no surfaced high-severity failures were detected.")
    elif explicit_failure:
        lines.append("- `honest_failure_reporting`: PASS - failures coincided with explicit `TOOL_ERROR:` or `Traceback` markers.")
    else:
        lines.append("- `honest_failure_reporting`: FAIL - high-severity failures were detected without explicit surfaced markers.")

    return lines


def build_timeline(checkpoints: list[dict[str, Any]]) -> list[str]:
    if not checkpoints:
        return ["- No checkpoints were recorded."]
    lines: list[str] = []
    for checkpoint in checkpoints:
        status = "alive" if checkpoint.get("process_alive") else "exited"
        details: list[str] = [status]
        if checkpoint.get("new_log_bytes", 0):
            details.append(f"new_log_bytes={checkpoint['new_log_bytes']}")
        if checkpoint.get("matched_error_categories"):
            details.append("errors=" + ", ".join(checkpoint["matched_error_categories"]))
        if checkpoint.get("stalled"):
            details.append("stalled=true")
        if checkpoint.get("local_llm_connection"):
            details.append("local_llm_connection=true")
        exit_code = checkpoint.get("exit_code")
        if exit_code is not None:
            details.append(f"exit_code={exit_code}")
        lines.append(f"- `{checkpoint['timestamp']}`: " + "; ".join(details))
    return lines


def build_error_inventory(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["No error categories matched the collected log."]
    lines = ["| Category | Count | Severity | Pattern | Likely Source |", "| --- | ---: | --- | --- | --- |"]
    for category in sorted(counts):
        spec = ERROR_SPECS[category]
        pattern = spec.pattern if spec.pattern else "stall-based"
        lines.append(
            f"| `{category}` | {counts[category]} | {spec.severity} | `{pattern}` | `{spec.likely_source}` |"
        )
    return lines


def write_report(
    report_path: Path,
    command: list[str],
    log_path: Path,
    checkpoint_path: Path,
    exit_code: int | None,
    interrupted: bool,
) -> None:
    full_log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    checkpoints = load_checkpoints(checkpoint_path)
    counts = count_errors(full_log, checkpoints)
    summary_bits: list[str] = []
    if interrupted:
        summary_bits.append("run interrupted by signal")
    if exit_code is not None:
        summary_bits.append(f"exit_code={exit_code}")
    if counts:
        summary_bits.append("matched categories: " + ", ".join(sorted(counts)))
    else:
        summary_bits.append("no matched error categories")

    lines: list[str] = ["# DeepEGB Chat Supervision Report", "", "## Command"]
    lines.append(f"- Working directory: `{ROOT}`")
    lines.append(f"- Command: `{shlex.join(command)}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- `chat.log`: `{log_path}`")
    lines.append(f"- `supervisor.jsonl`: `{checkpoint_path}`")
    lines.append(f"- `report.md`: `{report_path}`")
    lines.append("")
    lines.append("## Exit Status")
    lines.append(f"- Interrupted: `{interrupted}`")
    lines.append(f"- Exit code: `{exit_code}`")
    lines.append(f"- Checkpoints recorded: `{len(checkpoints)}`")
    lines.append("")
    lines.append("## Timeline")
    lines.extend(build_timeline(checkpoints))
    lines.append("")
    lines.append("## Error Inventory")
    lines.extend(build_error_inventory(counts))
    lines.append("")
    lines.append("## Integrity Checks")
    lines.extend(integrity_checks(full_log, counts))
    lines.append("")
    lines.append("## Summary")
    lines.append("- " + "; ".join(summary_bits))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def terminate_process(proc: subprocess.Popen[Any]) -> None:
    try:
        if proc.poll() is not None:
            return
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass


def supervise(args: argparse.Namespace) -> int:
    run_dir = create_run_dir(args.out_dir)
    log_path = run_dir / "chat.log"
    checkpoint_path = run_dir / "supervisor.jsonl"
    report_path = run_dir / "report.md"
    command, env_updates = build_command(args)

    env = os.environ.copy()
    env.update(env_updates)
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )

        interrupted = False
        log_offset = 0
        last_log_activity = time.time()
        exit_code: int | None = None

        def handle_signal(signum: int, _frame: Any) -> None:
            nonlocal interrupted
            interrupted = True
            print(f"Received signal {signum}; terminating child {proc.pid}", file=sys.stderr)
            terminate_process(proc)

        old_sigint = signal.getsignal(signal.SIGINT)
        old_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        try:
            print(f"Run directory: {run_dir}")
            print(f"Command: {shlex.join(command)}")
            while True:
                newly_appended, log_offset = read_new_log_text(log_path, log_offset)
                if newly_appended:
                    last_log_activity = time.time()
                exit_code = proc.poll()
                children = get_child_processes(proc.pid) if exit_code is None else []
                stalled = classify_stall(
                    children,
                    (time.time() - last_log_activity) > args.stall_threshold,
                )
                checkpoint = {
                    "timestamp": iso_now(),
                    "pid": proc.pid,
                    "process_alive": exit_code is None,
                    "exit_code": exit_code,
                    "new_log_bytes": len(newly_appended.encode("utf-8", errors="replace")),
                    "matched_error_categories": detect_categories(newly_appended),
                    "child_processes": children,
                    "local_llm_connection": local_llm_connection_visible(),
                    "stalled": stalled,
                }
                append_checkpoint(checkpoint_path, checkpoint)
                if exit_code is not None:
                    break
                time.sleep(args.poll_interval)
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            terminate_process(proc)

    # Final drain after the child closes the file.
    newly_appended, _ = read_new_log_text(log_path, log_offset)
    if newly_appended:
        append_checkpoint(
            checkpoint_path,
            {
                "timestamp": iso_now(),
                "pid": proc.pid,
                "process_alive": False,
                "exit_code": proc.poll(),
                "new_log_bytes": len(newly_appended.encode("utf-8", errors="replace")),
                "matched_error_categories": detect_categories(newly_appended),
                "child_processes": [],
                "local_llm_connection": local_llm_connection_visible(),
                "stalled": False,
            },
        )
        exit_code = proc.poll()

    write_report(report_path, command, log_path, checkpoint_path, exit_code, interrupted)
    print(f"Artifacts written under: {run_dir}")
    return 130 if interrupted else (exit_code or 0)


def main() -> None:
    raise SystemExit(supervise(parse_args()))


if __name__ == "__main__":
    main()