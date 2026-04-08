#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests


ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "7860"))
BASE_URL = os.environ.get("ENV_URL", f"http://127.0.0.1:{PORT}")
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _run(command: list[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def _wait_for_server(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Server at {base_url} did not become healthy within {timeout_s} seconds.")


def main() -> None:
    report: Dict[str, Any] = {}

    local_validation = _run(["openenv", "validate", str(ROOT), "--json"], allow_failure=True)
    report["local_validation"] = json.loads(local_validation.stdout)

    server = subprocess.Popen(
        [sys.executable, "-m", "server.app", "--host", "0.0.0.0", "--port", str(PORT)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_server(BASE_URL)
        runtime_validation = _run(
            ["openenv", "validate", "--url", BASE_URL],
            allow_failure=True,
        )
        report["runtime_validation"] = json.loads(runtime_validation.stdout)

        root_response = requests.get(BASE_URL, timeout=5)
        report["root_ping"] = {
            "status_code": root_response.status_code,
            "ok": root_response.status_code == 200,
        }

        inference_result = _run([sys.executable, "inference.py"], allow_failure=True)
        report["inference"] = {
            "returncode": inference_result.returncode,
            "stdout_tail": inference_result.stdout.strip().splitlines()[-12:],
            "stderr_tail": inference_result.stderr.strip().splitlines()[-12:],
        }

        if shutil.which("docker"):
            docker_result = _run(
                ["docker", "build", "-t", "hackathon-openenv-local", "."],
                allow_failure=True,
            )
            report["docker_build"] = {
                "returncode": docker_result.returncode,
                "stderr_tail": docker_result.stderr.strip().splitlines()[-20:],
            }
        else:
            report["docker_build"] = {"skipped": True, "reason": "docker not installed"}
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    output_path = OUTPUT_DIR / "validation_report.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
