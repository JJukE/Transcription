#!/usr/bin/env python3
"""Host-only launcher: isolate a dedicated container, run ASR, restore networking."""
from __future__ import annotations
import argparse
import csv
import datetime
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from node_resource_gate import sample_cpu_ram, add_gpu_evidence
from transcription_settings import load_settings

ROOT = Path(__file__).resolve().parent
CONTAINER_ROOT = "/workspace/transcription"
HOST_STATE = Path.home() / ".cache" / "transcription-network"


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def inspect(name):
    return json.loads(docker("inspect", name))[0]


def write_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def verify_isolated(container):
    config = inspect(container)
    if config["NetworkSettings"]["Networks"]:
        raise RuntimeError("Container still has Docker network endpoints")
    interfaces = docker("exec", container, "ls", "/sys/class/net").splitlines()
    ipv4 = docker("exec", container, "cat", "/proc/net/route").splitlines()
    ipv6 = docker("exec", container, "cat", "/proc/net/ipv6_route").splitlines()
    if interfaces != ["lo"] or len(ipv4) != 1 or any(row.split()[-1] != "lo" for row in ipv6):
        raise RuntimeError("External network interfaces or routes remain")
    return {"interfaces": interfaces, "ipv4_routes": 0, "external_ipv6_routes": 0}


def resource_gate(gpu=0):
    decision = sample_cpu_ram()
    inventory = list(csv.reader(subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits"], text=True).splitlines(), skipinitialspace=True))
    apps = list(csv.reader(subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits"], text=True).splitlines(), skipinitialspace=True))
    rows = [{"index": int(r[0]), "uuid": r[1], "name": r[2],
             "memory_used_mib": int(r[3]), "memory_total_mib": int(r[4]),
             "compute_processes": [a for a in apps if a[0] == r[1]]} for r in inventory]
    result = add_gpu_evidence(decision, rows, selected_gpu=gpu, gpu_required=True)
    if result["decision"] != "pass":
        raise RuntimeError("Resource gate blocked: " + json.dumps(result))
    return result


def task_finished(token):
    path = ROOT / ".network-state" / f"task-{token}.json"
    return path.exists() and json.loads(path.read_text()).get("status") == "finished"


def restore(container, state):
    if state.get("task_started") and not task_finished(state["token"]):
        raise RuntimeError("Transcription process exit is unconfirmed; keeping network isolated")
    networks = inspect(container)["NetworkSettings"]["Networks"]
    if any(name != "bridge" for name in networks):
        raise RuntimeError("Unexpected network change; refusing automatic recovery")
    if "bridge" not in networks:
        docker("network", "connect", "bridge", container)
    current = inspect(container)
    if set(current["NetworkSettings"]["Networks"]) != {"bridge"}:
        raise RuntimeError("Bridge restoration failed")
    if current["HostConfig"]["PortBindings"] != state["port_bindings"]:
        raise RuntimeError("Published port configuration changed")
    for port, bindings in state["port_bindings"].items():
        actual = current["NetworkSettings"]["Ports"].get(port) or []
        if {b["HostPort"] for b in actual} != {b["HostPort"] for b in bindings}:
            raise RuntimeError("Published port restoration failed: " + port)
    interfaces = docker("exec", container, "ls", "/sys/class/net").splitlines()
    routes = docker("exec", container, "cat", "/proc/net/route").splitlines()
    if "eth0" not in interfaces or not any(r.split()[1] == "00000000" for r in routes[1:]):
        raise RuntimeError("Bridge interface/default route was not restored")
    state.update(status="restored", restored_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 restored_ip=current["NetworkSettings"]["Networks"]["bridge"]["IPAddress"],
                 restored_ports=current["NetworkSettings"]["Ports"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", help="Dedicated container name (CLI, local config, then transcription)")
    parser.add_argument("--container-root", help="Absolute project path inside the container")
    parser.add_argument("--config", help="Host path to machine-local configuration")
    parser.add_argument("--backend", choices=["whisper", "qwen"], default="whisper")
    parser.add_argument("--entrypoint", choices=["transcribe", "transcribe_long"], default="transcribe_long")
    parser.add_argument("--receipt", help="Host path for the network lifecycle receipt")
    parser.add_argument("--recover", help="Recover a retained state JSON after confirming process exit")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    settings = load_settings(args.config)
    args.container = args.container or settings.get("container", "transcription")
    container_root = args.container_root or settings.get("container_root", CONTAINER_ROOT)
    if not Path(container_root).is_absolute():
        raise ValueError("--container-root must be an absolute container path")
    state_dir = HOST_STATE
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / f"{args.container}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.recover:
            path = Path(args.recover).resolve()
            state = json.loads(path.read_text())
            if state["container"] != args.container or state["host"] != socket.gethostname():
                raise RuntimeError("Recovery target mismatch")
            restore(args.container, state)
            write_state(path, state)
            print("Network restored")
            return 0
        for path in state_dir.glob("network-*.json"):
            old = json.loads(path.read_text())
            if old.get("container") == args.container and old.get("status") not in ("restored", "prepared"):
                raise RuntimeError(f"Unresolved network state: {path}; recover before a new run")
        before = inspect(args.container)
        if not before["State"]["Running"] or before["HostConfig"]["NetworkMode"] != "bridge" or set(before["NetworkSettings"]["Networks"]) != {"bridge"}:
            raise RuntimeError("Expected a running dedicated container on the default bridge")
        identifier = before["Id"]  # In-memory identity guard; never persist container IDs.
        for other_name in docker("ps", "--format", "{{.Names}}").splitlines():
            if other_name != args.container:
                mode = inspect(other_name)["HostConfig"]["NetworkMode"]
                if mode.startswith("container:") and mode.split(":", 1)[1] in (args.container, identifier, identifier[:12]):
                    raise RuntimeError("Another container shares this network namespace")
        token = uuid.uuid4().hex
        state_path = state_dir / f"network-{token}.json"
        state = {"container": args.container, "host": socket.gethostname(), "token": token,
                 "status": "prepared", "task_started": False, "port_bindings": before["HostConfig"]["PortBindings"],
                 "original_networks": ["bridge"], "original_ip": before["NetworkSettings"]["Networks"]["bridge"]["IPAddress"],
                 "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "resource_gate": resource_gate()}
        state["recovery_command"] = f"python3 {ROOT / 'run_container.py'} --container {args.container} --recover {state_path}"
        write_state(state_path, state)
        extra = args.args[1:] if args.args[:1] == ["--"] else args.args
        python = settings.get(f"{args.backend}_python", f"/opt/conda/envs/{'qwen3-asr' if args.backend=='qwen' else 'whisper'}/bin/python")
        model_args = ["--model-dir", settings[f"{args.backend}_model_dir"]] if f"{args.backend}_model_dir" in settings else []
        command = [python, f"{container_root}/{args.entrypoint}.py", "--backend", args.backend, *model_args, *extra]
        state["command"] = command
        process = None
        code = 1
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, interrupted)
        try:
            if inspect(args.container)["Id"] != identifier:
                raise RuntimeError("Container identity changed")
            state["status"] = "disconnecting"
            write_state(state_path, state)
            docker("network", "disconnect", "bridge", args.container)
            state.update(status="isolated", isolation_verification=verify_isolated(args.container))
            write_state(state_path, state)
            print("Container network isolated; starting ASR", flush=True)
            state["task_started"] = True
            write_state(state_path, state)
            try:
                process = subprocess.Popen(["docker", "exec", "-e", "HF_HUB_OFFLINE=1", "-e", "HF_HUB_DISABLE_TELEMETRY=1",
                    "-e", "DO_NOT_TRACK=1", "-e", "CUDA_VISIBLE_DEVICES=0", args.container,
                    python, f"{container_root}/container_task.py", "--token", token, "--", *command])
            except OSError:
                state["task_started"] = False
                write_state(state_path, state)
                raise
            code = process.wait()
            state["asr_returncode"] = code
        except KeyboardInterrupt:
            code = 130
            state["cancelled"] = True
        finally:
            # Ignore repeat interruptions while cleanup is in progress.
            for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
                signal.signal(signum, signal.SIG_IGN)
            if process is not None and not task_finished(token):
                for _ in range(40):
                    docker("exec", args.container, python, f"{container_root}/container_task.py", "--token", token, "--stop")
                    if task_finished(token):
                        break
                    time.sleep(0.5)
            try:
                if inspect(args.container)["Id"] != identifier:
                    raise RuntimeError("Container identity changed; recovery refused")
                restore(args.container, state)
                print("Container network restored", flush=True)
            except Exception as error:
                state.update(status="recovery_required", recovery_error=str(error))
                code = 1
                print(f"Network recovery required: {error}\n{state['recovery_command']}", file=sys.stderr)
            write_state(state_path, state)
            if args.receipt:
                receipt = Path(args.receipt).expanduser().resolve()
                if receipt.is_relative_to(ROOT):
                    target = str(Path(container_root) / receipt.relative_to(ROOT))
                    subprocess.run(["docker", "exec", "-i", args.container, python, "-c",
                        "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(sys.stdin.read())", target],
                        input=json.dumps(state, ensure_ascii=False, indent=2), text=True, check=True)
                else:
                    receipt.parent.mkdir(parents=True, exist_ok=True)
                    write_state(receipt, state)
        return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
