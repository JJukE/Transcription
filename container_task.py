"""Run one owned transcription process group and expose explicit cancellation."""
import argparse
import json
import os
import signal
import subprocess
from pathlib import Path


def save_state(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.token.isalnum():
        raise ValueError("Invalid task token")
    root = Path(__file__).resolve().parent / ".network-state"
    root.mkdir(exist_ok=True)
    path = root / f"task-{args.token}.json"
    if args.stop:
        if path.exists():
            state = json.loads(path.read_text())
            if state.get("status") == "running":
                try:
                    # Confirm this PID still belongs to this exact task before signalling.
                    actual = Path(f"/proc/{state['pid']}/stat").read_text().rsplit(")", 1)[1].split()[19]
                    if actual != state["start_ticks"]:
                        raise RuntimeError("PID was reused; cancellation refused")
                    os.killpg(state["pid"], signal.SIGTERM)
                except FileNotFoundError:
                    pass
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    process = subprocess.Popen(command, start_new_session=True)
    ticks = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    state = {"pid": process.pid, "start_ticks": ticks, "status": "running"}
    save_state(path, state)
    def cancel(signum, frame):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, cancel)
    try:
        code = process.wait()
    finally:
        state.update(status="finished", returncode=process.returncode)
        save_state(path, state)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
