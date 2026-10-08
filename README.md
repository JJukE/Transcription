# Local audio transcription

Transcribe audio/video locally with **Whisper** or **Qwen3-ASR**, export subtitles, and compare transcript outputs. Whisper `large-v3` is the default. Qwen uses an independent environment and its ForcedAligner for word timestamps.

The repository contains code, tests and generic configuration examples. Recordings, transcripts, review notes, receipts, machine settings and model weights stay local and are excluded by `.gitignore`.

## Requirements and installation

- Linux and Python 3.10+ for the optional host Docker launcher.
- Python 3.12 in the ASR environments and FFmpeg on their `PATH`.
- CUDA inference was exercised with PyTorch `2.7.1+cu118`. The launcher checks physical GPU 0, even for an ASR request with `--device cpu`.
- The launcher requires Docker access and a **dedicated, already running container on the default `bridge` network**. Mount this checkout and model caches into it. Direct entrypoints do not require Docker.

Create separate environments on the machine/container that performs ASR:

```bash
conda create -n whisper python=3.12 -y
conda activate whisper
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-whisper.txt

conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-qwen.lock
python -m pip check
```

The tested Whisper version is `openai-whisper==20250625`. The Qwen lock records the observed full environment, including transitive packages, for this Python/CUDA setup. Its dependencies use version pins instead of local `file://` wheel/build paths; it is not a universal cross-platform lock.

## Machine-local settings

```bash
mkdir -p .local
cp config.example.json .local/config.json
```

Edit these fields for your mounts/container:

| Setting | Meaning |
|---|---|
| `container` | Dedicated Docker container name |
| `container_root` | Absolute checkout path inside the container |
| `whisper_python`, `qwen_python` | Absolute interpreters inside the container |
| `whisper_model_dir`, `qwen_model_dir` | Absolute caches in the ASR environment |

Scripts load `.local/config.json` beside the code by default. Use `--config PATH` or `TRANSCRIPTION_CONFIG` to select another file. CLI values override settings. An explicitly selected missing configuration fails before the launcher changes networking.

Without settings, the launcher uses container `transcription`, checkout `/workspace/transcription`, and `/opt/conda/envs/{whisper,qwen3-asr}/bin/python`. Direct entrypoints use `~/.cache/transcription/models/{whisper,qwen}`. These cache paths are interpreted **inside the ASR environment**; host/container paths can differ.

## Prepare models before isolated inference

Download models while networking is available. Activate the Whisper environment and run from this checkout:

```bash
python - <<'PY'
import whisper
from transcription_settings import model_directory
whisper.load_model("large-v3", device="cpu", download_root=model_directory("whisper"))
PY
```

Activate the Qwen environment and download its pinned official assets:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
from transcription_settings import model_directory
cache = Path(model_directory("qwen"))
models = [
    ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5"),
    ("Qwen/Qwen3-ForcedAligner-0.6B", "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"),
]
for repository, revision in models:
    snapshot_download(repository, revision=revision,
                      local_dir=cache / repository.split("/")[-1],
                      allow_patterns=["*.json", "*.txt", "*.safetensors"])
PY
```

Qwen loads local assets with `local_files_only=True`, `trust_remote_code=False` and `use_safetensors=True`. Missing assets are not downloaded during inference. Preserve model revision/file hashes and package versions in local receipts. Weights and receipts remain outside the repository.

## Transcribe

Activate the matching environment and run:

```bash
python transcribe_long.py --backend whisper \
  -i input/example.m4a -o output/whisper_run --language ko
python transcribe_long.py --backend qwen \
  -i input/example.m4a -o output/qwen_run --language ko --device cuda
```

Use `transcribe.py` for the other entrypoint. Both accept multiple inputs, `--model`, `--model-dir`, `--output-name`, `--output-format`, `--initial-prompt` and `--config`; see `--help`. Outputs are TXT/SRT/VTT/TSV/JSON. Long runs include diagnostics; Qwen also preserves raw chunk text/alignment and an execution receipt.

| Behavior | Whisper | Qwen |
|---|---|---|
| Default model | `large-v3` | `Qwen/Qwen3-ASR-1.7B` |
| Long-input chunk / overlap per side | 600 seconds / 1 second | 120 seconds / 2 seconds |
| Word timestamps | Opt-in | ForcedAligner |
| Translation | `--task translate` | Unsupported; rejected |
| Confidence | Backend-provided fields | No fabricated confidence |

Qwen uses BF16 on CUDA (FP32 on CPU), SDPA, batch 1, greedy decoding and `max_new_tokens=4096`. `ko` maps to `Korean`; `auto` enables language detection. Initial prompt supplies SDK context. Explicit Whisper-only threshold options are rejected. Qwen alignment chunks including overlap must be at most 180 seconds; `--aligner-dir` selects a local aligner.

Qwen assigns overlap by word midpoint, maps aligner tokens to original punctuation/spacing, and saves raw alignment. Valid structure/timestamps do not establish transcription accuracy.

## Container-only network isolation and recovery

Run from the **Docker host**, adapting paths to your configured container mounts:

```bash
python3 run_container.py --backend qwen \
  --receipt output/qwen_run/network.receipt.json -- \
  -i /workspace/transcription/input/example.m4a \
  -o /workspace/transcription/output/qwen_run --language ko --device cuda
```

Options before `--` are host-runner options; those after it are ASR options using **container paths**. `--receipt`/`--recover` use **host paths**. The runner selects `transcribe_long.py`; `--entrypoint transcribe` selects `transcribe.py`. `--container`/`--container-root` override local settings. Configured caches are forwarded to ASR; an explicit ASR `--model-dir` wins.

The launcher locks the selected container and checks CPU/RAM plus physical GPU 0. CPU or RAM at least 85%, unknown resource state, or any GPU compute process blocks execution. Desktop graphics memory is reported separately; unrelated workloads are not terminated.

It records network/port state, disconnects only the selected container's default-bridge endpoint, and verifies that only loopback remains with no external routes/endpoints. Failed verification prevents ASR startup. Shared container network namespaces are refused. ASR runs via `docker exec` with `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1` and physical GPU 0 selected. After the owned task exits, bridge networking, published ports and the default route are restored. Ordinary errors and SIGINT/SIGTERM/SIGHUP cancellation use the same cleanup path.

**All services in that container, including SSH/Jupyter, become unreachable while isolated.** Host SSH/networking, `docker exec`, mounted files, GPU access and other containers remain available. Direct ASR entrypoints do not establish isolation. No container recreation or Docker daemon configuration is performed.

Hard termination or host/Docker failure may leave the container offline. State and the exact recovery command are retained in `~/.cache/transcription-network/`; task state stays in `.network-state/`. After task termination is confirmed, run the recorded command, for example:

```bash
python3 run_container.py --container transcription \
  --recover ~/.cache/transcription-network/network-TOKEN.json
```

Replace `TOKEN` with the retained file. Recovery refuses to reconnect while task exit is unconfirmed. Do not edit state to bypass guards. Loss of the controlling SSH session may cancel the run; persistence across disconnection is not guaranteed.

## Transcript comparison and supplied review

```bash
python compare_transcripts.py \
  --whisper output/whisper_run/example.json \
  --qwen output/qwen_run/example.json \
  --output-dir output/comparison --terms-file .local/terms.json
```

The command creates one-minute windows, structural checks, side-by-side TSV, raw/normalized diffs and a draft report. Optional `--terms-file` accepts a JSON array such as `["example term", "another term"]`; no project glossary is built in. An execution receipt beside the Qwen transcript supplies timing/memory when available; missing values are shown as unavailable.

Flags are disagreements and review candidates, not errors, CER/WER, accuracy scores or proof of omissions/hallucinations. A secondary ASR transcript is not a human reference. Without Whisper timing, speed superiority is not established.

Copy `review.example.json` into `.local/` and fill it with the work actually reviewed:

- `expected_sha256`: optional `whisper`/`qwen` transcript hashes.
- `reviewed_window_starts`: starts actually read from `analysis.json`, in seconds.
- `findings`: objects with `id`, `start`, `end`, `whisper_phrases`, `qwen_phrases`, `observation` and `interpretation`; optional categories/direction are supplied judgments.
- `window_notes`: objects with `start` and a `notes` array.
- `review`: supplied scope/method/observations/limitations/recommendation. `completed: true` requires coverage of every analysis window.
- `report_markdown`: optional already-authored local report text; otherwise a generic report is rendered.

Check without writing files:

```bash
python finalize_comparison.py \
  --analysis output/comparison/analysis.json \
  --review-json .local/review.json \
  --whisper output/whisper_run/example.json \
  --qwen output/qwen_run/example.json --check-only
```

Replace `--check-only` with `--output-dir output/reviewed` to write reviewed analysis, report and TSV. Use a new directory: transcripts, review JSON and source analysis cannot be overwritten. The verifier first matches analysis windows to the supplied source JSON, then checks fragments against those windows and marks interpretations as supplied, audio-unverified review. It does not perform a new review or decide which model is correct. Original private reports stay local.

## Validation and publication boundary

```bash
python -m unittest -v test_transcription test_public_preparation
python3 run_container.py --help
python compare_transcripts.py --help
python finalize_comparison.py --help
```

Tests cover alignment/text/timing, formats, network/recovery guards, resource gates, settings, supplied fragments/hashes, source-window consistency and partial review coverage. They do not certify model accuracy or hard-failure recovery.

Before the first commit, inspect `git add --dry-run .` and `git check-ignore -v`. `.gitignore` does not remove already tracked files. Include code, tests, README, configuration/review examples and dependency pins. Exclude input/output, `.local/`, runtime state, backups, weights, logs and OS metadata. Public demonstration data must be synthetic or explicitly cleared for publication.

Local ASR avoids an ASR-provider upload. Pinned revisions, safetensors and inference isolation reduce specific exposure without certifying packages/models, privileges or mounts as safe. External disclosure of private recordings/transcripts requires its own authorization.
