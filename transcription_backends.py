"""Local ASR backends. Importing this module never imports a model SDK."""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from transcription_settings import DEFAULT_MODEL_ROOT, model_directory

ASR_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
ALIGNER_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
QWEN_CACHE = DEFAULT_MODEL_ROOT / "qwen"
LANGUAGES = {"ko": "Korean", "en": "English", "zh": "Chinese", "ja": "Japanese",
             "de": "German", "fr": "French", "es": "Spanish", "ru": "Russian",
             "it": "Italian", "pt": "Portuguese", "ar": "Arabic"}


def whisper_models():
    import whisper
    return whisper.available_models()


def load_whisper(model, **kwargs):
    import whisper
    return whisper.load_model(model, **kwargs)


def get_writer(format, directory):
    # Preserve the existing Whisper writer byte/format behavior.
    from whisper.utils import get_writer as factory
    return factory(format, directory)


def qwen_language(value):
    if value.lower() == "auto":
        return None
    return LANGUAGES.get(value.lower(), value)


def normalized_offsets(text):
    chars, offsets = [], []
    for i, ch in enumerate(text):
        for normalized in ch.casefold():
            if normalized.isalnum():
                chars.append(normalized)
                offsets.append(i)
    return "".join(chars), offsets


def aligned_words(text, items):
    """Recover original punctuation/spacing discarded by the Korean aligner."""
    normalized, offsets = normalized_offsets(text)
    anchors, cursor = [], 0
    for item in items:
        token = item["text"]
        needle, _ = normalized_offsets(token)
        if not needle:
            continue
        start = normalized.find(needle, cursor)
        if start < 0:
            raise ValueError(f"Forced-aligner token does not match original text: {token!r}")
        end = start + len(needle)
        anchors.append((offsets[start], offsets[end-1]+1, item))
        cursor = end
    if normalized and (not anchors or cursor != len(normalized)):
        raise ValueError("Forced-aligner did not cover the entire ASR text")
    # Fail rather than silently discard words that precede the first match or gaps.
    if "".join(normalized_offsets(a[2]["text"])[0] for a in anchors) != normalized:
        raise ValueError("Forced-aligner text has unmatched words")
    words = []
    for i, (start, end, item) in enumerate(anchors):
        left = 0 if i == 0 else start
        right = anchors[i+1][0] if i+1 < len(anchors) else len(text)
        words.append({"word": text[left:right], "start": float(item["start"]),
                      "end": float(item["end"]), "aligner_text": item["text"]})
    return words


def crop_words(words, offset, nominal_start, nominal_end, last, duration):
    kept = []
    for word in words:
        start, end = offset + word["start"], offset + word["end"]
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Non-finite or reversed word timestamps")
        midpoint = (start + end) / 2
        if nominal_start <= midpoint < nominal_end or (last and midpoint == nominal_end):
            if end < start or start < -0.01 or end > duration + 0.05:
                raise ValueError(f"Invalid word timestamp: {start}~{end}")
            kept.append({**word, "start": max(0., start), "end": min(duration, end)})
    return kept


def word_segments(words):
    segments, group = [], []
    for word in words:
        if group and (word["start"] - group[-1]["end"] > 0.8 or
                      word["end"] - group[0]["start"] > 12 or
                      len("".join(w["word"] for w in group)) > 90):
            segments.append({"start": group[0]["start"], "end": group[-1]["end"],
                             "text": "".join(w["word"] for w in group), "words": group})
            group = []
        group.append(word)
        if re.search(r"[.!?。！？][\s\"'’”]*$", word["word"]):
            segments.append({"start": group[0]["start"], "end": group[-1]["end"],
                             "text": "".join(w["word"] for w in group), "words": group})
            group = []
    if group:
        segments.append({"start": group[0]["start"], "end": group[-1]["end"],
                         "text": "".join(w["word"] for w in group), "words": group})
    return segments


def timestamp(seconds, comma=False):
    milliseconds = round(seconds * 1000)
    hours, rest = divmod(milliseconds, 3600000)
    minutes, rest = divmod(rest, 60000)
    seconds, ms = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{',' if comma else '.'}{ms:03}"


def write_qwen(result, directory, name, format="all"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    formats = ["txt", "srt", "vtt", "tsv", "json"] if format == "all" else [format]
    segments = result["segments"]
    for fmt in formats:
        if fmt == "json":
            content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        elif fmt == "txt":
            content = "\n".join(s["text"].strip() for s in segments) + "\n"
        elif fmt == "tsv":
            content = "start\tend\ttext\n" + "".join(
                f"{round(s['start']*1000)}\t{round(s['end']*1000)}\t{s['text'].strip().replace(chr(9),' ')}\n" for s in segments)
        else:
            content = "WEBVTT\n\n" if fmt == "vtt" else ""
            for i, s in enumerate(segments, 1):
                if fmt == "srt":
                    content += f"{i}\n"
                content += f"{timestamp(s['start'],fmt=='srt')} --> {timestamp(s['end'],fmt=='srt')}\n{s['text'].strip()}\n\n"
        (directory / f"{name}.{fmt}").write_text(content, encoding="utf-8")


class QwenBackend:
    def __init__(self, model_dir, aligner_dir, device):
        import torch
        from qwen_asr import Qwen3ASRModel
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        for path in (model_dir, aligner_dir):
            if not Path(path, "config.json").is_file():
                raise FileNotFoundError(f"Download the pinned local model first: {path}")
        safe = dict(dtype=dtype, device_map=device, attn_implementation="sdpa",
                    local_files_only=True, trust_remote_code=False, use_safetensors=True)
        self.model = Qwen3ASRModel.from_pretrained(
            str(model_dir), **safe, max_inference_batch_size=1, max_new_tokens=4096,
            forced_aligner=str(aligner_dir), forced_aligner_kwargs=safe)
        self.model.model.generation_config.do_sample = False

    def transcribe(self, wav, language, context):
        result = self.model.transcribe(audio=str(wav), language=language,
                                       context=context, return_time_stamps=True)[0]
        items = [{"text": item.text, "start": item.start_time, "end": item.end_time}
                 for item in result.time_stamps]
        return {"text": result.text, "language": result.language, "alignment": items,
                "words": aligned_words(result.text, items)}


def run_qwen(args):
    import hashlib
    import importlib.metadata
    import sys
    import torch
    from transcribe_long import (build_chunk_ranges, extract_audio_chunk,
                                 find_repeat_runs, get_duration_seconds, require_ffmpeg)
    if args.task != "transcribe":
        raise ValueError("Qwen does not support --task translate")
    unsupported = {"--hallucination-silence-threshold", "--compression-ratio-threshold",
                   "--logprob-threshold", "--no-speech-threshold"}
    supplied = {a.split("=", 1)[0] for a in sys.argv[1:]}
    if supplied & unsupported:
        raise ValueError("Whisper-only options: " + ", ".join(sorted(supplied & unsupported)))
    require_ffmpeg()
    paths = [Path(value).expanduser().resolve() for value in args.input]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("An input audio file is missing")
    if args.output_name and len(paths) != 1:
        raise ValueError("--output-name requires exactly one input")
    cache = Path(model_directory("qwen", args.model_dir, getattr(args, "config", None))).expanduser().resolve()
    model = args.model or "Qwen/Qwen3-ASR-1.7B"
    model_dir = cache / "Qwen3-ASR-1.7B" if model == "Qwen/Qwen3-ASR-1.7B" else Path(model).expanduser().resolve()
    aligner_dir = Path(args.aligner_dir).expanduser().resolve() if args.aligner_dir else cache / "Qwen3-ForcedAligner-0.6B"
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    chunk_seconds = args.chunk_seconds if args.chunk_seconds is not None else 120.
    overlap = args.chunk_overlap if args.chunk_overlap is not None else 2.
    if not math.isfinite(chunk_seconds) or not math.isfinite(overlap):
        raise ValueError("Chunk seconds and overlap must be finite")
    if chunk_seconds + 2*overlap > 180:
        raise ValueError("Qwen alignment chunks including overlap must be <= 180 seconds")
    # Validate ranges before loading either model.
    build_chunk_ranges(1., chunk_seconds, overlap)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    backend = QwenBackend(model_dir, aligner_dir, device)
    any_repeat = False
    for path in paths:
        duration = get_duration_seconds(path)
        ranges = build_chunk_ranges(duration, chunk_seconds, overlap)
        segments, diagnostics, raw_chunks = [], [], []
        with tempfile.TemporaryDirectory(prefix="qwen_chunks_") as temporary:
            for index, (ns, ne, es, ee) in enumerate(ranges, 1):
                print(f"Qwen chunk {index}/{len(ranges)}: {ns:.2f}~{ne:.2f}s", flush=True)
                wav = Path(temporary, f"chunk_{index:05}.wav")
                extract_audio_chunk(path, wav, es, ee)
                if getattr(args, "keep_temp_chunks", False):
                    chunkdir = out / f"{path.stem}_chunks"
                    chunkdir.mkdir(exist_ok=True)
                    shutil.copy2(wav, chunkdir / wav.name)
                chunk_started = time.monotonic()
                raw = backend.transcribe(wav, qwen_language(args.language), args.initial_prompt or "")
                words = crop_words(raw["words"], es, ns, ne, index==len(ranges), duration)
                kept = word_segments(words)
                segments.extend(kept)
                raw_chunks.append({"chunk": index, "extraction_start": es, "extraction_end": ee,
                                   "text": raw["text"], "language": raw["language"], "alignment": raw["alignment"]})
                repeats = find_repeat_runs(kept)
                any_repeat |= bool(repeats)
                diagnostics.append({"chunk": index, "nominal_start": ns, "nominal_end": ne,
                                    "extraction_start": es, "extraction_end": ee,
                                    "raw_word_count": len(raw["words"]), "kept_word_count": len(words),
                                    "kept_segment_count": len(kept), "repeat_runs": repeats,
                                    "elapsed_seconds": time.monotonic()-chunk_started})
        segments.sort(key=lambda s: (s["start"], s["end"]))
        for i, segment in enumerate(segments):
            segment["id"] = i
        result = {"text": "".join(s["text"] for s in segments), "segments": segments,
                  "language": raw_chunks[0]["language"], "backend": "qwen", "duration": duration,
                  "raw_chunks": raw_chunks}
        name = args.output_name or path.stem
        write_qwen(result, out, name, args.output_format)
        (out / f"{name}.diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
        model_manifests = {label: json.loads((directory / "model_manifest.json").read_text())
                           if (directory / "model_manifest.json").is_file() else None
                           for label, directory in (("asr", model_dir), ("aligner", aligner_dir))}
        receipt = {"backend": "qwen", "input": str(path), "input_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
                   "argv": sys.argv, "model_path": str(model_dir), "aligner_path": str(aligner_dir),
                   "model_manifests": model_manifests,
                   "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "qwen-asr")},
                   "settings": {"device": device, "dtype": "bfloat16" if device=="cuda" else "float32", "attention": "sdpa",
                                "batch_size": 1, "max_new_tokens": 4096, "language": args.language,
                                "context": args.initial_prompt or "", "chunk_seconds": chunk_seconds, "overlap": overlap},
                   "elapsed_seconds_including_model_load": time.monotonic()-started,
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device=="cuda" else None,
                   "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device=="cuda" else None,
                   "source_sha256": source_hashes}
        (out / f"{name}.receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return 2 if any_repeat and getattr(args, "fail_on_repeat", False) else 0
