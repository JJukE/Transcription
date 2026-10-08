"""Verify supplied review fragments and render a local ASR comparison report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import unicodedata
from pathlib import Path
from compare_transcripts import window_text


def normalized_map(text):
    chars, positions = [], []
    for index, char in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", char).casefold():
            if normalized.isalnum():
                chars.append(normalized)
                positions.append(index)
    return "".join(chars), positions


def excerpt(text, phrase):
    normalized, positions = normalized_map(text)
    target, _ = normalized_map(phrase)
    offset = normalized.find(target)
    if not target or offset < 0:
        raise ValueError("Supplied review fragment is absent from the source window")
    return text[positions[offset]:positions[offset + len(target) - 1] + 1]


def verify_analysis_sources(analysis, sources):
    for backend, path in sources.items():
        transcript = json.loads(path.read_text(encoding="utf-8"))
        for row in analysis["windows"]:
            if row[backend] != window_text(transcript, row["start"], row["end"]):
                raise ValueError("Analysis window does not match the supplied source transcript")


def verify_review(analysis, review, sources):
    windows = analysis["windows"]
    starts = {row["start"] for row in windows}
    reviewed = set(review.get("reviewed_window_starts", []))
    if not reviewed <= starts:
        raise ValueError("Review contains an unknown window start")
    for backend, expected in review.get("expected_sha256", {}).items():
        if backend not in sources or hashlib.sha256(sources[backend].read_bytes()).hexdigest() != expected:
            raise ValueError("Source transcript hash does not match the supplied review")
    if review.get("review", {}).get("audio_adjudication"):
        raise ValueError("This transcript-only verifier cannot certify audio adjudication")
    findings, ids = [], set()
    duration = analysis["summary"]["duration_seconds"]
    for supplied in review.get("findings", []):
        item = dict(supplied)
        if item["id"] in ids:
            raise ValueError("Finding IDs must be unique")
        ids.add(item["id"])
        start, end = item["start"], item["end"]
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration + 0.05:
            raise ValueError("Finding timestamp is outside the audio range")
        covered = [row for row in windows if start <= row["start"] < end]
        for backend in ("whisper", "qwen"):
            text = " ".join(row[backend] for row in covered)
            phrases = item.pop(f"{backend}_phrases")
            if not isinstance(phrases, list) or not phrases:
                raise ValueError("Each finding needs source phrases for both transcripts")
            item[f"{backend}_excerpts"] = [excerpt(text, phrase) for phrase in phrases]
        item["evidence_status"] = "transcript_fragments_verified; contextual_interpretation; audio_unverified"
        findings.append(item)
    notes = {}
    for supplied in review.get("window_notes", []):
        if supplied["start"] not in starts:
            raise ValueError("Window note has an unknown start")
        notes.setdefault(supplied["start"], []).extend(supplied["notes"])
    for finding in findings:
        for row in windows:
            if finding["start"] <= row["start"] < finding["end"]:
                note = finding["observation"] + " " + finding["interpretation"]
                if note not in notes.setdefault(row["start"], []):
                    notes[row["start"]].append(note)
    completed = bool(windows) and reviewed == starts and review.get("review", {}).get("completed") is True
    return findings, reviewed, notes, completed


def label(seconds):
    rounded = round(seconds)
    return f"{rounded // 60:02}:{rounded % 60:02}"


def render_report(analysis, findings, completed):
    summary = analysis["summary"]
    lines = ["# ASR transcript comparison", "",
             f"Duration: {summary['duration_seconds']:.3f} seconds; windows: {len(analysis['windows'])}.",
             f"Supplied review covers all windows: {completed}.", "",
             "Fragments were checked against transcript windows. Interpretations remain supplied review judgments.",
             "No audio adjudication, gold reference, CER/WER or accuracy estimate is provided.", "",
             "| Time | Whisper | Qwen | Supplied observation and interpretation |",
             "|---|---|---|---|"]
    for item in findings:
        cells = [f"{label(item['start'])}–{label(item['end'])}",
                 " / ".join(item["whisper_excerpts"]), " / ".join(item["qwen_excerpts"]),
                 item["observation"] + " " + item["interpretation"]]
        lines.append("| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, help="analysis.json from compare_transcripts.py")
    parser.add_argument("--review-json", required=True, help="Local findings, review coverage and optional report text")
    parser.add_argument("--whisper", required=True)
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--output-dir", help="New directory for reviewed analysis, report and TSV")
    parser.add_argument("--check-only", action="store_true", help="Verify without writing files")
    args = parser.parse_args()
    if not args.check_only and not args.output_dir:
        parser.error("--output-dir is required unless --check-only is used")
    sources = {name: Path(getattr(args, name)).expanduser().resolve() for name in ("whisper", "qwen")}
    before = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    analysis_path = Path(args.analysis).expanduser().resolve()
    review_path = Path(args.review_json).expanduser().resolve()
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    verify_analysis_sources(analysis, sources)
    findings, reviewed, notes, completed = verify_review(analysis, review, sources)
    result = {"verified_findings": len(findings), "reviewed_windows": len(reviewed),
              "total_windows": len(analysis["windows"]), "review_completed": completed,
              "source_transcripts_unchanged": True}
    if args.check_only:
        print(json.dumps(result))
        return
    out = Path(args.output_dir).expanduser().resolve()
    destinations = {out / name for name in ("analysis.json", "comparison_report.md", "comparison_60s.tsv")}
    if destinations & {*sources.values(), analysis_path, review_path}:
        raise ValueError("Use a new output directory; source files cannot be overwritten")
    analysis["qualitative_review_status"] = ("completed" if completed else "partial") + "_transcript_comparison_audio_unverified"
    analysis["qualitative_review"] = {**review.get("review", {}), "completed": completed,
        "audio_adjudication": False, "reference_transcript": None, "findings": findings}
    for row in analysis["windows"]:
        row["qualitative_review"] = {"reviewed": row["start"] in reviewed, "audio_verified": False,
            "finding_ids": [f["id"] for f in findings if f["start"] <= row["start"] < f["end"]],
            "notes": notes.get(row["start"], [])}
    report = review.get("report_markdown") or render_report(analysis, findings, completed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "comparison_report.md").write_text(report, encoding="utf-8")
    with (out / "comparison_60s.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["start_seconds", "end_seconds", "whisper", "qwen", "review_candidates", "supplied_review_audio_unverified"])
        for row in analysis["windows"]:
            writer.writerow([row["start"], row["end"], row["whisper"], row["qwen"],
                             ",".join(row["flags"]), " ".join(row["qualitative_review"]["notes"])])
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != before[name] for name, path in sources.items()):
        raise ValueError("Source transcript changed during review")
    print(json.dumps({"report_written": True, **result}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
