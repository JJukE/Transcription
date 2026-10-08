#!/usr/bin/env python3
"""
장시간 영상용 Whisper / Qwen 전사 스크립트.

주요 변경점
- Whisper의 clip_timestamps로 원본 전체를 반복 슬라이싱하지 않음
- FFmpeg로 실제 WAV 청크를 만든 뒤 각 청크를 0초부터 독립 전사
- word_timestamps는 기본 비활성화(명시적으로 --word-timestamps를 줄 때만 사용)
- hallucination_silence_threshold는 기본 비활성화
- 청크 경계에는 짧은 오버랩을 주고, 세그먼트 중심점으로 중복 제거
- 빈 세그먼트와 0초 길이 세그먼트 제거
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from transcription_backends import get_writer, load_whisper, run_qwen, whisper_models
from transcription_settings import model_directory


DEFAULT_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
MIN_SEGMENT_DURATION = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whisper 또는 Qwen으로 장시간 영상/음성을 전사합니다."
    )
    parser.add_argument(
        "-i", "--input", nargs="+", required=True,
        help="입력 영상 또는 음성 파일 경로. 여러 파일 지정 가능",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="결과 저장 디렉터리",
    )
    parser.add_argument(
        "--model", default=None,
        help="Whisper 모델명 또는 로컬 .pt 체크포인트 경로",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="공식 모델 다운로드 및 캐시 디렉터리",
    )
    parser.add_argument(
        "--language", default="ko",
        help="언어 코드. 자동 감지는 auto",
    )
    parser.add_argument(
        "--task", choices=["transcribe", "translate"], default="transcribe",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto",
    )
    parser.add_argument(
        "--output-format",
        choices=["txt", "srt", "vtt", "tsv", "json", "all"],
        default="all",
    )
    parser.add_argument(
        "--output-name",
        help="입력 파일이 하나일 때 사용할 확장자 없는 출력 이름",
    )
    parser.add_argument(
        "--initial-prompt",
        help="회사명, 제품명, 전문용어 등 인식 참고 문장",
    )
    parser.add_argument(
        "--chunk-seconds", type=float, default=None,
        help="기본 청크 길이(초). 기본값 600초",
    )
    parser.add_argument(
        "--chunk-overlap", type=float, default=None,
        help="청크 경계 양쪽 오버랩 길이(초). 기본값 1초",
    )
    parser.add_argument(
        "--word-timestamps", action="store_true",
        help="단어 단위 타임스탬프를 활성화. 필요할 때만 사용 권장",
    )
    parser.add_argument(
        "--hallucination-silence-threshold", type=float, default=None,
        help=(
            "환각 의심 구간 주변 무음 건너뛰기 기준. "
            "--word-timestamps와 함께 사용하며 기본값은 비활성화"
        ),
    )
    parser.add_argument(
        "--compression-ratio-threshold", type=float, default=2.4,
    )
    parser.add_argument(
        "--logprob-threshold", type=float, default=-1.0,
    )
    parser.add_argument(
        "--no-speech-threshold", type=float, default=0.6,
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Whisper 내부 세그먼트를 터미널에 출력",
    )
    parser.add_argument(
        "--keep-temp-chunks", action="store_true",
        help="디버깅을 위해 추출된 WAV 청크를 출력 폴더에 보존",
    )
    parser.add_argument(
        "--fail-on-repeat", action="store_true",
        help="연속 동일 문장 반복이 탐지되면 종료 코드 2 반환",
    )
    parser.add_argument("--backend", choices=["whisper", "qwen"], default="whisper",
                        help="ASR backend (default: whisper / large-v3)")
    parser.add_argument("--aligner-dir", help="Qwen local ForcedAligner directory")
    parser.add_argument("--config", help="Local configuration path (default: .local/config.json)")
    return parser.parse_args()


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 PyTorch에서 GPU를 찾지 못했습니다.")
    return value


def resolve_model(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())

    looks_like_path = value.endswith(".pt") or "/" in value or "\\" in value
    if looks_like_path:
        raise FileNotFoundError(f"모델 체크포인트를 찾을 수 없습니다: {candidate}")

    if value not in whisper_models():
        raise ValueError(
            f"지원하지 않는 모델명: {value}\n"
            f"사용 가능한 모델: {', '.join(whisper_models())}"
        )
    return value


def validate_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
        paths.append(path)
    return paths


def require_ffmpeg() -> None:
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise RuntimeError(
                f"{command}를 찾을 수 없습니다. apt-get install ffmpeg로 설치하세요."
            )


def get_duration_seconds(path: Path) -> float:
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        output = subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
        duration = float(output)
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"영상 길이를 확인하지 못했습니다: {path}") from exc

    if duration <= 0:
        raise RuntimeError(f"유효하지 않은 영상 길이입니다: {duration}")
    return duration


def build_chunk_ranges(
    duration: float,
    chunk_seconds: float,
    overlap: float,
) -> list[tuple[float, float, float, float]]:
    if chunk_seconds <= 0:
        raise ValueError("--chunk-seconds는 0보다 커야 합니다.")
    if overlap < 0:
        raise ValueError("--chunk-overlap은 0 이상이어야 합니다.")
    if overlap * 2 >= chunk_seconds:
        raise ValueError("--chunk-overlap은 청크 길이의 절반보다 작아야 합니다.")

    ranges: list[tuple[float, float, float, float]] = []
    nominal_start = 0.0

    while nominal_start < duration:
        nominal_end = min(nominal_start + chunk_seconds, duration)
        extraction_start = max(0.0, nominal_start - overlap)
        extraction_end = min(duration, nominal_end + overlap)
        ranges.append(
            (nominal_start, nominal_end, extraction_start, extraction_end)
        )
        nominal_start = nominal_end

    return ranges


def extract_audio_chunk(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
) -> None:
    chunk_duration = max(0.0, end - start)
    if chunk_duration <= 0:
        raise ValueError(f"유효하지 않은 청크 범위: {start}~{end}")

    command = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_path),
        "-t", f"{chunk_duration:.3f}",
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"오디오 청크 추출에 실패했습니다: {input_path} "
            f"({start:.2f}~{end:.2f}초)"
        ) from exc


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[^\w가-힣]+", "", text)


def offset_and_filter_segments(
    segments: list[dict[str, Any]],
    extraction_start: float,
    nominal_start: float,
    nominal_end: float,
    is_last_chunk: bool,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []

    for segment in segments:
        local_start = float(segment.get("start", 0.0))
        local_end = float(segment.get("end", 0.0))
        text = str(segment.get("text", "")).strip()

        absolute_start = extraction_start + local_start
        absolute_end = extraction_start + local_end

        if not text or absolute_end - absolute_start < MIN_SEGMENT_DURATION:
            continue

        midpoint = (absolute_start + absolute_end) / 2.0
        in_nominal_range = (
            nominal_start <= midpoint < nominal_end
            or (is_last_chunk and nominal_start <= midpoint <= nominal_end)
        )
        if not in_nominal_range:
            continue

        copied = dict(segment)
        copied["start"] = max(0.0, absolute_start)
        copied["end"] = min(nominal_end, absolute_end) if is_last_chunk else absolute_end

        if copied["end"] - copied["start"] < MIN_SEGMENT_DURATION:
            continue

        if copied.get("words"):
            adjusted_words: list[dict[str, Any]] = []
            for word in copied["words"]:
                adjusted = dict(word)
                adjusted["start"] = extraction_start + float(word["start"])
                adjusted["end"] = extraction_start + float(word["end"])
                if adjusted["end"] > adjusted["start"]:
                    adjusted_words.append(adjusted)
            copied["words"] = adjusted_words

        kept.append(copied)

    return kept


def find_repeat_runs(
    segments: list[dict[str, Any]],
    minimum_run: int = 5,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    index = 0

    while index < len(segments):
        normalized = normalize_text(segments[index].get("text", ""))
        next_index = index + 1

        while (
            normalized
            and next_index < len(segments)
            and normalize_text(segments[next_index].get("text", "")) == normalized
        ):
            next_index += 1

        count = next_index - index
        if normalized and count >= minimum_run:
            runs.append({
                "count": count,
                "text": segments[index].get("text", "").strip(),
                "start": segments[index].get("start"),
                "end": segments[next_index - 1].get("end"),
            })

        index = next_index

    return runs


def transcribe_one(
    model: whisper.Whisper,
    input_path: Path,
    args: argparse.Namespace,
    device: str,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    language = None if args.language.lower() == "auto" else args.language
    duration = get_duration_seconds(input_path)
    ranges = build_chunk_ranges(
        duration=duration,
        chunk_seconds=args.chunk_seconds,
        overlap=args.chunk_overlap,
    )

    options: dict[str, Any] = {
        "language": language,
        "task": args.task,
        "fp16": device == "cuda",
        # None이면 Whisper 내부 세그먼트 출력과 progress bar를 모두 끕니다.
        "verbose": True if args.verbose else None,
        "temperature": DEFAULT_TEMPERATURES,
        "beam_size": 5,
        "best_of": 5,
        "compression_ratio_threshold": args.compression_ratio_threshold,
        "logprob_threshold": args.logprob_threshold,
        "no_speech_threshold": args.no_speech_threshold,
        "condition_on_previous_text": False,
        "carry_initial_prompt": False,
        "word_timestamps": args.word_timestamps,
    }

    if args.initial_prompt:
        options["initial_prompt"] = args.initial_prompt

    if args.word_timestamps and args.hallucination_silence_threshold is not None:
        options["hallucination_silence_threshold"] = (
            args.hallucination_silence_threshold
        )

    all_segments: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    detected_language = language or "unknown"

    persistent_chunk_dir: Path | None = None
    if args.keep_temp_chunks:
        persistent_chunk_dir = output_dir / f"{input_path.stem}_chunks"
        persistent_chunk_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="whisper_chunks_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for number, (
            nominal_start,
            nominal_end,
            extraction_start,
            extraction_end,
        ) in enumerate(ranges, start=1):
            print(
                f"[{input_path.name}] 청크 {number}/{len(ranges)} "
                f"전사 범위 {nominal_start:.2f}~{nominal_end:.2f}초 "
                f"(추출 {extraction_start:.2f}~{extraction_end:.2f}초)"
            )

            temp_wav = temp_dir / f"chunk_{number:05d}.wav"
            extract_audio_chunk(
                input_path=input_path,
                output_path=temp_wav,
                start=extraction_start,
                end=extraction_end,
            )

            if persistent_chunk_dir is not None:
                shutil.copy2(temp_wav, persistent_chunk_dir / temp_wav.name)

            result = model.transcribe(str(temp_wav), **options)
            detected_language = result.get("language", detected_language)

            filtered_segments = offset_and_filter_segments(
                segments=result.get("segments", []),
                extraction_start=extraction_start,
                nominal_start=nominal_start,
                nominal_end=nominal_end,
                is_last_chunk=number == len(ranges),
            )

            repeat_runs = find_repeat_runs(filtered_segments)
            diagnostics.append({
                "chunk": number,
                "nominal_start": nominal_start,
                "nominal_end": nominal_end,
                "extraction_start": extraction_start,
                "extraction_end": extraction_end,
                "raw_segment_count": len(result.get("segments", [])),
                "kept_segment_count": len(filtered_segments),
                "repeat_runs": repeat_runs,
                "max_compression_ratio": max(
                    (
                        float(segment.get("compression_ratio", 0.0))
                        for segment in result.get("segments", [])
                    ),
                    default=0.0,
                ),
            })
            all_segments.extend(filtered_segments)

    all_segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    for index, segment in enumerate(all_segments):
        segment["id"] = index

    result = {
        "text": "".join(segment.get("text", "") for segment in all_segments),
        "segments": all_segments,
        "language": detected_language,
    }
    return result, diagnostics


def write_outputs(
    result: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    input_path: Path,
    output_dir: Path,
    output_format: str,
    output_name: str | None,
) -> None:
    writer = get_writer(output_format, str(output_dir))

    if output_name:
        writer_name = f"{output_name}{input_path.suffix}"
        diagnostic_name = f"{output_name}.diagnostics.json"
    else:
        writer_name = str(input_path)
        diagnostic_name = f"{input_path.stem}.diagnostics.json"

    writer(
        result,
        writer_name,
        highlight_words=False,
        max_line_count=None,
        max_line_width=None,
        max_words_per_line=None,
    )

    (output_dir / diagnostic_name).write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    try:
        if args.backend == "qwen":
            return run_qwen(args)
        if args.aligner_dir:
            raise ValueError("--aligner-dir is only supported by Qwen")
        args.model = args.model or "large-v3"
        args.model_dir = model_directory("whisper", args.model_dir, args.config)
        args.chunk_seconds = 600.0 if args.chunk_seconds is None else args.chunk_seconds
        args.chunk_overlap = 1.0 if args.chunk_overlap is None else args.chunk_overlap
        require_ffmpeg()
        input_paths = validate_inputs(args.input)
        if args.output_name and len(input_paths) != 1:
            raise ValueError("--output-name은 입력 파일이 하나일 때만 사용할 수 있습니다.")

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = Path(args.model_dir).expanduser().resolve()
        model_dir.mkdir(parents=True, exist_ok=True)

        device = resolve_device(args.device)
        model_name_or_path = resolve_model(args.model)

        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch CUDA runtime: {torch.version.cuda}")
        print(f"장치: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"모델: {model_name_or_path}")
        print(f"모델 저장 경로: {model_dir}")
        print(f"청크: {args.chunk_seconds}초, 오버랩: {args.chunk_overlap}초")
        print(f"단어 타임스탬프: {args.word_timestamps}")
        print("이전 텍스트 조건 사용: False")

        model = load_whisper(
            model_name_or_path,
            device=device,
            download_root=str(model_dir),
        )

        any_repeat = False

        for input_path in input_paths:
            result, diagnostics = transcribe_one(
                model=model,
                input_path=input_path,
                args=args,
                device=device,
                output_dir=output_dir,
            )

            write_outputs(
                result=result,
                diagnostics=diagnostics,
                input_path=input_path,
                output_dir=output_dir,
                output_format=args.output_format,
                output_name=args.output_name,
            )

            repeat_runs = [
                run
                for diagnostic in diagnostics
                for run in diagnostic["repeat_runs"]
            ]
            if repeat_runs:
                any_repeat = True
                print(f"\n경고: {input_path.name}에서 반복 문장 후보가 탐지됐습니다.")
                for run in repeat_runs:
                    print(
                        f"  {run['start']:.2f}~{run['end']:.2f}초, "
                        f"{run['count']}회: {run['text']!r}"
                    )
            else:
                print(f"{input_path.name}: 뚜렷한 연속 반복 없음")

        if any_repeat and args.fail_on_repeat:
            return 2
        return 0

    except KeyboardInterrupt:
        print("\n사용자가 작업을 중단했습니다.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n오류: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())