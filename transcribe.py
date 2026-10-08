#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transcription_backends import get_writer, load_whisper, run_qwen, whisper_models
from transcription_settings import model_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whisper 또는 Qwen으로 영상 또는 음성을 텍스트로 변환합니다."
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="입력 영상 또는 음성 파일 경로. 여러 파일 지정 가능",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="전사 결과를 저장할 디렉터리",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Whisper 공식 모델명 또는 로컬 .pt 체크포인트 경로. "
            "기본값: large-v3"
        ),
    )

    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "공식 Whisper 모델을 다운로드하고 저장할 디렉터리. "
            "기본값: 로컬 설정 또는 ~/.cache/transcription/models/whisper"
        ),
    )

    parser.add_argument(
        "--language",
        default="auto",
        help=(
            "입력 언어 코드. 한국어는 ko, 자동 감지는 auto. "
            "기본값: auto"
        ),
    )

    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help=(
            "transcribe: 원래 언어로 전사, "
            "translate: 영어로 번역. 기본값: transcribe"
        ),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="추론 장치. 기본값: auto",
    )

    parser.add_argument(
        "--output-format",
        choices=["txt", "srt", "vtt", "tsv", "json", "all"],
        default="all",
        help="출력 형식. 기본값: all",
    )

    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "출력 파일명에서 사용할 확장자 없는 이름. "
            "입력 파일이 하나일 때만 사용 가능"
        ),
    )

    parser.add_argument(
        "--initial-prompt",
        default=None,
        help=(
            "전문용어, 사람 이름, 회사명 등 인식에 참고할 초기 문장"
        ),
    )

    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="단어 단위 타임스탬프 활성화",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="전사 진행 내용을 터미널에 자세히 출력",
    )

    parser.add_argument("--backend", choices=["whisper", "qwen"], default="whisper",
                        help="ASR backend (default: whisper / large-v3)")
    parser.add_argument("--aligner-dir", help="Qwen local ForcedAligner directory")
    parser.add_argument("--chunk-seconds", type=float, default=None,
                        help="Qwen chunk seconds (default: 120)")
    parser.add_argument("--chunk-overlap", type=float, default=None,
                        help="Qwen overlap seconds (default: 2)")
    parser.add_argument("--config", help="Local configuration path (default: .local/config.json)")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 장치를 요청했지만 PyTorch에서 GPU를 찾지 못했습니다."
        )

    return requested_device


def resolve_model(model_argument: str) -> str:
    """
    --model 값이 실제 파일이면 절대 경로로 변환합니다.
    그렇지 않으면 Whisper 공식 모델명으로 취급합니다.
    """
    candidate = Path(model_argument).expanduser()

    if candidate.is_file():
        return str(candidate.resolve())

    looks_like_checkpoint = (
        model_argument.endswith(".pt")
        or "/" in model_argument
        or "\\" in model_argument
    )

    if looks_like_checkpoint:
        raise FileNotFoundError(
            f"로컬 모델 체크포인트를 찾을 수 없습니다: {candidate}"
        )

    available_models = whisper_models()

    if model_argument not in available_models:
        raise ValueError(
            f"지원하지 않는 모델명입니다: {model_argument}\n"
            f"사용 가능한 모델: {', '.join(available_models)}"
        )

    return model_argument


def validate_inputs(input_arguments: list[str]) -> list[Path]:
    input_paths: list[Path] = []

    for input_argument in input_arguments:
        path = Path(input_argument).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

        if not path.is_file():
            raise ValueError(f"입력 경로가 파일이 아닙니다: {path}")

        input_paths.append(path.resolve())

    return input_paths


def transcribe_file(
    model: whisper.Whisper,
    input_path: Path,
    output_dir: Path,
    output_format: str,
    output_name: str | None,
    language: str | None,
    task: str,
    device: str,
    initial_prompt: str | None,
    word_timestamps: bool,
    verbose: bool,
) -> None:
    print()
    print("=" * 72)
    print(f"입력 파일: {input_path}")
    print(f"출력 폴더: {output_dir}")
    print("=" * 72)

    transcribe_options = {
        "language": language,
        "task": task,
        "fp16": device == "cuda",
        "verbose": verbose,

        # 0.0 하나만 사용하지 말고 fallback 온도들을 지정
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),

        "beam_size": 5,
        "best_of": 5,

        # 반복·저품질 결과 판정 기준
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,

        # 장시간 파일에서 반복 문맥이 다음 구간으로 전파되지 않도록 차단
        "condition_on_previous_text": False,

        # 무음 주변의 환각 구간 제거
        "word_timestamps": True,
        "hallucination_silence_threshold": 1.0,
    }

    if initial_prompt:
        transcribe_options["initial_prompt"] = initial_prompt

    result = model.transcribe(
        str(input_path),
        **transcribe_options,
    )

    writer = get_writer(output_format, str(output_dir))

    # Whisper writer는 두 번째 인자의 파일명을 기준으로
    # 출력 파일명을 생성합니다.
    if output_name:
        writer_input_name = f"{output_name}{input_path.suffix}"
    else:
        writer_input_name = str(input_path)

    writer(
        result,
        writer_input_name,
        highlight_words=False,
        max_line_count=None,
        max_line_width=None,
        max_words_per_line=None,
    )

    print(f"전사 완료: {input_path.name}")

    if output_name:
        print(f"출력 파일명: {output_name}.*")
    else:
        print(f"출력 파일명: {input_path.stem}.*")


def main() -> int:
    args = parse_args()

    try:
        if args.backend == "qwen":
            return run_qwen(args)
        if args.aligner_dir:
            raise ValueError("--aligner-dir is only supported by Qwen")
        args.model = args.model or "large-v3"
        args.model_dir = model_directory("whisper", args.model_dir, args.config)
        input_paths = validate_inputs(args.input)

        if args.output_name and len(input_paths) != 1:
            raise ValueError(
                "--output-name은 입력 파일이 하나일 때만 사용할 수 있습니다."
            )

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = Path(args.model_dir).expanduser().resolve()
        model_dir.mkdir(parents=True, exist_ok=True)

        device = resolve_device(args.device)
        model_name_or_path = resolve_model(args.model)

        language = (
            None
            if args.language.lower() == "auto"
            else args.language
        )

        print(f"PyTorch 버전: {torch.__version__}")
        print(f"PyTorch CUDA 런타임: {torch.version.cuda}")
        print(f"사용 장치: {device}")

        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print(f"모델 지정값: {model_name_or_path}")
        print(f"모델 저장 폴더: {model_dir}")
        print("Whisper 모델을 불러오는 중...")

        model = load_whisper(
            model_name_or_path,
            device=device,
            download_root=str(model_dir),
        )

        for input_path in input_paths:
            transcribe_file(
                model=model,
                input_path=input_path,
                output_dir=output_dir,
                output_format=args.output_format,
                output_name=args.output_name,
                language=language,
                task=args.task,
                device=device,
                initial_prompt=args.initial_prompt,
                word_timestamps=args.word_timestamps,
                verbose=args.verbose,
            )

        print()
        print(f"모든 작업 완료: {output_dir}")
        return 0

    except KeyboardInterrupt:
        print("\n사용자가 작업을 중단했습니다.", file=sys.stderr)
        return 130

    except Exception as exc:
        print(
            f"\n오류: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())