"""Transcribe a local audio file with faster-whisper and save timestamped JSON/text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("output_stem", type=Path)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--clip-timestamps")
    parser.add_argument("--initial-prompt")
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    segment_iter, info = model.transcribe(
        str(args.audio),
        language="zh",
        beam_size=5,
        best_of=5,
        vad_filter=not args.no_vad,
        vad_parameters={
            "threshold": args.vad_threshold,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 300,
        },
        condition_on_previous_text=not args.no_context,
        clip_timestamps=args.clip_timestamps or "0",
        initial_prompt=args.initial_prompt,
        word_timestamps=False,
    )

    segments = []
    for segment in segment_iter:
        item = {
            "id": segment.id,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
            "avg_logprob": round(segment.avg_logprob, 4),
            "no_speech_prob": round(segment.no_speech_prob, 4),
        }
        segments.append(item)
        print(
            f"[{format_timestamp(segment.start)}-{format_timestamp(segment.end)}] "
            f"{item['text']}",
            flush=True,
        )

    payload = {
        "audio": str(args.audio),
        "model": args.model,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
        "segments": segments,
    }

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    args.output_stem.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output_stem.with_suffix(".txt").write_text(
        "\n".join(
            f"[{format_timestamp(item['start'])}-{format_timestamp(item['end'])}] "
            f"{item['text']}"
            for item in segments
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
