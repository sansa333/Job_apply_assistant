"""Create a speech-focused mono WAV with soft gain for low-volume recordings."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import av
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gain", type=float, default=10.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(args.input))
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)

    with wave.open(str(args.output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)

        def write_frame(frame: av.AudioFrame) -> None:
            samples = frame.to_ndarray().reshape(-1).astype(np.float32)
            boosted = np.tanh(args.gain * samples) / np.tanh(args.gain)
            pcm = np.clip(np.rint(boosted * 32767), -32768, 32767).astype("<i2")
            wav.writeframes(pcm.tobytes())

        for source_frame in container.decode(audio=0):
            for output_frame in resampler.resample(source_frame):
                write_frame(output_frame)
        for output_frame in resampler.resample(None):
            write_frame(output_frame)


if __name__ == "__main__":
    main()
