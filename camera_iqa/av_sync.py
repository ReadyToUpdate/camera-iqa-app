from __future__ import annotations

import math
import csv
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class AvSyncPatternConfig:
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    sample_rate: int = 48000
    duration_seconds: float = 20.0
    active_seconds: float = 1.0
    silent_seconds: float = 3.0
    tone_hz: float = 1000.0
    amplitude: float = 0.75

    @property
    def cycle_seconds(self) -> float:
        return self.active_seconds + self.silent_seconds


@dataclass(frozen=True)
class Pulse:
    time_seconds: float
    strength: float


@dataclass(frozen=True)
class SyncPair:
    video_time_seconds: float
    audio_time_seconds: float
    offset_ms: float


@dataclass(frozen=True)
class AvSyncAnalysisResult:
    video_pulses: list[Pulse]
    audio_pulses: list[Pulse]
    pairs: list[SyncPair]

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    @property
    def mean_offset_ms(self) -> float:
        if not self.pairs:
            return math.nan
        return float(np.mean([pair.offset_ms for pair in self.pairs]))

    @property
    def std_offset_ms(self) -> float:
        if len(self.pairs) < 2:
            return 0.0 if self.pairs else math.nan
        return float(np.std([pair.offset_ms for pair in self.pairs], ddof=1))


def planned_pulse_times(config: AvSyncPatternConfig) -> list[float]:
    pulse_count = int(math.floor(config.duration_seconds / config.cycle_seconds))
    return [round(index * config.cycle_seconds, 9) for index in range(pulse_count)]


def generate_av_sync_assets(
    video_path: str | Path,
    audio_path: str | Path,
    config: AvSyncPatternConfig | None = None,
    muxed_output_path: str | Path | None = None,
    ffmpeg_path: str | None = None,
) -> Path | None:
    config = config or AvSyncPatternConfig()
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    _write_pattern_video(video_path, config)
    _write_pattern_wav(audio_path, config)

    if muxed_output_path is None:
        return None
    return mux_av_sync_assets(video_path, audio_path, muxed_output_path, ffmpeg_path=ffmpeg_path)


def mux_av_sync_assets(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    ffmpeg_path: str | None = None,
) -> Path:
    ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; install ffmpeg or keep the generated MP4 video and WAV audio separate")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return output_path


def detect_audio_pulses(
    samples: np.ndarray,
    sample_rate: int,
    min_gap_seconds: float = 2.0,
    threshold: float | None = None,
) -> list[Pulse]:
    mono = _to_mono_float(samples)
    if mono.size == 0:
        return []
    frame_length = max(1, int(sample_rate * 0.02))
    hop = max(1, int(sample_rate * 0.005))
    rms = _window_rms(mono, frame_length, hop)
    if rms.size == 0:
        return []
    threshold = threshold if threshold is not None else float(max(np.percentile(rms, 90) * 0.5, np.max(rms) * 0.2, 1e-4))
    active = rms >= threshold
    starts = np.flatnonzero(active & np.concatenate(([True], ~active[:-1])))
    raw_threshold = max(float(np.max(np.abs(mono))) * 0.2, 1e-4)
    min_gap_samples = max(1, int(round(min_gap_seconds * sample_rate)))
    pulses: list[Pulse] = []
    last_sample = -min_gap_samples
    for start in starts:
        search_start = max(0, int(start * hop) - frame_length)
        search_end = min(mono.size, int(start * hop) + frame_length * 2)
        candidates = np.flatnonzero(np.abs(mono[search_start:search_end]) >= raw_threshold)
        if candidates.size == 0:
            continue
        sample_index = search_start + int(candidates[0])
        if sample_index - last_sample < min_gap_samples:
            continue
        pulses.append(Pulse(time_seconds=float(sample_index / sample_rate), strength=float(rms[start])))
        last_sample = sample_index
    return pulses


def detect_video_pulses(
    brightness: np.ndarray,
    fps: float,
    min_gap_seconds: float = 2.0,
    threshold: float | None = None,
) -> list[Pulse]:
    values = np.asarray(brightness, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return []
    threshold = threshold if threshold is not None else float((np.percentile(values, 10) + np.percentile(values, 90)) / 2)
    active = values >= threshold
    return _pulses_from_active(active, fps=fps, strengths=values, min_gap_seconds=min_gap_seconds)


def analyze_av_sync(
    video_brightness: np.ndarray,
    fps: float,
    audio_samples: np.ndarray,
    sample_rate: int,
    min_gap_seconds: float = 2.0,
) -> AvSyncAnalysisResult:
    video_pulses = detect_video_pulses(video_brightness, fps, min_gap_seconds=min_gap_seconds)
    audio_pulses = detect_audio_pulses(audio_samples, sample_rate, min_gap_seconds=min_gap_seconds)
    pairs = [
        SyncPair(
            video_time_seconds=video.time_seconds,
            audio_time_seconds=audio.time_seconds,
            offset_ms=(audio.time_seconds - video.time_seconds) * 1000.0,
        )
        for video, audio in zip(video_pulses, audio_pulses)
    ]
    return AvSyncAnalysisResult(video_pulses=video_pulses, audio_pulses=audio_pulses, pairs=pairs)


def analyze_av_sync_files(video_path: str | Path, audio_wav_path: str | Path, min_gap_seconds: float = 2.0) -> AvSyncAnalysisResult:
    brightness, fps = video_brightness_series(video_path)
    samples, sample_rate = read_wav_mono(audio_wav_path)
    return analyze_av_sync(brightness, fps, samples, sample_rate, min_gap_seconds=min_gap_seconds)


def write_av_sync_csv(result: AvSyncAnalysisResult, output_path: str | Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "video_time_s", "audio_time_s", "audio_minus_video_ms"])
        for index, pair in enumerate(result.pairs, start=1):
            writer.writerow(
                [
                    index,
                    f"{pair.video_time_seconds:.6f}",
                    f"{pair.audio_time_seconds:.6f}",
                    f"{pair.offset_ms:.3f}",
                ]
            )


def video_brightness_series(video_path: str | Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    values: list[float] = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        values.append(float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
    capture.release()
    return np.asarray(values, dtype=np.float32), fps


def read_wav_mono(wav_path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        raw = wav_file.readframes(wav_file.getnframes())
    if width != 2:
        raise RuntimeError("Only 16-bit PCM WAV is supported")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def _write_pattern_video(video_path: Path, config: AvSyncPatternConfig) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, config.fps, (config.width, config.height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {video_path}")
    frame_count = int(round(config.duration_seconds * config.fps))
    for frame_index in range(frame_count):
        timestamp = frame_index / config.fps
        active = (timestamp % config.cycle_seconds) < config.active_seconds
        frame = np.full((config.height, config.width, 3), 255 if active else 0, dtype=np.uint8)
        _draw_time_overlay(frame, timestamp, active)
        writer.write(frame)
    writer.release()


def _write_pattern_wav(audio_path: Path, config: AvSyncPatternConfig) -> None:
    sample_count = int(round(config.duration_seconds * config.sample_rate))
    times = np.arange(sample_count, dtype=np.float32) / config.sample_rate
    active = np.mod(times, config.cycle_seconds) < config.active_seconds
    tone = np.sin(2.0 * np.pi * config.tone_hz * times) * config.amplitude
    samples = np.where(active, tone, 0.0)
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(config.sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _draw_time_overlay(frame: np.ndarray, timestamp: float, active: bool) -> None:
    color = (0, 0, 0) if active else (255, 255, 255)
    label = f"{timestamp:06.2f}s {'BEEP' if active else 'SILENT'}"
    cv2.putText(frame, label, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)


def _to_mono_float(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.float32) / float(np.iinfo(values.dtype).max)
    return values.astype(np.float32)


def _window_rms(samples: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    if samples.size < frame_length:
        return np.asarray([float(np.sqrt(np.mean(np.square(samples))))], dtype=np.float32)
    starts = range(0, samples.size - frame_length + 1, hop)
    return np.asarray([float(np.sqrt(np.mean(np.square(samples[start : start + frame_length])))) for start in starts], dtype=np.float32)


def _pulses_from_active(active: np.ndarray, fps: float, strengths: np.ndarray, min_gap_seconds: float) -> list[Pulse]:
    starts = np.flatnonzero(active & np.concatenate(([True], ~active[:-1])))
    min_gap = max(1, int(round(min_gap_seconds * fps)))
    pulses: list[Pulse] = []
    last_start = -min_gap
    for start in starts:
        if start - last_start < min_gap:
            continue
        pulses.append(Pulse(time_seconds=float(start / fps), strength=float(strengths[start])))
        last_start = int(start)
    return pulses
