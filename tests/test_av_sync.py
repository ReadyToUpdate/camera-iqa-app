import wave

import cv2
import numpy as np

from camera_iqa.av_sync import (
    AvSyncPatternConfig,
    analyze_av_sync,
    detect_audio_pulses,
    detect_video_pulses,
    generate_av_sync_assets,
    planned_pulse_times,
)


def test_planned_pulse_times_include_each_cycle_start():
    config = AvSyncPatternConfig(duration_seconds=12.0, active_seconds=1.0, silent_seconds=3.0)

    assert planned_pulse_times(config) == [0.0, 4.0, 8.0]


def test_detect_audio_pulses_finds_energy_onsets():
    sample_rate = 1000
    audio = np.zeros(5000, dtype=np.float32)
    audio[1000:1300] = 0.8
    audio[3000:3300] = 0.8

    pulses = detect_audio_pulses(audio, sample_rate, min_gap_seconds=1.0)

    assert [round(p.time_seconds, 2) for p in pulses] == [1.0, 3.0]


def test_detect_video_pulses_finds_brightness_onsets():
    brightness = np.zeros(120, dtype=np.float32)
    brightness[30:45] = 240
    brightness[90:105] = 240

    pulses = detect_video_pulses(brightness, fps=30.0, min_gap_seconds=1.0)

    assert [round(p.time_seconds, 2) for p in pulses] == [1.0, 3.0]


def test_analyze_av_sync_reports_audio_minus_video_offsets():
    video_brightness = np.zeros(180, dtype=np.float32)
    video_brightness[30:45] = 255
    video_brightness[90:105] = 255
    audio = np.zeros(6000, dtype=np.float32)
    audio[1100:1400] = 0.8
    audio[3100:3400] = 0.8

    result = analyze_av_sync(video_brightness, 30.0, audio, 1000)

    assert result.pair_count == 2
    assert [round(pair.offset_ms, 1) for pair in result.pairs] == [100.0, 100.0]
    assert round(result.mean_offset_ms, 1) == 100.0


def test_generate_av_sync_assets_writes_video_and_wav(tmp_path):
    video_path = tmp_path / "pattern.mp4"
    audio_path = tmp_path / "pattern.wav"

    generate_av_sync_assets(
        video_path,
        audio_path,
        AvSyncPatternConfig(width=160, height=90, fps=10.0, sample_rate=8000, duration_seconds=4.0),
    )

    capture = cv2.VideoCapture(str(video_path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    ret, first = capture.read()
    assert ret
    assert float(np.mean(first)) > 200
    capture.release()

    with wave.open(str(audio_path), "rb") as wav_file:
        assert wav_file.getframerate() == 8000
        assert wav_file.getnframes() == 32000
