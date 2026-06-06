from __future__ import annotations

import argparse

from camera_iqa.av_sync import (
    AvSyncPatternConfig,
    analyze_av_sync_files,
    generate_av_sync_assets,
    write_av_sync_csv,
)
from camera_iqa.config import load_config
from camera_iqa.grayscale_chart import analyze_grayscale_chart_path
from camera_iqa.ite_resolution import (
    ResolutionROI,
    analyze_ite_resolution_folder,
    calibrate_ite_roi,
    load_calibration,
    save_calibration,
    write_ite_resolution_csv,
)
from camera_iqa.metrics import read_image
from camera_iqa.pipeline import default_overlay_dir, list_images, process_image
from camera_iqa.stripe_fov import analyze_stripe_fov_folder, read_distance_csv, write_stripe_fov_csv, write_stripe_fov_excel


def parse_roi(value: str) -> ResolutionROI:
    try:
        x, y, width, height = (int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be formatted as x,y,width,height") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive")
    return ResolutionROI(x=x, y=y, width=width, height=height)


def main() -> None:
    parser = argparse.ArgumentParser(description="Security camera image quality assessment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--input", required=True, help="Input image folder")
    run_parser.add_argument("--output", required=True, help="Output Excel file")
    run_parser.add_argument("--config", default=None, help="Threshold YAML file")
    graychart_parser = subparsers.add_parser("graychart")
    graychart_parser.add_argument("--image", required=True, help="Input grayscale chart image")
    graychart_parser.add_argument("--annotated-output", default=None, help="Optional annotated output image")
    graychart_parser.add_argument("--expected-levels", type=int, default=11, help="Expected chart levels")
    graychart_parser.add_argument("--min-delta", type=float, default=5.0, help="Minimum adjacent gray delta to count as distinguishable")
    ite_calibrate_parser = subparsers.add_parser("ite-calibrate")
    ite_calibrate_parser.add_argument("--image", required=True, help="Baseline ITE Chart A image")
    ite_calibrate_parser.add_argument("--roi", required=True, type=parse_roi, help="Center line-group ROI as x,y,width,height")
    ite_calibrate_parser.add_argument("--output", required=True, help="Output ROI calibration JSON")
    ite_calibrate_parser.add_argument("--annotated-output", default=None, help="Optional annotated baseline image")
    ite_lines_parser = subparsers.add_parser("ite-lines")
    ite_lines_parser.add_argument("--input", required=True, help="Input folder with illumination sequence images")
    ite_lines_parser.add_argument("--roi", required=True, help="ROI calibration JSON from ite-calibrate")
    ite_lines_parser.add_argument("--output", required=True, help="Output CSV file")
    stripe_fov_parser = subparsers.add_parser("stripe-fov")
    stripe_fov_parser.add_argument("--input", required=True, help="Input folder with zoom test images")
    stripe_fov_parser.add_argument("--distances", required=True, help="CSV with image,distance_m columns")
    stripe_fov_parser.add_argument("--output", default=None, help="Optional output CSV file")
    stripe_fov_parser.add_argument("--excel-output", default=None, help="Optional output Excel file")
    stripe_fov_parser.add_argument("--physical-black-width-cm", type=float, default=1.0, help="Physical black stripe width in cm")
    stripe_fov_parser.add_argument("--annotated-dir", default=None, help="Optional folder for annotated images")
    avsync_generate_parser = subparsers.add_parser("avsync-generate")
    avsync_generate_parser.add_argument("--video-output", required=True, help="Output visual pattern MP4")
    avsync_generate_parser.add_argument("--audio-output", required=True, help="Output tone pattern WAV")
    avsync_generate_parser.add_argument("--muxed-output", default=None, help="Optional output MP4 with audio, requires ffmpeg")
    avsync_generate_parser.add_argument("--width", type=int, default=1280, help="Video width")
    avsync_generate_parser.add_argument("--height", type=int, default=720, help="Video height")
    avsync_generate_parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate")
    avsync_generate_parser.add_argument("--duration", type=float, default=20.0, help="Total pattern duration in seconds")
    avsync_generate_parser.add_argument("--active-seconds", type=float, default=1.0, help="Bright screen and tone duration per cycle")
    avsync_generate_parser.add_argument("--silent-seconds", type=float, default=3.0, help="Black screen and silence duration per cycle")
    avsync_generate_parser.add_argument("--tone-hz", type=float, default=1000.0, help="Tone frequency in Hz")
    avsync_generate_parser.add_argument("--sample-rate", type=int, default=48000, help="Audio sample rate")
    avsync_analyze_parser = subparsers.add_parser("avsync-analyze")
    avsync_analyze_parser.add_argument("--video", required=True, help="Recorded video file")
    avsync_analyze_parser.add_argument("--audio", required=True, help="Recorded 16-bit PCM WAV file")
    avsync_analyze_parser.add_argument("--output", default=None, help="Optional output CSV file")
    avsync_analyze_parser.add_argument("--min-gap-seconds", type=float, default=2.0, help="Minimum gap between detected pulses")
    args = parser.parse_args()

    if args.command == "run":
        from camera_iqa.report_excel import export_excel

        config = load_config(args.config)
        overlay_dir = default_overlay_dir()
        results = [process_image(path, config, overlay_dir) for path in list_images(args.input)]
        export_excel(results, args.output, config, args.input)
    elif args.command == "graychart":
        result = analyze_grayscale_chart_path(
            args.image,
            expected_levels=args.expected_levels,
            min_delta=args.min_delta,
            annotation_path=args.annotated_output,
        )
        print(f"可分辨级数: {result.distinguishable_levels}/{result.expected_levels}")
        print(f"判定阈值: 相邻灰度差 >= {result.min_delta:g}")
        print("最佳灰阶条均值:", ", ".join(f"{block.mean_gray:.1f}" for block in result.best_strip.blocks))
        print("相邻灰度差:", ", ".join(f"{delta:.1f}" for delta in result.best_strip.adjacent_deltas))
        if result.annotation_path is not None:
            print(f"标注图: {result.annotation_path}")
    elif args.command == "ite-calibrate":
        calibration = calibrate_ite_roi(
            read_image(args.image),
            args.roi,
            annotation_path=args.annotated_output,
        )
        save_calibration(calibration, args.output)
        print(f"ROI标定已保存: {args.output}")
        if args.annotated_output is not None:
            print(f"标注图: {args.annotated_output}")
    elif args.command == "ite-lines":
        calibration = load_calibration(args.roi)
        rows = analyze_ite_resolution_folder(args.input, calibration)
        write_ite_resolution_csv(rows, args.output)
        print(f"ITE线数结果: {args.output}")
        print(f"处理图片数: {len(rows)}")
    elif args.command == "stripe-fov":
        if args.output is None and args.excel_output is None:
            parser.error("stripe-fov requires --output and/or --excel-output")
        rows = analyze_stripe_fov_folder(
            args.input,
            distances=read_distance_csv(args.distances),
            physical_black_width_m=args.physical_black_width_cm / 100.0,
            annotation_dir=args.annotated_dir,
        )
        if args.output is not None:
            write_stripe_fov_csv(rows, args.output)
            print(f"条纹视场角CSV结果: {args.output}")
        if args.excel_output is not None:
            write_stripe_fov_excel(rows, args.excel_output)
            print(f"条纹视场角Excel结果: {args.excel_output}")
        print(f"处理图片数: {len(rows)}")
    elif args.command == "avsync-generate":
        config = AvSyncPatternConfig(
            width=args.width,
            height=args.height,
            fps=args.fps,
            sample_rate=args.sample_rate,
            duration_seconds=args.duration,
            active_seconds=args.active_seconds,
            silent_seconds=args.silent_seconds,
            tone_hz=args.tone_hz,
        )
        try:
            muxed = generate_av_sync_assets(args.video_output, args.audio_output, config, muxed_output_path=args.muxed_output)
        except RuntimeError as exc:
            print(f"测试画面视频: {args.video_output}")
            print(f"测试蜂鸣音频: {args.audio_output}")
            print(f"合成带音频MP4失败: {exc}")
        else:
            print(f"测试画面视频: {args.video_output}")
            print(f"测试蜂鸣音频: {args.audio_output}")
            if muxed is not None:
                print(f"带音频测试视频: {muxed}")
    elif args.command == "avsync-analyze":
        result = analyze_av_sync_files(args.video, args.audio, min_gap_seconds=args.min_gap_seconds)
        if args.output is not None:
            write_av_sync_csv(result, args.output)
            print(f"音视频同步CSV结果: {args.output}")
        print(f"检测到画面脉冲: {len(result.video_pulses)}")
        print(f"检测到音频脉冲: {len(result.audio_pulses)}")
        print(f"成功配对: {result.pair_count}")
        print(f"平均偏移(audio-video): {result.mean_offset_ms:.1f} ms")
        print(f"偏移标准差: {result.std_offset_ms:.1f} ms")


if __name__ == "__main__":
    main()
