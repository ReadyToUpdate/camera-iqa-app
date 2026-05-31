from __future__ import annotations

import argparse

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


if __name__ == "__main__":
    main()
