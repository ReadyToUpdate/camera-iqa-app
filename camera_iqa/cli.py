from __future__ import annotations

import argparse

from camera_iqa.config import load_config
from camera_iqa.pipeline import default_overlay_dir, list_images, process_image
from camera_iqa.report_excel import export_excel


def main() -> None:
    parser = argparse.ArgumentParser(description="Security camera image quality assessment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--input", required=True, help="Input image folder")
    run_parser.add_argument("--output", required=True, help="Output Excel file")
    run_parser.add_argument("--config", default=None, help="Threshold YAML file")
    args = parser.parse_args()

    if args.command == "run":
        config = load_config(args.config)
        overlay_dir = default_overlay_dir()
        results = [process_image(path, config, overlay_dir) for path in list_images(args.input)]
        export_excel(results, args.output, config, args.input)


if __name__ == "__main__":
    main()
