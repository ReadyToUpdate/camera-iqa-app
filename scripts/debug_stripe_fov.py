from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_iqa.stripe_fov import analyze_stripe_fov_path


# PyCharm 调试时可以直接改这里，然后点 Run/Debug。
IMAGE_PATH = "/Users/zhangbin/Downloads/u=1428198397,117153909&fm=193.jpeg"
DISTANCE_M = 5.0
PHYSICAL_BLACK_WIDTH_CM = 1.0
OUTPUT_DIR = "debug_outputs"
SHOW_WINDOW = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug black stripe width and horizontal FOV on one image")
    parser.add_argument("--image", default=IMAGE_PATH, help="Input stripe image")
    parser.add_argument("--distance-m", type=float, default=DISTANCE_M, help="Camera-to-chart distance in meters")
    parser.add_argument("--physical-black-width-cm", type=float, default=PHYSICAL_BLACK_WIDTH_CM, help="Physical black stripe width in cm")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Folder for annotated debug image")
    parser.add_argument("--show", action="store_true", default=SHOW_WINDOW, help="Show annotated image with OpenCV")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = output_dir / f"{image_path.stem}_stripe_fov_debug.jpg"

    result = analyze_stripe_fov_path(
        image_path,
        distance_m=args.distance_m,
        physical_black_width_m=args.physical_black_width_cm / 100.0,
        annotation_path=annotation_path,
    )

    print(
        json.dumps(
            {
                "image": result.image,
                "zoom": result.zoom,
                "distance_m": result.distance_m,
                "resolution": f"{result.image_width}x{result.image_height}",
                "chart_x": result.chart_x,
                "chart_width": result.chart_width,
                "black_stripe_px_mean": round(result.black_stripe_px_mean, 3),
                "black_stripe_px_std": round(result.black_stripe_px_std, 3),
                "black_stripe_count": result.black_stripe_count,
                "horizontal_fov_deg": round(result.horizontal_fov_deg, 6),
                "annotation_path": str(annotation_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.show:
        import cv2

        annotated = cv2.imread(str(annotation_path))
        cv2.imshow("stripe_fov_debug", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
