from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir

from camera_iqa.detectors import detect_defects
from camera_iqa.metrics import calculate_metrics, make_overlay, read_image, save_image
from camera_iqa.models import ImageResult, SUPPORTED_EXTENSIONS


def list_images(folder: str | Path) -> list[Path]:
    root = Path(folder)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def process_image(path: str | Path, config: dict, overlay_dir: str | Path | None = None) -> ImageResult:
    image_path = Path(path)
    try:
        image = read_image(image_path)
        metrics = calculate_metrics(image)
        verdict, severity, defects = detect_defects(metrics, config)
        height, width = image.shape[:2]
        overlay_path = None
        if overlay_dir:
            overlay_path = Path(overlay_dir) / f"{image_path.stem}_overlay.jpg"
            save_image(overlay_path, make_overlay(image, metrics))
        return ImageResult(
            path=image_path,
            status="done",
            verdict=verdict,
            severity=severity,
            metrics=metrics,
            defects=defects,
            width=width,
            height=height,
            overlay_path=overlay_path,
        )
    except Exception as exc:
        return ImageResult(path=image_path, status="error", verdict="error", severity="fail", error=str(exc))


def default_overlay_dir() -> Path:
    return Path(gettempdir()) / "camera_iqa_overlays"
