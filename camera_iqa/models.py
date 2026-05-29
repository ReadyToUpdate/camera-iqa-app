from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class Defect:
    code: str
    label: str
    severity: str
    reason: str
    evidence: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ImageResult:
    path: Path
    status: str
    verdict: str = "untested"
    severity: str = "pass"
    metrics: dict[str, float] = field(default_factory=dict)
    defects: list[Defect] = field(default_factory=list)
    width: int = 0
    height: int = 0
    error: str = ""
    overlay_path: Path | None = None

    @property
    def defect_codes(self) -> str:
        return ", ".join(defect.code for defect in self.defects)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "file": self.path.name,
            "path": str(self.path),
            "status": self.status,
            "verdict": self.verdict,
            "severity": self.severity,
            "width": self.width,
            "height": self.height,
            "defects": self.defect_codes,
            "error": self.error,
        }
        row.update(self.metrics)
        return row
