from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from camera_iqa.pipeline import default_overlay_dir, process_image


class BatchWorker(QThread):
    result_ready = pyqtSignal(object, int, int)
    log_ready = pyqtSignal(str)
    finished_batch = pyqtSignal(bool)

    def __init__(self, image_paths: list[Path], config: dict, parent=None) -> None:
        super().__init__(parent)
        self.image_paths = image_paths
        self.config = config
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        total = len(self.image_paths)
        overlay_dir = default_overlay_dir()
        self.log_ready.emit(f"开始检测 {total} 张图片")
        stopped = False
        for index, path in enumerate(self.image_paths, start=1):
            if self._stop_requested:
                stopped = True
                self.log_ready.emit("检测已停止，保留已完成结果")
                break
            result = process_image(path, self.config, overlay_dir)
            self.result_ready.emit(result, index, total)
        self.finished_batch.emit(stopped)
