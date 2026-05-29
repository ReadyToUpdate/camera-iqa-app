from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from camera_iqa_app.ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("安防摄像机图像客观测试工具")
    window = MainWindow()
    window.resize(1360, 840)
    window.show()
    sys.exit(app.exec())
