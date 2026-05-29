from __future__ import annotations

from collections import Counter
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from camera_iqa.config import DEFAULT_CONFIG_PATH, load_config
from camera_iqa.models import ImageResult
from camera_iqa.pipeline import list_images
from camera_iqa.report_excel import export_excel
from camera_iqa_app.workers.batch_worker import BatchWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("安防摄像机图像客观测试工具")
        self.input_folder: Path | None = None
        self.config_path: Path | None = DEFAULT_CONFIG_PATH
        self.config = load_config(DEFAULT_CONFIG_PATH)
        self.image_paths: list[Path] = []
        self.results: dict[Path, ImageResult] = {}
        self.worker: BatchWorker | None = None
        self.show_overlay = False

        self.setup_actions()
        self.setup_ui()
        self.apply_styles()
        self.log(f"已加载默认阈值配置：{DEFAULT_CONFIG_PATH}")

    def setup_actions(self) -> None:
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.open_action = QAction("选择图片文件夹", self)
        self.open_action.triggered.connect(self.choose_folder)
        toolbar.addAction(self.open_action)

        self.config_action = QAction("选择配置", self)
        self.config_action.triggered.connect(self.choose_config)
        toolbar.addAction(self.config_action)

        self.start_action = QAction("开始测试", self)
        self.start_action.triggered.connect(self.start_batch)
        toolbar.addAction(self.start_action)

        self.stop_action = QAction("停止", self)
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self.stop_batch)
        toolbar.addAction(self.stop_action)

        self.export_action = QAction("导出 Excel", self)
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self.export_report)
        toolbar.addAction(self.export_action)

    def setup_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        content = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(content, stretch=1)

        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(260)
        self.file_list.currentItemChanged.connect(self.on_current_item_changed)
        content.addWidget(self.wrap_panel("图片列表", self.file_list))

        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.summary_label = QLabel("请选择图片文件夹")
        self.summary_label.setObjectName("SummaryLabel")
        center_layout.addWidget(self.summary_label)
        self.preview_label = QLabel("图片预览")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(520, 420)
        self.preview_label.setFrameShape(QFrame.Shape.StyledPanel)
        center_layout.addWidget(self.preview_label, stretch=1)
        self.overlay_button = QPushButton("切换原图 / 缺陷高亮图")
        self.overlay_button.clicked.connect(self.toggle_overlay)
        center_layout.addWidget(self.overlay_button)
        content.addWidget(center)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.result_label = QLabel("当前图片：未选择")
        self.result_label.setObjectName("ResultLabel")
        right_layout.addWidget(self.result_label)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.metrics_table, stretch=2)
        self.defect_table = QTableWidget(0, 3)
        self.defect_table.setHorizontalHeaderLabels(["缺陷", "等级", "原因"])
        self.defect_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.defect_table, stretch=1)
        content.addWidget(right)
        content.setSizes([280, 720, 360])

        bottom = QSplitter(Qt.Orientation.Horizontal)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        bottom_left = QWidget()
        bottom_left_layout = QVBoxLayout(bottom_left)
        bottom_left_layout.addWidget(QLabel("批量进度"))
        bottom_left_layout.addWidget(self.progress)
        self.stats_label = QLabel("通过率：-    缺陷分布：-")
        bottom_left_layout.addWidget(self.stats_label)
        bottom.addWidget(bottom_left)
        bottom.addWidget(self.wrap_panel("日志", self.log_view))
        bottom.setSizes([500, 700])
        root_layout.addWidget(bottom)

        self.setCentralWidget(root)

    def wrap_panel(self, title: str, widget: QWidget) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        label = QLabel(title)
        label.setObjectName("PanelTitle")
        layout.addWidget(label)
        layout.addWidget(widget)
        return panel

    def apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f5f7fa; }
            QToolBar { background: #243447; spacing: 8px; padding: 8px; }
            QToolButton { color: white; padding: 7px 10px; background: #31506d; border-radius: 4px; }
            QToolButton:disabled { color: #aeb8c2; background: #415267; }
            QLabel#SummaryLabel, QLabel#ResultLabel { font-size: 16px; font-weight: 700; color: #1f2933; }
            QLabel#PanelTitle { font-weight: 700; color: #1f2933; padding: 4px 0; }
            QListWidget, QTableWidget, QTextEdit, QLabel[frameShape="6"] {
                background: white;
                border: 1px solid #d7dde5;
                border-radius: 6px;
            }
            QProgressBar { height: 18px; border: 1px solid #cfd7e3; border-radius: 4px; background: white; }
            QProgressBar::chunk { background: #2f80ed; border-radius: 4px; }
            QPushButton { padding: 8px 10px; border: 1px solid #cfd7e3; border-radius: 5px; background: white; }
            QPushButton:hover { background: #eef4ff; }
            """
        )

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        self.input_folder = Path(folder)
        self.image_paths = list_images(self.input_folder)
        self.results.clear()
        self.file_list.clear()
        for path in self.image_paths:
            self.file_list.addItem(QListWidgetItem(f"[待测] {path.name}"))
        self.summary_label.setText(f"已选择 {self.input_folder}，共 {len(self.image_paths)} 张图片")
        self.progress.setValue(0)
        self.export_action.setEnabled(False)
        self.log(f"已载入图片文件夹：{self.input_folder}")
        self.update_stats()

    def choose_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择阈值配置", str(DEFAULT_CONFIG_PATH.parent), "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            self.config_path = Path(path)
            self.config = load_config(self.config_path)
            self.log(f"已加载配置：{self.config_path}")
        except Exception as exc:
            QMessageBox.critical(self, "配置读取失败", str(exc))

    def start_batch(self) -> None:
        if not self.image_paths:
            QMessageBox.information(self, "没有图片", "请先选择包含图片的文件夹。")
            return
        if self.worker and self.worker.isRunning():
            return
        self.results.clear()
        self.progress.setValue(0)
        self.start_action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.export_action.setEnabled(False)
        self.worker = BatchWorker(self.image_paths, self.config, self)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker.log_ready.connect(self.log)
        self.worker.finished_batch.connect(self.on_finished_batch)
        self.worker.start()

    def stop_batch(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.stop_action.setEnabled(False)

    def on_result_ready(self, result: ImageResult, index: int, total: int) -> None:
        self.results[result.path] = result
        row = self.image_paths.index(result.path)
        prefix = result.severity.upper() if result.status == "done" else "ERROR"
        defects = result.defect_codes or "通过"
        self.file_list.item(row).setText(f"[{prefix}] {result.path.name}  {defects}")
        self.color_item(self.file_list.item(row), result.severity)
        self.progress.setValue(round(index / total * 100))
        self.update_stats()
        if self.file_list.currentRow() < 0:
            self.file_list.setCurrentRow(row)
        elif self.file_list.currentRow() == row:
            self.show_result(result)

    def on_finished_batch(self, stopped: bool) -> None:
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.export_action.setEnabled(bool(self.results))
        self.log("检测停止" if stopped else "检测完成")

    def on_current_item_changed(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        row = self.file_list.row(current)
        if row < 0 or row >= len(self.image_paths):
            return
        path = self.image_paths[row]
        result = self.results.get(path)
        if result:
            self.show_result(result)
        else:
            self.result_label.setText(f"当前图片：{path.name}（待测）")
            self.show_preview(path)
            self.metrics_table.setRowCount(0)
            self.defect_table.setRowCount(0)

    def show_result(self, result: ImageResult) -> None:
        self.result_label.setText(f"当前图片：{result.path.name}    结论：{result.verdict} / {result.severity}")
        self.show_preview(result.overlay_path if self.show_overlay and result.overlay_path else result.path)
        self.metrics_table.setRowCount(len(result.metrics))
        for row, (key, value) in enumerate(result.metrics.items()):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(key))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{value:.4f}"))
        self.defect_table.setRowCount(len(result.defects))
        for row, defect in enumerate(result.defects):
            self.defect_table.setItem(row, 0, QTableWidgetItem(defect.label))
            self.defect_table.setItem(row, 1, QTableWidgetItem(defect.severity))
            self.defect_table.setItem(row, 2, QTableWidgetItem(defect.reason))

    def show_preview(self, path: Path | None) -> None:
        if not path:
            self.preview_label.setText("无预览")
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.preview_label.setText("无法显示图片预览")
            return
        target = self.preview_label.size()
        self.preview_label.setPixmap(pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def toggle_overlay(self) -> None:
        self.show_overlay = not self.show_overlay
        current = self.file_list.currentItem()
        if current:
            self.on_current_item_changed(current)

    def export_report(self) -> None:
        if not self.results:
            QMessageBox.information(self, "没有结果", "请先运行测试。")
            return
        default_name = "camera_iqa_report.xlsx"
        path, _ = QFileDialog.getSaveFileName(self, "导出 Excel 报告", default_name, "Excel (*.xlsx)")
        if not path:
            return
        try:
            ordered = [self.results[p] for p in self.image_paths if p in self.results]
            export_excel(ordered, path, self.config, self.input_folder)
            self.log(f"已导出 Excel 报告：{path}")
            QMessageBox.information(self, "导出完成", f"报告已生成：\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def update_stats(self) -> None:
        total = len(self.results)
        if total == 0:
            self.stats_label.setText("通过率：-    缺陷分布：-")
            return
        passed = sum(1 for result in self.results.values() if result.verdict == "pass")
        counter = Counter(defect.code for result in self.results.values() for defect in result.defects)
        distribution = "，".join(f"{code}:{count}" for code, count in counter.most_common()) or "无"
        self.stats_label.setText(f"通过率：{passed / total * 100:.1f}%    缺陷分布：{distribution}")

    def color_item(self, item: QListWidgetItem, severity: str) -> None:
        if severity == "fail":
            item.setForeground(QColor("#b42318"))
        elif severity == "warn":
            item.setForeground(QColor("#b54708"))
        else:
            item.setForeground(QColor("#067647"))

    def log(self, message: str) -> None:
        self.log_view.append(message)
