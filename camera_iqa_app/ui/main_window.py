from __future__ import annotations

from collections import Counter
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
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

from camera_iqa.catalog import metric_group_for
from camera_iqa.config import DEFAULT_CONFIG_PATH, load_config
from camera_iqa.detectors import detect_defects
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
        self._syncing_tune_controls = False

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
        self.metrics_table = QTableWidget(0, 3)
        self.metrics_table.setHorizontalHeaderLabels(["类别", "指标", "数值"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.metrics_table, stretch=2)
        self.defect_table = QTableWidget(0, 4)
        self.defect_table.setHorizontalHeaderLabels(["类别", "缺陷", "等级", "原因"])
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
        bottom_left_layout.addWidget(self.create_tuning_panel())
        bottom.addWidget(bottom_left)
        bottom.addWidget(self.wrap_panel("日志", self.log_view))
        bottom.setSizes([500, 700])
        root_layout.addWidget(bottom)

        self.setCentralWidget(root)
        self.refresh_tune_metric_options()

    def create_tuning_panel(self) -> QWidget:
        group = QGroupBox("单指标调优")
        layout = QVBoxLayout(group)

        form = QFormLayout()
        self.tune_metric_combo = QComboBox()
        self.tune_metric_combo.currentTextChanged.connect(self.on_tune_metric_changed)
        form.addRow("指标", self.tune_metric_combo)

        self.tune_direction_combo = QComboBox()
        self.tune_direction_combo.currentTextChanged.connect(self.on_tune_direction_changed)
        form.addRow("方向", self.tune_direction_combo)

        self.warn_spin = QDoubleSpinBox()
        self.warn_spin.setDecimals(6)
        self.warn_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.warn_spin.setSingleStep(1.0)
        form.addRow("warn 阈值", self.warn_spin)

        self.fail_spin = QDoubleSpinBox()
        self.fail_spin.setDecimals(6)
        self.fail_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.fail_spin.setSingleStep(1.0)
        form.addRow("fail 阈值", self.fail_spin)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.apply_tune_button = QPushButton("应用到当前结果")
        self.apply_tune_button.clicked.connect(self.apply_tuning)
        button_row.addWidget(self.apply_tune_button)

        self.save_config_button = QPushButton("保存阈值配置")
        self.save_config_button.clicked.connect(self.save_tuned_config)
        button_row.addWidget(self.save_config_button)
        layout.addLayout(button_row)

        self.tune_stats_label = QLabel("先运行测试，再调单个指标。")
        self.tune_stats_label.setWordWrap(True)
        layout.addWidget(self.tune_stats_label)
        return group

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
            QGroupBox { font-weight: 700; color: #1f2933; border: 1px solid #d7dde5; border-radius: 6px; margin-top: 8px; padding: 8px; background: white; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
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
        self.refresh_tuning_stats()

    def choose_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择阈值配置", str(DEFAULT_CONFIG_PATH.parent), "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            self.config_path = Path(path)
            self.config = load_config(self.config_path)
            self.refresh_tune_metric_options()
            self.refresh_tune_controls()
            self.refresh_tuning_stats()
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
        self.refresh_file_list_item(row, result)
        self.progress.setValue(round(index / total * 100))
        self.update_stats()
        self.refresh_tuning_stats()
        if self.file_list.currentRow() < 0:
            self.file_list.setCurrentRow(row)
        elif self.file_list.currentRow() == row:
            self.show_result(result)

    def on_finished_batch(self, stopped: bool) -> None:
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.export_action.setEnabled(bool(self.results))
        self.log("检测停止" if stopped else "检测完成")
        self.refresh_tuning_stats()

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
            self.metrics_table.setItem(row, 0, QTableWidgetItem(metric_group_for(key)))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(key))
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{value:.4f}"))
        self.defect_table.setRowCount(len(result.defects))
        for row, defect in enumerate(result.defects):
            self.defect_table.setItem(row, 0, QTableWidgetItem(defect.category))
            self.defect_table.setItem(row, 1, QTableWidgetItem(defect.label))
            self.defect_table.setItem(row, 2, QTableWidgetItem(defect.severity))
            self.defect_table.setItem(row, 3, QTableWidgetItem(defect.reason))

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

    def refresh_file_list_item(self, row: int, result: ImageResult) -> None:
        item = self.file_list.item(row)
        if item is None:
            return
        prefix = result.severity.upper() if result.status == "done" else "ERROR"
        defects = result.defect_codes or "通过"
        item.setText(f"[{prefix}] {result.path.name}  {defects}")
        self.color_item(item, result.severity)

    def refresh_tune_metric_options(self) -> None:
        if not hasattr(self, "tune_metric_combo"):
            return
        current = self.tune_metric_combo.currentData()
        self._syncing_tune_controls = True
        self.tune_metric_combo.clear()
        for metric, rule in self.config.get("thresholds", {}).items():
            if self.rule_direction_pairs(rule):
                self.tune_metric_combo.addItem(f"{metric_group_for(metric)} / {metric}", metric)
        if current:
            index = self.tune_metric_combo.findData(current)
            if index >= 0:
                self.tune_metric_combo.setCurrentIndex(index)
        self._syncing_tune_controls = False
        self.refresh_tune_controls()

    def on_tune_metric_changed(self, *_args) -> None:
        if self._syncing_tune_controls:
            return
        self.refresh_tune_controls()
        self.refresh_tuning_stats()

    def on_tune_direction_changed(self, *_args) -> None:
        if self._syncing_tune_controls:
            return
        self.refresh_tune_values()
        self.refresh_tuning_stats()

    def refresh_tune_controls(self) -> None:
        metric = self.current_tune_metric()
        if not metric:
            return
        rule = self.config.get("thresholds", {}).get(metric, {})
        pairs = self.rule_direction_pairs(rule)
        current = self.tune_direction_combo.currentData()
        self._syncing_tune_controls = True
        self.tune_direction_combo.clear()
        for label, warn_key, fail_key in pairs:
            self.tune_direction_combo.addItem(label, (warn_key, fail_key))
        if current:
            index = self.tune_direction_combo.findData(current)
            if index >= 0:
                self.tune_direction_combo.setCurrentIndex(index)
        self._syncing_tune_controls = False
        self.refresh_tune_values()

    def refresh_tune_values(self) -> None:
        metric = self.current_tune_metric()
        keys = self.current_tune_rule_keys()
        if not metric or not keys:
            return
        rule = self.config.get("thresholds", {}).get(metric, {})
        warn_key, fail_key = keys
        self._syncing_tune_controls = True
        self.warn_spin.setValue(float(rule.get(warn_key, 0.0)))
        self.fail_spin.setValue(float(rule.get(fail_key, 0.0)))
        self._syncing_tune_controls = False

    def apply_tuning(self) -> None:
        metric = self.current_tune_metric()
        keys = self.current_tune_rule_keys()
        if not metric or not keys:
            QMessageBox.information(self, "无法调优", "当前配置没有可调的 warn/fail 阈值。")
            return
        warn_key, fail_key = keys
        rule = self.config.setdefault("thresholds", {}).setdefault(metric, {})
        rule[warn_key] = float(self.warn_spin.value())
        rule[fail_key] = float(self.fail_spin.value())
        self.reevaluate_results()
        self.log(f"已应用阈值：{metric} {warn_key}={rule[warn_key]:.6g}, {fail_key}={rule[fail_key]:.6g}")

    def reevaluate_results(self) -> None:
        for row, path in enumerate(self.image_paths):
            result = self.results.get(path)
            if not result or result.status != "done":
                continue
            verdict, severity, defects = detect_defects(result.metrics, self.config)
            result.verdict = verdict
            result.severity = severity
            result.defects = defects
            self.refresh_file_list_item(row, result)
        self.update_stats()
        self.refresh_tuning_stats()
        current = self.file_list.currentItem()
        if current:
            self.on_current_item_changed(current)

    def refresh_tuning_stats(self) -> None:
        if not hasattr(self, "tune_stats_label"):
            return
        metric = self.current_tune_metric()
        if not metric:
            self.tune_stats_label.setText("当前配置没有可调指标。")
            return
        values = [result.metrics[metric] for result in self.results.values() if result.status == "done" and metric in result.metrics]
        if not values:
            self.tune_stats_label.setText("先运行测试，再查看该指标的数值分布。")
            return
        warn_count = 0
        fail_count = 0
        keys = self.current_tune_rule_keys()
        rule = self.config.get("thresholds", {}).get(metric, {})
        if keys:
            warn_key, fail_key = keys
            warn_value = float(rule.get(warn_key, 0.0))
            fail_value = float(rule.get(fail_key, 0.0))
            if warn_key.endswith("_below"):
                warn_count = sum(1 for value in values if value < warn_value)
                fail_count = sum(1 for value in values if value < fail_value)
            else:
                warn_count = sum(1 for value in values if value > warn_value)
                fail_count = sum(1 for value in values if value > fail_value)
        avg = sum(values) / len(values)
        self.tune_stats_label.setText(
            f"样本数 {len(values)}，最小 {min(values):.4f}，平均 {avg:.4f}，最大 {max(values):.4f}；"
            f"按当前方向：warn 命中 {warn_count}，fail 命中 {fail_count}"
        )

    def save_tuned_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存阈值配置", str(self.config_path or DEFAULT_CONFIG_PATH), "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            import yaml

            data = {key: value for key, value in self.config.items() if not key.startswith("_")}
            with Path(path).open("w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
            self.config_path = Path(path)
            self.config["_source_path"] = str(self.config_path)
            self.log(f"已保存阈值配置：{self.config_path}")
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def current_tune_metric(self) -> str | None:
        if not hasattr(self, "tune_metric_combo"):
            return None
        return self.tune_metric_combo.currentData()

    def current_tune_rule_keys(self) -> tuple[str, str] | None:
        if not hasattr(self, "tune_direction_combo"):
            return None
        return self.tune_direction_combo.currentData()

    def rule_direction_pairs(self, rule: dict) -> list[tuple[str, str, str]]:
        pairs: list[tuple[str, str, str]] = []
        if "warn_below" in rule and "fail_below" in rule:
            pairs.append(("低于阈值判异常", "warn_below", "fail_below"))
        if "warn_above" in rule and "fail_above" in rule:
            pairs.append(("高于阈值判异常", "warn_above", "fail_above"))
        return pairs

    def color_item(self, item: QListWidgetItem, severity: str) -> None:
        if severity == "fail":
            item.setForeground(QColor("#b42318"))
        elif severity == "warn":
            item.setForeground(QColor("#b54708"))
        else:
            item.setForeground(QColor("#067647"))

    def log(self, message: str) -> None:
        self.log_view.append(message)
