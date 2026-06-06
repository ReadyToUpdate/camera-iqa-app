# Camera IQA App

安防摄像机图像客观指标和客观缺陷自动化测试工具。

## 运行

```bash
python3 -m pip install -e ".[dev]"
python3 -m camera_iqa_app
```

## 功能

- 批量读取图片文件夹：JPG、JPEG、PNG、BMP、TIF、TIFF
- 计算清晰度、亮度、曝光、对比度、噪声、色偏等客观指标
- 判定黑屏、白屏、模糊、过曝、欠曝、低对比度、高噪声、偏色、遮挡疑似、条纹疑似
- 基于 OpenCV 规则检出暗角、亮角、黑边、低照度条纹、孤立亮点和坏点
- 指标展示和报告按“客观指标 / 客观缺陷”两类管理
- PyQt6 单页质检台界面
- 导出 Excel 报告：Dashboard、Details、Defect Samples、Config

## 缺陷检测路线

第一版采用可解释的 OpenCV 指标和 YAML 阈值，适合快速落地、批量报表和现场阈值标定。后续如果纯规则阈值维护成本变高，可以在同一套 `metrics` / `defects` 输出结构上接入 YOLO 类模型，用标注数据学习局部缺陷并减少人工调参。

## 命令行批测

```bash
python3 -m camera_iqa.cli run --input ./images --output report.xlsx
```

## 变倍条纹视场角

准备一组黑白竖条测试图，文件名中包含倍率，例如 `zoom_1.0.jpg`、`zoom_1.1.jpg`、`camera-2.5x.png`。再准备距离表 `distances.csv`：

```csv
image,distance_m
zoom_1.0.jpg,5.0
zoom_1.1.jpg,5.2
camera-2.5x.png,8.0
```

运行：

```bash
python3 -m camera_iqa.cli stripe-fov \
  --input ./zoom_images \
  --distances ./distances.csv \
  --excel-output ./stripe_fov.xlsx \
  --physical-black-width-cm 1 \
  --annotated-dir ./stripe_annotations
```
