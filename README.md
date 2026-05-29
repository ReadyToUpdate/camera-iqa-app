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
- PyQt6 单页质检台界面
- 导出 Excel 报告：Dashboard、Details、Defect Samples、Config

## 命令行批测

```bash
python3 -m camera_iqa.cli run --input ./images --output report.xlsx
```
