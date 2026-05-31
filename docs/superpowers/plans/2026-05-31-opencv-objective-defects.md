# OpenCV Objective Defects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-version OpenCV detector for objective camera image defects: dark corners, bright corners, black borders, low-light banding, hot/dead pixels, and color cast.

**Architecture:** Extend `camera_iqa.metrics.calculate_metrics()` with interpretable image-derived scores, then map those scores to `Defect` records in `camera_iqa.detectors.detect_defects()`. Keep all results in the existing metrics/defects/overlay pipeline so a future YOLO-style model can plug into the same reporting surface.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, PyYAML, pytest.

---

### Task 1: Objective Defect Metrics

**Files:**
- Modify: `camera_iqa/metrics.py`
- Test: `tests/test_objective_defects.py`

- [ ] Write tests with synthetic images for corner falloff, corner glare, border bars, row banding, hot pixels, dead pixels, and color cast.
- [ ] Run `python3 -m pytest tests/test_objective_defects.py -q` and verify the new tests fail because the metric keys are missing.
- [ ] Implement metrics in `calculate_metrics()`:
  - `dark_corner_score`: center brightness compared with darkest corner patches.
  - `bright_corner_score`: brightest corner patches compared with center brightness.
  - `black_border_ratio`: fraction of edge strips that are near-black with low texture.
  - `low_light_stripe_score`: row/column periodic disturbance strength gated by low brightness.
  - `hot_pixel_ratio`: isolated very bright pixels that differ strongly from local median.
  - `dead_pixel_ratio`: isolated very dark pixels that differ strongly from local median.
- [ ] Re-run the focused tests and keep existing metric names unchanged.

### Task 2: Defect Classification

**Files:**
- Modify: `camera_iqa/detectors.py`
- Modify: `config/default_thresholds.yaml`
- Test: `tests/test_objective_defects.py`

- [ ] Add threshold rules for the new metrics.
- [ ] Add defect mappings:
  - `dark_corner` / `暗角`
  - `bright_corner` / `亮角`
  - `black_border` / `黑边`
  - `low_light_stripe` / `低照度条纹`
  - `hot_pixel` / `亮点`
  - `dead_pixel` / `坏点`
- [ ] Run `python3 -m pytest tests/test_objective_defects.py tests/test_detectors.py -q`.

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`

- [ ] Document that v1 uses OpenCV rules and reserves a future model-assisted path for trained local defects.
- [ ] Run `python3 -m pytest -q`.
