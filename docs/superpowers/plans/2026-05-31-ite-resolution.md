# ITE Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one-time ROI calibration and automated batch line-count reporting for ITE Chart A center vertical line groups.

**Architecture:** Create a focused `camera_iqa/ite_resolution.py` module for calibration JSON, ROI scaling, scoring, lux parsing, batch rows, and CSV export. Extend `camera_iqa/cli.py` with `ite-calibrate` and `ite-lines` commands that call the module without changing the existing IQA pipeline.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, standard-library JSON/CSV, pytest.

---

### Task 1: Core ROI and Scoring

**Files:**
- Create: `tests/test_ite_resolution.py`
- Create: `camera_iqa/ite_resolution.py`

- [ ] **Step 1: Write failing tests**

Add tests that create synthetic marker bands, assert the highest resolved marker, assert low-contrast high markers stop resolution, and assert ROI calibration JSON round-trips.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_ite_resolution.py -v`
Expected: FAIL with missing module or missing functions.

- [ ] **Step 3: Implement minimal core module**

Add `ResolutionROI`, `ITEResolutionCalibration`, `ITEResolutionResult`, `calibrate_ite_roi`, `save_calibration`, `load_calibration`, `scale_roi`, `analyze_ite_resolution`, and helper scoring functions.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ite_resolution.py -v`
Expected: PASS.

### Task 2: Batch CSV and Lux Parsing

**Files:**
- Modify: `tests/test_ite_resolution.py`
- Modify: `camera_iqa/ite_resolution.py`

- [ ] **Step 1: Write failing tests**

Add tests for filename lux parsing, empty folder CSV header, batch row generation, and CSV columns.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_ite_resolution.py -v`
Expected: FAIL on missing batch functions.

- [ ] **Step 3: Implement batch support**

Add `parse_lux_from_filename`, `analyze_ite_resolution_path`, `analyze_ite_resolution_folder`, and `write_ite_resolution_csv`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ite_resolution.py -v`
Expected: PASS.

### Task 3: CLI Commands

**Files:**
- Modify: `camera_iqa/cli.py`
- Modify: `tests/test_ite_resolution.py`

- [ ] **Step 1: Write failing CLI smoke tests**

Use direct module calls for behavior and run CLI commands manually for smoke verification.

- [ ] **Step 2: Implement CLI**

Add `ite-calibrate` and `ite-lines` subcommands. `ite-calibrate` accepts `--image`, `--roi x,y,w,h`, `--output`, and `--annotated-output`. `ite-lines` accepts `--input`, `--roi`, and `--output`.

- [ ] **Step 3: Verify**

Run: `pytest -q`
Expected: PASS.

Run sample CLI commands against generated synthetic images.
Expected: calibration JSON, annotation image, and CSV are created.
