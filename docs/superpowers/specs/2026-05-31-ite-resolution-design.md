# ITE Resolution Line Count Design

## Purpose

Build a fully automated Python workflow for ITE Chart A center vertical line-group resolution testing across images captured at different illumination levels.

The user can perform one ROI calibration on a clear baseline image. After that, the tool reuses the saved ROI for every image in a batch and reports the highest resolvable line count per illumination level.

## Scope

In scope:

- Analyze the center vertical line group of an ITE Chart A image.
- Calibrate the center line-group ROI once from a baseline image.
- Save and load ROI calibration as JSON.
- Batch-process a folder of images captured at different illumination levels.
- Parse illumination values from filenames when present.
- Output CSV with per-image resolved line count and per-marker scores.
- Support an optional annotated image during calibration.

Out of scope:

- Full-card semantic recognition.
- Four-corner circular fan analysis.
- OCR of printed chart labels.
- Direct lux-meter integration.
- GUI controls for ROI selection.

## Standards-Inspired Test Method

The workflow follows the practical intent of the security-camera minimum usable illumination method described in GA/T 1127-2013: reduce illumination, inspect the resolution chart image, and record the illumination where the required resolution can still be distinguished.

This implementation automates the chart-reading portion. It does not certify compliance by itself because the physical setup, illumination uniformity, camera output conditions, and pass/fail resolution requirement are controlled outside the software.

## User Workflow

1. Capture a clear baseline image of the ITE Chart A at sufficient illumination.
2. Run calibration:

   ```bash
   python3 -m camera_iqa.cli ite-calibrate --image baseline.jpg --roi 245,115,85,190 --output roi.json --annotated-output baseline_roi.jpg
   ```

3. Capture a sequence of images at lower illumination levels.
4. Run batch analysis:

   ```bash
   python3 -m camera_iqa.cli ite-lines --input ./lux_images --roi roi.json --output ite_lines.csv
   ```

5. Review `ite_lines.csv` to find the highest resolvable line count at each illumination level.

## ROI Calibration

Calibration stores:

- Source image size.
- ROI rectangle in baseline-image pixels.
- Marker line values used for scoring.
- Algorithm thresholds.

When a batch image has a different size from the baseline, the ROI scales proportionally by width and height. This supports repeated captures from the same setup where the camera output resolution may change but the chart framing remains geometrically similar.

Manual ROI input is the reliable path. Automatic ROI detection can be added later, but it is not required for the first automated workflow.

## Resolution Scoring

The center line-group ROI is split vertically into marker bands. Default marker values are:

```text
200, 300, 400, 500, 600, 700, 800, 1000
```

For each marker band:

- Convert ROI to grayscale.
- Sample the central portion of the band to avoid border noise.
- Build a horizontal intensity profile.
- Estimate black/white modulation using robust high and low percentiles.
- Count dark line runs in the profile.
- Count edge transitions to reject smooth blur that still has brightness variation.
- Mark the band resolved when modulation, dark-run count, and transition count pass thresholds.

The reported line count is the highest continuous resolved marker. If early low markers fail, the result is `0`.

## Batch Output

CSV columns:

```text
image,path,lux,resolved_lines,confidence,roi_x,roi_y,roi_width,roi_height,score_200,score_300,score_400,score_500,score_600,score_700,score_800,score_1000
```

`lux` is parsed from filename patterns such as:

- `0.1lux.jpg`
- `lux_0.05.png`
- `illum-2.5.jpg`

If no illumination value is found, `lux` is blank.

`confidence` is the resolved marker score. It is `0.0` when no marker resolves.

## Error Handling

- Missing or invalid ROI JSON raises a clear `ValueError`.
- Empty ROI crop raises a clear `ValueError`.
- Empty input folder produces a CSV header with no rows.
- Undecodable images are skipped with an error row containing the image name and message.

## Code Structure

- `camera_iqa/ite_resolution.py`: ROI model, calibration JSON, scoring, batch analysis, CSV export.
- `camera_iqa/cli.py`: `ite-calibrate` and `ite-lines` commands.
- `tests/test_ite_resolution.py`: synthetic line-card tests for scoring, ROI scaling, lux parsing, and CSV output.

## Testing Strategy

Use synthetic ITE center-line images so tests are deterministic:

- High-contrast marker bands resolve through a known marker.
- Blurred or low-contrast high markers do not resolve.
- Manual ROI calibration serializes and loads correctly.
- ROI scales when batch image size differs.
- Filename lux parsing handles common patterns.
- CSV export writes expected columns and values.

The implementation should also be tested manually with the provided ITE Chart A sample after the automated tests pass.
