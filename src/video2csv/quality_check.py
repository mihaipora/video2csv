"""Quality check: sample frames, use Claude CLI to visually verify OCR values.

Usage:
    python -m video2csv.quality_check -c CONFIG --csv CSV --video VIDEO [options]

Requires: claude CLI available in PATH.
Outputs: data/quality_check/report.md, report.json, and frame PNGs.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2

from video2csv.video import open_video, read_frame_at


def load_config(path: Path) -> dict:
    """Load and return the JSON config."""
    with open(path) as f:
        return json.load(f)


def make_short_name(roi_name: str) -> str:
    """Generate a short display label from an ROI name.

    Strip unit suffix in parentheses, then abbreviate: first char of each
    underscore-separated prefix word + last word truncated to 3 chars.
    E.g. 'boost_pressure_loop_diff (mbar)' -> 'bpl_dif'
    """
    # Strip unit suffix
    name = re.sub(r"\s*\(.*?\)\s*$", "", roi_name).strip()
    parts = name.split("_")
    if len(parts) == 1:
        return parts[0][:7]
    prefix = "".join(p[0] for p in parts[:-1])
    suffix = parts[-1][:3]
    return f"{prefix}_{suffix}"


def compute_crop_box(rois: list[dict], padding: int = 25) -> tuple[int, int, int, int]:
    """Compute bounding box (y1, y2, x1, x2) from all ROIs with vertical padding."""
    min_x = min(r["x"] for r in rois)
    min_y = min(r["y"] for r in rois)
    max_x = max(r["x"] + r["width"] for r in rois)
    max_y = max(r["y"] + r["height"] for r in rois)
    return (max(0, min_y - padding), max_y + padding, min_x, max_x)


def call_claude(image_path: Path, num_values: int, example_values: str) -> str:
    """Call claude CLI to read values from the image."""
    prompt = f"""Read the image file at {image_path}. It shows {num_values} numeric values with units in a table column, from top to bottom.

Read each value and output ONLY a JSON array of {num_values} numbers in order from top to bottom.
For example: {example_values}

Rules:
- Include the sign (negative values must have minus)
- Use decimal points where shown (e.g. 6.9, not 6)
- Output ONLY the JSON array, nothing else"""

    result = subprocess.run(
        ["claude", "-p", prompt, "--model", "haiku", "--allowed-tools", "Read",
         "--dangerously-skip-permissions", "--no-session-persistence"],
        capture_output=True, text=True, timeout=120,
    )
    return result.stdout.strip()


def parse_values(raw: str, expected_count: int) -> list | None:
    """Extract JSON array from Claude's response, validating length."""
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return None
    try:
        vals = json.loads(raw[start:end])
        if isinstance(vals, list) and len(vals) == expected_count:
            return vals
    except json.JSONDecodeError:
        pass
    return None


def values_match(csv_val: str, claude_val) -> bool:
    """Check if a CSV string value matches a Claude-extracted number."""
    try:
        csv_num = float(csv_val)
    except (ValueError, TypeError):
        return False
    claude_num = float(claude_val)
    if abs(csv_num - claude_num) < 0.011:
        return True
    return False


def make_example_values(rois: list[dict]) -> str:
    """Generate a plausible example JSON array for the prompt."""
    examples = []
    for roi in rois:
        vtype = roi.get("value_type", "int")
        signed = roi.get("signed", False)
        if vtype == "float":
            examples.append(-0.5 if signed else 6.9)
        else:
            examples.append(-84 if signed else 800)
    return json.dumps(examples)


def main():
    parser = argparse.ArgumentParser(
        description="Quality check: visually verify OCR extraction with Claude.")
    parser.add_argument("-c", "--config", type=Path, required=True,
                        help="Path to the JSON ROI configuration file.")
    parser.add_argument("--csv", type=Path, required=True,
                        help="Path to the small CSV (changes-only).")
    parser.add_argument("--video", type=Path, required=True,
                        help="Path to the video file.")
    parser.add_argument("-n", type=int, default=100,
                        help="Number of frames to sample.")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="Stop after checking N frames.")
    parser.add_argument("-o", type=Path, default=Path("data/quality_check"),
                        help="Output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    rois = config["rois"]
    num_params = len(rois)
    param_names = [r["name"] for r in rois]
    short_names = [make_short_name(name) for name in param_names]
    crop_box = compute_crop_box(rois)
    example_values = make_example_values(rois)

    out_dir = args.o
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    csv_data = {}
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            csv_data[int(row["frame"])] = row

    # Sample N evenly spaced frames
    all_frames = sorted(csv_data.keys())
    n = min(args.n, len(all_frames))
    step = max(1, len(all_frames) // n)
    sample = [all_frames[i * step] for i in range(n)]
    print(f"Sampling {len(sample)} frames from {len(all_frames)} unique OCR frames")
    print(f"Config: {args.config} ({num_params} parameters)")

    # Dump frame PNGs (value column crop)
    y1, y2, x1, x2 = crop_box
    cap, meta = open_video(args.video)
    for idx in sample:
        img_path = out_dir / f"frame_{idx}.png"
        if not img_path.exists():
            frame = read_frame_at(cap, idx)
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(str(img_path), crop)
    cap.release()
    print(f"Frame PNGs ready in {out_dir}/")

    # Print header
    header = "  ".join(f"{s:>7}" for s in short_names)
    print(f"\n{'':>16}{header}")
    print("-" * (16 + len(header)))

    # Run Claude checks
    error_frames = []
    total_errors = 0
    check_count = args.stop_after or len(sample)

    for i, idx in enumerate(sample[:check_count]):
        img_path = out_dir / f"frame_{idx}.png"
        csv_row = csv_data[idx]

        # Get CSV values
        csv_vals = [csv_row[name] for name in param_names]
        csv_display = "  ".join(f"{v:>7}" for v in csv_vals)

        # Get Claude values
        try:
            raw = call_claude(img_path, num_params, example_values)
            claude_vals = parse_values(raw, num_params)
        except subprocess.TimeoutExpired:
            claude_vals = None
        except Exception:
            claude_vals = None

        print(f"CSV values   : {csv_display}")

        if claude_vals is None:
            print(f"Claude values: {'PARSE ERROR':>7}")
            error_frames.append({"frame": idx, "errors": [
                {"param": "PARSE", "csv": "", "actual": "",
                 "issue": "Could not parse Claude response"}]})
            total_errors += 1
            print(f"[{i+1}/{check_count}] Frame {idx}... PARSE ERROR\n")
            continue

        claude_display = "  ".join(
            f"{v:>7}" if v is not None else "   None" for v in claude_vals)
        print(f"Claude values: {claude_display}")

        # Compare
        diffs = []
        for j, name in enumerate(param_names):
            if not values_match(csv_vals[j], claude_vals[j]):
                diffs.append((name, csv_vals[j], claude_vals[j]))
                total_errors += 1

        if diffs:
            for name, csv_v, claude_v in diffs:
                print(f"  !! {name} is different: {csv_v} vs {claude_v}")
            error_frames.append({"frame": idx, "errors": [
                {"param": name, "csv": str(csv_v), "actual": str(claude_v),
                 "issue": "value mismatch"}
                for name, csv_v, claude_v in diffs
            ]})
            print(f"[{i+1}/{check_count}] Frame {idx}... FAIL ({len(diffs)} errors)\n")
        else:
            print(f"[{i+1}/{check_count}] Frame {idx}... OK\n")

    # Save reports
    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"
    checked = min(check_count, len(sample))
    total_values = checked * num_params

    report_data = {
        "total_frames_checked": checked,
        "total_values_checked": total_values,
        "frames_with_errors": len(error_frames),
        "total_value_errors": total_errors,
        "accuracy_pct": round(
            (1 - total_errors / total_values) * 100, 2) if total_values else 0,
        "error_frames": error_frames,
    }
    with open(report_json, "w") as f:
        json.dump(report_data, f, indent=2)

    with open(report_md, "w") as f:
        f.write("# Quality Check Report\n\n")
        f.write(f"- Config: {args.config}\n")
        f.write(f"- Parameters: {num_params}\n")
        f.write(f"- Frames checked: {checked}\n")
        f.write(f"- Values checked: {total_values}\n")
        f.write(f"- Frames with errors: {len(error_frames)}\n")
        f.write(f"- Total value errors: {total_errors}\n")
        f.write(f"- Accuracy: {report_data['accuracy_pct']}%\n\n")

        if error_frames:
            f.write("## Errors\n\n")
            for ef in error_frames:
                f.write(f"### Frame {ef['frame']}\n\n")
                f.write(f"Image: `frame_{ef['frame']}.png`\n\n")
                f.write("| Parameter | CSV Value | Actual (image) | Issue |\n")
                f.write("|-----------|-----------|----------------|-------|\n")
                for err in ef["errors"]:
                    f.write(f"| {err.get('param', '')} | {err.get('csv', '')} "
                            f"| {err.get('actual', '')} | {err.get('issue', '')} |\n")
                f.write("\n")

    print(f"Done. Report saved to {report_md} and {report_json}")
    print(f"Accuracy: {report_data['accuracy_pct']}% "
          f"({total_errors} errors in {total_values} values)")


if __name__ == "__main__":
    main()
