"""
This script matches the frames by timestamp (nearest-neighbor, <50ms tolerance)
and outputs a unified dataset ready for BC training.
"""

import os
import json
import argparse
from pathlib import Path


def parse_frame_timestamp(filename: str) -> int:
    """Extract timestamp_ms from filename like '00000042_1716234567890.png'"""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return int(parts[1])
    raise ValueError(f"Can't parse timestamp from {filename}")


def parse_frame_index(filename: str) -> int:
    """Extract frame index from filename like '00000042_1716234567890.png'"""
    stem = Path(filename).stem
    parts = stem.split("_")
    return int(parts[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to mod's JSONL recording")
    parser.add_argument("--frames", required=True, help="Path to frames directory")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--tolerance-ms", type=int, default=50,
                        help="Max time difference for matching (default: 50ms)")
    parser.add_argument("--copy-frames", action="store_true",
                        help="Copy frames instead of symlinking")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_frames = output_dir / "frames"
    out_frames.mkdir(exist_ok=True)

    # Load JSONL entries
    print("Loading JSONL...")
    jsonl_entries = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "timestamp_ms" not in entry:
                print("ERROR: JSONL missing timestamp_ms. Was this recorded with the updated mod?")
                return
            jsonl_entries.append(entry)
    print(f"  {len(jsonl_entries)} state/action entries")

    # Load frame filenames and timestamps
    print("Loading frames...")
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith((".png", ".jpg")))
    frame_timestamps = []
    for f in frame_files:
        try:
            ts = parse_frame_timestamp(f)
            frame_timestamps.append((ts, f))
        except ValueError:
            continue
    frame_timestamps.sort(key=lambda x: x[0])
    print(f"  {len(frame_timestamps)} frames")

    if not jsonl_entries or not frame_timestamps:
        print("Nothing to sync.")
        return

    # Match by nearest timestamp
    print(f"Syncing (tolerance={args.tolerance_ms}ms)...")
    matched = 0
    unmatched = 0
    fi = 0

    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as mf:
        for entry in jsonl_entries:
            ts = entry["timestamp_ms"]

            # Binary search would be cleaner but this is fine for ~1M entries
            # Advance frame pointer to closest
            while fi < len(frame_timestamps) - 1 and frame_timestamps[fi + 1][0] <= ts:
                fi += 1

            # check neighbors
            best_diff = abs(frame_timestamps[fi][0] - ts)
            best_idx = fi
            if fi + 1 < len(frame_timestamps):
                diff_next = abs(frame_timestamps[fi + 1][0] - ts)
                if diff_next < best_diff:
                    best_diff = diff_next
                    best_idx = fi + 1

            if best_diff > args.tolerance_ms:
                unmatched += 1
                continue

            frame_file = frame_timestamps[best_idx][1]
            src = frames_dir / frame_file
            dst = out_frames / frame_file

            if args.copy_frames:
                if not dst.exists():
                    import shutil
                    shutil.copy2(src, dst)
            else:
                if not dst.exists():
                    dst.symlink_to(src.resolve())

            # Write manifest line 
            manifest_entry = {
                "frame": frame_file,
                "frame_idx": entry["frame"],
                "timestamp_ms": ts,
                "action": entry["action"],
                "state": entry["state"],
            }
            mf.write(json.dumps(manifest_entry) + "\n")
            matched += 1

    print(f"\nDone:")
    print(f"  Matched:   {matched}")
    print(f"  Unmatched: {unmatched}")
    print(f"  Manifest:  {manifest_path}")

    #quick stats
    if matched > 0:
        print(f"  Duration:  {matched / 15:.0f}s ({matched / 15 / 60:.1f} min)")
        sample_frame = out_frames / frame_files[0]
        if sample_frame.exists():
            size_kb = sample_frame.stat().st_size / 1024
            est_gb = (size_kb * matched) / (1024 * 1024)
            print(f"  Est. size: ~{est_gb:.1f} GB frames")


if __name__ == "__main__":
    main()
