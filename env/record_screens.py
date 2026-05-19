"""
Captures the Terraria window at 15hz, saves frames as compressed PNGs
in a session folder. Frames are keyed by timestamp so they can be synced
to the mod's JSONL output after the fact.
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime
from pathlib import Path

import mss
import numpy as np
from PIL import Image


def find_terraria_window():
    #find window bounds
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32

        #find Terraria window by title
        hwnd = user32.FindWindowW(None, "Terraria: tModLoader")
        if not hwnd:
            hwnd = user32.FindWindowW(None, "Terraria")
        if not hwnd:
            #try partial match
            def callback(hwnd, results):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    if "terraria" in buf.value.lower():
                        results.append(hwnd)
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
            results = []
            #fallback: just use primary monitor
            hwnd = None

        if hwnd:
            rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            point = wintypes.POINT(0, 0)
            ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
            return {
                "left": point.x,
                "top": point.y,
                "width": rect.right,
                "height": rect.bottom,
            }
    except Exception:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Capture Terraria screen frames for BC training")
    parser.add_argument("--session", type=str, default=None,
                        help="Session name (default: auto-generated timestamp)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Capture rate in Hz (default: 15, matches mod RECORD_INTERVAL)")
    parser.add_argument("--size", type=int, default=0,
                        help="Resize frames to NxN (0 = save full resolution)")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg"],
                        help="Image format (png=lossless, jpg=smaller)")
    parser.add_argument("--jpg-quality", type=int, default=85,
                        help="JPEG quality 1-100 (default: 85)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory (default: recordings/ next to this script)")
    args = parser.parse_args()

    #session name
    if args.session is None:
        args.session = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    #output directory
    if args.output_dir is None:
        base_dir = Path(__file__).parent / "recordings"
    else:
        base_dir = Path(args.output_dir)

    session_dir = base_dir / args.session / "frames"
    session_dir.mkdir(parents=True, exist_ok=True)

    #save metadata
    meta = {
        "session": args.session,
        "fps": args.fps,
        "size": args.size if args.size > 0 else "native",
        "format": args.format,
        "started_at": datetime.now().isoformat(),
    }
    meta_path = base_dir / args.session / "capture_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    #find window
    window = find_terraria_window()
    if window:
        print(f"Found Terraria window: {window['width']}x{window['height']} at ({window['left']}, {window['top']})")
    else:
        print("Terraria window not found — capturing primary monitor.")
        print("Tip: make sure Terraria is running in windowed/borderless mode.")

    interval = 1.0 / args.fps
    frame_count = 0
    ext = args.format

    print(f"Session: {args.session}")
    print(f"Output:  {session_dir}")
    print(f"Rate:    {args.fps} Hz ({interval*1000:.0f}ms per frame)")
    print(f"Format:  {args.format}" + (f" (quality={args.jpg_quality})" if args.format == "jpg" else ""))
    if args.size > 0:
        print(f"Resize:  {args.size}x{args.size}")
    print(f"\nCapturing... Press Ctrl+C to stop.\n")

    save_kwargs = {}
    if args.format == "jpg":
        save_kwargs = {"quality": args.jpg_quality}
    elif args.format == "png":
        save_kwargs = {"compress_level": 1}  

    try:
        with mss.mss() as sct:
            monitor = window if window else sct.monitors[1]  #primary monitor

            while True:
                t_start = time.perf_counter()

        
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                #tesize if requested
                if args.size > 0:
                    img = img.resize((args.size, args.size), Image.LANCZOS)

                #Save with frame number and timestamp
                timestamp_ms = int(time.time() * 1000)
                filename = f"{frame_count:08d}_{timestamp_ms}.{ext}"
                img.save(session_dir / filename, **save_kwargs)

                frame_count += 1


                if frame_count % (args.fps * 5) == 0:
                    elapsed = time.perf_counter() - t_start
                    print(f"  {frame_count} frames captured ({frame_count / args.fps:.0f}s of gameplay)")

                #sleep to maintain target fps
                elapsed = time.perf_counter() - t_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    meta["ended_at"] = datetime.now().isoformat()
    meta["total_frames"] = frame_count
    meta["duration_seconds"] = frame_count / args.fps
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {frame_count} frames saved to {session_dir}")
    print(f"Duration: {frame_count / args.fps:.1f}s ({frame_count / args.fps / 60:.1f} min)")

    if frame_count > 0:
        sample_file = next(session_dir.iterdir())
        avg_size = sample_file.stat().st_size
        total_gb = (avg_size * frame_count) / (1024**3)
        print(f"Storage: ~{total_gb:.2f} GB ({avg_size/1024:.0f} KB/frame)")


if __name__ == "__main__":
    main()
