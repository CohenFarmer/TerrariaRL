"""
Randomly teleports around the terraria world, then captures a screenshot
the goal is to collect many screenshots, to train the agents vision, the vision training
"""

import socket
import os
import json
import numpy as np
from mss import mss
from PIL import Image


def main():
    host = "127.0.0.1"
    port = 7555
    output_dir = "data/frames"
    target_size = (256, 256)
    max_frames = 5000

    os.makedirs(output_dir, exist_ok=True)

    # Connect to mod
    print(f"Connecting to TerrariaRL mod at {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock_file = sock.makefile("r")

    # Tell the mod we want data collection mode
    sock.sendall("collect\n".encode("utf-8"))
    print("Connected in COLLECT mode!\n")

    # Screenshot tool
    sct = mss()

    frame_count = 0
    metadata = []

    try:
        while frame_count < max_frames:
            #Wait for mod to signal CAPTURE
            line = sock_file.readline()
            if not line:
                print("Mod disconnected.")
                break

            line = line.strip()
            if not line.startswith("CAPTURE"):
                continue

            # Parse: CAPTURE,tileX,tileY,depth
            parts = line.split(",")
            tile_x = int(parts[1])
            tile_y = int(parts[2])
            depth = float(parts[3])

            # Capture screenshot of primary monitor
            monitor = sct.monitors[1]  # primary monitor
            screenshot = sct.grab(monitor)

            # Convert to PIL Image, resize, save
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            img = img.resize(target_size, Image.LANCZOS)
            
            filename = f"frame_{frame_count:05d}.png"
            img.save(os.path.join(output_dir, filename))

            # Store metadata
            metadata.append({
                "frame": filename,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "depth": depth
            })

            frame_count += 1

            # Tell mod we captured successfully
            sock.sendall("OK\n".encode("utf-8"))

            # Progress
            if frame_count % 50 == 0:
                print(f"Captured {frame_count}/{max_frames} frames | "
                      f"Last pos: ({tile_x}, {tile_y}) depth: {depth:.3f}")

        # Save metadata
        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nDone! {frame_count} frames saved to {output_dir}/")
        print(f"Metadata saved to {meta_path}")

    except KeyboardInterrupt:
        print(f"\nStopped. {frame_count} frames captured.")
        # Save whatever metadata we have
        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
