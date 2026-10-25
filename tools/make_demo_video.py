"""One-off helper: build a synthetic CCTV-style test video.

Lena slides across a grey corridor for 60 frames, disappears for 45 frames,
then reappears for 45 frames. 25 fps, so the expected demo events are:
frames ~0-60 person_001 MATCH, gap, frames ~105-150 person_001 MATCH again.
(Also proves event aggregation and the no-face stretches.)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT, FPS = 640, 480, 25
SEGMENTS = [
    (60, True),   # lena visible, panning
    (45, False),  # out of view
    (45, True),   # visible again
]

lena = cv2.imread("input/images/lena_test.jpg")
if lena is None:
    raise SystemExit("run tools/make_demo_assets.py first")

out_path = Path("input/videos/cctv_test.mp4")
out_path.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))

face = cv2.resize(lena, (150, 150))
frame_number = 0
for length, visible in SEGMENTS:
    for _ in range(length):
        frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
        if visible:
            t = frame_number / max(sum(seg[0] for seg in SEGMENTS), 1)
            x = int(frame_number * 3.2) % (WIDTH - 150)
            y = int(120 + 120 * abs(np.sin(t * np.pi * 2)))
            frame[y:y + 150, x:x + 150] = face
        writer.write(frame)
        frame_number += 1

writer.release()
print(f"wrote {out_path} ({frame_number} frames)")
