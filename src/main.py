"""Command-line entry point for the Missing Person Identification System.

Usage examples (spec sections 22 and 36):

    python src/main.py --mode image --input input/images/test_01.jpg
    python src/main.py --mode video --input input/videos/cctv_test.mp4
    python src/main.py --mode image --input input/images/test_01.jpg --threshold 0.60

Optionally, a custom arcface model path can be passed with --arcface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

from arcface import ArcFace
from aligner import Aligner
from database import ReferenceDatabase
from detector import SCRFDDetector
from image_pipeline import run_image_pipeline
from video_pipeline import run_video_pipeline

# Root of the project (src/main.py -> project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "arcface.onnx"
DEFAULT_DATABASE_DIR = PROJECT_ROOT / "database"
DEFAULT_PAIRS_CSV = PROJECT_ROOT / "evaluation" / "lfw" / "pairs.csv"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "evaluation" / "lfw" / "train"
DEFAULT_THRESHOLD = 0.30  # LFW-calibrated: max accuracy 0.9653 @ 0.28 (see results/lfw_threshold_report.txt)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Missing Person Identification System")
    parser.add_argument("--mode", choices=["image", "video"], required=True,
                        help="Process a single image or a CCTV video")
    parser.add_argument("--input", required=True,
                        help="Path to the input image or video")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine-similarity threshold (default: {DEFAULT_THRESHOLD}, LFW-calibrated)")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_DIR,
                        help="Reference database folder (default: database/)")
    parser.add_argument("--arcface", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Path to the ArcFace ONNX model")
    parser.add_argument("--frame-skip", type=int, default=5,
                        help="Process every Nth video frame (default: 5)")
    parser.add_argument("--det-size", type=int, nargs=2, default=[640, 640],
                        help="Face detection input size, width height (default: 640 640)")
    parser.add_argument("--det-thresh", type=float, default=0.5,
                        help="Face detection confidence threshold (default: 0.5)")
    parser.add_argument("--align-debug-dir", type=Path, default=None,
                        help="Optional folder to save aligned faces for visual inspection")
    return parser.parse_args(argv)


def print_startup_banner() -> None:
    print("========================================")
    print(" MISSING PERSON IDENTIFICATION SYSTEM")
    print("========================================")
    print()


def verify_environment() -> None:
    print("Loading face detector...")
    detector = SCRFDDetector(
        det_size=(640, 640),
        det_thresh=0.5,
    )
    print("Detector loaded.")
    print()
    return detector


def build_components(args: argparse.Namespace):
    print_startup_banner()

    print("Python/CV environment:")
    print(f"  OpenCV: {cv2.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  ONNX Runtime: {onnxruntime.__version__}")
    print()

    detector = verify_environment()

    print("Loading ArcFace model...")
    arcface = ArcFace(str(args.arcface))
    print("ArcFace loaded.")
    print()

    aligner = Aligner()
    reference_database = ReferenceDatabase(detector, aligner, arcface)
    database = reference_database.load(args.database)
    return detector, aligner, arcface, database


def main(argv=None) -> int:
    args = parse_args(argv)

    if not Path(args.arcface).is_file():
        print(f"ERROR: ArcFace model not found at {args.arcface}")
        print("Download it with: hf download onnx-community/arcface-onnx arcface.onnx --local-dir models")
        return 1

    detector, aligner, arcface, reference_database = build_components(args)

    if args.mode == "image":
        run_image_pipeline(args.input, detector, aligner, arcface, reference_database, args.threshold)
    else:
        run_video_pipeline(
            args.input, detector, aligner, arcface, reference_database,
            args.threshold, frame_skip=args.frame_skip,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
