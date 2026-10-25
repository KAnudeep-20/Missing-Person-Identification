"""LFW face-verification evaluation and threshold calibration (spec section 14).

Processes a balanced sample of the standard 6,000 LFW verification pairs:

    same-person pairs   -> positive similarity distribution
    different-person    -> negative similarity distribution

Reports accuracy / FAR / FRR over a threshold sweep, the EER, AUC, and a
recommended operating threshold, plus a histogram figure and a text report
under results/.

Usage (run from the project root):

    python src/evaluate_lfw.py --limit 2000
    python src/evaluate_lfw.py --limit 600 --seed 42
    python src/evaluate_lfw.py --limit 0          # all 6000 pairs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from aligner import Aligner
from arcface import ArcFace
from database import read_image
from detector import SCRFDDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS_CSV = PROJECT_ROOT / "evaluation" / "lfw" / "pairs.csv"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "evaluation" / "lfw" / "train"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LFW verification evaluation + threshold calibration")
    parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--limit", type=int, default=2000,
                        help="number of pairs to evaluate, half same / half different (0 = all 6000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-grid-max", type=float, default=1.01)
    parser.add_argument("--det-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--arcface", type=Path, default=PROJECT_ROOT / "models" / "arcface.onnx")
    return parser.parse_args(argv)


def select_pairs(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(df):
        return df
    half = max(limit // 2, 1)
    rng = np.random.RandomState(seed)
    same = df[df["is_same"] == 1]
    diff = df[df["is_same"] == 0]
    return pd.concat(
        [
            same.sample(n=min(half, len(same)), random_state=rng),
            diff.sample(n=min(half, len(diff)), random_state=rng),
        ],
        ignore_index=True,
    )


def build_embeddings(all_images: list[str], detector, aligner, arcface, images_dir: Path) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    no_face: list[str] = []
    for rel_path in tqdm(all_images, desc="Embedding LFW faces", unit="img"):
        if rel_path in embeddings:
            continue
        image = read_image(images_dir / rel_path)
        if image is None:
            no_face.append(rel_path)
            continue
        faces = detector.detect_faces(image)
        if not faces:
            no_face.append(rel_path)
            continue
        best = max(faces, key=lambda f: f.score)
        aligned = aligner.align_face(image, best.landmarks)
        embeddings[rel_path] = arcface.get_embedding(aligned)
    if no_face:
        print(f"WARNING: {len(no_face)} images had no detectable face and were skipped")
    return embeddings


def similarity_stats(pos: np.ndarray, neg: np.ndarray, grid_max: float = 1.01) -> dict:
    thresholds = np.arange(0.0, grid_max, 0.02)
    far = np.mean(neg[:, None] >= thresholds[None, :], axis=0)
    frr = np.mean(pos[:, None] < thresholds[None, :], axis=0)
    accuracy = (np.mean(pos[:, None] >= thresholds[None, :], axis=0) * 0.5
                + np.mean(neg[:, None] < thresholds[None, :], axis=0) * 0.5)

    # Error-free ROC: EER is where FAR == FRR.
    eer_index = int(np.argmin(np.abs(far - frr)))
    eer = float(abs(far[eer_index] - frr[eer_index]) / 2 * 2) if False else float(
        (far[eer_index] + frr[eer_index]) / 2
    )
    eer_threshold = float(thresholds[eer_index])

    # AUC via Mann-Whitney U: probability that a random positive scores above a random negative.
    from scipy import stats
    u = stats.mannwhitneyu(pos, neg, alternative="greater").statistic
    auc = float(u / (len(pos) * len(neg)))

    return {
        "thresholds": thresholds,
        "far": far,
        "frr": frr,
        "accuracy": accuracy,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": auc,
    }


def best_threshold(stats: dict) -> float:
    return float(stats["thresholds"][int(np.argmax(stats["accuracy"]))])


def plot_histogram(pos: np.ndarray, neg: np.ndarray, chosen: float, out_path: Path) -> None:
    """Overlapping similarity-distribution bars drawn with OpenCV (no matplotlib)."""

    bins = np.linspace(0.0, 1.0, 60)
    pos_hist, _ = np.histogram(np.clip(pos, 0.0, 1.0), bins)
    neg_hist, _ = np.histogram(np.clip(neg, 0.0, 1.0), bins)
    peak = max(pos_hist.max(), neg_hist.max(), 1)

    width, height = 900, 500
    margin_left, margin_bottom, top = 60, 50, 30
    plot_w = width - margin_left - 20
    base_y = height - margin_bottom

    def y_for(count: int) -> int:
        return base_y - int(count / peak * (base_y - top))

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    def half_width():
        return plot_w / 2

    # Left half: different-person (negative) bars; right half: same-person bars.
    for offsets, hist, edge in (
        ((0.0, half_width()), neg_hist, (200, 60, 40)),
        ((half_width(), half_width()), pos_hist, (40, 160, 60)),
    ):
        x_start, half = offsets
        for j in range(len(hist) - 1):
            if hist[j] <= 0:
                continue
            x1 = int(margin_left + x_start + j / (len(hist) - 2) * half)
            x2 = int(margin_left + x_start + (j + 1) / (len(hist) - 2) * half)
            cv2.rectangle(canvas, (x1, y_for(hist[j])), (x2, base_y), edge, thickness=-1)

    # Chosen threshold marker.
    x_line = int(margin_left + chosen * plot_w)
    cv2.line(canvas, (x_line, top), (x_line, base_y), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"threshold {chosen:.3f}", (min(x_line + 6, width - 190), top + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Legend + axes.
    cv2.rectangle(canvas, (margin_left, top - 22), (margin_left + 14, top - 8), (40, 160, 60), -1)
    cv2.putText(canvas, "same person", (margin_left + 20, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (margin_left + 140, top - 22), (margin_left + 154, top - 8), (200, 60, 40), -1)
    cv2.putText(canvas, "different person", (margin_left + 160, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(canvas, (margin_left, height - margin_bottom), (margin_left + plot_w, height - margin_bottom), (0, 0, 0), 1)
    cv2.putText(canvas, "cosine similarity ->", (margin_left + plot_w - 180, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def write_report(path: Path, args, pos, neg, stats, chosen) -> None:
    table_rows = []
    for t, far, frr, acc in zip(stats["thresholds"], stats["far"], stats["frr"], stats["accuracy"]):
        table_rows.append((t, far, frr, acc))

    interesting = [0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.68, 0.7, 0.75, 0.8, 0.85, 0.9, float(chosen)]
    lines = [
        "LFW THRESHOLD CALIBRATION REPORT",
        "=" * 60,
        f"pairs evaluated   : {len(pos)} same / {len(neg)} different",
        "",
        f"same-person       : mean {pos.mean():.4f}  std {pos.std():.4f}  min {pos.min():.4f}  max {pos.max():.4f}",
        f"different-person  : mean {neg.mean():.4f}  std {neg.std():.4f}  min {neg.min():.4f}  max {neg.max():.4f}",
        "",
        f"AUC               : {stats['auc']:.4f}",
        f"EER               : {stats['eer']:.4f} at threshold {stats['eer_threshold']:.3f}",
        f"max accuracy      : {stats['accuracy'].max():.4f} at threshold {chosen:.3f}",
        "",
        "selected thresholds:",
        f"{'t':>6} {'FAR':>8} {'FRR':>8} {'accuracy':>9}",
    ]
    for t in interesting:
        if 0 <= t <= stats["thresholds"][-1] + 1e-9:
            idx = int(np.clip(np.argmin(np.abs(stats["thresholds"] - t)), 0, len(stats["thresholds"]) - 1))
            lines.append(
                f"{stats['thresholds'][idx]:>6.2f} {stats['far'][idx]:>8.4f} {stats['frr'][idx]:>8.4f} {stats['accuracy'][idx]:>9.4f}"
            )
    lines += [
        "",
        "NOTE (PROJECT_SPEC 14/32): scores are potential matches for human",
        "verification; a similarity score alone does not prove identity.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_scores_csv(path: Path, pairs_df, pos_mask, neg_mask, scores):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "is_same", "similarity"])
        for row, score in zip(pairs_df.itertuples(), scores):
            writer.writerow([row.pair_id, row.is_same, f"{score:.6f}"])


def main(argv=None) -> int:
    args = parse_args(argv)

    print("Loading components (detector / aligner / arcface)...")
    detector = SCRFDDetector(det_size=tuple(args.det_size))
    aligner = Aligner()
    arcface = ArcFace(str(args.arcface))

    df = pd.read_csv(args.pairs_csv)
    df = select_pairs(df, args.limit, args.seed)
    print(f"Evaluating {len(df)} LFW pairs ({df['is_same'].sum():.0f} same / {(df['is_same'] == 0).sum():.0f} different)")

    unique_images = sorted(set(df["image_a_path"]) | set(df["image_b_path"]))
    embeddings = build_embeddings(unique_images, detector, aligner, arcface, args.images_dir)

    scores, labels = [], []
    for row in tqdm(df.itertuples(), total=len(df), desc="Scoring pairs", unit="pair"):
        emb_a, emb_b = embeddings.get(row.image_a_path), embeddings.get(row.image_b_path)
        if emb_a is None or emb_b is None:
            scores.append(None)
            labels.append(int(row.is_same))
            continue
        scores.append(float(np.dot(emb_a, emb_b)))
        labels.append(int(row.is_same))

    valid = np.array([s is not None for s in scores])
    labels = np.array(labels)[valid]
    scores = np.array([s for s in scores if s is not None], dtype=np.float64)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) < 10 or len(neg) < 10:
        print("Not enough valid pairs to calibrate; aborting.")
        return 1

    stats = similarity_stats(pos, neg, args.threshold_grid_max)
    chosen = best_threshold(stats)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    report_path = results_dir / "lfw_threshold_report.txt"
    write_report(report_path, args, pos, neg, stats, chosen)
    write_scores_csv(results_dir / "lfw_pair_scores.csv", df[valid], None, None, scores)
    plot_histogram(pos, neg, chosen, results_dir / "images" / "lfw_threshold_calibration.png")

    print()
    print(f"AUC {stats['auc']:.4f} | EER {stats['eer']:.4f} @ {stats['eer_threshold']:.3f}")
    print(f"Max accuracy {stats['accuracy'].max():.4f} -> recommended threshold {chosen:.3f}")
    print(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
