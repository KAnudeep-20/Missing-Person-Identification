"""CCTV/video identification pipeline.

Flow (spec section 20):

    video -> cv2.VideoCapture -> sample frames (FRAME_SKIP)
    -> detect faces -> align -> ArcFace embedding -> similarity search
    -> annotate frame -> write annotated output video

Consecutive detections of the same person are aggregated into detection
events (spec section 21). Scanning itself is silent (progress bar); for
every matched person the frame with the highest similarity is kept and
saved as a snapshot:

    results/videos/best_matches/<video>_<person>_frame<N>.jpg   (annotated full frame)
    results/videos/best_matches/<video>_<person>_face.jpg       (native-size face crop)
    results/videos/best_matches/<video>_<person>_face_2x.jpg    (2x Lanczos upscaled crop)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from matcher import match_embedding
from image_pipeline import GREEN, annotate_face
from detector import FaceDetection

FRAME_SKIP = 5
AGGREGATION_GAP = 4  # sampled frames a person may be missed before a new event starts
BEST_MATCH_DIR = Path("results") / "videos" / "best_matches"


def iter_sampled_frames(video_path, frame_skip: int = FRAME_SKIP) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_number, frame) for every sampled frame of a video."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % max(frame_skip, 1) == 0:
                yield frame_number, frame
            frame_number += 1
    finally:
        capture.release()


def run_video_pipeline(video_path, detector, aligner, arcface, reference_database, threshold, frame_skip=FRAME_SKIP):
    """Process a video and save an annotated result."""

    video_path = Path(video_path)
    annotated_path = Path("results") / "videos" / f"{video_path.stem}_result{video_path.suffix or '.mp4'}"
    annotated_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    print(f"Video: {video_path.name}")
    print(f"FPS: {fps:.1f}")
    print(f"Frames: {frame_count} (sampling every {max(frame_skip, 1)})")
    print()

    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(frame_skip, 1),
        (width, height),
    )

    open_events: dict[str, dict] = {}      # label -> currently running event
    finished_events: list[dict] = []
    best_matches: dict[str, dict] = {}     # person_id -> best similarity seen so far

    expected_samples = max(frame_count // max(frame_skip, 1), 1) if frame_count else None

    for frame_number, frame in tqdm(
        iter_sampled_frames(video_path, frame_skip),
        total=expected_samples,
        desc="Scanning frames",
        unit="frame",
    ):
        seen_labels: set[str] = set()
        annotated = frame.copy()

        for face in detector.detect_faces(frame):
            result = recognize_face(frame, face, aligner, arcface, reference_database, threshold)
            label = result["person_id"] or "unknown"
            seen_labels.add(label)

            open_event(open_events, finished_events, frame_number, label, result["similarity"])
            annotated = annotate_face(annotated, face, result)

            if result["matched"]:
                _update_best_match(best_matches, result["person_id"], result, face, frame_number, frame)

        # Close events for people no longer visible (after a short grace gap).
        for label in list(open_events):
            if label not in seen_labels:
                event = open_events[label]
                event["missed"] += 1
                if event["missed"] > AGGREGATION_GAP:
                    close_event(open_events, label, frame_number, finished_events)

        writer.write(annotated)

    writer.release()

    for label in list(open_events):
        close_event(open_events, label, frame_number, finished_events)
    _print_events(finished_events)

    _save_best_matches(best_matches, fps, video_path)

    print()
    print(f"Annotated video saved: {annotated_path}")
    return finished_events, annotated_path


def recognize_face(frame, face: FaceDetection, aligner, arcface, reference_database, threshold) -> dict:
    aligned = aligner.align_face(frame, face.landmarks)
    embedding = arcface.get_embedding(aligned)
    return match_embedding(embedding, reference_database, threshold)


def open_event(open_events: dict, finished_events: list, frame_number: int, label: str, similarity: float) -> None:
    """Continue an active event or start a new one for the given label."""

    event = open_events.get(label)
    if event is None:
        # Reopen a recently finished event instead of starting a new one.
        recent = next(
            (
                e
                for e in finished_events
                if e["person_id"] == label and frame_number - e["end_frame"] <= AGGREGATION_GAP
            ),
            None,
        )
        if recent is not None:
            finished_events.remove(recent)
            event = recent
        else:
            event = {
                "person_id": label,
                "start_frame": frame_number,
                "end_frame": frame_number,
                "similarities": [],
                "missed": 0,
            }
        open_events[label] = event

    event["end_frame"] = frame_number
    event["missed"] = 0
    event["similarities"].append(similarity)


def close_event(open_events: dict, label: str, frame_number: int, finished_events: list) -> None:
    event = open_events.pop(label, None)
    if event is None:
        return
    # Keep the last-seen end frame; do not extend to the grace-gap frame.
    finished_events.append(event)


def _print_events(events: list[dict]) -> None:
    if not events:
        return
    print()
    print("Detection events (aggregated):")
    for event in sorted(events, key=lambda e: e["start_frame"]):
        similarities = event["similarities"]
        average = sum(similarities) / len(similarities) if similarities else 0.0
        label = event["person_id"] or "unknown"
        state = "MATCH" if label != "unknown" else "NO MATCH"
        print(
            f"  Frames {event['start_frame']}-{event['end_frame']}"
            f" -> {label}"
            f" -> average similarity = {average:.3f}"
            f" -> {state}"
        )


def _update_best_match(
    best_matches: dict[str, dict],
    person_id: str,
    result: dict,
    face: FaceDetection,
    frame_number: int,
    frame: np.ndarray,
) -> None:
    """Keep the frame with the highest similarity seen so far for a person."""

    record = best_matches.get(person_id)
    if record is None or result["similarity"] > record["similarity"]:
        best_matches[person_id] = {
            "similarity": result["similarity"],
            "frame_number": frame_number,
            "bbox": face.bbox.copy(),
            "frame": frame.copy(),
        }


def _extract_face_crop(frame: np.ndarray, bbox: np.ndarray, margin: float = 1.5) -> np.ndarray:
    """Crop the face region with `margin`x context, clamped to frame bounds."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)
    face_w, face_h = x2 - x1, y2 - y1

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = face_w * margin / 2, face_h * margin / 2

    crop_x1 = max(int(round(cx - half_w)), 0)
    crop_y1 = max(int(round(cy - half_h)), 0)
    crop_x2 = min(int(round(cx + half_w)), width)
    crop_y2 = min(int(round(cy + half_h)), height)

    return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()


def _imwrite(path: Path, image: np.ndarray) -> None:
    """cv2.imwrite with a fallback for non-ASCII output paths on Windows."""

    try:
        cv2.imwrite(str(path), image)
    except cv2.error:
        ok, encoded = cv2.imencode(".jpg", image)
        if ok:
            encoded.tofile(str(path))


def _save_best_matches(best_matches: dict[str, dict], fps: float, video_path: Path) -> None:
    """Save one annotated frame + face crops per matched person and report paths."""

    if not best_matches:
        return

    BEST_MATCH_DIR.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    print()
    print("Best match per person:")
    for person_id in sorted(best_matches):
        record = best_matches[person_id]
        similarity = record["similarity"]
        frame_number = record["frame_number"]
        seconds = frame_number / fps if fps else 0.0

        frame = record["frame"]
        annotated = frame.copy()
        annotated = _annotate_best_match(annotated, record, person_id)

        base = BEST_MATCH_DIR / f"{stem}_{person_id}_frame{frame_number:05d}"
        _imwrite(base.with_suffix(".jpg"), annotated)

        crop = _extract_face_crop(frame, record["bbox"])
        if crop.size:
            _imwrite(BEST_MATCH_DIR / f"{stem}_{person_id}_face.jpg", crop)
            two_x = cv2.resize(
                crop,
                (crop.shape[1] * 2, crop.shape[0] * 2),
                interpolation=cv2.INTER_LANCZOS4,
            )
            _imwrite(BEST_MATCH_DIR / f"{stem}_{person_id}_face_2x.jpg", two_x)

        print(
            f"  {person_id} -> similarity {similarity:.3f}"
            f" @ frame {frame_number} ({seconds:.1f}s)"
        )
        print(f"    annotated frame   -> {base.with_suffix('.jpg')}")
        print(f"    face crop (native) -> {BEST_MATCH_DIR / (f'{stem}_{person_id}_face.jpg')}")
        print(f"    face crop (2x Lanczos) -> {BEST_MATCH_DIR / (f'{stem}_{person_id}_face_2x.jpg')}")


def _annotate_best_match(image: np.ndarray, record: dict, person_id: str) -> np.ndarray:
    """Annotate the saved best-match frame with person, similarity and frame number."""

    x1, y1, x2, y2 = record["bbox"].astype(int)
    similarity = record["similarity"]

    cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 2)
    lines = [
        person_id,
        f"similarity: {similarity:.3f}",
        f"frame: {record['frame_number']}",
    ]
    y = y2 + 22 if y2 + 22 * len(lines) < image.shape[0] else y1 - 6
    for line in lines:
        ty = max(int(y), 18)
        cv2.putText(image, line, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2, cv2.LINE_AA)
        y += 34
    return image
