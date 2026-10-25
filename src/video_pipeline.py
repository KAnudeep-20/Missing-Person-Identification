"""CCTV/video identification pipeline.

Flow (spec section 20):

    video -> cv2.VideoCapture -> sample frames (FRAME_SKIP)
    -> detect faces -> align -> ArcFace embedding -> similarity search
    -> annotate frame -> write annotated output video

Consecutive detections of the same person are aggregated into detection
events so the console does not repeat MATCH lines for every sampled frame
(spec section 21).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from aligner import align_face
from matcher import match_embedding
from image_pipeline import GREEN, RED, annotate_face
from detector import FaceDetection

FRAME_SKIP = 5
AGGREGATION_GAP = 4  # sampled frames a person may be missed before a new event starts


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

    for frame_number, frame in iter_sampled_frames(video_path, frame_skip):
        seen_labels: set[str] = set()
        annotated = frame.copy()

        for face in detector.detect_faces(frame):
            result = recognize_face(frame, face, aligner, arcface, reference_database, threshold)
            label = result["person_id"] or "unknown"
            seen_labels.add(label)

            state = "MATCH" if result["matched"] else "NO MATCH"
            print(f"Frame {frame_number}:")
            print(f"  Face detected (best candidate: {label}, similarity: {result['similarity']:.3f}, {state})")
            open_event(open_events, finished_events, frame_number, label, result["similarity"])
            annotated = annotate_face(annotated, face, result)

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
