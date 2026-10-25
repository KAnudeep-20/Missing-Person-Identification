"""Single-image identification pipeline.

Flow (spec section 19):

    input image -> detect all faces -> for each face:
        landmarks -> align -> ArcFace embedding -> similarity search
        -> threshold decision -> draw result
    -> save annotated output

Console output mirrors spec section 24.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from aligner import align_face
from arcface import ArcFace
from database import read_image
from detector import FaceDetection
from matcher import match_embedding

GREEN = (60, 200, 60)
RED = (60, 60, 220)


def run_image_pipeline(image_path, detector, aligner, arcface, reference_database, threshold):
    """Process one image and save an annotated result."""

    image = read_image(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")

    image_path = Path(image_path)
    faces = detector.detect_faces(image)

    print(f"Input image: {image_path.name}")
    print()
    print(f"Faces detected: {len(faces)}")
    print()

    annotated = image.copy()
    results = []
    for index, face in enumerate(faces, start=1):
        aligned = aligner.align_face(image, face.landmarks)
        embedding = arcface.get_embedding(aligned)
        result = match_embedding(embedding, reference_database, threshold)

        _, report = _print_face_report(index, result)
        annotated = annotate_face(annotated, face, result)
        results.append(report)

    output_dir = Path("results") / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_result{image_path.suffix or '.jpg'}"

    try:
        cv2.imwrite(str(output_path), annotated)
    except cv2.error:  # non-ASCII output paths on Windows
        ok, encoded = cv2.imencode(str(image_path.suffix or Path(image_path.suffix).suffix or ".jpg"), annotated)
        if ok:
            encoded.tofile(str(output_path))

    print(f"Output saved: {output_path}")
    return results, output_path


def _print_face_report(index: int, result: dict) -> tuple[bool, dict]:
    matched = result["matched"]
    state = "MATCH FOUND" if matched else "NO MATCH"
    label = result["person_id"] or "unknown"
    print(f"Face #{index}")
    print(f"  Best candidate: {label}")
    print(f"  Similarity: {result['similarity']:.3f}")
    print(f"  Result: {state}")
    print()
    return matched, {"face_index": index, **result}


def annotate_face(image: np.ndarray, face: FaceDetection, result: dict) -> np.ndarray:
    """Draw bounding box, person ID, similarity and match state on the image."""

    x1, y1, x2, y2 = face.bbox.astype(int)
    person = result["person_id"] or "unknown"
    color = GREEN if result["matched"] else RED
    similarity = result["similarity"]

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    lines = [person, f"similarity: {similarity:.3f}", "MATCH" if result["matched"] else "UNKNOWN"]
    y = y2 + 22 if y2 + 22 * len(lines) < image.shape[0] else y1 - 6
    for line in lines:
        ty = max(int(y), 18)
        cv2.putText(image, line, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22
    return image
