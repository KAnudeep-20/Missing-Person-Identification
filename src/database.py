"""Local reference "database" built from a plain folder tree.

The database substitute (spec section 11):

    database/
        person_001/ front.jpg left.jpg ...
        person_002/ front.jpg ...
        ...

Each subdirectory name becomes the identity label and every image inside it
becomes one reference embedding (best-of comparison at query time, option 1
in spec section 12).

Public API (spec section 26):
    scroll_directory(database_dir) -> {person_id: [embedding, ...]}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ReferenceDatabase:
    """Builds and stores reference embeddings for every person folder."""

    def __init__(self, detector, aligner, arcface):
        self._detector = detector
        self._aligner = aligner
        self._arcface = arcface
        self.embeddings: dict[str, list[np.ndarray]] = {}
        self.image_counts: dict[str, int] = {}

    def load(self, database_dir) -> dict[str, list[np.ndarray]]:
        """Scan ``database_dir``, generate embeddings, associate with person IDs."""

        database_dir = Path(database_dir)
        if not database_dir.is_dir():
            raise FileNotFoundError(f"Reference database folder not found: {database_dir}")

        person_dirs = sorted(
            path for path in database_dir.iterdir() if path.is_dir()
        )
        if not person_dirs:
            raise FileNotFoundError(
                f"No person folders inside {database_dir}. Expected folders like database/person_001/."
            )

        print("Loading reference database...")
        print()

        for person_dir in person_dirs:
            person_id = person_dir.name
            embeddings: list[np.ndarray] = []
            images = sorted(
                path for path in person_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for image_path in images:
                image = read_image(image_path)
                if image is None:
                    print(f"  ! could not read {image_path.name}, skipping")
                    continue

                faces = self._detector.detect_faces(image)
                if not faces:
                    print(f"  ! no face found in {image_path.name}, skipping")
                    continue

                # Use the most confident face in the reference photograph.
                best_face = max(faces, key=lambda f: f.score)
                aligned = self._aligner.align_face(image, best_face.landmarks)
                embeddings.append(self._arcface.get_embedding(aligned))

            self.image_counts[person_id] = len(images)
            self.embeddings[person_id] = embeddings
            print(f"{person_id}:")
            print(f"  {len(images)} reference images")
            print(f"  {len(embeddings)} embeddings")
            print()

        total = sum(len(v) for v in self.embeddings.values())
        empty = [pid for pid, v in self.embeddings.items() if not v]
        if empty:
            print(f"WARNING: no usable reference faces for: {', '.join(empty)} (ignored during matching)")
        print(f"Reference database ready ({len([p for p in self.embeddings if self.embeddings[p]])} persons, {total} embeddings).")
        print()
        return self.embeddings


def read_image(image_path) -> np.ndarray | None:
    """Read an image as BGR, tolerating non-ASCII Windows paths."""
    import cv2

    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
