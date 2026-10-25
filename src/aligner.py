"""Five-point face alignment to the ArcFace 112x112 canonical geometry.

The detector supplies five facial landmarks (left eye, right eye, nose, left
mouth corner, right mouth corner). A least-squares similarity transform maps
them onto the canonical ArcFace template, and the source image is warped so
that eyes end up approximately horizontal with the face centered and sized
112x112.

Public API (spec section 26):
    align_face(image, landmarks) -> aligned_face
"""

from __future__ import annotations

import cv2
import numpy as np

# Canonical ArcFace 112x112 landmark template (InsightFace convention).
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7297, 92.2044],  # right mouth corner
    ],
    dtype=np.float32,
)


class Aligner:
    """Callable wrapper so pipelines can share one aligner instance."""

    def align_face(self, image, landmarks, size=(112, 112)):
        return align_face(image, landmarks, size)


def align_face(image, landmarks, size=(112, 112)):
    """Warp ``image`` (BGR) so the face defined by ``landmarks`` matches the
    ArcFace canonical geometry. Returns a 112x112 (BGR) aligned face."""

    landmarks = np.asarray(landmarks, dtype=np.float32)
    template = ARCFACE_TEMPLATE.astype(np.float32)

    if tuple(size) != (112, 112):
        scale = np.array([size[0] / 112.0, size[1] / 112.0], dtype=np.float32)
        template = template * scale

    # estimateAffinePartial2D fits a similarity transform
    # (rotation + uniform scale + translation) which is exactly what the
    # ArcFace alignment needs.
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks.reshape(-1, 1, 2),
        template.reshape(-1, 1, 2),
        method=cv2.LMEDS,
    )
    if matrix is None:
        raise ValueError("Could not estimate a similarity transform for the face landmarks.")

    aligned = cv2.warpAffine(
        image,
        matrix,
        dsize=tuple(size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned
