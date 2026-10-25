"""SCRFD face detector with five-point facial landmarks.

Uses the InsightFace package (model zoo SCRFD weights) for detection only.
Face *recognition* is deliberately kept out of this module: embeddings come
from the project's own ArcFace ONNX module (see arcface.py).

Public API (spec section 26):
    detect_faces(image) -> list[FaceDetection]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime


@dataclass
class FaceDetection:
    """One detected face in a source image."""

    bbox: np.ndarray       # (4,) float: x1, y1, x2, y2
    score: float           # detector confidence
    landmarks: np.ndarray  # (5, 2) float: left eye, right eye, nose,
                           #             left mouth corner, right mouth corner


class SCRFDDetector:
    """Thin wrapper around insightface's FaceAnalysis running only detection.

    The buffalo_l pack contains an SCRFD-10G detector plus several
    recognition/marker models. With ``allowed_modules=["detection"]`` only the
    detection ONNX is loaded from the pack, so recognition stays explicitly
    with the project's own ArcFace module.
    """

    def __init__(
        self,
        det_size=(640, 640),
        det_thresh=0.5,
        pack_name="buffalo_l",
        providers=None,
        root=None,
    ):
        from insightface.app import FaceAnalysis  # imported lazily

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self._app = FaceAnalysis(
            name=pack_name,
            allowed_modules=["detection"],
            providers=providers,
            **({"root": root} if root is not None else {}),
        )
        self._app.prepare(
            ctx_id=-1,  # CPU
            det_size=tuple(det_size),
            det_thresh=det_thresh,
        )

    def detect_faces(self, image) -> list[FaceDetection]:
        """Detect faces in a BGR image (as loaded by OpenCV)."""
        detections: list[FaceDetection] = []
        for face in self._app.get(image):
            bbox = np.asarray(face.bbox, dtype=np.float32).flatten()
            if bbox.size != 4:
                continue
            detections.append(
                FaceDetection(
                    bbox=bbox,
                    score=float(face.det_score),
                    landmarks=np.asarray(face.kps, dtype=np.float32),
                )
            )
        return detections
