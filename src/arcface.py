"""ArcFace ONNX embedding extractor.

Loads ``models/arcface.onnx`` (Hugging Face: onnx-community/arcface-onnx),
preprocesses an aligned face, runs ONNX Runtime inference, and returns an
L2-normalized 512-dimensional embedding.

Preprocessing follows the model card (spec section 9):
    BGR -> RGB -> resize 112x112 -> float32 -> (x - 127.5) / 128.0 -> batch dim

Public API (spec section 26):
    get_embedding(aligned_face) -> np.ndarray of shape (512,)
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime


class ArcFace:
    """Face embedding extractor backed by an ArcFace ONNX model."""

    def __init__(self, model_path, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = onnxruntime.InferenceSession(model_path, providers=providers)

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_name = self.session.get_outputs()[0].name

        # Accept either NCHW (1, 3, 112, 112) or NHWC (1, 112, 112, 3) inputs.
        shape = [dim if isinstance(dim, int) else None for dim in model_input.shape]
        if len(shape) == 4 and shape[1] == 3:
            self._nchw = True
            self.input_size = (shape[3], shape[2])
        elif len(shape) == 4:
            self._nchw = False
            self.input_size = (shape[2], shape[1])
        else:
            raise ValueError(f"Unexpected ArcFace input layout: {model_input.shape}")

    def get_embedding(self, aligned_face) -> np.ndarray:
        """Generate a normalized 512-D embedding from an aligned BGR face."""

        image = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = (image - 127.5) / 128.0

        if self._nchw:
            blob = image.transpose(2, 0, 1)[np.newaxis, ...]
        else:
            blob = image[np.newaxis, ...]

        output = self.session.run([self.output_name], {self.input_name: blob})[0]
        embedding = np.asarray(output).flatten()[:512].astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("ArcFace produced a zero-length embedding.")
        return embedding / norm
