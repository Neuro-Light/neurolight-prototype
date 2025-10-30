from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from core.experiment_manager import Experiment


class ImageProcessor:
    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def preprocess_image(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        # Simple placeholder: Gaussian blur
        ksize = int(params.get("ksize", 3))
        out = cv2.GaussianBlur(image, (ksize, ksize), 0)
        self.log_processing_step("preprocess", {"ksize": ksize})
        return out

    def apply_opencv_filter(self, image: np.ndarray, filter_type: str) -> np.ndarray:
        if filter_type == "edges":
            out = cv2.Canny(image, 100, 200)
        else:
            out = image
        self.log_processing_step("filter", {"type": filter_type})
        return out

    def log_processing_step(self, operation: str, params: Dict[str, Any]) -> None:
        self.experiment.processing_history.append({
            "timestamp": self.experiment.modified_date.isoformat(),
            "operation": operation,
            "parameters": params,
        })

    # Placeholders for future expansion
    def detect_objects(self, image: np.ndarray):  # YOLOv8 placeholder
        return []

    def extract_features(self, image: np.ndarray):
        return {}

