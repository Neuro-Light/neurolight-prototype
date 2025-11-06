from __future__ import annotations

from typing import Any, Dict, Optional

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
        self.experiment.processing_history.append(
            {
                "timestamp": self.experiment.modified_date.isoformat(),
                "operation": operation,
                "parameters": params,
            }
        )

    # Placeholders for future expansion
    def detect_objects(self, image: np.ndarray):  # YOLOv8 placeholder
        return []

    def extract_features(self, image: np.ndarray):
        return {}

    def crop_image(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        shape: str = "rectangle",
        polygon_points: Optional[list] = None,
    ) -> np.ndarray:
        """
        Crop an image using ROI coordinates.

        Args:
            image: Input image as numpy array
            x: X coordinate of ROI top-left corner
            y: Y coordinate of ROI top-left corner
            width: Width of ROI
            height: Height of ROI
            shape: "rectangle" or "ellipse"

        Returns:
            Cropped image as numpy array
        """
        if image.ndim < 2:
            raise ValueError("Image must be at least 2D")

        img_height, img_width = image.shape[0], image.shape[1]

        # Clamp ROI to image bounds
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_width, x1 + int(width))
        y2 = min(img_height, y1 + int(height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid ROI dimensions")

        if shape == "ellipse":
            # For ellipse, create a mask and apply it
            # Create coordinate grids
            h, w = y2 - y1, x2 - x1
            center_x, center_y = w / 2.0, h / 2.0
            radius_x, radius_y = w / 2.0, h / 2.0

            # Safety check: avoid division by zero
            if radius_x <= 0 or radius_y <= 0:
                raise ValueError("Invalid ellipse dimensions: width and height must be > 0")

            # Create coordinate grids for mask
            y_coords, x_coords = np.ogrid[:h, :w]
            # Calculate ellipse mask: pixels inside ellipse have value <= 1
            dx = (x_coords - center_x) / radius_x
            dy = (y_coords - center_y) / radius_y
            mask = (dx * dx + dy * dy) <= 1.0

            # Extract rectangular region first (this contains the actual image data)
            if image.ndim == 2:
                cropped = image[y1:y2, x1:x2].copy()
                # Only set pixels OUTSIDE the ellipse to 0 (preserve pixels inside)
                cropped[~mask] = 0
            else:
                cropped = image[y1:y2, x1:x2].copy()
                # Apply mask to each channel
                for c in range(image.shape[2]):
                    cropped[:, :, c][~mask] = 0
            return cropped
        else:
            # Rectangle crop
            if image.ndim == 2:
                return image[y1:y2, x1:x2].copy()
            else:
                return image[y1:y2, x1:x2, :].copy()

    def crop_image_stack(
        self,
        image_stack: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        shape: str = "rectangle",
        polygon_points: Optional[list] = None,
    ) -> np.ndarray:
        """
        Crop an image stack (3D array: frames, height, width) using ROI coordinates.

        Args:
            image_stack: 3D numpy array (frames, height, width)
            x: X coordinate of ROI top-left corner
            y: Y coordinate of ROI top-left corner
            width: Width of ROI
            height: Height of ROI
            shape: "rectangle" or "ellipse"

        Returns:
            Cropped image stack as numpy array
        """
        if image_stack.ndim != 3:
            raise ValueError("Image stack must be 3D (frames, height, width)")

        num_frames = image_stack.shape[0]
        img_height, img_width = image_stack.shape[1], image_stack.shape[2]

        # Clamp ROI to image bounds
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_width, x1 + int(width))
        y2 = min(img_height, y1 + int(height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid ROI dimensions")

        if shape == "ellipse":
            # Create ellipse mask
            h, w = y2 - y1, x2 - x1
            center_x, center_y = w / 2.0, h / 2.0
            radius_x, radius_y = w / 2.0, h / 2.0

            # Safety check: avoid division by zero
            if radius_x <= 0 or radius_y <= 0:
                raise ValueError("Invalid ellipse dimensions: width and height must be > 0")

            # Create coordinate grids for mask
            y_coords, x_coords = np.ogrid[:h, :w]
            # Calculate ellipse mask: pixels inside ellipse have value <= 1
            dx = (x_coords - center_x) / radius_x
            dy = (y_coords - center_y) / radius_y
            mask = (dx * dx + dy * dy) <= 1.0

            # Crop each frame - extract the rectangular region first
            cropped_stack = image_stack[:, y1:y2, x1:x2].copy()
            # Apply mask to each frame: set pixels OUTSIDE ellipse to 0 (preserve pixels inside)
            for t in range(num_frames):
                cropped_stack[t][~mask] = 0
            return cropped_stack
        else:
            # Rectangle crop
            return image_stack[:, y1:y2, x1:x2].copy()
