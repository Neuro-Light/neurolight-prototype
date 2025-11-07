from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from core.experiment_manager import Experiment
from core.roi import ROI, ROIShape


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

    def crop_to_roi(
        self, 
        image: np.ndarray, 
        roi: ROI, 
        apply_mask: bool = True
    ) -> np.ndarray:
        """
        Crop image to ROI region.
        
        Args:
            image: Input image (2D or 3D numpy array)
            roi: ROI object defining the region
            apply_mask: If True and ROI is ellipse, apply ellipse mask
                       (pixels outside ellipse are set to 0)
                       
        Returns:
            Cropped image. For ellipse ROI with apply_mask=True,
            this is the bounding box with mask applied.
        """
        # Clamp ROI to image bounds
        if image.ndim == 2:
            height, width = image.shape
        elif image.ndim == 3:
            height, width = image.shape[0], image.shape[1]
        else:
            raise ValueError("Image must be 2D or 3D array")
        
        x1 = max(0, roi.x)
        y1 = max(0, roi.y)
        x2 = min(width, roi.x + roi.width)
        y2 = min(height, roi.y + roi.height)
        
        # Crop to bounding box
        if image.ndim == 2:
            cropped = image[y1:y2, x1:x2].copy()
        else:
            cropped = image[y1:y2, x1:x2, :].copy()
        
        # Apply ellipse mask if needed
        if apply_mask and roi.shape == ROIShape.ELLIPSE:
            # Create mask for the cropped region
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            
            # Calculate ellipse center and radii
            # The ellipse center in original image coordinates
            ellipse_center_x = roi.x + roi.width / 2
            ellipse_center_y = roi.y + roi.height / 2
            
            # Convert to coordinates relative to the cropped region
            # Account for the offset when ROI is clamped to image bounds
            cx = ellipse_center_x - x1
            cy = ellipse_center_y - y1
            
            # Radii remain the same (based on full ROI dimensions)
            rx = roi.width / 2
            ry = roi.height / 2
            
            # Create coordinate grids
            y_coords, x_coords = np.ogrid[:y2-y1, :x2-x1]
            
            # Ellipse equation
            if rx > 0 and ry > 0:
                ellipse_mask = ((x_coords - cx) / rx) ** 2 + ((y_coords - cy) / ry) ** 2 <= 1
                mask[ellipse_mask] = 255
                
                # Apply mask
                if image.ndim == 2:
                    cropped = cv2.bitwise_and(cropped, cropped, mask=mask)
                else:
                    cropped = cv2.bitwise_and(cropped, cropped, mask=mask)
        
        self.log_processing_step("crop", {
            "roi": roi.to_dict(),
            "apply_mask": apply_mask
        })
        return cropped
    
    def crop_stack_to_roi(
        self,
        image_stack: np.ndarray,
        roi: ROI,
        apply_mask: bool = True
    ) -> np.ndarray:
        """
        Crop an entire image stack to ROI region.
        
        Args:
            image_stack: 3D numpy array (frames, height, width)
            roi: ROI object defining the region
            apply_mask: If True and ROI is ellipse, apply ellipse mask
            
        Returns:
            Cropped image stack (frames, cropped_height, cropped_width)
        """
        if image_stack.ndim != 3:
            raise ValueError("Image stack must be 3D array (frames, height, width)")
        
        num_frames = image_stack.shape[0]
        
        # Crop first frame to get dimensions
        first_cropped = self.crop_to_roi(image_stack[0], roi, apply_mask)
        
        # Allocate output array
        cropped_stack = np.zeros(
            (num_frames, first_cropped.shape[0], first_cropped.shape[1]),
            dtype=image_stack.dtype
        )
        cropped_stack[0] = first_cropped
        
        # Crop remaining frames
        for i in range(1, num_frames):
            cropped_stack[i] = self.crop_to_roi(image_stack[i], roi, apply_mask)
        
        return cropped_stack

    # Placeholders for future expansion
    def detect_objects(self, image: np.ndarray):  # YOLOv8 placeholder
        return []

    def extract_features(self, image: np.ndarray):
        return {}

