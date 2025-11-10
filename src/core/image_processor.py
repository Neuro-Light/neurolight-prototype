from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

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

    def detect_neurons(
        self, 
        image: np.ndarray, 
        threshold_percentile: float = 95.0,
        min_area: int = 2,
        max_area: int = 100
    ) -> List[Tuple[int, int]]:
        """
        Detect neuron positions (bright spots) in an image.
        
        Args:
            image: Input grayscale image
            threshold_percentile: Percentile for thresholding (default 95.0)
            min_area: Minimum neuron area in pixels
            max_area: Maximum neuron area in pixels
            
        Returns:
            List of (x, y) coordinates of detected neurons
        """
        # Normalize image to 0-255 range if needed
        if image.dtype != np.uint8:
            img_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            img_normalized = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img_normalized, (5, 5), 0)
        
        # Threshold based on percentile
        threshold_value = np.percentile(blurred, threshold_percentile)
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area and get centroids
        neurons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    neurons.append((cx, cy))
        
        return neurons

    def load_image_for_alignment(self, img_path: str) -> np.ndarray:
        """
        Load a single image file for alignment, ensuring it's 2D grayscale.
        
        Args:
            img_path: Path to image file
            
        Returns:
            2D numpy array (height, width)
        """
        import tifffile
        from PIL import Image
        
        path = Path(img_path)
        suffix = path.suffix.lower()
        
        # Use tifffile for TIFF files (supports 16-bit, multi-page, etc.)
        if suffix in ['.tif', '.tiff']:
            try:
                img = tifffile.imread(str(path))
            except Exception as e:
                raise ValueError(f"Failed to load TIFF file {path}: {e}")
        else:
            # Use PIL/Pillow for other formats
            try:
                pil_img = Image.open(str(path))
                # Convert to grayscale if needed
                if pil_img.mode != 'L':
                    pil_img = pil_img.convert('L')
                img = np.array(pil_img)
            except Exception as e:
                raise ValueError(f"Failed to load image file {path}: {e}")
        
        # Ensure 2D array
        if img.ndim == 2:
            pass  # Already 2D
        elif img.ndim == 3:
            # Need to distinguish between:
            # - Multi-page TIFF: (frames, height, width) - frames on first axis
            # - Multi-channel image: (height, width, channels) - channels on last axis
            
            # Heuristic to detect multi-page TIFF:
            # - First dimension is small (reasonable number of frames, <= 10000)
            # - Last two dimensions are large (both > 10, typical image dimensions)
            # - Last dimension (width) is typically larger than first dimension (frames)
            # OR first dimension is 1 (single-page TIFF) and last two dimensions look like image H/W
            is_likely_frames = (
                img.shape[0] <= 10000 and  # Reasonable number of frames
                img.shape[1] > 10 and      # Height looks like image dimension
                img.shape[2] > 10 and      # Width looks like image dimension
                img.shape[2] > img.shape[0]  # Width typically > number of frames
            ) or (
                img.shape[0] == 1 and      # Single-page TIFF
                img.shape[1] > 4 and       # Height looks like image dimension
                img.shape[2] > 4           # Width looks like image dimension
            )
            # Heuristic to detect multi-channel image:
            # - Last dimension is small (<= 4 for RGB/RGBA)
            # - But only if first dimension is NOT 1 (to avoid misclassifying single-page TIFFs)
            is_likely_channels = img.shape[2] <= 4 and img.shape[0] != 1
            
            if is_likely_frames and not is_likely_channels:
                # Multi-page TIFF: (frames, height, width)
                if img.shape[0] > 1:
                    raise ValueError(
                        f"Multi-page TIFF detected with {img.shape[0]} pages. "
                        f"Please select a single page or use a different image. "
                        f"Shape: {img.shape} (frames, height, width)"
                    )
                # Single frame: extract it
                img = img[0]
            else:
                # Multi-channel image: (height, width, channels)
                if img.shape[2] == 3:
                    # RGB - use luminance formula
                    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(img.dtype)
                elif img.shape[2] == 4:
                    # RGBA - use RGB channels with luminance formula
                    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(img.dtype)
                else:
                    # Multi-channel - take first channel
                    img = img[:, :, 0]
        elif img.ndim == 1:
            raise ValueError(f"1D array not supported for image: {path}")
        elif img.ndim > 3:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}D (expected 2D or 3D)")
        
        if img.ndim != 2:
            raise ValueError(f"Failed to convert image to 2D array. Final shape: {img.shape}")
        
        return img

    def align_image_stack(
        self,
        image_stack: np.ndarray,
        transform_type: str = 'RIGID_BODY',
        reference: str = 'first',
        progress_callback: Optional[Callable[[int, int, str], bool]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Align images using PyStackReg.
        
        Args:
            image_stack: 3D numpy array (frames, height, width)
            transform_type: Transformation type ('TRANSLATION', 'RIGID_BODY', 'SCALED_ROTATION', 'AFFINE', 'BILINEAR')
            reference: Reference strategy ('first', 'previous', 'mean')
            progress_callback: Optional callback function(completed, total, status_message) -> bool
            
        Returns:
            Tuple of (aligned_stack, transformation_matrices, confidence_scores)
        """
        from pystackreg import StackReg
        from pystackreg.util import to_uint16
        
        if image_stack.ndim != 3:
            raise ValueError("Image stack must be 3D array (frames, height, width)")
        
        num_frames = image_stack.shape[0]
        
        # Map string to StackReg constant
        transform_map = {
            'translation': StackReg.TRANSLATION,
            'rigid_body': StackReg.RIGID_BODY,
            'scaled_rotation': StackReg.SCALED_ROTATION,
            'affine': StackReg.AFFINE,
            'bilinear': StackReg.BILINEAR,
        }
        
        transform_const = transform_map.get(transform_type.lower(), StackReg.RIGID_BODY)
        
        if progress_callback:
            progress_callback(0, num_frames, "Initializing StackReg...")
        
        # Initialize StackReg
        sr = StackReg(transform_const)
        
        # Store original data range for each frame to preserve brightness/contrast
        original_mins = []
        original_maxs = []
        for i in range(num_frames):
            original_mins.append(float(np.min(image_stack[i])))
            original_maxs.append(float(np.max(image_stack[i])))
        
        # Normalize all frames to a consistent range (0-65535) for alignment
        # This ensures consistent brightness across frames during alignment
        image_stack_normalized = np.zeros_like(image_stack, dtype=np.float32)
        global_min = float(np.min(image_stack))
        global_max = float(np.max(image_stack))
        global_range = global_max - global_min
        
        if global_range > 0:
            for i in range(num_frames):
                image_stack_normalized[i] = ((image_stack[i].astype(np.float32) - global_min) / global_range) * 65535.0
        else:
            image_stack_normalized = image_stack.astype(np.float32)
        
        # Convert to uint16 for StackReg (it works best with uint16)
        image_stack_uint16 = image_stack_normalized.astype(np.uint16)
        
        # Register to get transformation matrices
        if progress_callback:
            progress_callback(0, num_frames, "Computing transformation matrices...")
        
        tmats = sr.register_stack(image_stack_uint16, reference=reference)
        
        # Apply transformations
        if progress_callback:
            progress_callback(0, num_frames, "Applying transformations...")
        
        aligned_stack_uint16 = sr.transform_stack(image_stack_uint16, tmats=tmats)
        
        # Convert back to original data range to preserve brightness/contrast
        # Scale from normalized uint16 back to original range
        aligned_stack = np.zeros_like(image_stack, dtype=image_stack.dtype)
        
        if global_range > 0:
            for i in range(num_frames):
                # Convert from uint16 (0-65535) back to original range
                aligned_float = aligned_stack_uint16[i].astype(np.float32) / 65535.0
                aligned_float = aligned_float * global_range + global_min
                
                # Convert to original dtype with proper clipping
                if image_stack.dtype == np.uint8:
                    aligned_stack[i] = np.clip(aligned_float, 0, 255).astype(np.uint8)
                elif image_stack.dtype == np.uint16:
                    aligned_stack[i] = np.clip(aligned_float, 0, 65535).astype(np.uint16)
                else:
                    aligned_stack[i] = aligned_float.astype(image_stack.dtype)
        else:
            # If all pixels are the same, just copy the original
            aligned_stack = image_stack.copy()
        
        # Calculate confidence scores using normalized cross-correlation
        confidence_scores = []
        mean_reference_frame = None
        if reference == 'mean':
            mean_reference_frame = np.mean(aligned_stack.astype(np.float32), axis=0)
        
        for i in range(num_frames):
            if reference == 'first':
                if i == 0:
                    confidence_scores.append(1.0)
                    continue
                reference_frame = aligned_stack[0]
            elif reference == 'previous':
                if i == 0:
                    confidence_scores.append(1.0)
                    continue
                reference_frame = aligned_stack[i - 1]
            elif reference == 'mean':
                reference_frame = mean_reference_frame
            else:
                reference_frame = aligned_stack[0]
            
            if progress_callback:
                progress_callback(i, num_frames, f"Calculating confidence for frame {i+1}/{num_frames}...")
            
            # Convert to float32 for calculations
            ref_float = reference_frame.astype(np.float32)
            aligned_float = aligned_stack[i].astype(np.float32)
            
            # Normalized Cross-Correlation (NCC)
            ref_norm = (ref_float - ref_float.mean()) / (ref_float.std() + 1e-10)
            aligned_norm = (aligned_float - aligned_float.mean()) / (aligned_float.std() + 1e-10)
            ncc = np.mean(ref_norm * aligned_norm)
            
            # Use NCC as confidence (clamp to [0, 1])
            confidence = max(0.0, min(1.0, ncc))
            confidence_scores.append(confidence)
        
        if progress_callback:
            progress_callback(num_frames, num_frames, "Alignment complete!")
        
        return aligned_stack, tmats, confidence_scores

