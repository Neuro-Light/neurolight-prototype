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
        self.experiment.processing_history.append(
            {
                "timestamp": self.experiment.modified_date.isoformat(),
                "operation": operation,
                "parameters": params,
            }
        )

    def crop_to_roi(self, image: np.ndarray, roi: ROI, apply_mask: bool = True) -> np.ndarray:
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
            y_coords, x_coords = np.ogrid[: y2 - y1, : x2 - x1]

            # Ellipse equation
            if rx > 0 and ry > 0:
                ellipse_mask = ((x_coords - cx) / rx) ** 2 + ((y_coords - cy) / ry) ** 2 <= 1
                mask[ellipse_mask] = 255

                # Apply mask
                if image.ndim == 2:
                    cropped = cv2.bitwise_and(cropped, cropped, mask=mask)
                else:
                    cropped = cv2.bitwise_and(cropped, cropped, mask=mask)

        self.log_processing_step("crop", {"roi": roi.to_dict(), "apply_mask": apply_mask})
        return cropped

    def crop_stack_to_roi(
        self, image_stack: np.ndarray, roi: ROI, apply_mask: bool = True
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
            dtype=image_stack.dtype,
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
        max_area: int = 100,
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
        if suffix in [".tif", ".tiff"]:
            try:
                img = tifffile.imread(str(path))
            except Exception as e:
                raise ValueError(f"Failed to load TIFF file {path}: {e}") from e
        else:
            # Use PIL/Pillow for other formats
            try:
                pil_img = Image.open(str(path))
                # Convert to grayscale if needed
                if pil_img.mode != "L":
                    pil_img = pil_img.convert("L")
                img = np.array(pil_img)
            except Exception as e:
                raise ValueError(f"Failed to load image file {path}: {e}") from e

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
                img.shape[0] <= 10000  # Reasonable number of frames
                and img.shape[1] > 10  # Height looks like image dimension
                and img.shape[2] > 10  # Width looks like image dimension
                and img.shape[2] > img.shape[0]  # Width typically > number of frames
            ) or (
                img.shape[0] == 1  # Single-page TIFF
                and img.shape[1] > 4  # Height looks like image dimension
                and img.shape[2] > 4  # Width looks like image dimension
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
        transform_type: str = "RIGID_BODY",
        reference: str = "first",
        progress_callback: Optional[Callable[[int, int, str], bool]] = None,
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

        if image_stack.ndim != 3:
            raise ValueError("Image stack must be 3D array (frames, height, width)")

        num_frames = image_stack.shape[0]

        # Map string to StackReg constant
        transform_map = {
            "translation": StackReg.TRANSLATION,
            "rigid_body": StackReg.RIGID_BODY,
            "scaled_rotation": StackReg.SCALED_ROTATION,
            "affine": StackReg.AFFINE,
            "bilinear": StackReg.BILINEAR,
        }

        transform_const = transform_map.get(transform_type.lower(), StackReg.RIGID_BODY)

        if progress_callback:
            progress_callback(0, num_frames, "Initializing StackReg...")

        # Initialize StackReg
        sr = StackReg(transform_const)

        # Normalize all frames to a consistent range (0-65535) for alignment
        # This ensures consistent brightness across frames during alignment
        image_stack_normalized = np.zeros_like(image_stack, dtype=np.float32)
        global_min = float(np.min(image_stack))
        global_max = float(np.max(image_stack))
        global_range = global_max - global_min

        if global_range > 0:
            for i in range(num_frames):
                image_stack_normalized[i] = (
                    (image_stack[i].astype(np.float32) - global_min) / global_range
                ) * 65535.0
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
        if reference == "mean":
            mean_reference_frame = np.mean(aligned_stack.astype(np.float32), axis=0)

        for i in range(num_frames):
            if reference == "first":
                if i == 0:
                    confidence_scores.append(1.0)
                    continue
                reference_frame = aligned_stack[0]
            elif reference == "previous":
                if i == 0:
                    confidence_scores.append(1.0)
                    continue
                reference_frame = aligned_stack[i - 1]
            elif reference == "mean":
                reference_frame = mean_reference_frame
            else:
                reference_frame = aligned_stack[0]

            if progress_callback:
                progress_callback(
                    i,
                    num_frames,
                    f"Calculating confidence for frame {i + 1}/{num_frames}...",
                )

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

    def detect_neurons_in_roi(
        self,
        frame_data: np.ndarray,
        roi_mask: np.ndarray,
        cell_size: int = 6,
        num_peaks: int = 400,
        correlation_threshold: float = 0.4,
        threshold_rel: float = 0.1,
        apply_detrending: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect neurons within a specified ROI using local maxima detection.

        This implements a complete neuron detection pipeline:
        1. Extract and preprocess ROI region from frames
        2. Detect local maxima (neuron centers)
        3. Extract intensity trajectories for each neuron
        4. Filter neurons based on correlation quality
        5. Optionally apply detrending to remove slow drift

        Parameters:
        -----------
        frame_data : np.ndarray
            3D array (frames, height, width) of image stack
        roi_mask : np.ndarray
            2D boolean array (height, width) where True = inside ROI
        cell_size : int
            Neuron diameter in pixels (default: 6)
        num_peaks : int
            Maximum number of neurons to detect (default: 400)
        correlation_threshold : float
            Threshold for filtering neurons by correlation (default: 0.4)
        threshold_rel : float
            Relative threshold for peak detection (0.0-1.0, default: 0.1)
        apply_detrending : bool
            Whether to apply Savitzky-Golay detrending (default: True)

        Returns:
        --------
        neuron_locations : np.ndarray
            Array of (y, x) coordinates for detected neurons (in image coordinates)
        neuron_trajectories : np.ndarray
            2D array of intensity time-series (neurons x frames)
        quality_mask : np.ndarray
            Boolean array indicating good (True) vs bad (False) neurons
        """
        from skimage.feature import peak_local_max
        from scipy.signal import savgol_filter

        if frame_data.ndim != 3:
            raise ValueError("frame_data must be a 3D array (frames, height, width)")

        if roi_mask.ndim != 2:
            raise ValueError("roi_mask must be a 2D array (height, width)")

        num_frames, height, width = frame_data.shape

        if roi_mask.shape != (height, width):
            raise ValueError(
                f"roi_mask shape {roi_mask.shape} must match image dimensions ({height}, {width})"
            )

        # ============================================================
        # Step 1: Image Preprocessing
        # ============================================================
        # Extract ROI region from all frames
        roi_region_stack = np.zeros((num_frames, height, width), dtype=frame_data.dtype)
        for t in range(num_frames):
            roi_region_stack[t] = frame_data[t] * roi_mask.astype(frame_data.dtype)

        # Rescale pixel values to 0.0-1.0 range
        frame_min = np.min(roi_region_stack)
        frame_max = np.max(roi_region_stack)
        if frame_max > frame_min:
            roi_region_stack = (roi_region_stack - frame_min) / (frame_max - frame_min)
        else:
            # All pixels are the same value
            roi_region_stack = np.zeros_like(roi_region_stack)

        # Calculate mean frame across all time points for the ROI region
        mean_frame = np.mean(roi_region_stack, axis=0)

        # ============================================================
        # Step 2: Neuron Detection (Local Maxima)
        # ============================================================
        # Use peak_local_max to find local maxima
        # Note: peak_local_max returns (row, col) = (y, x) coordinates
        peaks = peak_local_max(
            mean_frame,
            min_distance=cell_size,
            num_peaks=num_peaks,
            threshold_rel=threshold_rel,
            exclude_border=cell_size // 2,  # Exclude border to avoid edge artifacts
        )

        if len(peaks) == 0:
            # No neurons detected
            return (
                np.array([], dtype=np.int32).reshape(0, 2),  # Empty array with shape (0, 2)
                np.array([]).reshape(0, num_frames),  # Empty trajectories
                np.array([], dtype=bool),  # Empty quality mask
            )

        # Convert peaks to (y, x) format
        # peak_local_max returns coordinates as (row, col) = (y, x)
        neuron_locations = peaks  # Shape: (num_neurons, 2) with columns [y, x]

        # Filter to only include neurons within ROI mask
        valid_neurons = []
        for i, (y, x) in enumerate(neuron_locations):
            if 0 <= y < height and 0 <= x < width and roi_mask[y, x]:
                valid_neurons.append(i)

        if len(valid_neurons) == 0:
            # No valid neurons in ROI
            return (
                np.array([], dtype=np.int32).reshape(0, 2),
                np.array([]).reshape(0, num_frames),
                np.array([], dtype=bool),
            )

        neuron_locations = neuron_locations[valid_neurons]
        num_neurons = len(neuron_locations)

        # ============================================================
        # Step 3: Trajectory Extraction
        # ============================================================
        # For each detected neuron, extract mean intensity over time
        # within a circular region around the center
        radius = cell_size / 2
        neuron_trajectories = np.zeros((num_neurons, num_frames), dtype=np.float32)

        # Create coordinate grids for circular mask
        y_coords, x_coords = np.ogrid[:height, :width]

        for neuron_idx, (y_center, x_center) in enumerate(neuron_locations):
            # Create circular mask around neuron center
            dist_sq = (y_coords - y_center) ** 2 + (x_coords - x_center) ** 2
            circle_mask = dist_sq <= (radius**2)

            # Extract mean intensity for each frame within circular region
            for t in range(num_frames):
                # Only consider pixels within both circle and ROI
                valid_pixels = roi_region_stack[t][circle_mask & roi_mask]
                if len(valid_pixels) > 0:
                    neuron_trajectories[neuron_idx, t] = np.mean(valid_pixels)
                else:
                    neuron_trajectories[neuron_idx, t] = 0.0

        # ============================================================
        # Step 4: Quality Filtering (Correlation-based)
        # ============================================================
        # Calculate correlation matrix between all neuron trajectories
        if num_neurons > 1:
            # Compute pairwise correlations
            correlation_matrix = np.corrcoef(neuron_trajectories)

            # For each neuron, compute mean correlation with all other neurons
            # (excluding self-correlation which is always 1.0)
            mean_correlations = np.zeros(num_neurons)
            for i in range(num_neurons):
                # Get correlations with all other neurons (exclude self)
                other_correlations = np.delete(correlation_matrix[i], i)
                mean_correlations[i] = np.mean(other_correlations)

            # Filter neurons based on correlation threshold
            quality_mask = mean_correlations > correlation_threshold
        else:
            # Single neuron: can't compute correlation, mark as good
            quality_mask = np.array([True])

        # ============================================================
        # Step 5: Detrending (Optional)
        # ============================================================
        if apply_detrending and num_frames > 71:  # Need enough frames for Savitzky-Golay
            window_length = min(71, num_frames if num_frames % 2 == 1 else num_frames - 1)
            if window_length >= 5:  # Minimum window length for polyorder=3
                polyorder = min(3, window_length // 2)
                for neuron_idx in range(num_neurons):
                    trajectory = neuron_trajectories[neuron_idx]
                    try:
                        # Apply Savitzky-Golay filter to remove slow drift
                        smoothed = savgol_filter(trajectory, window_length, polyorder)
                        # Subtract smoothed trend from original signal
                        neuron_trajectories[neuron_idx] = trajectory - smoothed
                    except ValueError:
                        # If filtering fails (e.g., too few points), skip detrending
                        pass

        return neuron_locations, neuron_trajectories, quality_mask
