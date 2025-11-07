from __future__ import annotations

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

    def align_image_ecc(
        self,
        template: np.ndarray,
        image: np.ndarray,
        warp_mode: int = cv2.MOTION_EUCLIDEAN,
        max_iterations: int = 5000,
        termination_eps: float = 1e-10
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Align image to template using Enhanced Correlation Coefficient (ECC) algorithm.
        
        Args:
            template: Reference template image
            image: Image to align
            warp_mode: Warp mode (MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE, MOTION_HOMOGRAPHY)
            max_iterations: Maximum iterations for ECC algorithm
            termination_eps: Termination threshold
            
        Returns:
            Tuple of (aligned_image, transformation_matrix, confidence_score)
        """
        # Ensure images are grayscale and uint8
        if template.ndim == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()
        if image.ndim == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image.copy()
        
        # Normalize to uint8
        if template_gray.dtype != np.uint8:
            template_gray = cv2.normalize(template_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if image_gray.dtype != np.uint8:
            image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Initialize transformation matrix
        if warp_mode == cv2.MOTION_TRANSLATION:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        elif warp_mode == cv2.MOTION_EUCLIDEAN:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        elif warp_mode == cv2.MOTION_AFFINE:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:  # MOTION_HOMOGRAPHY
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, termination_eps)
        
        try:
            # Run ECC algorithm
            cc, warp_matrix = cv2.findTransformECC(
                template_gray,
                image_gray,
                warp_matrix,
                warp_mode,
                criteria,
                None,
                1
            )
            
            # Apply transformation to original image (preserving data type and range)
            # Use BORDER_REPLICATE to avoid black borders, or BORDER_CONSTANT with a better fill value
            # Store original dtype for output
            output_dtype = image.dtype
            output_shape = (template_gray.shape[1], template_gray.shape[0])
            
            # For warping, we need to handle different data types properly
            # Convert to float32 for warping to avoid clipping issues
            if image.dtype != np.float32:
                image_float = image.astype(np.float32)
            else:
                image_float = image.copy()
            
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                aligned_float = cv2.warpPerspective(
                    image_float, 
                    warp_matrix, 
                    output_shape,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE  # Replicate edge pixels instead of black
                )
            else:
                aligned_float = cv2.warpAffine(
                    image_float, 
                    warp_matrix, 
                    output_shape,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE  # Replicate edge pixels instead of black
                )
            
            # Convert back to original dtype, preserving the original value range
            if output_dtype == np.uint8:
                aligned = np.clip(aligned_float, 0, 255).astype(np.uint8)
            elif output_dtype == np.uint16:
                aligned = np.clip(aligned_float, 0, 65535).astype(np.uint16)
            else:
                aligned = aligned_float.astype(output_dtype)
            
            return aligned, warp_matrix, float(cc)
        except cv2.error as e:
            # If alignment fails, return original image with identity matrix
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, 3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
            return image.copy(), warp_matrix, 0.0

    def align_image_orb(
        self,
        template: np.ndarray,
        image: np.ndarray,
        max_features: int = 500,
        good_match_percent: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Align image to template using ORB feature detection and matching.
        
        Args:
            template: Reference template image
            image: Image to align
            max_features: Maximum number of features to detect
            good_match_percent: Percentage of best matches to use
            
        Returns:
            Tuple of (aligned_image, transformation_matrix, confidence_score)
        """
        # Ensure images are grayscale and uint8
        if template.ndim == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()
        if image.ndim == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image.copy()
        
        # Normalize to uint8
        if template_gray.dtype != np.uint8:
            template_gray = cv2.normalize(template_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if image_gray.dtype != np.uint8:
            image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Detect ORB features
        orb = cv2.ORB_create(max_features)
        kp1, des1 = orb.detectAndCompute(template_gray, None)
        kp2, des2 = orb.detectAndCompute(image_gray, None)
        
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            # Not enough features, return original image
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            return image.copy(), warp_matrix, 0.0
        
        # Match features
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # Check if we have enough matches
        num_good_matches = int(len(good_matches) * good_match_percent)
        if num_good_matches < 4:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            return image.copy(), warp_matrix, 0.0
        
        # Get matching points
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:num_good_matches]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        warp_matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if warp_matrix is None:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
            return image.copy(), warp_matrix, 0.0
        
        # Apply transformation to original image (preserving data type and range)
        h, w = template_gray.shape
        output_shape = (w, h)
        
        # Store original dtype for output
        output_dtype = image.dtype
        
        # Convert to float32 for warping to avoid clipping issues
        if image.dtype != np.float32:
            image_float = image.astype(np.float32)
        else:
            image_float = image.copy()
        
        aligned_float = cv2.warpPerspective(
            image_float,
            warp_matrix,
            output_shape,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE  # Replicate edge pixels instead of black
        )
        
        # Convert back to original dtype, preserving the original value range
        if output_dtype == np.uint8:
            aligned = np.clip(aligned_float, 0, 255).astype(np.uint8)
        elif output_dtype == np.uint16:
            aligned = np.clip(aligned_float, 0, 65535).astype(np.uint16)
        else:
            aligned = aligned_float.astype(output_dtype)
        
        # Calculate confidence based on number of inliers
        confidence = float(np.sum(mask)) / len(good_matches) if len(good_matches) > 0 else 0.0
        
        return aligned, warp_matrix, confidence

    def align_image_stack(
        self,
        image_stack: np.ndarray,
        reference_index: int = 0,
        method: str = "ecc",
        warp_mode: str = "euclidean",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
        """
        Align a stack of images to a reference image.
        
        Args:
            image_stack: 3D numpy array (frames, height, width)
            reference_index: Index of reference image (default 0)
            method: Alignment method ("ecc" or "orb")
            warp_mode: Warp mode for ECC ("translation", "euclidean", "affine", "homography")
            progress_callback: Optional callback function(completed, total, status_message)
            
        Returns:
            Tuple of (aligned_stack, transformation_matrices, confidence_scores)
        """
        if image_stack.ndim != 3:
            raise ValueError("Image stack must be 3D array (frames, height, width)")
        
        num_frames = image_stack.shape[0]
        if reference_index < 0 or reference_index >= num_frames:
            raise ValueError(f"Reference index {reference_index} out of range [0, {num_frames-1}]")
        
        # Get reference image
        template = image_stack[reference_index]
        
        # Initialize output arrays
        aligned_stack = np.zeros_like(image_stack)
        aligned_stack[reference_index] = template.copy()
        
        transformation_matrices = []
        confidence_scores = []
        
        # Convert warp_mode string to OpenCV constant
        warp_mode_map = {
            "translation": cv2.MOTION_TRANSLATION,
            "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE,
            "homography": cv2.MOTION_HOMOGRAPHY
        }
        warp_mode_cv = warp_mode_map.get(warp_mode.lower(), cv2.MOTION_EUCLIDEAN)
        
        # Initialize transformation matrices for all frames in correct order
        for i in range(num_frames):
            if i == reference_index:
                # Reference frame has identity transformation
                if warp_mode_cv == cv2.MOTION_HOMOGRAPHY:
                    transformation_matrices.append(np.eye(3, 3, dtype=np.float32))
                else:
                    transformation_matrices.append(np.eye(2, 3, dtype=np.float32))
                confidence_scores.append(1.0)
            else:
                # Will be filled during alignment
                if warp_mode_cv == cv2.MOTION_HOMOGRAPHY:
                    transformation_matrices.append(np.eye(3, 3, dtype=np.float32))
                else:
                    transformation_matrices.append(np.eye(2, 3, dtype=np.float32))
                confidence_scores.append(0.0)
        
        # Align each frame (in order, skipping reference)
        frames_processed = 0
        for i in range(num_frames):
            if i == reference_index:
                continue
            
            frames_processed += 1
            if progress_callback:
                progress_callback(frames_processed, num_frames - 1, f"Aligning frame {i+1}/{num_frames}")
            
            image = image_stack[i]
            
            try:
                if method.lower() == "orb":
                    aligned, warp_matrix, confidence = self.align_image_orb(template, image)
                else:  # ECC
                    aligned, warp_matrix, confidence = self.align_image_ecc(template, image, warp_mode_cv)
                
                aligned_stack[i] = aligned
                transformation_matrices[i] = warp_matrix
                confidence_scores[i] = confidence
            except Exception as e:
                # If alignment fails, use original image with identity transform
                if progress_callback:
                    progress_callback(frames_processed, num_frames - 1, f"Warning: Alignment failed for frame {i+1}, using original")
                aligned_stack[i] = image.copy()
                if warp_mode_cv == cv2.MOTION_HOMOGRAPHY:
                    transformation_matrices[i] = np.eye(3, 3, dtype=np.float32)
                else:
                    transformation_matrices[i] = np.eye(2, 3, dtype=np.float32)
                confidence_scores[i] = 0.0
        
        if progress_callback:
            progress_callback(num_frames - 1, num_frames - 1, "Alignment complete")
        
        return aligned_stack, transformation_matrices, confidence_scores

