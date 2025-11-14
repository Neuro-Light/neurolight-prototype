from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from core.experiment_manager import Experiment


class DataAnalyzer:
    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def calculate_statistics(self, data: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }

    def generate_plots(self, data: np.ndarray, plot_type: str = "hist"):
        fig, ax = plt.subplots()
        if plot_type == "hist":
            ax.hist(data.flatten(), bins=50)
            ax.set_title("Histogram")
        else:
            ax.plot(data)
        return fig

    def save_results_to_experiment(self, experiment: Experiment) -> None:
        experiment.analysis_results.setdefault("runs", []).append(
            {
                "summary": "Placeholder analysis",
            }
        )

    # Placeholders for future expansion
    def time_series_analysis(self, data: np.ndarray):
        return {}

    def correlation_analysis(self, data_a: np.ndarray, data_b: np.ndarray):
        return {}

    def extract_roi_intensity_time_series(
        self,
        frame_data: np.ndarray,
        roi_x: int,
        roi_y: int,
        roi_width: int,
        roi_height: int,
    ) -> np.ndarray:
        """
        Extract mean pixel intensity within an ROI across all frames.
        Reuses approach from Jupyter notebook (similar to BTS calculation).

        Args:
            frame_data: 3D numpy array (frames, height, width)
            roi_x: X coordinate of ROI top-left corner
            roi_y: Y coordinate of ROI top-left corner
            roi_width: Width of ROI
            roi_height: Height of ROI

        Returns:
            1D numpy array of mean intensities across frames
        """
        if frame_data.ndim != 3:
            raise ValueError("frame_data must be a 3D array (frames, height, width)")

        num_frames = frame_data.shape[0]
        frame_height = frame_data.shape[1]
        frame_width = frame_data.shape[2]

        # Clamp ROI to image bounds
        x1: int = max(0, int(roi_x))
        y1: int = max(0, int(roi_y))
        x2 = min(frame_width, x1 + int(roi_width))
        y2 = min(frame_height, y1 + int(roi_height))

        if x2 <= x1 or y2 <= y1:
            # Invalid ROI, return zeros
            return np.zeros(num_frames)

        # Extract ROI region and calculate mean intensity for each frame
        # Similar to: BTS = frame_data.mean(axis=1).mean(axis=1) but for ROI only
        roi_intensities = np.zeros(num_frames)
        for t in range(num_frames):
            roi_region = frame_data[t, y1:y2, x1:x2]
            roi_intensities[t] = np.mean(roi_region)

        return roi_intensities
