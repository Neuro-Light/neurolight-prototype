from __future__ import annotations

import csv
from typing import Optional, Dict, Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QFormLayout,
)
from PySide6.QtCore import Qt


class NeuronDetectionWidget(QWidget):
    """Widget for detecting and visualizing neurons within a selected ROI."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.neuron_locations: Optional[np.ndarray] = None
        self.neuron_trajectories: Optional[np.ndarray] = None
        self.quality_mask: Optional[np.ndarray] = None
        self.mean_frame: Optional[np.ndarray] = None
        self.roi_mask: Optional[np.ndarray] = None
        self.experiment: Optional["Experiment"] = None
        self.image_processor: Optional["ImageProcessor"] = None
        self.frame_data: Optional[np.ndarray] = None

        layout = QVBoxLayout(self)

        # Status label
        self.status_label = QLabel("No ROI selected. Select an ROI in the image viewer.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Parameters group
        params_group = QGroupBox("Detection Parameters")
        params_layout = QFormLayout()

        # Cell size (diameter in pixels)
        self.cell_size_spin = QSpinBox()
        self.cell_size_spin.setRange(2, 50)
        self.cell_size_spin.setValue(6)
        self.cell_size_spin.setToolTip("Neuron diameter in pixels")
        params_layout.addRow("Cell Size (pixels):", self.cell_size_spin)

        # Number of peaks
        self.num_peaks_spin = QSpinBox()
        self.num_peaks_spin.setRange(1, 1000)
        self.num_peaks_spin.setValue(400)
        self.num_peaks_spin.setToolTip("Maximum number of neurons to detect")
        params_layout.addRow("Max Neurons:", self.num_peaks_spin)

        # Correlation threshold
        self.correlation_threshold_spin = QDoubleSpinBox()
        self.correlation_threshold_spin.setRange(0.0, 1.0)
        self.correlation_threshold_spin.setSingleStep(0.1)
        self.correlation_threshold_spin.setValue(0.4)
        self.correlation_threshold_spin.setDecimals(2)
        self.correlation_threshold_spin.setToolTip(
            "Threshold for filtering neurons by correlation quality"
        )
        params_layout.addRow("Correlation Threshold:", self.correlation_threshold_spin)

        # Relative threshold
        self.threshold_rel_spin = QDoubleSpinBox()
        self.threshold_rel_spin.setRange(0.0, 1.0)
        self.threshold_rel_spin.setSingleStep(0.01)
        self.threshold_rel_spin.setValue(0.1)
        self.threshold_rel_spin.setDecimals(2)
        self.threshold_rel_spin.setToolTip("Relative threshold for peak detection (0.0-1.0)")
        params_layout.addRow("Peak Threshold:", self.threshold_rel_spin)

        # Detrending checkbox
        self.detrending_checkbox = QCheckBox()
        self.detrending_checkbox.setChecked(True)
        self.detrending_checkbox.setToolTip("Apply Savitzky-Golay filter to remove slow drift")
        params_layout.addRow("Apply Detrending:", self.detrending_checkbox)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Detection button
        self.detect_btn = QPushButton("Detect Neurons")
        self.detect_btn.clicked.connect(self._run_detection)
        self.detect_btn.setEnabled(False)
        layout.addWidget(self.detect_btn)

        # Statistics label
        self.stats_label = QLabel("")
        self.stats_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.stats_label)

        # Matplotlib figure and canvas for visualization
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Buttons layout
        buttons_layout = QHBoxLayout()

        self.export_locations_btn = QPushButton("Export Locations (CSV)")
        self.export_locations_btn.clicked.connect(self._export_locations)
        self.export_locations_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_locations_btn)

        self.export_trajectories_btn = QPushButton("Export Trajectories (NPY)")
        self.export_trajectories_btn.clicked.connect(self._export_trajectories)
        self.export_trajectories_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_trajectories_btn)

        self.export_all_btn = QPushButton("Export All (CSV)")
        self.export_all_btn.clicked.connect(self._export_all)
        self.export_all_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_all_btn)

        layout.addLayout(buttons_layout)

    def set_image_processor(self, image_processor: "ImageProcessor") -> None:
        """Set the image processor for detection."""
        self.image_processor = image_processor

    def set_frame_data(self, frame_data: Optional[np.ndarray]) -> None:
        """Set the frame data (3D array: frames, height, width)."""
        self.frame_data = frame_data
        self._update_ui_state()

    def set_roi_mask(self, roi_mask: Optional[np.ndarray]) -> None:
        """Set the ROI mask (2D boolean array)."""
        self.roi_mask = roi_mask
        self._update_ui_state()

    def set_trajectory_plot_callback(self, callback) -> None:
        """Set callback function to notify when trajectories are available."""
        self.trajectory_plot_callback = callback

    def set_save_experiment_callback(self, callback) -> None:
        """Set callback function to save experiment when detection completes."""
        self.save_experiment_callback = callback

    def load_detection_data(
        self,
        neuron_locations: np.ndarray,
        neuron_trajectories: np.ndarray,
        quality_mask: np.ndarray,
        mean_frame: Optional[np.ndarray] = None,
        detection_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load previously saved detection data."""
        self.neuron_locations = neuron_locations
        self.neuron_trajectories = neuron_trajectories
        self.quality_mask = quality_mask

        # Recalculate mean_frame if not provided (it's not saved to reduce file size)
        if mean_frame is not None:
            self.mean_frame = mean_frame
        elif self.frame_data is not None and self.roi_mask is not None:
            # Recalculate mean_frame from frame_data and ROI mask
            roi_region_stack = np.zeros_like(self.frame_data)
            for t in range(self.frame_data.shape[0]):
                roi_region_stack[t] = self.frame_data[t] * self.roi_mask.astype(
                    self.frame_data.dtype
                )

            # Rescale to 0-1 for visualization
            frame_min = np.min(roi_region_stack)
            frame_max = np.max(roi_region_stack)
            if frame_max > frame_min:
                roi_region_stack = (roi_region_stack - frame_min) / (frame_max - frame_min)

            self.mean_frame = np.mean(roi_region_stack, axis=0)
        else:
            self.mean_frame = None

        # Restore detection parameters if available
        if detection_params:
            self.cell_size_spin.setValue(detection_params.get("cell_size", 6))
            self.num_peaks_spin.setValue(detection_params.get("num_peaks", 400))
            self.correlation_threshold_spin.setValue(
                detection_params.get("correlation_threshold", 0.4)
            )
            self.threshold_rel_spin.setValue(detection_params.get("threshold_rel", 0.1))
            self.detrending_checkbox.setChecked(detection_params.get("apply_detrending", True))

        # Update visualization
        self._visualize_results()

        # Update statistics
        num_neurons = len(neuron_locations)
        num_good = np.sum(quality_mask) if quality_mask is not None else 0
        num_bad = num_neurons - num_good

        self.stats_label.setText(
            f"Total Neurons: {num_neurons} | Good: {num_good} | Bad: {num_bad}"
        )

        # Enable export buttons
        self.export_locations_btn.setEnabled(num_neurons > 0)
        self.export_trajectories_btn.setEnabled(num_neurons > 0)
        self.export_all_btn.setEnabled(num_neurons > 0)

        self.status_label.setText(
            f"Loaded {num_neurons} neurons from saved experiment ({num_good} good, {num_bad} bad)"
        )

        # Notify trajectory plot widget
        if hasattr(self, "trajectory_plot_callback"):
            self.trajectory_plot_callback(
                self.neuron_trajectories, self.quality_mask, self.neuron_locations
            )

    def _update_ui_state(self) -> None:
        """Update UI state based on available data."""
        has_data = (
            self.frame_data is not None
            and self.roi_mask is not None
            and self.image_processor is not None
        )
        self.detect_btn.setEnabled(has_data)

        if not has_data:
            if self.frame_data is None:
                self.status_label.setText("No image stack loaded. Load an image stack first.")
            elif self.roi_mask is None:
                self.status_label.setText("No ROI selected. Select an ROI in the image viewer.")
            else:
                self.status_label.setText("Ready to detect neurons.")

    def _run_detection(self) -> None:
        """Run neuron detection on the current ROI."""
        if self.frame_data is None or self.roi_mask is None or self.image_processor is None:
            QMessageBox.warning(
                self,
                "Missing Data",
                "Please load an image stack and select an ROI first.",
            )
            return

        try:
            # Get parameters
            cell_size = self.cell_size_spin.value()
            num_peaks = self.num_peaks_spin.value()
            correlation_threshold = self.correlation_threshold_spin.value()
            threshold_rel = self.threshold_rel_spin.value()
            apply_detrending = self.detrending_checkbox.isChecked()

            # Run detection
            self.detect_btn.setEnabled(False)
            self.status_label.setText("Detecting neurons...")
            QMessageBox.information(
                self,
                "Detection Started",
                "Neuron detection is running. This may take a moment...",
            )

            (self.neuron_locations, self.neuron_trajectories, self.quality_mask) = (
                self.image_processor.detect_neurons_in_roi(
                    self.frame_data,
                    self.roi_mask,
                    cell_size=cell_size,
                    num_peaks=num_peaks,
                    correlation_threshold=correlation_threshold,
                    threshold_rel=threshold_rel,
                    apply_detrending=apply_detrending,
                )
            )

            # Calculate mean frame for visualization
            roi_region_stack = np.zeros_like(self.frame_data)
            for t in range(self.frame_data.shape[0]):
                roi_region_stack[t] = self.frame_data[t] * self.roi_mask.astype(
                    self.frame_data.dtype
                )

            # Rescale to 0-1 for visualization
            frame_min = np.min(roi_region_stack)
            frame_max = np.max(roi_region_stack)
            if frame_max > frame_min:
                roi_region_stack = (roi_region_stack - frame_min) / (frame_max - frame_min)

            self.mean_frame = np.mean(roi_region_stack, axis=0)

            # Update visualization
            self._visualize_results()

            # Update statistics
            num_neurons = len(self.neuron_locations)
            num_good = np.sum(self.quality_mask) if self.quality_mask is not None else 0
            num_bad = num_neurons - num_good

            self.stats_label.setText(
                f"Total Neurons: {num_neurons} | Good: {num_good} | Bad: {num_bad}"
            )

            # Enable export buttons
            self.export_locations_btn.setEnabled(num_neurons > 0)
            self.export_trajectories_btn.setEnabled(num_neurons > 0)
            self.export_all_btn.setEnabled(num_neurons > 0)

            self.status_label.setText(
                f"Detection complete: {num_neurons} neurons detected "
                f"({num_good} good, {num_bad} bad)"
            )

            # Save detection data to experiment
            if self.experiment is not None:
                # Get detection parameters
                detection_params = {
                    "cell_size": cell_size,
                    "num_peaks": num_peaks,
                    "correlation_threshold": correlation_threshold,
                    "threshold_rel": threshold_rel,
                    "apply_detrending": apply_detrending,
                }

                # Set detection data in experiment
                # Note: mean_frame is NOT saved to reduce file size (can be recalculated)
                self.experiment.set_neuron_detection_data(
                    neuron_locations=self.neuron_locations,
                    neuron_trajectories=self.neuron_trajectories,
                    quality_mask=self.quality_mask,
                    mean_frame=None,  # Don't save mean_frame - it can be recalculated
                    detection_params=detection_params,
                )

                # Verify data was set (for debugging)
                saved_data = self.experiment.get_neuron_detection_data()
                if saved_data is None:
                    # Data wasn't set, log error
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.error("Failed to set neuron detection data in experiment")

                # Auto-save experiment if path is available
                if hasattr(self, "save_experiment_callback"):
                    self.save_experiment_callback()

            # Emit signal or notify trajectory plot widget (if connected)
            if hasattr(self, "trajectory_plot_callback"):
                self.trajectory_plot_callback(
                    self.neuron_trajectories, self.quality_mask, self.neuron_locations
                )

        except Exception as e:
            QMessageBox.critical(self, "Detection Failed", f"Failed to detect neurons:\n{str(e)}")
            self.status_label.setText(f"Detection failed: {str(e)}")
        finally:
            self.detect_btn.setEnabled(True)

    def _visualize_results(self) -> None:
        """Visualize detected neurons overlaid on the mean frame."""
        if self.mean_frame is None or self.neuron_locations is None:
            return

        # Clear previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Display mean frame
        ax.imshow(self.mean_frame, cmap="gray", origin="upper")

        # Overlay detected neurons
        if len(self.neuron_locations) > 0:
            # Plot good neurons in green
            good_neurons = (
                self.neuron_locations[self.quality_mask]
                if self.quality_mask is not None
                else self.neuron_locations
            )
            if len(good_neurons) > 0:
                ax.scatter(
                    good_neurons[:, 1],  # x coordinates
                    good_neurons[:, 0],  # y coordinates
                    c="green",
                    s=50,
                    marker="o",
                    edgecolors="darkgreen",
                    linewidths=1,
                    alpha=0.7,
                    label=f"Good Neurons ({len(good_neurons)})",
                )

            # Plot bad neurons in red
            if self.quality_mask is not None:
                bad_neurons = self.neuron_locations[~self.quality_mask]
                if len(bad_neurons) > 0:
                    ax.scatter(
                        bad_neurons[:, 1],  # x coordinates
                        bad_neurons[:, 0],  # y coordinates
                        c="red",
                        s=50,
                        marker="x",
                        linewidths=2,
                        alpha=0.7,
                        label=f"Bad Neurons ({len(bad_neurons)})",
                    )

            ax.legend(loc="upper right")

        ax.set_title("Detected Neurons Overlaid on Mean Frame", fontsize=14)
        ax.set_xlabel("X (pixels)", fontsize=12)
        ax.set_ylabel("Y (pixels)", fontsize=12)

        # Refresh canvas
        self.canvas.draw()

    def _export_locations(self) -> None:
        """Export neuron locations to CSV."""
        if self.neuron_locations is None or len(self.neuron_locations) == 0:
            QMessageBox.warning(self, "No Data", "No neuron locations to export.")
            return

        experiment_name = self.experiment.name if self.experiment else "Experiment"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Neuron Locations",
            f"{experiment_name}_neuron_locations.csv",
            "CSV Files (*.csv)",
        )

        if not file_path:
            return

        try:
            # Create CSV with y, x coordinates and quality
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(["Y", "X", "Quality"])
                # Write data rows
                for i, (y, x) in enumerate(self.neuron_locations):
                    quality = (
                        "Good"
                        if (self.quality_mask[i] if self.quality_mask is not None else True)
                        else "Bad"
                    )
                    writer.writerow([int(y), int(x), quality])

            QMessageBox.information(
                self, "Export Successful", f"Neuron locations exported to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export locations:\n{str(e)}")

    def _export_trajectories(self) -> None:
        """Export neuron trajectories to NPY file."""
        if self.neuron_trajectories is None or len(self.neuron_trajectories) == 0:
            QMessageBox.warning(self, "No Data", "No neuron trajectories to export.")
            return

        experiment_name = self.experiment.name if self.experiment else "Experiment"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Neuron Trajectories",
            f"{experiment_name}_neuron_trajectories.npy",
            "NumPy Files (*.npy)",
        )

        if not file_path:
            return

        try:
            np.save(file_path, self.neuron_trajectories)
            QMessageBox.information(
                self,
                "Export Successful",
                f"Neuron trajectories exported to:\n{file_path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export trajectories:\n{str(e)}")

    def _export_all(self) -> None:
        """Export all data (locations + trajectories) to CSV."""
        if (
            self.neuron_locations is None
            or self.neuron_trajectories is None
            or len(self.neuron_locations) == 0
        ):
            QMessageBox.warning(self, "No Data", "No neuron data to export.")
            return

        experiment_name = self.experiment.name if self.experiment else "Experiment"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save All Neuron Data",
            f"{experiment_name}_neuron_data.csv",
            "CSV Files (*.csv)",
        )

        if not file_path:
            return

        try:
            # Create CSV with y, x, quality, and all trajectory values
            num_neurons, num_frames = self.neuron_trajectories.shape

            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                # Write header: Y, X, Quality, Frame_0, Frame_1, ..., Frame_N
                header = ["Y", "X", "Quality"]
                header.extend([f"Frame_{i}" for i in range(num_frames)])
                writer.writerow(header)

                # Write data rows
                for i, (y, x) in enumerate(self.neuron_locations):
                    quality = (
                        "Good"
                        if (self.quality_mask[i] if self.quality_mask is not None else True)
                        else "Bad"
                    )
                    row = [int(y), int(x), quality]
                    # Add trajectory values as floats
                    row.extend([float(val) for val in self.neuron_trajectories[i]])
                    writer.writerow(row)

            QMessageBox.information(
                self, "Export Successful", f"All neuron data exported to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{str(e)}")

    def clear_results(self) -> None:
        """Clear detection results and reset state."""
        self.neuron_locations = None
        self.neuron_trajectories = None
        self.quality_mask = None
        self.mean_frame = None
        self.figure.clear()
        self.canvas.draw()
        self.stats_label.setText("")
        self.status_label.setText("No ROI selected. Select an ROI in the image viewer.")
        self.export_locations_btn.setEnabled(False)
        self.export_trajectories_btn.setEnabled(False)
        self.export_all_btn.setEnabled(False)
