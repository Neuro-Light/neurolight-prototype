from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QLabel,
    QMessageBox,
    QApplication,
    QDialog,
)
from PySide6.QtGui import QAction

import numpy as np

from core.experiment_manager import Experiment, ExperimentManager
from core.data_analyzer import DataAnalyzer
from core.image_processor import ImageProcessor
from utils.file_handler import ImageStackHandler
from ui.image_viewer import ImageViewer
from ui.analysis_panel import AnalysisPanel
from ui.startup_dialog import StartupDialog


class MainWindow(QMainWindow):
    def __init__(self, experiment: Experiment) -> None:
        super().__init__()
        self.experiment = experiment
        self.manager = ExperimentManager()
        self.current_experiment_path: Optional[str] = None

        self.setWindowTitle(f"Neurolight - {self.experiment.name}")
        self.resize(1200, 800)

        self._init_menu()
        self._init_layout()

    def _init_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        save_action = QAction("Save Experiment", self)
        save_as_action = QAction("Save Experiment As...", self)
        close_action = QAction("Close Experiment", self)
        exit_action = QAction("Exit Experiment", self)
        open_stack_action = QAction("Open Image Stack", self)
        export_results_action = QAction("Export Results", self)

        save_action.setShortcut("Ctrl+S")

        save_action.triggered.connect(self._save)
        save_as_action.triggered.connect(self._save_as)
        close_action.triggered.connect(self._close_experiment)
        exit_action.triggered.connect(self._exit_experiment)
        open_stack_action.triggered.connect(self._open_image_stack)

        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(open_stack_action)
        file_menu.addAction(export_results_action)
        file_menu.addSeparator()
        file_menu.addAction(close_action)
        file_menu.addAction(exit_action)

        menubar.addMenu("Edit").addAction("Experiment Settings")
        tools_menu = menubar.addMenu("Tools")
        crop_roi_action = QAction("Crop ROI", self)
        crop_roi_action.triggered.connect(self._crop_roi)
        tools_menu.addAction(crop_roi_action)
        tools_menu.addAction("Generate GIF")
        tools_menu.addAction("Run Analysis")
        menubar.addMenu("Help").addAction("About")

    def _init_layout(self) -> None:
        splitter = QSplitter()

        # Left panel: image viewer
        self.stack_handler = ImageStackHandler()
        self.stack_handler.associate_with_experiment(self.experiment)
        self.viewer = ImageViewer(self.stack_handler)
        self.viewer.stackLoaded.connect(self._on_stack_loaded)

        # Right panel: analysis dashboard
        self.analysis = AnalysisPanel()

        # Connect ROI selection to analysis and saving
        self.viewer.roiSelected.connect(self._on_roi_selected)
        self.viewer.roiSelected.connect(self._save_roi_to_experiment)

        # Create data analyzer and image processor
        self.data_analyzer = DataAnalyzer(self.experiment)
        self.image_processor = ImageProcessor(self.experiment)

        splitter.addWidget(self.viewer)
        splitter.addWidget(self.analysis)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

        self.setCentralWidget(splitter)

        # Auto-load image stack/ROI if experiment has saved data
        self._auto_load_experiment_data()

    def _auto_load_experiment_data(self) -> None:
        """Auto-load image stack and ROI if experiment has saved data."""
        try:
            path = self.experiment.image_stack_path
            if path:

                def load_stack_and_roi(p=path):
                    self.viewer.set_stack(p)
                    if self.experiment.roi:
                        roi = self.experiment.roi
                        x = roi.get("x", 0)
                        y = roi.get("y", 0)
                        width = roi.get("width", 0)
                        height = roi.get("height", 0)
                        roi_shape = roi.get("shape", "ellipse")

                        def load_roi_and_plot():
                            # Load ROI with the correct shape from saved data
                            self.viewer.set_roi(x, y, width, height)
                            self.viewer.roi_shape = roi_shape
                            self._on_roi_selected(x, y, width, height)

                        QTimer.singleShot(200, load_roi_and_plot)

                QTimer.singleShot(0, load_stack_and_roi)
        except Exception:
            pass

    def _open_image_stack(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select Image Stack Folder", ""
        )
        if not directory:
            return
        self.viewer.set_stack(directory)

    def _on_stack_loaded(self, directory_path: str) -> None:
        # ImageStackHandler already updates experiment association for path/count
        self.stack_handler.associate_with_experiment(self.experiment)
        # Persist immediately if we know the path to the .nexp
        if self.current_experiment_path:
            try:
                self.manager.save_experiment(
                    self.experiment, self.current_experiment_path
                )
            except Exception:
                pass

    def _save(self) -> None:
        if not self.current_experiment_path:
            self._save_as()
            return
        try:
            self.manager.save_experiment(self.experiment, self.current_experiment_path)
            QMessageBox.information(self, "Saved", "Experiment saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _save_as(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Experiment As", "", "Neurolight Experiment (*.nexp)"
        )
        if not file_path:
            return
        if not file_path.endswith(".nexp"):
            file_path += ".nexp"
        try:
            self.manager.save_experiment(self.experiment, file_path)
            self.current_experiment_path = file_path
            QMessageBox.information(self, "Saved", "Experiment saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _close_experiment(self) -> None:
        """
        Close the current experiment and navigate to the home page (StartupDialog).
        This keeps the user in the application and allows them to select a new experiment.
        """
        # Prompt user if there are unsaved changes
        reply = QMessageBox.question(
            self,
            "Close Experiment",
            "Are you sure you want to close this experiment and return to the home page?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.No:
            return

        # Hide the main window
        self.hide()

        # Show startup dialog
        startup = StartupDialog()
        result = startup.exec()

        if result == QDialog.Accepted and startup.experiment is not None:
            # User selected a new experiment - replace current experiment
            self.experiment = startup.experiment
            self.current_experiment_path = startup.experiment_path
            self.setWindowTitle(f"Neurolight - {self.experiment.name}")

            # Reset viewer state
            self.viewer.reset()

            # Clear analysis panel
            self.analysis.roi_plot_widget.clear_plot()

            # Reassociate handler and data analyzer with new experiment
            self.stack_handler.associate_with_experiment(self.experiment)
            self.data_analyzer = DataAnalyzer(self.experiment)

            # Auto-load image stack/ROI if experiment has saved data
            self._auto_load_experiment_data()

            # Show the window again
            self.show()
        else:
            # User canceled - exit the application
            QApplication.quit()

    def _exit_experiment(self) -> None:
        """
        Exit the entire application.
        This closes the application completely.
        """
        reply = QMessageBox.question(
            self,
            "Exit Experiment",
            "Are you sure you want to exit the application?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            QApplication.quit()

    def _on_roi_selected(self, x: int, y: int, width: int, height: int) -> None:
        """Handle ROI selection and extract intensity time series."""
        try:
            # Load all frames as numpy array (reusing Jupyter notebook approach)
            frame_data = self.stack_handler.get_all_frames_as_array()
            if frame_data is None:
                QMessageBox.warning(
                    self,
                    "No Image Data",
                    "No image stack loaded. Please load an image stack first.",
                )
                return

            # Rescale frame data (reusing approach from Jupyter notebook)
            # frame_data = NTF.rescale(frames, 0.0, 1.0)
            # We'll keep it in original range but normalize if needed
            frame_min = np.min(frame_data)
            frame_max = np.max(frame_data)
            if frame_max > 1.0:
                # Normalize to 0-1 range like in Jupyter notebook
                frame_data = (
                    (frame_data - frame_min) / (frame_max - frame_min)
                    if frame_max != frame_min
                    else frame_data
                )

            # Extract ROI intensity time series
            intensity_data = self.data_analyzer.extract_roi_intensity_time_series(
                frame_data, x, y, width, height
            )

            # Plot in the ROI Intensity tab
            roi_plot_widget = self.analysis.get_roi_plot_widget()
            roi_plot_widget.plot_intensity_time_series(
                intensity_data, (x, y, width, height)
            )

            # Switch to ROI Intensity tab
            for i in range(self.analysis.count()):
                if self.analysis.tabText(i) == "ROI Intensity":
                    self.analysis.setCurrentIndex(i)
                    break

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze ROI:\n{str(e)}")

    def _save_roi_to_experiment(self, x: int, y: int, width: int, height: int) -> None:
        """
        Save ROI coordinates to experiment and persist to .nexp file.

        This method is called when a user selects an ROI in the image viewer.
        Coordinates are in original image pixel space (not widget/display space).
        This ensures the ROI stays fixed to the correct image region when:
        - The window is resized
        - The experiment is loaded on a different screen resolution
        - The image scaling changes

        The ROI is automatically saved to the .nexp file so it persists across sessions.
        """
        # Store coordinates in image pixel space (not display coordinates)
        # These coordinates are saved to the .nexp file and remain constant
        roi_shape = getattr(self.viewer, "roi_shape", "ellipse")

        # Save ellipse ROI
        self.experiment.roi = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "shape": roi_shape,
        }
        if self.current_experiment_path:
            try:
                # Persist ROI to .nexp file immediately
                self.manager.save_experiment(
                    self.experiment, self.current_experiment_path
                )
            except Exception:
                pass

    def autosave_experiment(self) -> None:
        if not self.experiment.settings.get("processing", {}).get("auto_save", True):
            return
        if not self.current_experiment_path:
            return
        try:
            self.manager.save_experiment(self.experiment, self.current_experiment_path)
        except Exception:
            pass

    def _crop_roi(self) -> None:
        """Crop the image stack using the current ROI and save cropped images."""
        if self.viewer.current_roi is None:
            QMessageBox.warning(
                self, "No ROI Selected", "Please select an ROI before cropping."
            )
            return

        if self.stack_handler.get_image_count() == 0:
            QMessageBox.warning(
                self, "No Images", "Please load an image stack before cropping."
            )
            return

        # Get ROI coordinates
        if (
            not isinstance(self.viewer.current_roi, tuple)
            or len(self.viewer.current_roi) != 4
        ):
            QMessageBox.warning(
                self, "Invalid ROI", "Current ROI is not valid for cropping."
            )
            return

        # Ellipse ROI
        roi_tuple = self.viewer.current_roi
        assert all(isinstance(v, (int, float)) for v in roi_tuple), (
            "ROI tuple must contain numbers"
        )
        x = int(roi_tuple[0])  # type: ignore
        y = int(roi_tuple[1])  # type: ignore
        width = int(roi_tuple[2])  # type: ignore
        height = int(roi_tuple[3])  # type: ignore
        roi_shape = getattr(self.viewer, "roi_shape", "ellipse")

        # Ask user for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Cropped Images", ""
        )
        if not output_dir:
            return

        try:
            # Load all frames
            frame_data = self.stack_handler.get_all_frames_as_array()
            if frame_data is None:
                QMessageBox.critical(self, "Error", "Failed to load image stack.")
                return

            # Crop the stack
            cropped_stack = self.image_processor.crop_image_stack(
                frame_data, x, y, width, height, roi_shape, None
            )

            # Save cropped images
            from pathlib import Path
            import cv2

            output_path = Path(output_dir)
            base_name = (
                Path(self.experiment.name).stem if self.experiment.name else "cropped"
            )

            num_frames = cropped_stack.shape[0]
            for i in range(num_frames):
                frame = cropped_stack[i]
                # Normalize to 0-255 if needed
                if frame.dtype != np.uint8:
                    frame_min = np.min(frame)
                    frame_max = np.max(frame)
                    if frame_max > frame_min:
                        frame = (
                            (frame - frame_min) / (frame_max - frame_min) * 255
                        ).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)

                output_file = output_path / f"{base_name}_cropped_{i:04d}.tif"
                cv2.imwrite(str(output_file), frame)

            QMessageBox.information(
                self,
                "Crop Complete",
                f"Successfully cropped and saved {num_frames} images to:\n{output_dir}",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to crop images:\n{str(e)}")
