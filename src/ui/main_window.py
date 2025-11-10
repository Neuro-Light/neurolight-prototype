from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

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
from PySide6.QtGui import QAction, QCloseEvent

import numpy as np

from core.experiment_manager import Experiment, ExperimentManager
from core.data_analyzer import DataAnalyzer
from core.image_processor import ImageProcessor
from core.roi import ROI, ROIShape
from utils.file_handler import ImageStackHandler
from ui.image_viewer import ImageViewer
from ui.analysis_panel import AnalysisPanel
from ui.startup_dialog import StartupDialog
from ui.alignment_dialog import AlignmentDialog
from ui.alignment_progress_dialog import AlignmentProgressDialog

# Set up logger for main window
logger = logging.getLogger(__name__)

# Configure logging to file if not already configured
_log_file = Path.home() / ".neurolight" / "neurolight.log"
_log_file.parent.mkdir(parents=True, exist_ok=True)

# Check if logging is already configured at the module level
if not logger.handlers:
    # Create file handler with append mode
    file_handler = logging.FileHandler(_log_file, encoding="utf-8", mode='a')
    file_handler.setLevel(logging.ERROR)
    
    # Create formatter - logger.exception() automatically includes traceback
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(logging.ERROR)
    logger.propagate = False  # Prevent duplicate logs


class MainWindow(QMainWindow):
    def __init__(self, experiment: Experiment) -> None:
        super().__init__()
        self.experiment = experiment
        self.manager = ExperimentManager()
        self.current_experiment_path: Optional[str] = None
        self.image_processor = ImageProcessor(experiment)

        # Initialize debounced save timer for display settings
        self._display_settings_timer = QTimer()
        self._display_settings_timer.setSingleShot(True)
        self._display_settings_timer.timeout.connect(self._persist_display_settings)
        self._pending_exposure: Optional[int] = None
        self._pending_contrast: Optional[int] = None

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
        export_results_action.triggered.connect(self._export_experiment)

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
        
        # Add crop action
        crop_action = QAction("Crop Stack to ROI", self)
        crop_action.triggered.connect(self._crop_stack_to_roi)
        tools_menu.addAction(crop_action)
        
        # Add alignment action
        align_action = QAction("Align Images", self)
        align_action.triggered.connect(self._align_images)
        tools_menu.addAction(align_action)
        
        tools_menu.addAction("Generate GIF")
        tools_menu.addAction("Run Analysis")
        menubar.addMenu("Help").addAction("About")

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Handle window close event (when user clicks X button).
        Shows a confirmation dialog before closing.
        """
        reply = QMessageBox.question(
            self,
            "Exit Application",
            "Are you sure you want to exit the application?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Flush any pending display settings before exiting
            self._flush_pending_display_settings()
            # Save current ROI to experiment before exiting
            current_roi = self.viewer.get_current_roi()
            if current_roi is not None:
                self.experiment.roi = current_roi.to_dict()
            # Capture current display settings before exiting
            self._capture_display_settings()
            # Save to file if we have a path
            if self.current_experiment_path:
                try:
                    self.manager.save_experiment(
                        self.experiment, self.current_experiment_path
                    )
                except Exception as e:
                    # Log the full exception with traceback
                    logger.exception(
                        f"Failed to save experiment during close: {self.current_experiment_path}"
                    )
                    # Show non-blocking user feedback
                    self._show_save_error_feedback(str(e))
            event.accept()
        else:
            event.ignore()

    def _show_save_error_feedback(self, error_message: str) -> None:
        """
        Show non-blocking feedback when save fails during close.
        Uses status bar message to inform user without blocking the close flow.
        
        Args:
            error_message: The error message string (also logged via logger.exception)
        """
        log_path = _log_file
        status_message = (
            f"Save failed during close. Error logged to: {log_path}"
        )
        
        # Show status bar message (non-blocking, brief display)
        # This will be visible briefly before the window closes if there's any delay
        status_bar = self.statusBar()
        if status_bar:
            # Show message for 5 seconds to increase visibility
            status_bar.showMessage(status_message, 5000)
        
        # Note: The full exception with traceback is already logged via logger.exception()
        # in the closeEvent handler. We intentionally don't show a modal dialog here
        # as it would block the close flow. The error is fully logged to file, and
        # the status bar message provides immediate feedback. Users can check the
        # log file at ~/.neurolight/neurolight.log for full error details.

    def _init_layout(self) -> None:
        splitter = QSplitter()

        # Left panel: image viewer
        self.stack_handler = ImageStackHandler()
        self.stack_handler.associate_with_experiment(self.experiment)
        self.viewer = ImageViewer(self.stack_handler)
        self.viewer.stackLoaded.connect(self._on_stack_loaded)

        # Right panel: analysis dashboard
        self.analysis = AnalysisPanel()
        self.analysis.get_roi_plot_widget().experiment = self.experiment

        # Connect ROI selection to analysis and saving
        self.viewer.roiSelected.connect(self._on_roi_selected)
        self.viewer.roiSelected.connect(self._save_roi_to_experiment)
        
        # Connect display settings changes to debounced saving
        self.viewer.displaySettingsChanged.connect(self._on_display_settings_changed)

        # Create data analyzer
        self.data_analyzer = DataAnalyzer(self.experiment)

        splitter.addWidget(self.viewer)
        splitter.addWidget(self.analysis)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

        self.setCentralWidget(splitter)

        # Auto-load image stack/ROI if experiment has saved data
        self._auto_load_experiment_data()

    def _auto_load_experiment_data(self) -> None:
        """Auto-load image stack, ROI, and display settings if experiment has saved data."""
        try:
            path = self.experiment.image_stack_path
            if path:

                def load_stack_and_roi(p=path):
                    self.viewer.set_stack(p)
                    # Load display settings (exposure, contrast)
                    display_settings = self.experiment.settings.get("display", {})
                    exposure = display_settings.get("exposure", 0)
                    contrast = display_settings.get("contrast", 0)
                    self.viewer.set_exposure(exposure)
                    self.viewer.set_contrast(contrast)
                    
                    if self.experiment.roi:
                        roi_data = self.experiment.roi
                        
                        # Convert dict to ROI object
                        try:
                            roi = ROI.from_dict(roi_data)
                        except Exception:
                            # Fallback for malformed data - use same defaults as from_dict()
                            roi = ROI(
                                x=roi_data.get("x", 0),
                                y=roi_data.get("y", 0),
                                width=roi_data.get("width", 100),
                                height=roi_data.get("height", 100),
                                shape=ROIShape.ELLIPSE
                            )

                        def load_roi_and_plot():
                            self.viewer.set_roi(roi)
                            self._on_roi_selected(roi)

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
        # Flush any pending display settings immediately
        self._flush_pending_display_settings()
        # Save current ROI to experiment before saving
        current_roi = self.viewer.get_current_roi()
        if current_roi is not None:
            self.experiment.roi = current_roi.to_dict()
        # Capture current display settings before saving
        self._capture_display_settings()
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
        # Flush any pending display settings immediately
        self._flush_pending_display_settings()
        # Save current ROI to experiment before saving
        current_roi = self.viewer.get_current_roi()
        if current_roi is not None:
            self.experiment.roi = current_roi.to_dict()
        # Capture current display settings before saving
        self._capture_display_settings()
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

        # Flush any pending display settings immediately
        self._flush_pending_display_settings()
        # Save current ROI to experiment before closing
        current_roi = self.viewer.get_current_roi()
        if current_roi is not None:
            self.experiment.roi = current_roi.to_dict()
        # Capture current display settings before closing
        self._capture_display_settings()
        # Save to file if we have a path
        if self.current_experiment_path:
            try:
                self.manager.save_experiment(
                    self.experiment, self.current_experiment_path
                )
            except Exception:
                pass

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

            # Auto-load image stack/ROI/display settings if experiment has saved data
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
            # Flush any pending display settings immediately
            self._flush_pending_display_settings()
            # Save current ROI to experiment before exiting
            current_roi = self.viewer.get_current_roi()
            if current_roi is not None:
                self.experiment.roi = current_roi.to_dict()
            # Capture current display settings before exiting
            self._capture_display_settings()
            # Save to file if we have a path
            if self.current_experiment_path:
                try:
                    self.manager.save_experiment(
                        self.experiment, self.current_experiment_path
                    )
                except Exception:
                    pass
            QApplication.quit()

    def _on_roi_selected(self, roi: ROI) -> None:
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
                frame_data, roi.x, roi.y, roi.width, roi.height
            )

            # Plot in the ROI Intensity tab
            roi_plot_widget = self.analysis.get_roi_plot_widget()
            roi_plot_widget.plot_intensity_time_series(
                intensity_data, (roi.x, roi.y, roi.width, roi.height)
            )

            # Switch to ROI Intensity tab
            for i in range(self.analysis.count()):
                if self.analysis.tabText(i) == "ROI Intensity":
                    self.analysis.setCurrentIndex(i)
                    break

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze ROI:\n{str(e)}")

    def _capture_display_settings(self) -> None:
        """
        Capture current display settings (exposure, contrast) from viewer into experiment.
        
        This method consolidates the logic for updating display settings in the experiment
        and should be called before saving, closing, or exiting.
        """
        if "display" not in self.experiment.settings:
            self.experiment.settings["display"] = {}
        self.experiment.settings["display"]["exposure"] = self.viewer.get_exposure()
        self.experiment.settings["display"]["contrast"] = self.viewer.get_contrast()
    
    def _on_display_settings_changed(self, exposure: int, contrast: int) -> None:
        """
        Handle display settings changes with debounced saving.
        
        This method is called when the user adjusts exposure or contrast sliders.
        It stores the pending values and starts/resets a timer for debounced saving.
        """
        self._pending_exposure = exposure
        self._pending_contrast = contrast
        # Restart the timer (debounce: wait 500ms after last change before saving)
        self._display_settings_timer.stop()
        self._display_settings_timer.start(500)
    
    def _flush_pending_display_settings(self) -> None:
        """
        Immediately flush any pending display settings without waiting for the timer.
        
        This is called when manually saving, closing, or exiting to ensure
        all pending changes are persisted immediately.
        """
        # Stop the timer and flush immediately
        self._display_settings_timer.stop()
        if self._pending_exposure is not None and self._pending_contrast is not None:
            self._persist_display_settings()
    
    def _persist_display_settings(self) -> None:
        """
        Persist pending display settings to experiment and save to file.
        
        This method is called by the debounce timer after the user stops adjusting sliders,
        or immediately when flushing pending settings.
        """
        if self._pending_exposure is not None and self._pending_contrast is not None:
            # Update experiment with pending values
            if "display" not in self.experiment.settings:
                self.experiment.settings["display"] = {}
            self.experiment.settings["display"]["exposure"] = self._pending_exposure
            self.experiment.settings["display"]["contrast"] = self._pending_contrast
            
            # Clear pending values
            self._pending_exposure = None
            self._pending_contrast = None
            
            # Persist to file if we have a path
            if self.current_experiment_path:
                try:
                    self.manager.save_experiment(
                        self.experiment, self.current_experiment_path
                    )
                except Exception:
                    pass

    def _save_roi_to_experiment(self, roi: ROI) -> None:
        """
        Save ROI to experiment and persist to .nexp file.

        This method is called when a user selects an ROI in the image viewer.
        Coordinates are in original image pixel space (not widget/display space).
        This ensures the ROI stays fixed to the correct image region when:
        - The window is resized
        - The experiment is loaded on a different screen resolution
        - The image scaling changes

        The ROI is automatically saved to the .nexp file so it persists across sessions.
        """
        # Store ROI in image pixel space (not display coordinates)
        # These coordinates are saved to the .nexp file and remain constant
        self.experiment.roi = roi.to_dict()
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
        # Flush any pending display settings immediately
        self._flush_pending_display_settings()
        # Save current ROI to experiment before auto-saving
        current_roi = self.viewer.get_current_roi()
        if current_roi is not None:
            self.experiment.roi = current_roi.to_dict()
        # Capture current display settings before auto-saving
        self._capture_display_settings()
        try:
            self.manager.save_experiment(self.experiment, self.current_experiment_path)
        except Exception:
            pass
    
    def _crop_stack_to_roi(self) -> None:
        """Crop the image stack to the current ROI and save as new stack."""
        current_roi = self.viewer.get_current_roi()
        if current_roi is None:
            QMessageBox.warning(
                self,
                "No ROI Selected",
                "Please select an ROI before cropping."
            )
            return
        
        try:
            # Load all frames
            frame_data = self.stack_handler.get_all_frames_as_array()
            if frame_data is None:
                QMessageBox.warning(
                    self,
                    "No Image Data",
                    "No image stack loaded."
                )
                return
            
            # Ask user for output directory
            output_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Output Directory for Cropped Stack",
                ""
            )
            if not output_dir:
                return
            
            # Crop stack
            cropped_stack = self.image_processor.crop_stack_to_roi(
                frame_data, current_roi, apply_mask=(current_roi.shape == ROIShape.ELLIPSE)
            )
            
            # Save cropped frames
            from PIL import Image
            output_path = Path(output_dir)
            original_files = self.stack_handler.files
            
            for i, cropped_frame in enumerate(cropped_stack):
                # Generate output filename
                if i < len(original_files):
                    original_name = Path(original_files[i]).stem
                    output_file = output_path / f"{original_name}_cropped.tif"
                else:
                    output_file = output_path / f"frame_{i:04d}_cropped.tif"
                
                # Convert frame to uint8 if needed
                if cropped_frame.dtype != np.uint8:
                    # Normalize to 0-255 range
                    frame_min = np.min(cropped_frame)
                    frame_max = np.max(cropped_frame)
                    
                    if frame_max > frame_min:
                        # Scale to 0-255
                        normalized = (cropped_frame - frame_min) / (frame_max - frame_min)
                        cropped_frame = (normalized * 255).astype(np.uint8)
                    else:
                        # Constant frame - convert to uint8 with proper scaling
                        if np.issubdtype(cropped_frame.dtype, np.floating):
                            # For float dtypes, scale to 0-255 range
                            uint8_value = np.clip(np.round(frame_min * 255), 0, 255).astype(np.uint8)
                        else:
                            # For integer dtypes, just clip to 0-255
                            uint8_value = np.clip(frame_min, 0, 255).astype(np.uint8)
                        cropped_frame = np.full_like(cropped_frame, uint8_value, dtype=np.uint8)
                
                # Save frame
                img = Image.fromarray(cropped_frame)
                img.save(str(output_file))
            
            QMessageBox.information(
                self,
                "Cropping Complete",
                f"Cropped {len(cropped_stack)} frames to {output_dir}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Cropping Error",
                f"Failed to crop image stack:\n{str(e)}"
            )
    
    def _align_images(self) -> None:
        """Align images in the stack."""
        # Check if images are loaded
        num_frames = self.stack_handler.get_image_count()
        if num_frames == 0:
            QMessageBox.warning(
                self,
                "No Images",
                "No image stack loaded. Please load an image stack first."
            )
            return
        
        if num_frames < 2:
            QMessageBox.warning(
                self,
                "Not Enough Images",
                "At least 2 images are required for alignment."
            )
            return
        
        # Show alignment dialog
        dialog = AlignmentDialog(self, num_frames)
        if dialog.exec() != QDialog.Accepted:
            return
        
        params = dialog.get_parameters()
        
        # Show progress dialog (we align num_frames - 1 frames, skipping reference)
        frames_to_align = num_frames - 1
        progress_dialog = AlignmentProgressDialog(self, frames_to_align)
        progress_dialog.show()
        QApplication.processEvents()
        
        try:
            # Load all frames
            progress_dialog.update_progress(0, frames_to_align, "Loading image stack...")
            QApplication.processEvents()
            
            frame_data = self.stack_handler.get_all_frames_as_array()
            if frame_data is None:
                progress_dialog.close()
                QMessageBox.warning(
                    self,
                    "No Image Data",
                    "Failed to load image stack."
                )
                return
            
            # Progress callback - returns False if cancelled
            def progress_callback(completed: int, total: int, message: str) -> bool:
                progress_dialog.update_progress(completed, total, message)
                QApplication.processEvents()
                # Return False if cancelled (to stop alignment), True to continue
                return not progress_dialog.is_cancelled()
            
            # Perform alignment
            progress_dialog.update_progress(0, frames_to_align, "Starting alignment...")
            QApplication.processEvents()
            
            aligned_stack, transformation_matrices, confidence_scores = (
                self.image_processor.align_image_stack(
                    frame_data,
                    reference_index=params["reference_index"],
                    method=params["method"],
                    warp_mode=params["warp_mode"],
                    progress_callback=progress_callback
                )
            )
            
            # Check if alignment was cancelled
            if progress_dialog.is_cancelled():
                progress_dialog.close()
                QMessageBox.information(
                    self,
                    "Alignment Cancelled",
                    "Image alignment was cancelled. No changes were saved."
                )
                return
            
            # Check alignment quality
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            low_confidence_frames = [
                i for i, conf in enumerate(confidence_scores) if conf < 0.5
            ]
            
            progress_dialog.close()
            
            # Show alignment results
            result_message = (
                f"Alignment complete!\n\n"
                f"Average confidence: {avg_confidence:.2%}\n"
                f"Frames with low confidence (<50%): {len(low_confidence_frames)}"
            )
            
            if low_confidence_frames:
                result_message += f"\n\nLow confidence frames: {low_confidence_frames[:10]}"
                if len(low_confidence_frames) > 10:
                    result_message += f" (+{len(low_confidence_frames) - 10} more)"
            
            reply = QMessageBox.question(
                self,
                "Alignment Complete",
                result_message + "\n\nWould you like to save the aligned images?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # Ask user for output directory
                output_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Select Output Directory for Aligned Stack",
                    ""
                )
                if not output_dir:
                    return
                
                # Save aligned images
                self._save_aligned_stack(
                    aligned_stack,
                    transformation_matrices,
                    confidence_scores,
                    output_dir,
                    params
                )
                
                # Ask if user wants to load aligned images
                load_reply = QMessageBox.question(
                    self,
                    "Load Aligned Images?",
                    "Would you like to load the aligned images into the viewer?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if load_reply == QMessageBox.Yes:
                    self.viewer.set_stack(output_dir)
            
        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self,
                "Alignment Error",
                f"Failed to align images:\n{str(e)}"
            )
    
    def _save_aligned_stack(
        self,
        aligned_stack: np.ndarray,
        transformation_matrices: list,
        confidence_scores: list,
        output_dir: str,
        params: dict
    ) -> None:
        """Save aligned image stack to disk."""
        from PIL import Image
        import json
        
        output_path = Path(output_dir)
        
        # Save aligned images
        for i, aligned_frame in enumerate(aligned_stack):
            # Generate output filename
            if i < len(self.stack_handler.files):
                original_name = Path(self.stack_handler.files[i]).stem
                output_file = output_path / f"{original_name}_aligned.tif"
            else:
                output_file = output_path / f"frame_{i:04d}_aligned.tif"
            
            # Convert frame to appropriate format for saving
            if aligned_frame.dtype != np.uint8 and aligned_frame.dtype != np.uint16:
                # Normalize to uint16 for better precision
                frame_min = np.min(aligned_frame)
                frame_max = np.max(aligned_frame)
                
                if frame_max > frame_min:
                    # Scale to 0-65535 range
                    normalized = (aligned_frame - frame_min) / (frame_max - frame_min)
                    aligned_frame = (normalized * 65535).astype(np.uint16)
                else:
                    aligned_frame = np.full_like(aligned_frame, frame_min, dtype=np.uint16)
            
            # Save frame
            img = Image.fromarray(aligned_frame)
            img.save(str(output_file))
        
        # Save transformation matrices and confidence scores as JSON
        transform_data = {
            "reference_index": params.get("reference_index", 0),
            "method": params.get("method", "ecc"),
            "warp_mode": params.get("warp_mode", "euclidean"),
            "num_frames": len(aligned_stack),
            "transformations": [
                {
                    "frame_index": i,
                    "matrix": matrix.tolist() if matrix is not None else None,
                    "confidence": float(conf)
                }
                for i, (matrix, conf) in enumerate(zip(transformation_matrices, confidence_scores))
            ]
        }
        
        transform_file = output_path / "alignment_transforms.json"
        with open(transform_file, 'w') as f:
            json.dump(transform_data, f, indent=2)
        
        QMessageBox.information(
            self,
            "Save Complete",
            f"Aligned {len(aligned_stack)} images saved to {output_dir}\n"
            f"Transformation matrices saved to {transform_file.name}"
        )

    def _export_experiment(self) -> None:
        """Export the current experiment to a .nexp file."""
        try:
            # Flush any pending display settings before exporting
            self._flush_pending_display_settings()
            
            # Save current ROI to experiment before exporting
            current_roi = self.viewer.get_current_roi()
            if current_roi is not None:
                self.experiment.roi = current_roi.to_dict()
            
            # Capture current display settings before exporting
            self._capture_display_settings()
            
            # Get export location
            default_name = f"{self.experiment.name}_export.nexp"
            if self.current_experiment_path:
                # Suggest a location near the original file
                original_path = Path(self.current_experiment_path)
                default_name = str(original_path.parent / f"{self.experiment.name}_export.nexp")
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Experiment",
                default_name,
                "Neurolight Experiment (*.nexp);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Ensure .nexp extension
            if not file_path.endswith(".nexp"):
                file_path += ".nexp"
            
            # Export experiment data using the manager's save method
            # This ensures the file format matches the native .nexp format
            if self.manager.save_experiment(self.experiment, file_path):
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Experiment exported to:\n{file_path}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Export Failed",
                    "Failed to export experiment."
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export experiment:\n{str(e)}"
            )
