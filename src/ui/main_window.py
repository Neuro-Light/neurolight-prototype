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
from utils.file_handler import ImageStackHandler
from ui.image_viewer import ImageViewer
from ui.analysis_panel import AnalysisPanel
from ui.startup_dialog import StartupDialog
from ui.error_popup import ErrorPopup


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

        # Keep references to menus on the instance to avoid PySide GC issues
        self.file_menu = menubar.addMenu("File")
        save_action = QAction("Save Experiment", self)
        save_as_action = QAction("Save Experiment As...", self)
        close_action = QAction("Close Experiment", self)
        exit_action = QAction("Exit Experiment", self)
        open_stack_action = QAction("Open Image Stack", self)
        export_results_action = QAction("Export Results", self)

        # Keep references to the actions on the instance to avoid GC of underlying C++ objects
        self.save_action = save_action
        self.save_as_action = save_as_action
        self.close_action = close_action
        self.exit_action = exit_action
        self.open_stack_action = open_stack_action
        self.export_results_action = export_results_action

        save_action.setShortcut("Ctrl+S")

        save_action.triggered.connect(self._save)
        save_as_action.triggered.connect(self._save_as)
        close_action.triggered.connect(self._close_experiment)
        exit_action.triggered.connect(self._exit_experiment)
        open_stack_action.triggered.connect(self._open_image_stack)

        self.file_menu.addAction(save_action)
        self.file_menu.addAction(save_as_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(open_stack_action)
        self.file_menu.addAction(export_results_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(close_action)
        self.file_menu.addAction(exit_action)

        # Keep a reference to the menubar's actions list
        try:
            self._menubar_actions = menubar.actions()
        except Exception:
            self._menubar_actions = None

        self.edit_menu = menubar.addMenu("Edit")
        self.edit_menu.addAction("Experiment Settings")
        self.tools_menu = menubar.addMenu("Tools")
        self.tools_menu.addAction("Generate GIF")
        self.tools_menu.addAction("Run Analysis")
        self.help_menu = menubar.addMenu("Help")
        self.help_menu.addAction("About")

    def _ensure_qwidget(self, widget_obj):
        """Return a QWidget suitable for adding to layouts/splitters.

        If widget_obj is already a QWidget, return it. Otherwise create a simple
        placeholder QWidget so tests that patch UI classes with mocks don't fail
        when the real Qt APIs expect actual QWidget instances.
        """
        if isinstance(widget_obj, QWidget):
            return widget_obj
        # Create a lightweight container to host a non-QWidget (e.g. a Mock in tests)
        container = QWidget()
        # Attach the original object to the container for test access if needed
        try:
            setattr(container, "_wrapped", widget_obj)
        except Exception:
            pass
        return container

    def _init_layout(self) -> None:
        splitter = QSplitter()

        # Left panel: image viewer
        self.stack_handler = ImageStackHandler()
        self.stack_handler.associate_with_experiment(self.experiment)
        self.viewer = ImageViewer(self.stack_handler)
        # Connect signals if available (mocks may not have connectable signals)
        try:
            self.viewer.stackLoaded.connect(self._on_stack_loaded)
        except Exception:
            pass

        # Right panel: analysis dashboard
        self.analysis = AnalysisPanel()

        # Connect ROI selection to analysis and saving
        try:
            self.viewer.roiSelected.connect(self._on_roi_selected)
            self.viewer.roiSelected.connect(self._save_roi_to_experiment)
        except Exception:
            # If viewer is a test mock it may not support Qt signals/connect
            pass

        # Create data analyzer
        self.data_analyzer = DataAnalyzer(self.experiment)

        # Add widgets to splitter, wrapping non-QWidget test doubles when necessary
        splitter.addWidget(self._ensure_qwidget(self.viewer))
        splitter.addWidget(self._ensure_qwidget(self.analysis))
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

                        def load_roi_and_plot():
                            self.viewer.set_roi(x, y, width, height)
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
        # Save current ROI to experiment before saving
        current_roi = self.viewer.get_current_roi()
        if current_roi is not None:
            x, y, width, height = current_roi
            self.experiment.roi = {"x": x, "y": y, "width": width, "height": height}
        try:
            self.manager.save_experiment(self.experiment, self.current_experiment_path)
            QMessageBox.information(self, "Saved", "Experiment saved successfully.")
        except Exception as e:
            # Use ErrorPopup to provide a consistent, user-friendly error dialog
            ErrorPopup.show_error(self, "Save Error", "Failed to save experiment.", details=str(e))

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
            ErrorPopup.show_error(self, "Save Error", "Failed to save experiment.", details=str(e))

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

        # Save current ROI to experiment before closing
        current_roi = None
        try:
            current_roi = self.viewer.get_current_roi()
        except Exception:
            # Viewer may be a mock without the method; fall back to attribute if present
            current_roi = getattr(self.viewer, "current_roi", None)

        # Safely unpack ROI if it's an iterable of four numbers (x,y,width,height)
        roi_vals = None
        try:
            if current_roi is not None and hasattr(current_roi, "__iter__"):
                x, y, width, height = current_roi
                roi_vals = (x, y, width, height)
        except Exception:
            roi_vals = None

        if roi_vals is not None:
            x, y, width, height = roi_vals
            self.experiment.roi = {"x": x, "y": y, "width": width, "height": height}
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
            # Save current ROI to experiment before exiting
            current_roi = None
            try:
                current_roi = self.viewer.get_current_roi()
            except Exception:
                current_roi = getattr(self.viewer, "current_roi", None)

            roi_vals = None
            try:
                if current_roi is not None and hasattr(current_roi, "__iter__"):
                    x, y, width, height = current_roi
                    roi_vals = (x, y, width, height)
            except Exception:
                roi_vals = None

            if roi_vals is not None:
                x, y, width, height = roi_vals
                self.experiment.roi = {"x": x, "y": y, "width": width, "height": height}
                # Save to file if we have a path
                if self.current_experiment_path:
                    try:
                        self.manager.save_experiment(
                            self.experiment, self.current_experiment_path
                        )
                    except Exception:
                        pass
            QApplication.quit()

    def _on_roi_selected(self, x: int, y: int, width: int, height: int) -> None:
        """Handle ROI selection and extract intensity time series."""
        try:
            # Load all frames as numpy array (reusing Jupyter notebook approach)
            frame_data = self.stack_handler.get_all_frames_as_array()
            if frame_data is None:
                ErrorPopup.show_warning(
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
            ErrorPopup.show_error(self, "Analysis Error", "Failed to analyze ROI.", details=str(e))

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
        self.experiment.roi = {"x": x, "y": y, "width": width, "height": height}
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
        # Save current ROI to experiment before auto-saving
        current_roi = None
        try:
            current_roi = self.viewer.get_current_roi()
        except Exception:
            current_roi = getattr(self.viewer, "current_roi", None)

        roi_vals = None
        try:
            if current_roi is not None and hasattr(current_roi, "__iter__"):
                x, y, width, height = current_roi
                roi_vals = (x, y, width, height)
        except Exception:
            roi_vals = None

        if roi_vals is not None:
            x, y, width, height = roi_vals
            self.experiment.roi = {"x": x, "y": y, "width": width, "height": height}
        try:
            self.manager.save_experiment(self.experiment, self.current_experiment_path)
        except Exception:
            pass
