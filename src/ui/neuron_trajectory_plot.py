from __future__ import annotations

from typing import Optional

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
    QCheckBox,
    QSpinBox,
    QGroupBox,
    QFormLayout,
)
from PySide6.QtCore import Qt


class NeuronTrajectoryPlotWidget(QWidget):
    """Widget for plotting individual neuron intensity trajectories over time."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.neuron_trajectories: Optional[np.ndarray] = None
        self.quality_mask: Optional[np.ndarray] = None
        self.neuron_locations: Optional[np.ndarray] = None
        
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("No neuron trajectories available. Run detection first.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # Display options group
        options_group = QGroupBox("Display Options")
        options_layout = QFormLayout()
        
        # Show good neurons checkbox
        self.show_good_checkbox = QCheckBox()
        self.show_good_checkbox.setChecked(True)
        self.show_good_checkbox.stateChanged.connect(self._update_plot)
        options_layout.addRow("Show Good Neurons:", self.show_good_checkbox)
        
        # Show bad neurons checkbox
        self.show_bad_checkbox = QCheckBox()
        self.show_bad_checkbox.setChecked(False)
        self.show_bad_checkbox.stateChanged.connect(self._update_plot)
        options_layout.addRow("Show Bad Neurons:", self.show_bad_checkbox)
        
        # Max neurons to display
        self.max_neurons_spin = QSpinBox()
        self.max_neurons_spin.setRange(1, 1000)
        self.max_neurons_spin.setValue(50)
        self.max_neurons_spin.setToolTip("Maximum number of neurons to display (for performance)")
        self.max_neurons_spin.valueChanged.connect(self._update_plot)
        options_layout.addRow("Max Neurons to Display:", self.max_neurons_spin)
        
        # Show average checkbox
        self.show_average_checkbox = QCheckBox()
        self.show_average_checkbox.setChecked(True)
        self.show_average_checkbox.stateChanged.connect(self._update_plot)
        options_layout.addRow("Show Average:", self.show_average_checkbox)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Export Plot Data (CSV)")
        self.export_btn.clicked.connect(self._export_to_csv)
        self.export_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_btn)
        
        layout.addLayout(buttons_layout)

    def plot_trajectories(
        self,
        neuron_trajectories: np.ndarray,
        quality_mask: Optional[np.ndarray] = None,
        neuron_locations: Optional[np.ndarray] = None
    ) -> None:
        """
        Plot intensity trajectories for each detected neuron.
        
        Args:
            neuron_trajectories: 2D array (neurons x frames) of intensity time-series
            quality_mask: Boolean array indicating good (True) vs bad (False) neurons
            neuron_locations: Array of (y, x) coordinates for neurons (optional, for labeling)
        """
        self.neuron_trajectories = neuron_trajectories
        self.quality_mask = quality_mask
        self.neuron_locations = neuron_locations
        
        if neuron_trajectories is None or len(neuron_trajectories) == 0:
            self.status_label.setText("No neuron trajectories to display.")
            self.export_btn.setEnabled(False)
            return
        
        num_neurons, num_frames = neuron_trajectories.shape
        
        # Update status
        if quality_mask is not None:
            num_good = np.sum(quality_mask)
            num_bad = num_neurons - num_good
            self.status_label.setText(
                f"Displaying {num_neurons} neuron trajectories "
                f"({num_good} good, {num_bad} bad) across {num_frames} frames"
            )
        else:
            self.status_label.setText(
                f"Displaying {num_neurons} neuron trajectories across {num_frames} frames"
            )
        
        # Enable export button
        self.export_btn.setEnabled(True)
        
        # Update plot
        self._update_plot()

    def _update_plot(self) -> None:
        """Update the trajectory plot based on current display options."""
        if self.neuron_trajectories is None or len(self.neuron_trajectories) == 0:
            return
        
        # Clear previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        num_neurons, num_frames = self.neuron_trajectories.shape
        frames = np.arange(num_frames)
        
        # Get display options
        show_good = self.show_good_checkbox.isChecked()
        show_bad = self.show_bad_checkbox.isChecked()
        max_neurons = self.max_neurons_spin.value()
        show_average = self.show_average_checkbox.isChecked()
        
        # Determine which neurons to display
        neurons_to_plot = []
        if self.quality_mask is not None:
            if show_good:
                good_indices = np.where(self.quality_mask)[0]
                neurons_to_plot.extend(good_indices[:max_neurons].tolist())
            if show_bad:
                bad_indices = np.where(~self.quality_mask)[0]
                neurons_to_plot.extend(bad_indices[:max_neurons].tolist())
        else:
            # No quality mask, show all neurons
            neurons_to_plot = list(range(min(num_neurons, max_neurons)))
        
        # Limit total number of neurons to display
        if len(neurons_to_plot) > max_neurons:
            neurons_to_plot = neurons_to_plot[:max_neurons]
        
        # Plot individual neuron trajectories
        if self.quality_mask is not None:
            # Plot good neurons in green
            good_to_plot = [i for i in neurons_to_plot if self.quality_mask[i]]
            if good_to_plot and show_good:
                for idx in good_to_plot:
                    ax.plot(
                        frames,
                        self.neuron_trajectories[idx],
                        color='green',
                        alpha=0.3,
                        linewidth=0.8,
                        label='Good Neurons' if idx == good_to_plot[0] else ''
                    )
            
            # Plot bad neurons in red
            bad_to_plot = [i for i in neurons_to_plot if not self.quality_mask[i]]
            if bad_to_plot and show_bad:
                for idx in bad_to_plot:
                    ax.plot(
                        frames,
                        self.neuron_trajectories[idx],
                        color='red',
                        alpha=0.3,
                        linewidth=0.8,
                        label='Bad Neurons' if idx == bad_to_plot[0] else ''
                    )
        else:
            # No quality mask, plot all in blue
            for idx in neurons_to_plot:
                ax.plot(
                    frames,
                    self.neuron_trajectories[idx],
                    color='blue',
                    alpha=0.3,
                    linewidth=0.8,
                    label='Neurons' if idx == neurons_to_plot[0] else ''
                )
        
        # Plot average trajectory
        if show_average:
            if self.quality_mask is not None and show_good:
                # Average of good neurons
                good_indices = np.where(self.quality_mask)[0]
                if len(good_indices) > 0:
                    avg_trajectory = np.mean(self.neuron_trajectories[good_indices], axis=0)
                    ax.plot(
                        frames,
                        avg_trajectory,
                        color='darkgreen',
                        linewidth=2.5,
                        label='Average (Good Neurons)'
                    )
            else:
                # Average of all displayed neurons
                if len(neurons_to_plot) > 0:
                    avg_trajectory = np.mean(self.neuron_trajectories[neurons_to_plot], axis=0)
                    ax.plot(
                        frames,
                        avg_trajectory,
                        color='black',
                        linewidth=2.5,
                        label='Average'
                    )
        
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.set_title('Neuron Intensity Trajectories Over Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Refresh canvas
        self.canvas.draw()

    def _export_to_csv(self) -> None:
        """Export trajectory data to CSV file."""
        if self.neuron_trajectories is None or len(self.neuron_trajectories) == 0:
            QMessageBox.warning(self, "No Data", "No trajectory data to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Trajectory Data",
            "neuron_trajectories.csv",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            num_neurons, num_frames = self.neuron_trajectories.shape
            
            # Create CSV with frame numbers and all neuron trajectories
            # Format: Frame, Neuron_0, Neuron_1, ..., Neuron_N
            header_parts = ["Frame"]
            if self.quality_mask is not None:
                for i in range(num_neurons):
                    quality = "Good" if self.quality_mask[i] else "Bad"
                    header_parts.append(f"Neuron_{i}_{quality}")
            else:
                header_parts.extend([f"Neuron_{i}" for i in range(num_neurons)])
            
            header = ",".join(header_parts)
            
            # Create data array
            data = []
            for frame_idx in range(num_frames):
                row = [frame_idx]
                row.extend(self.neuron_trajectories[:, frame_idx])
                data.append(row)
            
            # Format string
            fmt_parts = ["%d"]  # Frame number
            fmt_parts.extend(["%.6f"] * num_neurons)  # Trajectory values
            fmt = ",".join(fmt_parts)
            
            np.savetxt(
                file_path,
                data,
                delimiter=',',
                header=header,
                comments='',
                fmt=fmt
            )
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Trajectory data exported to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export data:\n{str(e)}"
            )

    def clear_plot(self) -> None:
        """Clear the plot and reset state."""
        self.figure.clear()
        self.canvas.draw()
        self.neuron_trajectories = None
        self.quality_mask = None
        self.neuron_locations = None
        self.status_label.setText("No neuron trajectories available. Run detection first.")
        self.export_btn.setEnabled(False)

