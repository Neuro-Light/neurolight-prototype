from PySide6.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QLabel

from ui.roi_intensity_plot import ROIIntensityPlotWidget
from ui.neuron_detection_widget import NeuronDetectionWidget
from ui.neuron_trajectory_plot import NeuronTrajectoryPlotWidget


class AnalysisPanel(QTabWidget):
    def __init__(self) -> None:
        super().__init__()
        self.roi_plot_widget = ROIIntensityPlotWidget()
        self.neuron_detection_widget = NeuronDetectionWidget()
        self.neuron_trajectory_plot_widget = NeuronTrajectoryPlotWidget()
        self._add_tab("Statistics")
        self._add_tab("Graphs", self.neuron_trajectory_plot_widget)
        self._add_tab("ROI Intensity", self.roi_plot_widget)
        self._add_tab("Detection", self.neuron_detection_widget)

    def _add_tab(self, title: str, widget: QWidget | None = None) -> None:
        if widget is None:
            w = QWidget()
            l = QVBoxLayout(w)
            l.addWidget(QLabel(f"{title} (placeholder)"))
            self.addTab(w, title)
        else:
            self.addTab(widget, title)

    def get_roi_plot_widget(self) -> ROIIntensityPlotWidget:
        """Get the ROI intensity plot widget."""
        return self.roi_plot_widget

    def get_neuron_detection_widget(self) -> NeuronDetectionWidget:
        """Get the neuron detection widget."""
        return self.neuron_detection_widget

    def get_neuron_trajectory_plot_widget(self) -> NeuronTrajectoryPlotWidget:
        """Get the neuron trajectory plot widget."""
        return self.neuron_trajectory_plot_widget
