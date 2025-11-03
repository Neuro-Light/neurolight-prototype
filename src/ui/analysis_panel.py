from PySide6.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QLabel

from ui.roi_intensity_plot import ROIIntensityPlotWidget


class AnalysisPanel(QTabWidget):
    def __init__(self) -> None:
        super().__init__()
        self.roi_plot_widget = ROIIntensityPlotWidget()
        self._add_tab("Statistics")
        self._add_tab("Graphs")
        self._add_tab("ROI Intensity", self.roi_plot_widget)
        self._add_tab("Detection")

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

