from PySide6.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QLabel


class AnalysisPanel(QTabWidget):
    def __init__(self) -> None:
        super().__init__()
        self._add_tab("Statistics")
        self._add_tab("Graphs")
        self._add_tab("Detection")

    def _add_tab(self, title: str) -> None:
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel(f"{title} (placeholder)"))
        self.addTab(w, title)

