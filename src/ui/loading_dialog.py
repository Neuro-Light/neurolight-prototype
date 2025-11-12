from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
)
from PySide6.QtCore import Qt


class LoadingDialog(QDialog):
    """Dialog showing progress during experiment loading."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading Experiment...")
        self.setModal(False)  # Non-modal so async operations can continue
        self.setMinimumWidth(400)
        self.setMinimumHeight(150)
        
        # Make it stay on top and visible
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Status label
        self.status_label = QLabel("Loading experiment data...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress bar (indeterminate)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)
        
        # Additional info label
        self.info_label = QLabel("Please wait...")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
    
    def update_status(self, message: str, info: str = "") -> None:
        """Update status message and optional info text."""
        self.status_label.setText(message)
        if info:
            self.info_label.setText(info)
        # Process events to update the UI
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
    
    def close_dialog(self) -> None:
        """Close the loading dialog."""
        self.accept()

