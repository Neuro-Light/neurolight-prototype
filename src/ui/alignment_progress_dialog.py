from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
)


class AlignmentProgressDialog(QDialog):
    """Dialog showing progress during image alignment."""

    def __init__(self, parent=None, total_frames: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Aligning Images...")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(200)

        layout = QVBoxLayout(self)

        # Status label
        self.status_label = QLabel("Initializing alignment...")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(total_frames)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)

        self._cancelled = False

    def update_progress(self, completed: int, total: int, message: str):
        """Update progress bar and status message."""
        self.progress_bar.setValue(completed)
        self.progress_bar.setMaximum(total)
        self.status_label.setText(message)
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def is_cancelled(self) -> bool:
        """Check if user cancelled the operation."""
        return self._cancelled

    def reject(self) -> None:
        """Handle cancel button."""
        self._cancelled = True
        self.status_label.setText("Cancelling...")
        self.log_text.append("Cancellation requested by user.")
        super().reject()
