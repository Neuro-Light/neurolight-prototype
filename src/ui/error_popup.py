from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QPlainTextEdit,
    QWidget,
)


class ErrorPopup(QDialog):
    """Reusable error/warning/info popup used across the application.

    Usage:
        ErrorPopup.show_error(parent, "Title", "Friendly message", details=str(exc))

    The popup has consistent styling, a toggleable details area, and a copy-to-clipboard
    button to help users report issues.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setMinimumWidth(420)

        self._main_layout = QVBoxLayout(self)

        self._title_label = QLabel()
        self._title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._main_layout.addWidget(self._title_label)

        self._message_label = QLabel()
        self._message_label.setWordWrap(True)
        self._main_layout.addWidget(self._message_label)

        self._details = QPlainTextEdit()
        self._details.setReadOnly(True)
        self._details.setVisible(False)
        self._main_layout.addWidget(self._details)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._toggle_btn = QPushButton("Show details…")
        self._toggle_btn.clicked.connect(self._toggle_details)
        btn_row.addWidget(self._toggle_btn)

        self._copy_btn = QPushButton("Copy")
        self._copy_btn.clicked.connect(self._copy_details)
        btn_row.addWidget(self._copy_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._close_btn)

        self._main_layout.addLayout(btn_row)

    def _toggle_details(self) -> None:
        visible = not self._details.isVisible()
        self._details.setVisible(visible)
        self._toggle_btn.setText("Hide details" if visible else "Show details…")

    def _copy_details(self) -> None:
        txt = self._details.toPlainText() or self._message_label.text()
        try:
            # Import here to avoid top-level dependency for tests that patch QApplication
            from PySide6.QtWidgets import QApplication as _QApp
            app = _QApp.instance()
            if app:
                app.clipboard().setText(txt)
        except Exception:
            # Clipboard not available or running headless in tests
            pass

    def set_title(self, title: str) -> None:
        self._title_label.setText(title)

    def set_message(self, message: str) -> None:
        self._message_label.setText(message)

    def set_details(self, details: Optional[str]) -> None:
        if details:
            self._details.setPlainText(details)
            self._toggle_btn.setVisible(True)
            self._copy_btn.setVisible(True)
        else:
            self._details.setPlainText("")
            self._toggle_btn.setVisible(False)
            self._copy_btn.setVisible(False)

    # Convenience static methods -------------------------------------------------
    @staticmethod
    def _show(parent: Optional[QWidget], title: str, message: str, details: Optional[str]) -> None:
        dlg = ErrorPopup(parent)
        dlg.setWindowTitle(title)
        dlg.set_title(title)
        dlg.set_message(message)
        dlg.set_details(details)
        dlg.exec()

    @staticmethod
    def show_error(parent: Optional[QWidget], title: str, message: str, details: Optional[str] = None) -> None:
        ErrorPopup._show(parent, title, message, details)

    @staticmethod
    def show_warning(parent: Optional[QWidget], title: str, message: str, details: Optional[str] = None) -> None:
        ErrorPopup._show(parent, title, message, details)

    @staticmethod
    def show_info(parent: Optional[QWidget], title: str, message: str, details: Optional[str] = None) -> None:
        ErrorPopup._show(parent, title, message, details)
