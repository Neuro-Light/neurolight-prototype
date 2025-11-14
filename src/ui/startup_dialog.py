from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QMenu,
)

from core.experiment_manager import ExperimentManager, Experiment


EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


class NewExperimentDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Neurolight - New Experiment")
        self.setModal(True)
        self.setMinimumWidth(500)

        self.name_edit = QLineEdit()
        self.pi_edit = QLineEdit()
        self.desc_edit = QPlainTextEdit()
        self.date_edit = QLineEdit(datetime.utcnow().strftime("%Y-%m-%d"))

        self.path_edit = QLineEdit(str(EXPERIMENTS_DIR))
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse_btn)

        form = QFormLayout()
        form.addRow("Experiment Name*", self.name_edit)
        form.addRow("Principal Investigator", self.pi_edit)
        form.addRow("Description", self.desc_edit)
        form.addRow("Date", self.date_edit)

        container = QVBoxLayout()
        container.addLayout(form)

        path_container = QVBoxLayout()
        path_container.addWidget(QLabel("Save Location"))
        path_container.addLayout(path_row)
        container.addLayout(path_container)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.button(QDialogButtonBox.Ok).setText("Create")
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self._accept)
        container.addWidget(buttons)

        self.setLayout(container)

        self.output_path: Optional[str] = None
        self.metadata: dict = {}

    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select Save Location", self.path_edit.text()
        )
        if directory:
            self.path_edit.setText(directory)

    def _accept(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            self.name_edit.setFocus()
            return
        base_dir = Path(self.path_edit.text().strip() or str(EXPERIMENTS_DIR))
        base_dir.mkdir(parents=True, exist_ok=True)
        file_path = base_dir / f"{name}.nexp"
        if file_path.exists():
            self.name_edit.setFocus()
            return
        self.output_path = str(file_path)
        self.metadata = {
            "name": name,
            "description": self.desc_edit.toPlainText().strip(),
            "principal_investigator": self.pi_edit.text().strip(),
            "created_date": datetime.utcnow(),
        }
        self.accept()


class StartupDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Neurolight - Experiment Manager")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.experiment: Optional[Experiment] = None
        self.experiment_path: Optional[str] = None
        self.manager = ExperimentManager()

        title = QLabel("Neurolight - Experiment Manager")
        title.setAlignment(Qt.AlignCenter)

        new_btn = QPushButton("Start New Experiment")
        load_btn = QPushButton("Load Existing Experiment")

        new_btn.clicked.connect(self._start_new)
        load_btn.clicked.connect(self._load_existing)

        self.recent_list = QListWidget()
        self.recent_list.itemDoubleClicked.connect(self._open_recent)
        self.recent_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.recent_list.customContextMenuRequested.connect(self._show_context_menu)
        self._refresh_recent()

        # Buttons for selected experiment actions
        buttons_layout = QHBoxLayout()
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        self.delete_btn.setEnabled(False)
        self.export_btn = QPushButton("Export Selected")
        self.export_btn.clicked.connect(self._export_selected)
        self.export_btn.setEnabled(False)
        buttons_layout.addWidget(self.delete_btn)
        buttons_layout.addWidget(self.export_btn)

        # Connect selection changed to enable/disable buttons
        self.recent_list.itemSelectionChanged.connect(self._on_selection_changed)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(new_btn)
        layout.addWidget(load_btn)
        layout.addWidget(QLabel("Recent Experiments"))
        layout.addWidget(self.recent_list)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def _refresh_recent(self) -> None:
        self.recent_list.clear()
        for item in self.manager.get_recent_experiments():
            label = f"{item.get('name', '')} — {item.get('path', '')}"
            list_item = QListWidgetItem(label)
            list_item.setData(Qt.UserRole, item.get("path"))
            self.recent_list.addItem(list_item)
        # Clear selection after refresh to disable buttons
        self.recent_list.clearSelection()

    def _start_new(self) -> None:
        dlg = NewExperimentDialog(self)
        if dlg.exec() == QDialog.Accepted and dlg.output_path:
            exp = self.manager.create_new_experiment(dlg.metadata)
            try:
                self.manager.save_experiment(exp, dlg.output_path)
                # Refresh recent list to show the new experiment
                self._refresh_recent()
            except Exception:
                return
            self.experiment = exp
            self.experiment_path = dlg.output_path
            self.accept()

    def _load_existing(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Experiment",
            str(EXPERIMENTS_DIR),
            "Neurolight Experiment (*.nexp)",
        )
        if not file_path:
            return
        try:
            self.experiment = self.manager.load_experiment(file_path)
            self.experiment_path = file_path
            # Refresh recent list to update the order
            self._refresh_recent()
            self.accept()
        except Exception as e:
            QMessageBox.warning(
                self,
                "Load Failed",
                f"Failed to load experiment:\n{file_path}\n\n{str(e)}",
            )
            self._refresh_recent()

    def _open_recent(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if not path:
            return
        try:
            self.experiment = self.manager.load_experiment(path)
            self.experiment_path = path
            self.accept()
        except Exception:
            QMessageBox.warning(
                self,
                "Load Failed",
                f"Failed to load experiment:\n{path}\n\nThe file may have been deleted or is corrupted.",
            )
            self._refresh_recent()

    def _on_selection_changed(self) -> None:
        """Enable/disable delete and export buttons based on selection."""
        has_selection = len(self.recent_list.selectedItems()) > 0
        self.delete_btn.setEnabled(has_selection)
        self.export_btn.setEnabled(has_selection)

    def _show_context_menu(self, position) -> None:
        """Show context menu for right-click on experiment list."""
        item = self.recent_list.itemAt(position)
        if item is None:
            return

        menu = QMenu(self)
        open_action = menu.addAction("Open")
        delete_action = menu.addAction("Delete from List")
        delete_file_action = menu.addAction("Delete File and Remove from List")
        export_action = menu.addAction("Export")

        action = menu.exec(self.recent_list.mapToGlobal(position))

        if action == open_action:
            self._open_recent(item)
        elif action == delete_action:
            self._delete_experiment(item, delete_file=False)
        elif action == delete_file_action:
            self._delete_experiment(item, delete_file=True)
        elif action == export_action:
            self._export_experiment(item)

    def _delete_selected(self) -> None:
        """Delete the selected experiment from the list."""
        selected_items = self.recent_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        reply = QMessageBox.question(
            self,
            "Delete Experiment",
            f"Remove '{item.text()}' from recent experiments list?\n\n"
            "This will not delete the experiment file from disk.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self._delete_experiment(item, delete_file=False)

    def _delete_experiment(
        self, item: QListWidgetItem, delete_file: bool = False
    ) -> None:
        """Delete an experiment from the list and optionally from disk."""
        path = item.data(Qt.UserRole)
        if not path:
            return

        if delete_file:
            reply = QMessageBox.warning(
                self,
                "Delete Experiment File",
                f"Are you sure you want to permanently delete:\n{path}\n\n"
                "This action cannot be undone!",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        try:
            if self.manager.delete_experiment(path, delete_file=delete_file):
                self._refresh_recent()
                if delete_file:
                    QMessageBox.information(
                        self, "Deleted", "Experiment file and entry have been deleted."
                    )
                else:
                    QMessageBox.information(
                        self,
                        "Removed",
                        "Experiment has been removed from the recent list.",
                    )
            else:
                QMessageBox.warning(
                    self, "Delete Failed", "Failed to delete experiment."
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while deleting:\n{str(e)}"
            )

    def _export_selected(self) -> None:
        """Export the selected experiment."""
        selected_items = self.recent_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        self._export_experiment(item)

    def _export_experiment(self, item: QListWidgetItem) -> None:
        """Export an experiment to a .nexp file."""
        path = item.data(Qt.UserRole)
        if not path:
            return

        try:
            # Load the experiment
            experiment = self.manager.load_experiment(path)

            # Get export location
            default_name = f"{experiment.name}_export.nexp"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Experiment",
                default_name,
                "Neurolight Experiment (*.nexp);;All Files (*)",
            )

            if not file_path:
                return

            # Ensure .nexp extension
            if not file_path.endswith(".nexp"):
                file_path += ".nexp"

            # Export experiment data using the manager's save method
            # This ensures the file format matches the native .nexp format
            if self.manager.save_experiment(experiment, file_path):
                QMessageBox.information(
                    self, "Export Successful", f"Experiment exported to:\n{file_path}"
                )
            else:
                QMessageBox.warning(
                    self, "Export Failed", "Failed to export experiment."
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Failed", f"Failed to export experiment:\n{str(e)}"
            )
