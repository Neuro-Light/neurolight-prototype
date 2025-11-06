from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
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
        directory = QFileDialog.getExistingDirectory(self, "Select Save Location", self.path_edit.text())
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
        self._refresh_recent()

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(new_btn)
        layout.addWidget(load_btn)
        layout.addWidget(QLabel("Recent Experiments"))
        layout.addWidget(self.recent_list)
        self.setLayout(layout)

    def _refresh_recent(self) -> None:
        self.recent_list.clear()
        for item in self.manager.get_recent_experiments():
            label = f"{item.get('name','')} — {item.get('path','')}"
            list_item = QListWidgetItem(label)
            list_item.setData(Qt.UserRole, item.get("path"))
            self.recent_list.addItem(list_item)

    def _start_new(self) -> None:
        dlg = NewExperimentDialog(self)
        if dlg.exec() == QDialog.Accepted and dlg.output_path:
            exp = self.manager.create_new_experiment(dlg.metadata)
            try:
                self.manager.save_experiment(exp, dlg.output_path)
            except Exception:
                return
            self.experiment = exp
            self.experiment_path = dlg.output_path
            self.accept()

    def _load_existing(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Experiment", str(EXPERIMENTS_DIR), "Neurolight Experiment (*.nexp)")
        if not file_path:
            return
        try:
            self.experiment = self.manager.load_experiment(file_path)
            self.experiment_path = file_path
            self.accept()
        except Exception:
            pass

    def _open_recent(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if not path:
            return
        try:
            self.experiment = self.manager.load_experiment(path)
            self.experiment_path = path
            self.accept()
        except Exception:
            pass
