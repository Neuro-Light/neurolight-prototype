from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QSpinBox,
    QComboBox,
    QDialogButtonBox,
    QGroupBox,
    QFormLayout,
    QMessageBox,
)


class AlignmentDialog(QDialog):
    """Dialog for configuring image alignment parameters."""

    def __init__(self, parent=None, num_frames: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Align Images")
        self.setModal(True)
        self.setMinimumWidth(400)

        self.num_frames = num_frames
        self.reference_index = 0
        self.transform_type = "rigid_body"
        self.reference_strategy = "first"

        layout = QVBoxLayout(self)

        # Reference frame selection
        ref_group = QGroupBox("Reference Frame")
        ref_layout = QFormLayout()

        self.reference_spinbox = QSpinBox()
        self.reference_spinbox.setMinimum(0)
        self.reference_spinbox.setMaximum(max(0, num_frames - 1))
        self.reference_spinbox.setValue(0)
        if num_frames > 0:
            self.reference_spinbox.setSuffix(f" (of {num_frames} frames)")
        self.reference_spinbox.valueChanged.connect(self._on_reference_changed)

        ref_layout.addRow("Reference Frame Index:", self.reference_spinbox)
        ref_group.setLayout(ref_layout)
        layout.addWidget(ref_group)

        # Transformation type selection
        transform_group = QGroupBox("Transformation Type")
        transform_layout = QFormLayout()

        self.transform_combo = QComboBox()
        self.transform_combo.addItems(
            [
                "Rigid Body (Translation + Rotation)",
                "Translation",
                "Scaled Rotation",
                "Affine",
                "Bilinear",
            ]
        )
        self.transform_combo.setCurrentIndex(0)  # Default to Rigid Body
        self.transform_combo.currentIndexChanged.connect(self._on_transform_changed)

        transform_layout.addRow("Transformation:", self.transform_combo)
        transform_group.setLayout(transform_layout)
        layout.addWidget(transform_group)

        # Reference strategy selection
        reference_group = QGroupBox("Reference Strategy")
        reference_layout = QFormLayout()

        self.reference_combo = QComboBox()
        self.reference_combo.addItems(["First Frame", "Previous Frame", "Mean of All Frames"])
        self.reference_combo.setCurrentIndex(0)  # Default to First Frame
        self.reference_combo.currentIndexChanged.connect(self._on_reference_strategy_changed)

        reference_layout.addRow("Reference:", self.reference_combo)
        reference_group.setLayout(reference_layout)
        layout.addWidget(reference_group)

        # Info label
        info_label = QLabel(
            "This will align all images in the stack using PyStackReg (ImageJ StackReg).\n"
            "The original images will be preserved, and aligned copies will be created."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_reference_changed(self, value: int):
        self.reference_index = value

    def _on_transform_changed(self, index: int):
        transform_types = [
            "rigid_body",
            "translation",
            "scaled_rotation",
            "affine",
            "bilinear",
        ]
        self.transform_type = transform_types[index]

    def _on_reference_strategy_changed(self, index: int):
        reference_strategies = ["first", "previous", "mean"]
        self.reference_strategy = reference_strategies[index]

    def get_parameters(self) -> dict:
        """Get alignment parameters."""
        return {
            "reference_index": self.reference_index,
            "transform_type": self.transform_type,
            "reference": self.reference_strategy,
        }

    def accept(self) -> None:
        """Validate and accept the dialog."""
        if self.num_frames == 0:
            QMessageBox.warning(
                self, "No Images", "No images loaded. Please load an image stack first."
            )
            return

        if self.reference_index < 0 or self.reference_index >= self.num_frames:
            QMessageBox.warning(
                self,
                "Invalid Reference",
                f"Reference frame index must be between 0 and {self.num_frames - 1}.",
            )
            return

        super().accept()
