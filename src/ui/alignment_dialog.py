from __future__ import annotations

from typing import Optional
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
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
        self.method = "ecc"
        self.warp_mode = "euclidean"
        
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
        
        # Alignment method selection
        method_group = QGroupBox("Alignment Method")
        method_layout = QFormLayout()
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["ECC (Enhanced Correlation Coefficient)", "ORB (Feature-based)"])
        self.method_combo.setCurrentIndex(0)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        
        method_layout.addRow("Method:", self.method_combo)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # Warp mode selection (only for ECC)
        warp_group = QGroupBox("Transformation Type")
        warp_layout = QFormLayout()
        
        self.warp_combo = QComboBox()
        self.warp_combo.addItems([
            "Translation",
            "Euclidean (Translation + Rotation)",
            "Affine (Translation + Rotation + Scaling)",
            "Homography (Full Perspective)"
        ])
        self.warp_combo.setCurrentIndex(1)  # Default to Euclidean
        self.warp_combo.currentIndexChanged.connect(self._on_warp_mode_changed)
        
        warp_layout.addRow("Transformation:", self.warp_combo)
        warp_group.setLayout(warp_layout)
        layout.addWidget(warp_group)
        
        # Info label
        info_label = QLabel(
            "This will align all images in the stack to the reference frame.\n"
            "The original images will be preserved, and aligned copies will be created."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self._update_warp_mode_visibility()
    
    def _on_reference_changed(self, value: int):
        self.reference_index = value
    
    def _on_method_changed(self, index: int):
        self.method = "ecc" if index == 0 else "orb"
        self._update_warp_mode_visibility()
    
    def _on_warp_mode_changed(self, index: int):
        warp_modes = ["translation", "euclidean", "affine", "homography"]
        self.warp_mode = warp_modes[index]
    
    def _update_warp_mode_visibility(self):
        # Warp mode only applies to ECC method
        warp_group = self.findChild(QGroupBox, "Transformation Type")
        if warp_group:
            warp_group.setEnabled(self.method == "ecc")
    
    def get_parameters(self) -> dict:
        """Get alignment parameters."""
        return {
            "reference_index": self.reference_index,
            "method": self.method,
            "warp_mode": self.warp_mode
        }
    
    def accept(self) -> None:
        """Validate and accept the dialog."""
        if self.num_frames == 0:
            QMessageBox.warning(
                self,
                "No Images",
                "No images loaded. Please load an image stack first."
            )
            return
        
        if self.reference_index < 0 or self.reference_index >= self.num_frames:
            QMessageBox.warning(
                self,
                "Invalid Reference",
                f"Reference frame index must be between 0 and {self.num_frames - 1}."
            )
            return
        
        super().accept()

