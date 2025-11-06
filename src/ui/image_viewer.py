from __future__ import annotations

from collections import OrderedDict
from typing import Optional
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
)

from utils.file_handler import ImageStackHandler


class _LRUCache:
    def __init__(self, capacity: int = 20) -> None:
        self.capacity = capacity
        self.store: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def get(self, key: int) -> Optional[np.ndarray]:
        if key not in self.store:
            return None
        value = self.store.pop(key)
        self.store[key] = value
        return value

    def set(self, key: int, value: np.ndarray) -> None:
        if key in self.store:
            self.store.pop(key)
        elif len(self.store) >= self.capacity:
            self.store.popitem(last=False)
        self.store[key] = value


class ImageViewer(QWidget):
    stackLoaded = Signal(str)
    roiSelected = Signal(int, int, int, int)  # x, y, width, height

    def __init__(self, handler: ImageStackHandler) -> None:
        super().__init__()
        self.handler = handler
        self.index = 0
        self.cache = _LRUCache(20)

        # ROI selection state
        self.roi_selection_mode = False
        self.roi_adjustment_mode = False
        self.roi_start_point = None
        self.roi_end_point = None
        self.current_roi = None  # (x, y, width, height) for ellipse
        self.roi_shape = "ellipse"
        self.dragging_handle = (
            None  # For ROI adjustment (handle name or "move" for moving)
        )
        self.drag_start_pos = None  # Starting position when dragging
        self.handle_size = 8  # Size of adjustment handles

        self.filename_label = QLabel(
            "Load image to see data"
        )  # label for user to see if no image are selected
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setWordWrap(True)
        self.filename_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Create container widget for image display with overlay button
        self.image_container = QWidget()
        self.image_container.setMinimumSize(320, 240)
        image_container_layout = QVBoxLayout(self.image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)

        # Create image label
        self.image_label = QLabel("Drop TIF files or open a folder…")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self._on_mouse_press
        self.image_label.mouseMoveEvent = self._on_mouse_move
        self.image_label.mouseReleaseEvent = self._on_mouse_release

        # Create styled button for opening files (appears when no images loaded)
        self.open_files_btn = QPushButton("📁 Open Image Files", self.image_container)
        self.open_files_btn.setMinimumSize(200, 50)
        self.open_files_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
            QPushButton:pressed {
                background-color: #2A5F8F;
            }
        """)
        self.open_files_btn.clicked.connect(self._on_open_files_clicked)
        self.open_files_btn.hide()  # Hidden initially, shown when no images

        # Add image label to container
        image_container_layout.addWidget(self.image_label)

        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.roi_btn = QPushButton("Select ROI")
        self.adjust_roi_btn = QPushButton("Adjust ROI")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.roi_btn.clicked.connect(self._toggle_roi_mode)
        self.adjust_roi_btn.clicked.connect(self._toggle_adjustment_mode)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)

        nav = QHBoxLayout()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.roi_btn)
        nav.addWidget(self.adjust_roi_btn)

        layout = QVBoxLayout(self)
        # Image container gets most of the space (stretch factor 1)
        layout.addWidget(self.image_container, 1)
        layout.addLayout(nav)
        layout.addWidget(self.slider)
        # Metadata label should be compact (stretch factor 0, max height)
        self.filename_label.setMaximumHeight(50)
        layout.addWidget(self.filename_label, 0)

        self.setAcceptDrops(True)

        # Initial button state (no images loaded)
        self._update_button_states()

    def set_stack(self, files) -> None:
        self.handler.load_image_stack(files)
        self.slider.setRange(0, max(0, self.handler.get_image_count() - 1))
        self.index = 0
        self._update_button_states()
        self._show_current()
        # Determine directory path and emit
        directory: Optional[str] = None
        if isinstance(files, (list, tuple)) and files:
            directory = str(Path(files[0]).parent)
        elif isinstance(files, str):
            p = Path(files)
            directory = str(p if p.is_dir() else p.parent)
        if directory:
            self.stackLoaded.emit(directory)

    def reset_cache(self) -> None:
        """Reset the image cache."""
        self.cache = _LRUCache(20)

    def reset(self) -> None:
        """Reset the viewer to initial state."""
        # Clear handler files and reset navigation
        self.handler.files = []
        self.index = 0
        # Reset cache and ROI-related state
        self.cache = _LRUCache(20)
        self.current_roi = None
        self.roi_selection_mode = False
        self.roi_adjustment_mode = False
        self.roi_start_point = None
        self.roi_end_point = None
        self.roi_shape = "ellipse"
        self.dragging_handle = None
        self.drag_start_pos = None
        # Reset UI labels and slider
        self.image_label.setText("Drop TIF files or open a folder…")
        self.filename_label.setText("Load image to see data")
        self.slider.setRange(0, 0)
        self._update_button_states()

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        md = event.mimeData()
        if md.hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls]
        if not paths:
            return
        # If a single directory dropped, use directory
        if len(paths) == 1 and Path(paths[0]).is_dir():
            self.set_stack(paths[0])
        else:
            self.set_stack(paths)

    def _numpy_to_qimage(self, arr: np.ndarray) -> QImage:
        if arr.ndim == 2:
            h, w = arr.shape
            fmt = (
                QImage.Format_Grayscale8
                if arr.dtype != np.uint16
                else QImage.Format_Grayscale16
            )
            bytes_per_line = arr.strides[0]
            return QImage(arr.data, w, h, bytes_per_line, fmt)
        if arr.ndim == 3:
            h, w, c = arr.shape
            if c == 3:
                return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
            if c == 4:
                return QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
        raise ValueError("Unsupported image shape")

    def _show_current(self) -> None:
        count = self.handler.get_image_count()
        if count == 0:
            # Clear any pixmap and show placeholder text
            self.image_label.clear()
            self.image_label.setText("Drop TIF files or open a folder…")
            self.filename_label.setText("Load image to see data")
            self._update_button_states()
            return
        img = self.cache.get(self.index)
        if img is None:
            img = self.handler.get_image_at_index(self.index)
            self.cache.set(self.index, img)
        qimg = self._numpy_to_qimage(img)
        pix = QPixmap.fromImage(qimg)
        scaled_pix = pix.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # ============================================================
        # Draw ROI selection during selection mode
        # ============================================================
        if (
            self.roi_selection_mode
            and self.roi_start_point is not None
            and self.roi_end_point is not None
        ):
            painter = QPainter(scaled_pix)
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)

            if img.ndim >= 2:
                original_height, original_width = img.shape[0], img.shape[1]
                label_size = self.image_label.size()

                label_aspect = label_size.width() / label_size.height()
                original_aspect = original_width / original_height

                if label_aspect > original_aspect:
                    scale = scaled_pix.height() / original_height
                else:
                    scale = scaled_pix.width() / original_width

                # Draw ellipse
                x1 = min(self.roi_start_point.x(), self.roi_end_point.x())
                y1 = min(self.roi_start_point.y(), self.roi_end_point.y())
                x2 = max(self.roi_start_point.x(), self.roi_end_point.x())
                y2 = max(self.roi_start_point.y(), self.roi_end_point.y())

                x1_scaled = int(x1 * scale)
                y1_scaled = int(y1 * scale)
                w_scaled = int((x2 - x1 + 1) * scale)
                h_scaled = int((y2 - y1 + 1) * scale)
                painter.drawEllipse(x1_scaled, y1_scaled, w_scaled, h_scaled)
            painter.end()

        # ============================================================
        # Draw saved ROI when not in selection mode
        # ============================================================
        elif self.current_roi is not None and not self.roi_selection_mode:
            painter = QPainter(scaled_pix)
            pen = QPen(Qt.green, 2, Qt.SolidLine)
            painter.setPen(pen)

            if img.ndim >= 2:
                original_height, original_width = img.shape[0], img.shape[1]
                label_size = self.image_label.size()

                label_aspect = label_size.width() / label_size.height()
                original_aspect = original_width / original_height

                if label_aspect > original_aspect:
                    scale = scaled_pix.height() / original_height
                else:
                    scale = scaled_pix.width() / original_width

                x, y, w, h = self.current_roi

                x_scaled = int(x * scale)
                y_scaled = int(y * scale)
                w_scaled = int(w * scale)
                h_scaled = int(h * scale)

                # Draw ellipse ROI
                painter.drawEllipse(x_scaled, y_scaled, w_scaled, h_scaled)

                # Draw adjustment handles if in adjustment mode
                if self.roi_adjustment_mode:
                    self._draw_adjustment_handles(
                        painter, x_scaled, y_scaled, w_scaled, h_scaled, scale
                    )
            painter.end()

        self.image_label.setPixmap(scaled_pix)
        current_path = Path(self.handler.files[self.index])
        # label for the image that is been viewed
        self.filename_label.setText(f"{self.index + 1}/{count}: \n{current_path.name}")

    def resizeEvent(self, event) -> None:  # noqa: N802
        # ROI FIX: Redraw on resize so ROI scale updates correctly
        # When the window/pane is resized, the scaled_pix size changes
        # By calling _show_current(), we recalculate the scale factor and redraw the ROI
        # at the correct position for the new display size
        super().resizeEvent(event)
        self._show_current()
        # Reposition button if visible
        if self.open_files_btn.isVisible():
            self._position_open_button()

    def prev_image(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.index)
            self.slider.blockSignals(False)
            self._show_current()

    def next_image(self) -> None:
        if self.index < max(0, self.handler.get_image_count() - 1):
            self.index += 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.index)
            self.slider.blockSignals(False)
            self._show_current()

    def _on_slider(self, value: int) -> None:
        self.index = value
        self._show_current()

    def _toggle_roi_mode(self) -> None:
        """Toggle ROI selection mode."""
        # Check if images are loaded
        if self.handler.get_image_count() == 0:
            QMessageBox.warning(
                self, "No Images Loaded", "Please load images before selecting an ROI."
            )
            return

        # If already in selection mode, cancel it
        if self.roi_selection_mode:
            self.roi_selection_mode = False
            self.roi_btn.setText("Select ROI")
            self.roi_start_point = None
            self.roi_end_point = None
            self._show_current()
            return

        # If there's an existing ROI, ask for confirmation
        if self.current_roi is not None:
            reply = QMessageBox.question(
                self,
                "Replace Existing ROI",
                "An ellipse ROI already exists. Selecting a new ROI will remove the current ellipse.\n\nDo you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return
            # Clear existing ROI
            self.current_roi = None
            self.roi_adjustment_mode = False
            self.adjust_roi_btn.setText("Adjust ROI")
            self._update_button_states()

        # Enter ROI selection mode
        self.roi_selection_mode = True
        self.roi_btn.setText("Cancel ROI")
        self.roi_adjustment_mode = False
        self.adjust_roi_btn.setText("Adjust ROI")

    def _toggle_adjustment_mode(self) -> None:
        """Toggle ROI adjustment mode."""
        if self.current_roi is None:
            return
        self.roi_adjustment_mode = not self.roi_adjustment_mode
        self.adjust_roi_btn.setText(
            "Cancel Adjust" if self.roi_adjustment_mode else "Adjust ROI"
        )
        if self.roi_adjustment_mode:
            self.roi_selection_mode = False
            self.roi_btn.setText("Select ROI")
            self.dragging_handle = None
            self.drag_start_pos = None
        self._show_current()

    def _on_mouse_press(self, event) -> None:
        """Handle mouse press for ROI selection and adjustment."""
        if event.button() != Qt.LeftButton:
            return

        # Handle ROI adjustment
        if self.roi_adjustment_mode and self.current_roi is not None:
            # Check if clicking on a handle
            handle = self._get_handle_at_position(
                event.position().x(), event.position().y()
            )
            if handle:
                self.dragging_handle = handle
                self.drag_start_pos = QPoint(
                    int(event.position().x()), int(event.position().y())
                )
                return

            # Check if clicking inside the ellipse (for moving)
            if self._is_point_inside_ellipse(
                event.position().x(), event.position().y()
            ):
                self.dragging_handle = "move"
                self.drag_start_pos = QPoint(
                    int(event.position().x()), int(event.position().y())
                )
                return

        # Handle ROI selection
        if self.roi_selection_mode:
            # Get the actual image dimensions
            img = self.cache.get(self.index)
            if img is None:
                img = self.handler.get_image_at_index(self.index)
            if img is not None and img.ndim >= 2:
                original_height, original_width = img.shape[0], img.shape[1]

                # Get label and scaled pixmap sizes
                label_size = self.image_label.size()
                pixmap = self.image_label.pixmap()
                if pixmap:
                    scaled_pixmap_size = pixmap.size()

                    # Calculate aspect ratio scaling
                    label_aspect = label_size.width() / label_size.height()
                    original_aspect = original_width / original_height

                    # Determine actual scaled dimensions (with aspect ratio preserved)
                    if label_aspect > original_aspect:
                        # Label is wider - height determines scale
                        scale = scaled_pixmap_size.height() / original_height
                    else:
                        # Label is taller - width determines scale
                        scale = scaled_pixmap_size.width() / original_width

                    # Get mouse position relative to label
                    mouse_x = event.position().x()
                    mouse_y = event.position().y()

                    # Account for centering if image doesn't fill entire label
                    offset_x = (label_size.width() - scaled_pixmap_size.width()) / 2
                    offset_y = (label_size.height() - scaled_pixmap_size.height()) / 2

                    # Convert to original image coordinates
                    x = int((mouse_x - offset_x) / scale)
                    y = int((mouse_y - offset_y) / scale)

                    # Clamp to image bounds
                    x = max(0, min(original_width - 1, x))
                    y = max(0, min(original_height - 1, y))

                    self.roi_start_point = QPoint(x, y)
                    self.roi_end_point = QPoint(x, y)

    def _on_mouse_move(self, event) -> None:
        """Handle mouse move for ROI selection and adjustment."""
        # Handle ROI adjustment
        if (
            self.roi_adjustment_mode
            and self.current_roi is not None
            and self.dragging_handle
        ):
            if self.dragging_handle == "move":
                self._move_roi(event.position().x(), event.position().y())
            else:
                self._adjust_roi(event.position().x(), event.position().y())
            return

        # Handle ROI selection (ellipse)
        if self.roi_selection_mode and self.roi_start_point is not None:
            img = self.cache.get(self.index)
            if img is None:
                img = self.handler.get_image_at_index(self.index)
            if img is not None and img.ndim >= 2:
                original_height, original_width = img.shape[0], img.shape[1]

                label_size = self.image_label.size()
                pixmap = self.image_label.pixmap()
                if pixmap:
                    scaled_pixmap_size = pixmap.size()

                    # Calculate aspect ratio scaling
                    label_aspect = label_size.width() / label_size.height()
                    original_aspect = original_width / original_height

                    if label_aspect > original_aspect:
                        scale = scaled_pixmap_size.height() / original_height
                    else:
                        scale = scaled_pixmap_size.width() / original_width

                    mouse_x = event.position().x()
                    mouse_y = event.position().y()

                    offset_x = (label_size.width() - scaled_pixmap_size.width()) / 2
                    offset_y = (label_size.height() - scaled_pixmap_size.height()) / 2

                    x = int((mouse_x - offset_x) / scale)
                    y = int((mouse_y - offset_y) / scale)

                    # Clamp to image bounds
                    x = max(0, min(original_width - 1, x))
                    y = max(0, min(original_height - 1, y))

                    self.roi_end_point = QPoint(x, y)
                    self._show_current()

    def _on_mouse_release(self, event) -> None:
        """Handle mouse release for ROI selection and adjustment."""
        if event.button() != Qt.LeftButton:
            return

        # Handle ROI adjustment
        if self.roi_adjustment_mode and self.current_roi is not None:
            if self.dragging_handle:
                # Emit signal with updated ROI
                x, y, w, h = self.current_roi
                self.roiSelected.emit(x, y, w, h)
            self.dragging_handle = None
            self.drag_start_pos = None
            self._show_current()
            return

        # Handle ROI selection (ellipse)
        if self.roi_selection_mode and self.roi_start_point is not None:
            img = self.cache.get(self.index)
            if img is None:
                img = self.handler.get_image_at_index(self.index)
            if img is not None and img.ndim >= 2:
                original_height, original_width = img.shape[0], img.shape[1]

                label_size = self.image_label.size()
                pixmap = self.image_label.pixmap()
                if pixmap:
                    scaled_pixmap_size = pixmap.size()

                    label_aspect = label_size.width() / label_size.height()
                    original_aspect = original_width / original_height

                    if label_aspect > original_aspect:
                        scale = scaled_pixmap_size.height() / original_height
                    else:
                        scale = scaled_pixmap_size.width() / original_width

                    mouse_x = event.position().x()
                    mouse_y = event.position().y()

                    offset_x = (label_size.width() - scaled_pixmap_size.width()) / 2
                    offset_y = (label_size.height() - scaled_pixmap_size.height()) / 2

                    x = int((mouse_x - offset_x) / scale)
                    y = int((mouse_y - offset_y) / scale)

                    x = max(0, min(original_width - 1, x))
                    y = max(0, min(original_height - 1, y))

                    self.roi_end_point = QPoint(x, y)

                    # Create ROI in image coordinates
                    x1 = min(self.roi_start_point.x(), self.roi_end_point.x())
                    y1 = min(self.roi_start_point.y(), self.roi_end_point.y())
                    x2 = max(self.roi_start_point.x(), self.roi_end_point.x())
                    y2 = max(self.roi_start_point.y(), self.roi_end_point.y())

                    width = max(1, x2 - x1 + 1)
                    height = max(1, y2 - y1 + 1)

                    # Store ROI in image coordinates
                    self.current_roi = (x1, y1, width, height)
                    self.roi_shape = "ellipse"

                    # Emit signal with ROI coordinates (x, y, width, height)
                    self.roiSelected.emit(x1, y1, width, height)

                    # Exit selection mode without clearing the ROI
                    self.roi_selection_mode = False
                    self.roi_btn.setText("Select ROI")
                    self.roi_start_point = None
                    self.roi_end_point = None
                    self._update_button_states()

                    self._show_current()

    def _is_point_inside_ellipse(self, mouse_x: float, mouse_y: float) -> bool:
        """Check if mouse point is inside the ellipse ROI."""
        if self.current_roi is None:
            return False

        img = self.cache.get(self.index)
        if img is None:
            img = self.handler.get_image_at_index(self.index)
        if img is None or img.ndim < 2:
            return False

        original_height, original_width = img.shape[0], img.shape[1]
        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return False

        scaled_pixmap_size = pixmap.size()
        label_aspect = label_size.width() / label_size.height()
        original_aspect = original_width / original_height

        if label_aspect > original_aspect:
            scale = scaled_pixmap_size.height() / original_height
        else:
            scale = scaled_pixmap_size.width() / original_width

        offset_x = (label_size.width() - scaled_pixmap_size.width()) / 2
        offset_y = (label_size.height() - scaled_pixmap_size.height()) / 2

        # Convert mouse position to image coordinates
        img_x = (mouse_x - offset_x) / scale
        img_y = (mouse_y - offset_y) / scale

        # Get ellipse parameters
        x, y, w, h = self.current_roi
        center_x = x + w / 2
        center_y = y + h / 2
        radius_x = w / 2
        radius_y = h / 2

        # Check if point is inside ellipse: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
        dx = (img_x - center_x) / radius_x if radius_x > 0 else 0
        dy = (img_y - center_y) / radius_y if radius_y > 0 else 0
        return (dx * dx + dy * dy) <= 1.0

    def _draw_adjustment_handles(
        self, painter: QPainter, x: int, y: int, w: int, h: int, scale: float
    ) -> None:
        """Draw adjustment handles around the ROI."""
        handle_size = int(self.handle_size * scale)
        half_handle = handle_size // 2

        # Define handle positions (corners and midpoints)
        handles = [
            (x, y),  # Top-left
            (x + w // 2, y),  # Top-center
            (x + w, y),  # Top-right
            (x + w, y + h // 2),  # Right-center
            (x + w, y + h),  # Bottom-right
            (x + w // 2, y + h),  # Bottom-center
            (x, y + h),  # Bottom-left
            (x, y + h // 2),  # Left-center
        ]

        painter.setPen(QPen(Qt.blue, 2))
        painter.setBrush(Qt.blue)
        for hx, hy in handles:
            painter.drawRect(
                hx - half_handle, hy - half_handle, handle_size, handle_size
            )

    def _get_handle_at_position(self, mouse_x: float, mouse_y: float) -> Optional[str]:
        """Check if mouse is over an adjustment handle. Returns handle name or None."""
        if self.current_roi is None:
            return None

        img = self.cache.get(self.index)
        if img is None:
            img = self.handler.get_image_at_index(self.index)
        if img is None or img.ndim < 2:
            return None

        original_height, original_width = img.shape[0], img.shape[1]
        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return None

        scaled_pixmap_size = pixmap.size()
        label_aspect = label_size.width() / label_size.height()
        original_aspect = original_width / original_height

        if label_aspect > original_aspect:
            scale = scaled_pixmap_size.height() / original_height
        else:
            scale = scaled_pixmap_size.width() / original_width

        offset_x = (label_size.width() - scaled_pixmap_size.width()) / 2
        offset_y = (label_size.height() - scaled_pixmap_size.height()) / 2

        x, y, w, h = self.current_roi
        # Calculate handle positions in display coordinates
        x_scaled = int(x * scale) + offset_x
        y_scaled = int(y * scale) + offset_y
        w_scaled = int(w * scale)
        h_scaled = int(h * scale)

        handle_size = max(8, int(self.handle_size * scale))
        half_handle = handle_size // 2

        handles = {
            "tl": (x_scaled, y_scaled),
            "tc": (x_scaled + w_scaled // 2, y_scaled),
            "tr": (x_scaled + w_scaled, y_scaled),
            "rc": (x_scaled + w_scaled, y_scaled + h_scaled // 2),
            "br": (x_scaled + w_scaled, y_scaled + h_scaled),
            "bc": (x_scaled + w_scaled // 2, y_scaled + h_scaled),
            "bl": (x_scaled, y_scaled + h_scaled),
            "lc": (x_scaled, y_scaled + h_scaled // 2),
        }

        for handle_name, (hx, hy) in handles.items():
            if abs(mouse_x - hx) <= half_handle and abs(mouse_y - hy) <= half_handle:
                return handle_name
        return None

    def _move_roi(self, mouse_x: float, mouse_y: float) -> None:
        """Move the entire ROI ellipse."""
        if self.current_roi is None or self.drag_start_pos is None:
            return

        img = self.cache.get(self.index)
        if img is None:
            img = self.handler.get_image_at_index(self.index)
        if img is None or img.ndim < 2:
            return

        original_height, original_width = img.shape[0], img.shape[1]
        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return

        scaled_pixmap_size = pixmap.size()
        label_aspect = label_size.width() / label_size.height()
        original_aspect = original_width / original_height

        if label_aspect > original_aspect:
            scale = scaled_pixmap_size.height() / original_height
        else:
            scale = scaled_pixmap_size.width() / original_width

        offset_x = (label_size.width() - scaled_pixmap_size.width()) / 2
        offset_y = (label_size.height() - scaled_pixmap_size.height()) / 2

        # Convert mouse position to image coordinates
        img_x = (mouse_x - offset_x) / scale
        img_y = (mouse_y - offset_y) / scale

        # Convert start position to image coordinates
        start_img_x = (self.drag_start_pos.x() - offset_x) / scale
        start_img_y = (self.drag_start_pos.y() - offset_y) / scale

        # Calculate delta movement
        dx = img_x - start_img_x
        dy = img_y - start_img_y

        # Get current ROI
        x, y, w, h = self.current_roi

        # Calculate new position
        new_x = int(x + dx)
        new_y = int(y + dy)

        # Clamp to image bounds
        new_x = max(0, min(original_width - w, new_x))
        new_y = max(0, min(original_height - h, new_y))

        # Update ROI
        self.current_roi = (new_x, new_y, w, h)

        # Update drag start position for smooth continuous movement
        self.drag_start_pos = QPoint(int(mouse_x), int(mouse_y))

        self._show_current()

    def _adjust_roi(self, mouse_x: float, mouse_y: float) -> None:
        """Adjust ROI size based on dragged handle."""
        if self.current_roi is None or self.dragging_handle is None:
            return

        img = self.cache.get(self.index)
        if img is None:
            img = self.handler.get_image_at_index(self.index)
        if img is None or img.ndim < 2:
            return

        original_height, original_width = img.shape[0], img.shape[1]
        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return

        scaled_pixmap_size = pixmap.size()
        label_aspect = label_size.width() / label_size.height()
        original_aspect = original_width / original_height

        if label_aspect > original_aspect:
            scale = scaled_pixmap_size.height() / original_height
        else:
            scale = scaled_pixmap_size.width() / original_width

        offset_x = (label_size.width() - scaled_pixmap_size.width()) / 2
        offset_y = (label_size.height() - scaled_pixmap_size.height()) / 2

        # Convert mouse position to image coordinates
        img_x = int((mouse_x - offset_x) / scale)
        img_y = int((mouse_y - offset_y) / scale)
        img_x = max(0, min(original_width - 1, img_x))
        img_y = max(0, min(original_height - 1, img_y))

        x, y, w, h = self.current_roi
        x2 = x + w
        y2 = y + h

        # Handle resizing via handles
        if self.dragging_handle == "tl":
            x, y = img_x, img_y
        elif self.dragging_handle == "tc":
            y = img_y
        elif self.dragging_handle == "tr":
            x2, y = img_x, img_y
        elif self.dragging_handle == "rc":
            x2 = img_x
        elif self.dragging_handle == "br":
            x2, y2 = img_x, img_y
        elif self.dragging_handle == "bc":
            y2 = img_y
        elif self.dragging_handle == "bl":
            x, y2 = img_x, img_y
        elif self.dragging_handle == "lc":
            x = img_x

        # Ensure valid dimensions
        x = max(0, min(x, original_width - 1))
        y = max(0, min(y, original_height - 1))
        x2 = max(x + 1, min(x2, original_width))
        y2 = max(y + 1, min(y2, original_height))

        w = x2 - x
        h = y2 - y
        self.current_roi = (x, y, w, h)

        self._show_current()

    def get_current_roi(self) -> Optional[tuple]:
        """Get the current ROI coordinates (x, y, width, height) in image space."""
        return self.current_roi

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """
        Set the ROI from saved coordinates (x, y, width, height) in image space.

        This method is called when loading an experiment with a saved ROI.
        The coordinates are in original image pixel space, not display/widget space.
        This ensures the ROI stays fixed to the correct image region regardless of
        window size or scaling.
        """
        self.current_roi = (x, y, width, height)
        self._update_button_states()
        # Redraw to show the ROI with correct scaling
        self._show_current()

    def _update_button_states(self) -> None:
        """Update button enabled/disabled states based on whether images are loaded."""
        has_images = self.handler.get_image_count() > 0
        self.roi_btn.setEnabled(has_images)
        self.adjust_roi_btn.setEnabled(has_images and self.current_roi is not None)
        self.prev_btn.setEnabled(has_images)
        self.next_btn.setEnabled(has_images)
        self.slider.setEnabled(has_images)

        # Show/hide and position open files button
        if not has_images:
            # Show button and center it in the container
            self.open_files_btn.show()
            self.open_files_btn.raise_()
            # Position button in center (use timer to ensure layout is complete)
            QTimer.singleShot(10, self._position_open_button)
        else:
            self.open_files_btn.hide()

    def _position_open_button(self) -> None:
        """Position the open files button in the center of the image container."""
        if not self.open_files_btn.isVisible():
            return
        container_rect = self.image_container.rect()
        btn_size = self.open_files_btn.size()
        x = (container_rect.width() - btn_size.width()) // 2
        y = (container_rect.height() - btn_size.height()) // 2
        self.open_files_btn.move(x, y)

    def _on_open_files_clicked(self) -> None:
        """Handle click on the open files button (placeholder for future implementation)."""
        # Placeholder - does nothing for now
        pass
