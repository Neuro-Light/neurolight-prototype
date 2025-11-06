from __future__ import annotations

from collections import OrderedDict
from typing import Optional
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
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
        self.roi_start_point = None
        self.roi_end_point = None
        self.current_roi = None

        self.filename_label = QLabel("Load image to see data") #label for user to see if no image are selected
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setWordWrap(True)
        self.filename_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.image_label = QLabel("Drop TIF files or open a folder…")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self._on_mouse_press
        self.image_label.mouseMoveEvent = self._on_mouse_move
        self.image_label.mouseReleaseEvent = self._on_mouse_release

        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.roi_btn = QPushButton("Select ROI")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.roi_btn.clicked.connect(self._toggle_roi_mode)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)

        ## Exposure/Contrast slider controls for the exposure
        # Label text at inital load
        self.exposure_label = QLabel("Exposure: 0")
        # Adds the slider movement as a horizontal slider "vertical setting also"
        self.exposure_slider = QSlider(Qt.Horizontal)
        # Setting the value in the slider, set value to -100 - 100 for more accuracy
        self.exposure_slider.setRange(-100, 100)
        # The slider will start 0 at the load in
        self.exposure_slider.setValue(0)
        # Handle the changeing value for the exposure slider
        self.exposure_slider.valueChanged.connect(self._on_adjustment_changed)

        # Label text at inital load for the contrast
        self.contrast_label = QLabel("Contrast: 0")
        # Adds the slider movement as a horizontal slider "vertical setting also"
        self.contrast_slider = QSlider(Qt.Horizontal)
        # Setting the values in the slider, set value to -100 - 100 for more accuracy
        self.contrast_slider.setRange(-100, 100)
        # Slider will start at 0 on load in
        self.contrast_slider.setValue(0)
        #handle the changing value for the contrast slider
        self.contrast_slider.valueChanged.connect(self._on_adjustment_changed)
        

        nav = QHBoxLayout()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.roi_btn)

        # New box for the exposure and  contrast slider
        adjustments_layout = QVBoxLayout()
        # Add the label to the slider
        adjustments_layout.addWidget(self.exposure_label)
        # Add the slider to the container
        adjustments_layout.addWidget(self.exposure_slider)
        # Added the label to the slider
        adjustments_layout.addWidget(self.contrast_label)
        # Add the slider to the container
        adjustments_layout.addWidget(self.contrast_slider)

        layout = QVBoxLayout(self)
        # Image gets most of the space (stretch factor 1)
        layout.addWidget(self.image_label, 1)
        layout.addLayout(nav)
        layout.addWidget(self.slider)
        # Added the layout for the exposure and contrast
        layout.addLayout(adjustments_layout)
        # Metadata label should be compact (stretch factor 0, max height)
        self.filename_label.setMaximumHeight(50)
        layout.addWidget(self.filename_label, 0)
        # adjust the labels on sliders to go to 0 on load
        self._update_adjustment_labels()

        self.setAcceptDrops(True)

    def set_stack(self, files) -> None:
        self.handler.load_image_stack(files)
        self.slider.setRange(0, max(0, self.handler.get_image_count() - 1))
        self.index = 0
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
        self.roi_start_point = None
        self.roi_end_point = None
        # Reset UI labels and slider
        self.image_label.setText("Drop TIF files or open a folder…")
        self.filename_label.setText("Load image to see data")
        self.slider.setRange(0, 0)

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

    # Function to update the silder value so the user can see what value they have
    def _update_adjustment_labels(self) -> None:
        # For exposure
        self.exposure_label.setText(f"Exposure: {self.exposure_slider.value()}")
        # For contrast
        self.contrast_label.setText(f"Contrast: {self.contrast_slider.value()}")

    '''Next three function convert to 8 bit and do exposure and contrast
    calculation. The order is a stated: Exposure/Contrast, then 8 bit converstion
    This will preserve the appearance and will allow for more adjustment'''
    #function to calculate teh amount of exposure and contrast the imahe
    # should get based on the slider value input
    def _apply_adjustments(self, arr: np.ndarray) -> np.ndarray:

        # Set the exposure and contrast sliders value to ev and cv
        ev = self.exposure_slider.value()
        cv = self.contrast_slider.value()
        #stores the orignal image data type
        orig_dtype = arr.dtype
        # convert the image to float32 and saves it as a new array
        # copy=false stop the automatic saving of this array from astype(),
        # if step to true then astype() will save a array if one value is off.
        new_arr = arr.astype(np.float32, copy=False)

        # If/else statement normailizing the image
        # if the orignal image data type is not a float
        # This is because float and integer data type normalize differently
        if np.issubdtype(orig_dtype, np.floating):
            # get the minimum pixel value in the array
            min_pixel = np.min(new_arr)
            #get the pixel range
            pixel_range = np.max(new_arr) - min_pixel 
            # check to see if the pixel arent all equal
            # cant divied by 0 
            if pixel_range != 0:
                # normalize all the images in the array
                new_arr = (new_arr - min_pixel) / pixel_range
        # else the data type is a float
        else:
            # normalizing a integer image
            # the array divided by the max color value
            # the value would be between 0 and 1 for every pixel in the image
            new_arr = new_arr / np.iinfo(orig_dtype).max

        #creating exposure and contrast scalers
        exposure = 2 ** (ev / 50)               
        contrast = 1 + (cv / 100) 
        #0.5 to preserve the greyscale... if higher turns black...if lower turns white      
        new_arr = ((new_arr - 0.5) * contrast + 0.5) * exposure
        # this will set the max and min values to 0 to 1
        # you will get more uniformed ranges in contrast and exposure
        new_arr = np.clip(new_arr, 0, 1)
        # return the normalized array with edits
        return new_arr


    def _on_adjustment_changed(self, _value: int) -> None:
        self._update_adjustment_labels()
        self._show_current()


    # Function to convert to 8 bits
    def _ensure_uint8(self, arr: np.ndarray) -> np.ndarray:
        # convert to unit 8... 8 bit
        unit_8 = cv2.convertScaleAbs(new_arr, alpha=255.0, beta=0.0)
        #return the 8 bit image
        return unit_8

    def _show_current(self) -> None:
        count = self.handler.get_image_count()
        if count == 0:
            self.image_label.setText("No images loaded")
            self.filename_label.setText("Load image to see data")
            return
        img = self.cache.get(self.index)
        if img is None:
            img = self.handler.get_image_at_index(self.index)
            self.cache.set(self.index, img)
        #show the adjusted image on the workbench
        adjusted_img = self._apply_adjustments(img)
        #show the 8 bit image on the workbench
        preview_img = self._ensure_uint8(adjusted_img)
        qimg = self._numpy_to_qimage(preview_img)
        pix = QPixmap.fromImage(qimg)
        scaled_pix = pix.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # ============================================================
        # ROI FIX: Draw ROI selection rectangle during selection mode
        # ============================================================
        # FIX DETAILS:
        # - ROI coordinates are stored in ORIGINAL IMAGE PIXEL SPACE (not widget/display space)
        # - This ensures ROI stays fixed to the image region regardless of window resizing
        # - When drawing, we recalculate scale from the ACTUAL scaled pixmap size (not label size)
        # - This prevents the ROI from moving when the pane is resized
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

                # ROI FIX: Calculate aspect ratio to determine which dimension constrains scaling
                # This determines whether width or height of the label determines the scale factor
                label_aspect = label_size.width() / label_size.height()
                original_aspect = original_width / original_height

                # ROI FIX: Calculate scale from ACTUAL scaled pixmap dimensions (not label size)
                # This is critical - we use scaled_pix.height()/width() directly because:
                # 1. The pixmap is already scaled to fit the label while maintaining aspect ratio
                # 2. Using the pixmap size ensures we get the correct scale regardless of window size
                # 3. This prevents ROI drift when resizing the window
                if label_aspect > original_aspect:
                    # Label is wider than image - height constrains the scale
                    scale = scaled_pix.height() / original_height
                else:
                    # Label is taller than image - width constrains the scale
                    scale = scaled_pix.width() / original_width

                # ROI FIX: ROI coordinates are in ORIGINAL IMAGE SPACE (from mouse event conversion)
                # These were converted from mouse coordinates to image pixel coordinates in
                # _on_mouse_press/_on_mouse_move/_on_mouse_release methods
                x1 = min(self.roi_start_point.x(), self.roi_end_point.x())
                y1 = min(self.roi_start_point.y(), self.roi_end_point.y())
                x2 = max(self.roi_start_point.x(), self.roi_end_point.x())
                y2 = max(self.roi_start_point.y(), self.roi_end_point.y())

                # ROI FIX: Convert from image pixel coordinates to scaled display coordinates
                # Multiply by the scale factor to get the correct display position
                # The +1 ensures we include both endpoints in the selection
                x1_scaled = int(x1 * scale)
                y1_scaled = int(y1 * scale)
                w_scaled = int((x2 - x1 + 1) * scale)
                h_scaled = int((y2 - y1 + 1) * scale)

                # ROI FIX: Draw directly on the scaled pixmap (no offset needed)
                # Since we're drawing on the pixmap itself (not the label), we don't need
                # to account for centering offsets. The pixmap will be centered by Qt when displayed.
                painter.drawRect(x1_scaled, y1_scaled, w_scaled, h_scaled)
            painter.end()

        # ============================================================
        # ROI FIX: Draw saved ROI rectangle when not in selection mode
        # ============================================================
        # FIX DETAILS:
        # - This is the key fix for keeping ROI anchored to image region on resize
        # - ROI is stored in image pixel coordinates (self.current_roi)
        # - Scale is recalculated EVERY time _show_current() is called (including on resize)
        # - This ensures ROI position updates correctly when window/pane size changes
        elif self.current_roi is not None and not self.roi_selection_mode:
            painter = QPainter(scaled_pix)
            pen = QPen(Qt.green, 2, Qt.SolidLine)
            painter.setPen(pen)

            # Get original image dimensions (in pixels)
            if img.ndim >= 2:
                original_height, original_width = img.shape[0], img.shape[1]
                label_size = self.image_label.size()

                # ROI FIX: Calculate aspect ratio to determine scaling constraint
                # This tells us whether the label is wider or taller relative to the image
                label_aspect = label_size.width() / label_size.height()
                original_aspect = original_width / original_height

                # ROI FIX: Calculate scale from ACTUAL scaled pixmap size (not label size)
                # CRITICAL: We use scaled_pix.height()/width() NOT label_size because:
                # - The pixmap is already scaled to fit the label with aspect ratio preserved
                # - The pixmap size reflects the actual displayed image size
                # - This ensures correct scale calculation even when window is resized
                # - Using label_size would cause incorrect offsets and ROI drift
                if label_aspect > original_aspect:
                    # Label is wider than image - height constrains the scale
                    scale = scaled_pix.height() / original_height
                else:
                    # Label is taller than image - width constrains the scale
                    scale = scaled_pix.width() / original_width

                # ROI FIX: ROI coordinates stored in ORIGINAL IMAGE PIXEL SPACE
                # These coordinates are saved to the .nexp file and remain constant
                # regardless of window size or image scaling
                x, y, w, h = self.current_roi
                
                # ROI FIX: Convert from image pixel coordinates to scaled display coordinates
                # Multiply by the dynamically calculated scale factor
                # This conversion happens every time the image is redrawn (including on resize)
                x_scaled = int(x * scale)
                y_scaled = int(y * scale)
                w_scaled = int(w * scale)
                h_scaled = int(h * scale)

                # ROI FIX: Draw directly on the scaled pixmap (no centering offset needed)
                # Since we're drawing on the pixmap itself, coordinates are relative to the pixmap
                # Qt will handle centering the pixmap in the label if needed
                painter.drawRect(x_scaled, y_scaled, w_scaled, h_scaled)
            painter.end()

        self.image_label.setPixmap(scaled_pix)
        current_path = Path(self.handler.files[self.index])
        #label for the image that is been viewed
        self.filename_label.setText(f"{self.index + 1}/{count}: \n{current_path.name}")

    def resizeEvent(self, event) -> None:  # noqa: N802
        # ROI FIX: Redraw on resize so ROI scale updates correctly
        # When the window/pane is resized, the scaled_pix size changes
        # By calling _show_current(), we recalculate the scale factor and redraw the ROI
        # at the correct position for the new display size
        super().resizeEvent(event)
        self._show_current()

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
        self.roi_selection_mode = not self.roi_selection_mode
        self.roi_btn.setText("Cancel ROI" if self.roi_selection_mode else "Select ROI")
        if not self.roi_selection_mode:
            self.roi_start_point = None
            self.roi_end_point = None
            self.current_roi = None
            self._show_current()

    def _on_mouse_press(self, event) -> None:
        """Handle mouse press for ROI selection."""
        if self.roi_selection_mode and event.button() == Qt.LeftButton:
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
        """Handle mouse move for ROI selection."""
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
        """Handle mouse release for ROI selection."""
        if (
            self.roi_selection_mode
            and event.button() == Qt.LeftButton
            and self.roi_start_point is not None
        ):
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

                    # Create ROI rectangle in image coordinates
                    x1 = min(self.roi_start_point.x(), self.roi_end_point.x())
                    y1 = min(self.roi_start_point.y(), self.roi_end_point.y())
                    x2 = max(self.roi_start_point.x(), self.roi_end_point.x())
                    y2 = max(self.roi_start_point.y(), self.roi_end_point.y())

                    # Add 1 to include both endpoints (inclusive selection)
                    # Python slicing [x1:x2] is exclusive of x2, so we need width = x2-x1+1
                    width = max(1, x2 - x1 + 1)
                    height = max(1, y2 - y1 + 1)

                    # Store ROI in image coordinates
                    self.current_roi = (x1, y1, width, height)

                    # Emit signal with ROI coordinates (x, y, width, height)
                    self.roiSelected.emit(x1, y1, width, height)

                    # Exit selection mode without clearing the ROI
                    self.roi_selection_mode = False
                    self.roi_btn.setText("Select ROI")
                    # Clear temporary selection points but keep current_roi
                    self.roi_start_point = None
                    self.roi_end_point = None

                    self._show_current()

    def get_current_roi(self) -> Optional[tuple]:
        """Get the current ROI coordinates (x, y, width, height) in image space."""
        return self.current_roi

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """
        ROI FIX: Set the ROI from saved coordinates (x, y, width, height) in image space.
        
        This method is called when loading an experiment with a saved ROI.
        The coordinates are in original image pixel space, not display/widget space.
        This ensures the ROI stays fixed to the correct image region regardless of
        window size or scaling.
        """
        self.current_roi = (x, y, width, height)
        # Redraw to show the ROI with correct scaling
        self._show_current()
