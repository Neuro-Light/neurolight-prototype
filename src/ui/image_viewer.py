from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple, List
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QMovie
from PySide6.QtWidgets import (
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QStyle,
    QMessageBox,
    QFileDialog,
    QDialog,
)
# Direct Pillow access for GIF encoding
from PIL import Image  

from utils.file_handler import ImageStackHandler
from core.roi import ROI, ROIShape, ROIHandle


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
    # Emits ROI object
    roiSelected = Signal(object)
    # Emits when ROI is changed
    roiChanged = Signal(object)
    # Emits when exposure or contrast settings are changed, with new values
    displaySettingsChanged = Signal(int, int)

    def __init__(self, handler: ImageStackHandler) -> None:
        super().__init__()
        self.handler = handler
        self.index = 0
        self.cache = _LRUCache(20)
        # Store the path of the last generated GIF for previewing
        self._last_gif_path: Optional[str] = None

        # ROI selection state
        self.roi_selection_mode = False
        self.roi_adjustment_mode = False  # User must explicitly enable adjustment
        self.roi_start_point = None
        self.roi_end_point = None
        self.current_roi: Optional[ROI] = None
        self.selected_shape = ROIShape.ELLIPSE  # Only ellipse shape supported
        
        # ROI adjustment state
        self.active_handle = ROIHandle.NONE
        self.last_mouse_pos = None
        self.can_adjust_roi = False  # Only true when user clicks "Adjust ROI"

        self.filename_label = QLabel("Load image to see data") #label for user to see if no image are selected
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setWordWrap(True)
        self.filename_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # Create image display area with upload button
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self._on_mouse_press
        self.image_label.mouseMoveEvent = self._on_mouse_move
        self.image_label.mouseReleaseEvent = self._on_mouse_release
        
        # Upload button (visible when no images loaded)
        self.upload_btn = QPushButton("Open Images")
        # Add standard Qt file open icon
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        self.upload_btn.setIcon(icon)
        self.upload_btn.clicked.connect(self._open_upload_dialog)
        
        # Container for image label with upload button overlay
        self.image_container = QWidget()
        image_layout = QVBoxLayout(self.image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addWidget(self.image_label)
        
        # Overlay upload button on image label
        self.upload_btn.setParent(self.image_label)
        self.upload_btn.show()
        
        # Schedule button centering after layout is complete
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._center_upload_button)

        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.roi_btn = QPushButton("Select ROI")
        self.adjust_roi_btn = QPushButton("Adjust ROI")
        # Initially hide the adjust ROI button until an ROI is created
        self.adjust_roi_btn.setVisible(False) 
        # Make a button o create GIF
        self.create_gif_btn = QPushButton("Create GIF")
        # Make a button to preview GIF
        self.preview_gif_btn = QPushButton("Preview GIF")
        self.preview_gif_btn.setEnabled(False)
        
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.roi_btn.clicked.connect(self._toggle_roi_mode)
        self.adjust_roi_btn.clicked.connect(self._toggle_adjustment_mode)
        # Connect the create GIF and preview GIF buttons to their respective functions
        self.create_gif_btn.clicked.connect(self._create_gif)
        self.preview_gif_btn.clicked.connect(self._preview_gif)

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
        nav.addWidget(self.adjust_roi_btn)
        # Add the create and preview GIF buttons to the navigation layout
        gif_layout = QHBoxLayout()
        gif_layout.addWidget(self.create_gif_btn)
        gif_layout.addWidget(self.preview_gif_btn)

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
        layout.addWidget(self.image_container, 1)
        layout.addLayout(nav)
        # Add the GIF buttons below the navigation buttons
        layout.addLayout(gif_layout)
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
        
        # Hide upload button when images are loaded
        if self.handler.get_image_count() > 0:
            self.upload_btn.hide()
        
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
        self.can_adjust_roi = False
        self.roi_start_point = None
        self.roi_end_point = None
        self.active_handle = ROIHandle.NONE
        self.last_mouse_pos = None
        # Reset UI labels and slider
        self.image_label.clear()
        self.filename_label.setText("Load image to see data")
        self.slider.setRange(0, 0)
        self._update_roi_button_text()
        self.adjust_roi_btn.setVisible(False)
        # Re-enable all controls
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.slider.setEnabled(True)
        self.roi_btn.setEnabled(True)
        # Show upload button again
        self.upload_btn.show()
        self._center_upload_button()

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
            # Filter to only allow TIF and GIF files
            allowed_extensions = {'.tif', '.tiff', '.gif'}
            filtered_paths = [
                p for p in paths 
                if Path(p).suffix.lower() in allowed_extensions
            ]
            
            if filtered_paths:
                self.set_stack(filtered_paths)
            else:
                # Show message if no valid files were dropped
                QMessageBox.warning(
                    self,
                    "Invalid Files",
                    "Only TIF and GIF files are supported."
                )

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

    # Helper function to prepare frames for GIF encoding by normalizing and converting to uint8
    def _prepare_frame_for_gif(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            frame = frame.mean(axis=2)
        # Normalize to 0-255 to avoid washed out previews
        frame = frame.astype(np.float32)
        min_val = float(frame.min())
        max_val = float(frame.max())
        if max_val > min_val:
            # Normalize the frame to the range [0, 1]
            frame = (frame - min_val) / (max_val - min_val)
        else:
            # If all values are the same, just return a blank image
            frame = np.zeros_like(frame, dtype=np.float32)
        return (frame * 255).clip(0, 255).astype(np.uint8)

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
        new_arr = arr.astype(np.float32, copy=True)
        min_pixel = float(np.min(new_arr))
        max_pixel = float(np.max(new_arr))
        pixel_range = max_pixel - min_pixel
        # check to see if the pixel arent all equal
        # cant divied by 0 
        if pixel_range != 0:
            # normalize all the images in the array
            new_arr = (new_arr - min_pixel) / pixel_range
        else:
            # if the orignal image data type is a integer image
            # This is because float and integer data type normalize differently
            if np.issubdtype(orig_dtype, np.integer):
                #get the max out of the image arr
                max_possible = float(np.iinfo(orig_dtype).max)
                #normailze the image
                new_arr = new_arr / max_possible
            else:
                #the image is normalized
                new_arr = np.clip(new_arr, 0, 1)

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
        # Emit signal so MainWindow can save to experiment
        self.displaySettingsChanged.emit(self.exposure_slider.value(), self.contrast_slider.value())


    # Function to convert to 8 bits
    def _ensure_uint8(self, arr: np.ndarray) -> np.ndarray:
        # Do the contrast and exposure edits first
        new_arr = self._apply_adjustments(arr)
        # convert to unit 8... 8 bit
        unit_8 = cv2.convertScaleAbs(new_arr, alpha=255.0, beta=0.0)
        #return the 8 bit image
        return unit_8

    def _show_current(self) -> None:
        count = self.handler.get_image_count()
        if count == 0:
            self.image_label.clear()
            self.filename_label.setText("Load image to see data")
            # Make sure upload button is visible and centered
            if not self.upload_btn.isVisible():
                self.upload_btn.show()
                # Delay centering to ensure layout is complete
                from PySide6.QtCore import QTimer
                QTimer.singleShot(10, self._center_upload_button)
            else:
                self._center_upload_button()
            return
        img = self.cache.get(self.index)
        if img is None:
            img = self.handler.get_image_at_index(self.index)
            self.cache.set(self.index, img)
        #show the 8 bit image on the workbench
        preview_img = self._ensure_uint8(img)
        qimg = self._numpy_to_qimage(preview_img)
        pix = QPixmap.fromImage(qimg)
        scaled_pix = pix.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # ============================================================
        # ROI: Draw ROI selection during selection mode
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

                x1 = min(self.roi_start_point.x(), self.roi_end_point.x())
                y1 = min(self.roi_start_point.y(), self.roi_end_point.y())
                x2 = max(self.roi_start_point.x(), self.roi_end_point.x())
                y2 = max(self.roi_start_point.y(), self.roi_end_point.y())

                x1_scaled = int(x1 * scale)
                y1_scaled = int(y1 * scale)
                w_scaled = int((x2 - x1 + 1) * scale)
                h_scaled = int((y2 - y1 + 1) * scale)

                # Draw ellipse shape
                painter.drawEllipse(x1_scaled, y1_scaled, w_scaled, h_scaled)
            painter.end()

        # ============================================================
        # ROI: Draw saved ROI when not in selection mode
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

                x_scaled = int(self.current_roi.x * scale)
                y_scaled = int(self.current_roi.y * scale)
                w_scaled = int(self.current_roi.width * scale)
                h_scaled = int(self.current_roi.height * scale)

                # Draw ellipse shape
                painter.drawEllipse(x_scaled, y_scaled, w_scaled, h_scaled)
                
                # Draw adjustment handles only when in adjustment mode
                if self.can_adjust_roi:
                    handle_size = 10
                    # Use cyan/yellow color for adjustment handles
                    painter.setPen(QPen(QColor(255, 255, 0), 2))  # Yellow border
                    painter.setBrush(QBrush(QColor(0, 255, 255)))  # Cyan fill
                    
                    # Corner handles
                    corners = [
                        (x_scaled, y_scaled),
                        (x_scaled + w_scaled, y_scaled),
                        (x_scaled, y_scaled + h_scaled),
                        (x_scaled + w_scaled, y_scaled + h_scaled),
                    ]
                    for cx, cy in corners:
                        painter.drawRect(cx - handle_size//2, cy - handle_size//2, 
                                       handle_size, handle_size)
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
        # Center upload button when window resizes
        if self.upload_btn.isVisible():
            self._center_upload_button()
    
    def _center_upload_button(self) -> None:
        """Center the upload button in the image label."""
        if not self.upload_btn.isVisible():
            return
            
        # Use parent (image_label) size for centering
        parent_width = self.image_label.width()
        parent_height = self.image_label.height()
        btn_width = self.upload_btn.width()
        btn_height = self.upload_btn.height()
        
        # Calculate center position
        x = max(0, (parent_width - btn_width) // 2)
        y = max(0, (parent_height - btn_height) // 2)
        
        self.upload_btn.move(x, y)
    
    def _open_upload_dialog(self) -> None:
        """Open file dialog to select TIF or GIF images."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Image Files",
            "",
            "Image Files (*.tif *.tiff *.gif);;TIF Files (*.tif *.tiff);;GIF Files (*.gif);;All Files (*.*)"
        )
        
        if files:
            self.set_stack(files)

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


    def _create_gif(self) -> None:
        # Check if there are images to create a GIF from
        image_count = self.handler.get_image_count()
        if image_count == 0:
            # Inform the user that no images are loaded for GIF creation
            QMessageBox.information(self, "No Images", "Load images before creating a GIF.")
            return
        # Check if the first image file exists to determine the output directory
        if not self.handler.files:
            # Inform the user that no image files are loaded for GIF creation
            QMessageBox.warning(self, "GIF Error", "No image files loaded.")
            return
        # Determine the output path for the GIF based on the first image's directory
        image_dir = Path(self.handler.files[0]).parent
        # Set the output path for the GIF in the same directory as the input images
        output_path = image_dir / "animation.gif"

        # Generate the frames for the GIF by normalizing and converting each image to uint8
        try:
            frames: List[np.ndarray] = [
                self._prepare_frame_for_gif(self.handler.get_image_at_index(i))
                for i in range(image_count)
            ]
            # Check if frames were generated successfully
            if not frames:
                raise ValueError("No frames available.")
            # Convert each frame to a Pillow Image object in grayscale mode for GIF encoding
            pil_frames = [Image.fromarray(frame, mode="L") for frame in frames]  
            # Save the frames as a GIF using Pillow
            base_frame, *extra_frames = pil_frames 
            # frame rate 
            fps = 10  
            duration_ms = int(1000 / max(1, fps))  
            # Save the GIF with the specified duration and loop settings
            base_frame.save(  
                output_path,  
                format="GIF",  
                save_all=True,  
                append_images=extra_frames,  
                duration=duration_ms,  
                loop=0,  
                disposal=2, 
            )
            #error message if the gif could not be created
        except Exception as exc:
            QMessageBox.warning(self, "GIF Error", f"Could not create GIF:\n{exc}")
            return

        self._last_gif_path = output_path
        self.preview_gif_btn.setEnabled(True)
        QMessageBox.information(self, "GIF Created", f"Saved GIF to:\n{output_path}")

    def _preview_gif(self) -> None:
        # Check if the last generated GIF path is valid before attempting to preview
        if not self._last_gif_path or not Path(self._last_gif_path).exists():
            #error message if the gif could not be found
            QMessageBox.information(self, "No GIF", "Create a GIF first to preview it.")
            self.preview_gif_btn.setEnabled(False)
            return
        # for preview of the gif window
        dialog = QDialog(self)
        dialog.setWindowTitle("GIF Preview")
        dialog_layout = QVBoxLayout(dialog)

        preview_label = QLabel(alignment=Qt.AlignCenter)
        movie = QMovie(str(self._last_gif_path))  # QMovie expects a plain string path, not Path objects
        # Check if the movie loaded successfully
        if not movie.isValid():
            QMessageBox.warning(self, "Preview Error", "Could not load the GIF for preview.")
            return
        # Set the movie to the label and start playing the GIF
        preview_label.setMovie(movie)
        dialog_layout.addWidget(preview_label)
        # Add a close button to the dialog
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        dialog_layout.addWidget(close_btn, alignment=Qt.AlignCenter)
        # start the gif
        movie.start()
        dialog._movie = movie  
        dialog.exec()

    def _toggle_roi_mode(self) -> None:
        """Toggle ROI selection mode."""
        # If we're not in selection mode and there's an existing ROI, confirm before starting new selection
        if not self.roi_selection_mode and self.current_roi is not None:
            reply = QMessageBox.question(
                self,
                "Create New ROI",
                "Creating a new ROI will replace the existing one. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            # Clear existing ROI to start fresh
            self.current_roi = None
            self.can_adjust_roi = False
            self.adjust_roi_btn.setVisible(False)
        
        self.roi_selection_mode = not self.roi_selection_mode
        self.roi_btn.setText("Cancel ROI" if self.roi_selection_mode else (
            "New ROI" if self.current_roi is not None else "Select ROI"
        ))
        if not self.roi_selection_mode:
            self.roi_start_point = None
            self.roi_end_point = None
        self._show_current()
    
    def _toggle_adjustment_mode(self) -> None:
        """Toggle ROI adjustment mode."""
        self.can_adjust_roi = not self.can_adjust_roi
        self.adjust_roi_btn.setText("Finish Adjusting" if self.can_adjust_roi else "Adjust ROI")
        
        # Disable/enable other controls based on adjustment mode
        self.prev_btn.setEnabled(not self.can_adjust_roi)
        self.next_btn.setEnabled(not self.can_adjust_roi)
        self.slider.setEnabled(not self.can_adjust_roi)
        self.roi_btn.setEnabled(not self.can_adjust_roi)
        
        # Exit adjustment mode properly
        if not self.can_adjust_roi:
            self.roi_adjustment_mode = False
            self.active_handle = ROIHandle.NONE
            self.last_mouse_pos = None
            # Emit final ROI state after adjustment is complete
            if self.current_roi is not None:
                self.roiSelected.emit(self.current_roi)
        
        self._show_current()
    
    def _update_roi_button_text(self) -> None:
        """Update ROI button text based on current state."""
        if self.roi_selection_mode:
            self.roi_btn.setText("Cancel ROI")
        elif self.current_roi is not None:
            self.roi_btn.setText("New ROI")
        else:
            self.roi_btn.setText("Select ROI")
        
        # Show/hide adjust button based on ROI existence
        if self.current_roi is not None and not self.roi_selection_mode:
            self.adjust_roi_btn.setVisible(True)
        else:
            self.adjust_roi_btn.setVisible(False)

    def _get_image_coords_from_mouse(self, event) -> Optional[Tuple[int, int, float]]:
        """Convert mouse coordinates to image coordinates and return scale."""
        # Check if there are any images loaded
        if self.handler.get_image_count() == 0:
            return None
        
        img = self.cache.get(self.index)
        if img is None:
            try:
                img = self.handler.get_image_at_index(self.index)
            except (IndexError, Exception):
                return None
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

        mouse_x = event.position().x()
        mouse_y = event.position().y()
        offset_x = (label_size.width() - scaled_pixmap_size.width()) / 2
        offset_y = (label_size.height() - scaled_pixmap_size.height()) / 2

        x = int((mouse_x - offset_x) / scale)
        y = int((mouse_y - offset_y) / scale)
        x = max(0, min(original_width - 1, x))
        y = max(0, min(original_height - 1, y))
        
        return (x, y, scale)
    
    def _on_mouse_press(self, event) -> None:
        """Handle mouse press for ROI selection and adjustment."""
        if event.button() != Qt.LeftButton:
            return
        
        # Don't process mouse events if no images loaded
        if self.handler.get_image_count() == 0:
            return
            
        coords = self._get_image_coords_from_mouse(event)
        if coords is None:
            return
        x, y, scale = coords
        
        # Check if adjusting existing ROI (only if adjustment mode is enabled)
        if self.current_roi is not None and not self.roi_selection_mode and self.can_adjust_roi:
            # Check which handle was clicked
            handle_size_image = int(10 / scale)  # Convert handle size to image coords
            self.active_handle = self.current_roi.get_handle_at_point(x, y, handle_size_image)
            if self.active_handle != ROIHandle.NONE:
                self.roi_adjustment_mode = True
                self.last_mouse_pos = QPoint(x, y)
                return
        
        # Otherwise, start new ROI selection
        if self.roi_selection_mode:
            self.roi_start_point = QPoint(x, y)
            self.roi_end_point = QPoint(x, y)

    def _on_mouse_move(self, event) -> None:
        """Handle mouse move for ROI selection and adjustment."""
        # Don't process mouse events if no images loaded
        if self.handler.get_image_count() == 0:
            return
            
        coords = self._get_image_coords_from_mouse(event)
        if coords is None:
            return
        x, y, _ = coords
        
        # Handle ROI adjustment (only if adjustment mode is enabled)
        if self.roi_adjustment_mode and self.last_mouse_pos is not None and self.can_adjust_roi:
            img = self.cache.get(self.index)
            if img is None:
                img = self.handler.get_image_at_index(self.index)
            if img is not None and img.ndim >= 2:
                original_height, original_width = img.shape[0], img.shape[1]
                
                dx = x - self.last_mouse_pos.x()
                dy = y - self.last_mouse_pos.y()
                
                self.current_roi.adjust_with_handle(
                    self.active_handle, dx, dy, original_width, original_height
                )
                
                self.last_mouse_pos = QPoint(x, y)
                self._show_current()
                # Emit change signal for live updates
                self.roiChanged.emit(self.current_roi)
        
        # Handle new ROI selection
        elif self.roi_selection_mode and self.roi_start_point is not None:
            self.roi_end_point = QPoint(x, y)
            self._show_current()

    def _on_mouse_release(self, event) -> None:
        """Handle mouse release for ROI selection and adjustment."""
        if event.button() != Qt.LeftButton:
            return
        
        # Don't process mouse events if no images loaded
        if self.handler.get_image_count() == 0:
            return
        
        # Handle ROI adjustment completion
        if self.roi_adjustment_mode:
            self.roi_adjustment_mode = False
            self.active_handle = ROIHandle.NONE
            self.last_mouse_pos = None
            # Emit final ROI after adjustment
            if self.current_roi is not None:
                self.roiSelected.emit(self.current_roi)
            return
        
        # Handle new ROI selection completion
        if self.roi_selection_mode and self.roi_start_point is not None:
            coords = self._get_image_coords_from_mouse(event)
            if coords is None:
                return
            x, y, _ = coords
            
            self.roi_end_point = QPoint(x, y)

            # Create ROI in image coordinates
            x1 = min(self.roi_start_point.x(), self.roi_end_point.x())
            y1 = min(self.roi_start_point.y(), self.roi_end_point.y())
            x2 = max(self.roi_start_point.x(), self.roi_end_point.x())
            y2 = max(self.roi_start_point.y(), self.roi_end_point.y())

            width = max(1, x2 - x1 + 1)
            height = max(1, y2 - y1 + 1)

            # Create ROI object with selected shape
            self.current_roi = ROI(x=x1, y=y1, width=width, height=height, shape=self.selected_shape)

            # Emit signal with ROI object
            self.roiSelected.emit(self.current_roi)

            # Exit selection mode
            self.roi_selection_mode = False
            self.roi_start_point = None
            self.roi_end_point = None
            self.can_adjust_roi = False  # Start with adjustment disabled
            self._update_roi_button_text()  # This will show the Adjust ROI button
            self._show_current()

    def get_current_roi(self) -> Optional[ROI]:
        """Get the current ROI object."""
        return self.current_roi

    def get_exposure(self) -> int:
        """Get the current exposure value (-100 to 100)."""
        return self.exposure_slider.value()

    def get_contrast(self) -> int:
        """Get the current contrast value (-100 to 100)."""
        return self.contrast_slider.value()

    def set_exposure(self, value: int) -> None:
        """Set the exposure value (-100 to 100)."""
        # Block signals to avoid triggering save during load
        self.exposure_slider.blockSignals(True)
        self.exposure_slider.setValue(value)
        self.exposure_slider.blockSignals(False)
        self._update_adjustment_labels()

    def set_contrast(self, value: int) -> None:
        """Set the contrast value (-100 to 100)."""
        # Block signals to avoid triggering save during load
        self.contrast_slider.blockSignals(True)
        self.contrast_slider.setValue(value)
        self.contrast_slider.blockSignals(False)
        self._update_adjustment_labels()

    def set_roi(self, roi: ROI) -> None:
        """
        Set the ROI from a saved ROI object.
        
        This method is called when loading an experiment with a saved ROI.
        The coordinates are in original image pixel space, not display/widget space.
        """
        self.current_roi = roi
        self.can_adjust_roi = False  # Start with adjustment disabled
        # Update button text
        self._update_roi_button_text()
        # Redraw to show the ROI with correct scaling
        self._show_current()
