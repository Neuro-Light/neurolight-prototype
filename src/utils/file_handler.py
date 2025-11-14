from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from core.experiment_manager import Experiment


class ImageStackHandler:
    def __init__(self) -> None:
        self.files: List[str] = []
        self._experiment: Optional[Experiment] = None

    def load_image_stack(self, directory_or_files) -> List[str]:
        paths: List[str] = []
        if isinstance(directory_or_files, (list, tuple)):
            for p in directory_or_files:
                if str(p).lower().endswith((".tif", ".tiff")):
                    paths.append(str(p))
        else:
            base = Path(directory_or_files)
            if base.is_dir():
                # Case-insensitive filter for .tif/.tiff files
                for p in sorted(base.iterdir()):
                    if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
                        paths.append(str(p))
        self.files = paths
        return self.files

    def validate_tif_files(self, file_paths: List[str]) -> bool:
        return all(str(p).lower().endswith((".tif", ".tiff")) for p in file_paths)

    def get_image_count(self) -> int:
        return len(self.files)

    def get_image_at_index(self, index: int) -> np.ndarray:
        if index < 0 or index >= len(self.files):
            raise IndexError("Image index out of range")
        with Image.open(self.files[index]) as img:
            return np.array(img)

    def get_all_frames_as_array(self) -> Optional[np.ndarray]:
        """Load all frames as a 3D numpy array (frames, height, width).
        Reuses approach from Jupyter notebook.
        Preserves original image dtype to avoid precision loss (consistent with get_image_at_index).
        """
        if not self.files:
            return None
        frame_list = []
        for file_path in self.files:
            with Image.open(file_path) as img:
                # Use np.array(img) directly to preserve dtype (consistent with get_image_at_index)
                pixels = np.array(img)

                # Handle different image modes
                if pixels.ndim == 2:  # Grayscale (mode 'L')
                    # Already in correct shape (height, width)
                    pass
                elif pixels.ndim == 3:  # Color image (RGB, RGBA, etc.)
                    # Convert to grayscale by taking mean of color channels
                    # Preserve dtype during conversion
                    pixels = pixels.mean(axis=2)
                else:
                    # Fallback: try to reshape if needed
                    if img.mode == "L":  # Grayscale
                        pixels = pixels.reshape(img.size[1], img.size[0])

                frame_list.append(pixels)
        return np.array(frame_list)

    def associate_with_experiment(self, experiment: Experiment) -> None:
        self._experiment = experiment
        # keeps that path and loads the path to images when loading an expirement
        experiment.image_count = len(self.files)
        if self.files:
            experiment.image_stack_path = str(Path(self.files[0]).parent)
            # Save the actual list of selected files
            experiment.image_stack_files = self.files.copy()
