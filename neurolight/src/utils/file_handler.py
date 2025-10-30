from __future__ import annotations

import os
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

    def associate_with_experiment(self, experiment: Experiment) -> None:
        self._experiment = experiment
        if experiment:
            experiment.image_stack_path = str(Path(self.files[0]).parent) if self.files else None
            experiment.image_count = len(self.files)

