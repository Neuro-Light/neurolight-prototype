from __future__ import annotations

from typing import List

import imageio
import numpy as np


class GifGenerator:
    def __init__(self, fps: int = 10) -> None:
        self.frames: List[np.ndarray] = []
        self.fps = fps

    def generate_gif(
        self, image_stack: List[np.ndarray], output_path: str, fps: int = 10
    ) -> str:
        imageio.mimsave(output_path, image_stack, duration=1.0 / max(1, fps))
        return output_path

    def add_frame(self, image: np.ndarray) -> None:
        self.frames.append(image)

    def optimize_gif(self, input_path: str, output_path: str) -> None:
        # Placeholder: no-op copy in MVP
        with open(input_path, "rb") as src, open(output_path, "wb") as dst:
            dst.write(src.read())
