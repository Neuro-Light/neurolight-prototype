from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image


class GifGenerator:
    def __init__(self, fps: int = 10) -> None:
        self.frames: List[np.ndarray] = []
        self.fps = fps
    
    # Function to generate a GIF from the frames collected in the image stack
    def generate_gif(self, image_stack: List[np.ndarray], output_path: str, fps: int = 10) -> str:
        # If there is no frame in the image stack, raise an error
        if not image_stack:
            raise ValueError("Cannot generate GIF without frames")
        # Convert each frame to uint8 format    
        processed_frames = [self._to_uint8(frame) for frame in image_stack]
        # Convert each frame to a PIL Image object "pillow image"
        pil_frames = [Image.fromarray(frame) for frame in processed_frames]
        # speed of each frame
        duration_ms = int(1000 / max(1, fps))
        # Save the frames as a GIF file using the first frame as the base and appending the rest
        first, *rest = pil_frames
        first.save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=rest,
            loop=0,
            duration=duration_ms,
            optimize=True,
        )
        return output_path

    def add_frame(self, image: np.ndarray) -> None:
        self.frames.append(image)

    def optimize_gif(self, input_path: str, output_path: str) -> None:
        # Placeholder: no-op copy in MVP
        with open(input_path, "rb") as src, open(output_path, "wb") as dst:
            dst.write(src.read())
    # Helper function to convert arbitrary frame data to uint8 for GIF encoding
    # Has to be done again from the image_viewer since the frames are stored as numpy arrays and not PIL images
    def _to_uint8(self, frame: np.ndarray) -> np.ndarray:
        # If the frame is already in uint8 format, return it as is
        arr = np.asarray(frame)
        if arr.dtype == np.uint8:
            return arr
        # Normalize the array to the range [0, 255] and convert to uint8
        arr = arr.astype(np.float32, copy=True)
        arr -= arr.min()
        max_val = arr.max()
        if max_val > 0:
            arr /= max_val
        arr = np.clip(arr * 255.0, 0, 255)
        return arr.astype(np.uint8)
