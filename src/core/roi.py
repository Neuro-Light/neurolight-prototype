"""ROI (Region of Interest) data structures and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np


class ROIShape(Enum):
    """ROI shape types."""

    ELLIPSE = "ellipse"  # Legacy support only, not used in UI


class ROIHandle(Enum):
    """ROI adjustment handles."""

    NONE = 0
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4
    MOVE = 5  # For dragging the entire ROI


@dataclass
class ROI:
    """
    Ellipse Region of Interest.

    Coordinates are stored in original image pixel space, not display space.
    This ensures ROI stays fixed to image region regardless of scaling.

    Structure:
    - x, y: top-left corner of bounding box
    - width, height: dimensions of bounding box
    - shape: always ELLIPSE

    The ellipse center is at (x + width/2, y + height/2)
    with semi-axes of width/2 and height/2.
    """

    x: int
    y: int
    width: int
    height: int
    shape: ROIShape = ROIShape.ELLIPSE

    def to_dict(self) -> dict:
        """Convert ROI to dictionary for serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "shape": self.shape.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ROI:
        """Create ROI from dictionary with safe defaults for missing keys."""
        shape_str = data.get("shape", "ellipse")
        try:
            shape = ROIShape(shape_str)
        except ValueError:
            shape = ROIShape.ELLIPSE

        return cls(
            x=data.get("x", 0),
            y=data.get("y", 0),
            width=data.get("width", 100),
            height=data.get("height", 100),
            shape=shape,
        )

    def get_center(self) -> Tuple[float, float]:
        """Get center coordinates of ROI."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def contains_point(self, px: int, py: int) -> bool:
        """Check if a point is inside the ellipse ROI."""
        # Check if point is inside ellipse using equation:
        # ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
        cx, cy = self.get_center()
        rx = self.width / 2
        ry = self.height / 2
        if rx == 0 or ry == 0:
            return False
        normalized = ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2
        return normalized <= 1.0

    def get_handle_at_point(self, px: int, py: int, handle_size: int = 8) -> ROIHandle:
        """
        Determine which handle (if any) is at the given point.

        Args:
            px, py: Point coordinates in image space
            handle_size: Size of handle hit area in image pixels

        Returns:
            ROIHandle indicating which handle was clicked
        """
        # Check corner handles
        corners = [
            (self.x, self.y, ROIHandle.TOP_LEFT),
            (self.x + self.width, self.y, ROIHandle.TOP_RIGHT),
            (self.x, self.y + self.height, ROIHandle.BOTTOM_LEFT),
            (self.x + self.width, self.y + self.height, ROIHandle.BOTTOM_RIGHT),
        ]

        for cx, cy, handle in corners:
            if abs(px - cx) <= handle_size and abs(py - cy) <= handle_size:
                return handle

        # Check if inside ROI for move
        if self.contains_point(px, py):
            return ROIHandle.MOVE

        return ROIHandle.NONE

    def adjust_with_handle(
        self, handle: ROIHandle, dx: int, dy: int, image_width: int, image_height: int
    ) -> None:
        """
        Adjust ROI based on handle being dragged.

        Args:
            handle: Which handle is being dragged
            dx, dy: Delta movement in image pixels
            image_width, image_height: Image dimensions for clamping
        """
        if handle == ROIHandle.MOVE:
            # Move entire ROI
            self.x = max(0, min(image_width - self.width, self.x + dx))
            self.y = max(0, min(image_height - self.height, self.y + dy))

        elif handle == ROIHandle.TOP_LEFT:
            # Adjust top-left corner
            new_x = max(0, min(self.x + self.width - 10, self.x + dx))
            new_y = max(0, min(self.y + self.height - 10, self.y + dy))
            self.width += self.x - new_x
            self.height += self.y - new_y
            self.x = new_x
            self.y = new_y

        elif handle == ROIHandle.TOP_RIGHT:
            # Adjust top-right corner
            new_y = max(0, min(self.y + self.height - 10, self.y + dy))
            self.width = max(10, min(image_width - self.x, self.width + dx))
            self.height += self.y - new_y
            self.y = new_y

        elif handle == ROIHandle.BOTTOM_LEFT:
            # Adjust bottom-left corner
            new_x = max(0, min(self.x + self.width - 10, self.x + dx))
            self.width += self.x - new_x
            self.height = max(10, min(image_height - self.y, self.height + dy))
            self.x = new_x

        elif handle == ROIHandle.BOTTOM_RIGHT:
            # Adjust bottom-right corner
            self.width = max(10, min(image_width - self.x, self.width + dx))
            self.height = max(10, min(image_height - self.y, self.height + dy))

    def create_mask(self, image_width: int, image_height: int) -> np.ndarray:
        """
        Create a binary mask for the ellipse ROI.

        Args:
            image_width: Width of the image
            image_height: Height of the image

        Returns:
            Binary mask (uint8) where ROI region is 255, rest is 0
        """
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Create ellipse mask
        cx, cy = self.get_center()
        rx = self.width / 2
        ry = self.height / 2

        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:image_height, :image_width]

        # Ellipse equation: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
        if rx > 0 and ry > 0:
            ellipse_mask = ((x_coords - cx) / rx) ** 2 + ((y_coords - cy) / ry) ** 2 <= 1
            mask[ellipse_mask] = 255

        return mask
