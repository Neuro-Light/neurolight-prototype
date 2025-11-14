from __future__ import annotations

import json
import os
import base64
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


CONFIG_DIR = Path.home() / ".neurolight"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
RECENT_FILE = CONFIG_DIR / "recent_experiments.json"


@dataclass
class Experiment:
    name: str
    description: str = ""
    principal_investigator: str = ""
    created_date: datetime = field(default_factory=datetime.utcnow)
    modified_date: datetime = field(default_factory=datetime.utcnow)
    image_stack_path: Optional[str] = None
    image_count: int = 0
    image_stack_files: List[str] = field(default_factory=list)  # List of selected file paths
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(
        default_factory=lambda: {
            "display": {
                "colormap": "gray",
                "brightness": 1.0,
                "exposure": 0,
                "contrast": 0,
            },
            "processing": {"auto_save": True},
        }
    )
    # Store ROI coordinates in image pixel space (not widget/display space)
    # Format: {"x": int, "y": int, "width": int, "height": int, "shape": str}
    # where shape is "ellipse" (rectangle kept for legacy compatibility only)
    # These coordinates are in original image pixels, ensuring ROI stays fixed to
    # the image region regardless of window size or scaling
    roi: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "experiment": {
                "name": self.name,
                "description": self.description,
                "principal_investigator": self.principal_investigator,
                "created_date": self.created_date.isoformat(timespec="seconds"),
                "modified_date": self.modified_date.isoformat(timespec="seconds"),
                "image_stack": {
                    "path": self.image_stack_path or "",
                    "file_list": self.image_stack_files,  # Save actual selected files
                    "count": self.image_count,
                    "format": "tif",
                    "dimensions": [],
                    "bit_depth": None,
                },
                "processing": {"history": self.processing_history},
                "analysis": {"results": self.analysis_results, "plots": []},
                "settings": self.settings,
                # Save ROI coordinates to .nexp file
                # Coordinates are in image pixel space, not display/widget space
                "roi": self.roi,
                # Save neuron detection data if available
                "neuron_detection": self._serialize_neuron_detection(),
            },
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Experiment":
        exp = data.get("experiment", {})
        created = exp.get("created_date")
        modified = exp.get("modified_date")
        created_dt = datetime.fromisoformat(created) if created else datetime.utcnow()
        modified_dt = datetime.fromisoformat(modified) if modified else datetime.utcnow()
        image_stack = exp.get("image_stack", {})
        experiment = Experiment(
            name=exp.get("name", "Unnamed"),
            description=exp.get("description", ""),
            principal_investigator=exp.get("principal_investigator", ""),
            created_date=created_dt,
            modified_date=modified_dt,
            image_stack_path=image_stack.get("path") or None,
            image_count=int(image_stack.get("count") or 0),
            image_stack_files=image_stack.get("file_list", []),  # Load saved file list
            processing_history=exp.get("processing", {}).get("history", []),
            analysis_results=exp.get("analysis", {}).get("results", {}),
            settings=exp.get("settings", {}),
            # Load ROI coordinates from .nexp file
            # Coordinates are in image pixel space and will be converted to display
            # coordinates when drawing (see image_viewer.py _show_current method)
            roi=exp.get("roi"),
        )
        # Load neuron detection data if available
        neuron_detection = exp.get("neuron_detection")
        if neuron_detection:
            experiment._deserialize_neuron_detection(neuron_detection)
        return experiment

    def update_modified_date(self) -> None:
        self.modified_date = datetime.utcnow()

    def _serialize_neuron_detection(self) -> Optional[Dict[str, Any]]:
        """Serialize neuron detection data to JSON-serializable format."""
        # Check if data exists
        if not hasattr(self, "_neuron_detection_data"):
            return None

        data = self._neuron_detection_data
        if data is None:
            return None

        # Ensure it's a dict
        if not isinstance(data, dict):
            return None

        result = {}

        # Serialize neuron locations (2D array: neurons x 2)
        if "neuron_locations" in data and data["neuron_locations"] is not None:
            locations = data["neuron_locations"]
            if isinstance(locations, np.ndarray):
                if locations.size > 0:  # Only serialize if not empty
                    result["neuron_locations"] = locations.tolist()
            else:
                result["neuron_locations"] = locations

        # Serialize neuron trajectories (2D array: neurons x frames)
        # Use base64 encoding for large arrays to reduce JSON size
        if "neuron_trajectories" in data and data["neuron_trajectories"] is not None:
            trajectories = data["neuron_trajectories"]
            if isinstance(trajectories, np.ndarray):
                if trajectories.size > 0:  # Only serialize if not empty
                    # Convert to base64-encoded string for efficiency
                    trajectories_bytes = trajectories.tobytes()
                    trajectories_b64 = base64.b64encode(trajectories_bytes).decode("utf-8")
                    result["neuron_trajectories"] = {
                        "data": trajectories_b64,
                        "shape": list(trajectories.shape),
                        "dtype": str(trajectories.dtype),
                    }
            else:
                result["neuron_trajectories"] = trajectories

        # Serialize quality mask (1D boolean array)
        if "quality_mask" in data and data["quality_mask"] is not None:
            quality_mask = data["quality_mask"]
            if isinstance(quality_mask, np.ndarray):
                if quality_mask.size > 0:  # Only serialize if not empty
                    result["quality_mask"] = quality_mask.tolist()
            else:
                result["quality_mask"] = quality_mask

        # Note: mean_frame is NOT saved - it can be recalculated from frame_data and ROI mask
        # This significantly reduces file size since mean_frame is often mostly zeros
        # and would produce long base64 strings of "A" characters

        # Save detection parameters
        if "detection_params" in data:
            result["detection_params"] = data["detection_params"]

        return result if result else None

    def _deserialize_neuron_detection(self, data: Dict[str, Any]) -> None:
        """Deserialize neuron detection data from JSON format."""
        if not hasattr(self, "_neuron_detection_data"):
            self._neuron_detection_data = {}

        # Deserialize neuron locations
        if "neuron_locations" in data:
            self._neuron_detection_data["neuron_locations"] = np.array(
                data["neuron_locations"], dtype=np.int32
            )

        # Deserialize neuron trajectories (base64-encoded)
        if "neuron_trajectories" in data:
            traj_data = data["neuron_trajectories"]
            if isinstance(traj_data, dict) and "data" in traj_data:
                # Base64-encoded format
                trajectories_b64 = traj_data["data"]
                shape = tuple(traj_data["shape"])
                dtype = np.dtype(traj_data["dtype"])
                trajectories_bytes = base64.b64decode(traj_data["data"])
                self._neuron_detection_data["neuron_trajectories"] = np.frombuffer(
                    trajectories_bytes, dtype=dtype
                ).reshape(shape)
            else:
                # Legacy format (list)
                self._neuron_detection_data["neuron_trajectories"] = np.array(
                    traj_data, dtype=np.float32
                )

        # Deserialize quality mask
        if "quality_mask" in data:
            self._neuron_detection_data["quality_mask"] = np.array(data["quality_mask"], dtype=bool)

        # Note: mean_frame is NOT loaded - it will be recalculated when needed
        # This is fine since mean_frame is only used for visualization and can be
        # recalculated from frame_data and ROI mask

        # Load detection parameters
        if "detection_params" in data:
            self._neuron_detection_data["detection_params"] = data["detection_params"]

    def set_neuron_detection_data(
        self,
        neuron_locations: Optional[np.ndarray] = None,
        neuron_trajectories: Optional[np.ndarray] = None,
        quality_mask: Optional[np.ndarray] = None,
        mean_frame: Optional[np.ndarray] = None,
        detection_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store neuron detection data in the experiment."""
        # Always initialize the dict if it doesn't exist
        if not hasattr(self, "_neuron_detection_data"):
            self._neuron_detection_data = {}
        elif self._neuron_detection_data is None:
            self._neuron_detection_data = {}

        # Set data only if provided (even if empty arrays)
        if neuron_locations is not None:
            self._neuron_detection_data["neuron_locations"] = neuron_locations
        if neuron_trajectories is not None:
            self._neuron_detection_data["neuron_trajectories"] = neuron_trajectories
        if quality_mask is not None:
            self._neuron_detection_data["quality_mask"] = quality_mask
        if mean_frame is not None:
            self._neuron_detection_data["mean_frame"] = mean_frame
        if detection_params is not None:
            self._neuron_detection_data["detection_params"] = detection_params

    def get_neuron_detection_data(self) -> Optional[Dict[str, Any]]:
        """Get stored neuron detection data."""
        if hasattr(self, "_neuron_detection_data"):
            return self._neuron_detection_data
        return None


class ExperimentManager:
    def __init__(self) -> None:
        RECENT_FILE.touch(exist_ok=True)
        if RECENT_FILE.stat().st_size == 0:
            RECENT_FILE.write_text(json.dumps({"recent": []}, indent=2))

    def create_new_experiment(self, metadata: Dict[str, Any]) -> Experiment:
        name = metadata.get("name", "").strip()
        if not name:
            raise ValueError("Experiment name cannot be empty")
        experiment = Experiment(
            name=name,
            description=metadata.get("description", ""),
            principal_investigator=metadata.get("principal_investigator", ""),
            created_date=metadata.get("created_date", datetime.utcnow()),
            modified_date=datetime.utcnow(),
        )
        return experiment

    def load_experiment(self, file_path: str) -> Experiment:
        if not self.validate_experiment_file(file_path):
            raise ValueError("Invalid experiment file")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        experiment = Experiment.from_json(data)
        self.add_to_recent(file_path, experiment.name)
        return experiment

    def save_experiment(self, experiment: Experiment, file_path: Optional[str] = None) -> bool:
        if not file_path:
            raise ValueError("file_path is required for saving experiment")
        experiment.update_modified_date()
        payload = experiment.to_json()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.add_to_recent(file_path, experiment.name)
        return True

    def validate_experiment_file(self, file_path: str) -> bool:
        try:
            if not os.path.isfile(file_path):
                return False
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return False
            if data.get("version") != "1.0":
                return False
            exp = data.get("experiment", {})
            return "name" in exp
        except Exception:
            return False

    def get_recent_experiments(self) -> List[Dict[str, Any]]:
        try:
            with open(RECENT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or {"recent": []}
            items = data.get("recent", [])

            # Filter out experiments that no longer exist on disk
            valid_items = []
            invalid_paths = []
            for item in items:
                path = item.get("path", "")
                if path and os.path.isfile(path) and self.validate_experiment_file(path):
                    valid_items.append(item)
                else:
                    invalid_paths.append(path)

            # Remove invalid entries from recent file if any were found
            if invalid_paths:
                data["recent"] = valid_items
                with open(RECENT_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            # Return most recent first, limit 5
            valid_items.sort(key=lambda x: x.get("last_opened", ""), reverse=True)
            return valid_items[:5]
        except Exception:
            return []

    def add_to_recent(self, file_path: str, name: Optional[str] = None) -> None:
        file_path = str(Path(file_path).resolve())
        entry = {
            "path": file_path,
            "name": name or Path(file_path).stem,
            "last_opened": datetime.utcnow().isoformat(timespec="seconds"),
        }
        try:
            with open(RECENT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or {"recent": []}
        except Exception:
            data = {"recent": []}
        # Remove duplicates
        data["recent"] = [e for e in data.get("recent", []) if e.get("path") != file_path]
        data["recent"].insert(0, entry)
        data["recent"] = data["recent"][:20]
        with open(RECENT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def remove_from_recent(self, file_path: str) -> None:
        """Remove an experiment from the recent experiments list."""
        file_path = str(Path(file_path).resolve())
        try:
            with open(RECENT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or {"recent": []}
        except Exception:
            data = {"recent": []}
        # Remove the entry
        data["recent"] = [e for e in data.get("recent", []) if e.get("path") != file_path]
        with open(RECENT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def delete_experiment(self, file_path: str, delete_file: bool = False) -> bool:
        """
        Delete an experiment from recent list and optionally delete the file.

        Args:
            file_path: Path to the experiment file
            delete_file: If True, also delete the experiment file from disk

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = str(Path(file_path).resolve())

            # Remove from recent list
            self.remove_from_recent(file_path)

            # Optionally delete the file
            if delete_file and os.path.isfile(file_path):
                os.remove(file_path)
                return True

            return True
        except Exception:
            return False
