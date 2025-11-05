from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "display": {"colormap": "gray", "brightness": 1.0},
        "processing": {"auto_save": True},
    })
    roi: Optional[Dict[str, int]] = None  # {"x": int, "y": int, "width": int, "height": int}

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
                    "file_list": [],
                    "count": self.image_count,
                    "format": "tif",
                    "dimensions": [],
                    "bit_depth": None,
                },
                "processing": {"history": self.processing_history},
                "analysis": {"results": self.analysis_results, "plots": []},
                "settings": self.settings,
                "roi": self.roi,
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
        return Experiment(
            name=exp.get("name", "Unnamed"),
            description=exp.get("description", ""),
            principal_investigator=exp.get("principal_investigator", ""),
            created_date=created_dt,
            modified_date=modified_dt,
            image_stack_path=image_stack.get("path") or None,
            image_count=int(image_stack.get("count") or 0),
            processing_history=exp.get("processing", {}).get("history", []),
            analysis_results=exp.get("analysis", {}).get("results", {}),
            settings=exp.get("settings", {}),
            roi=exp.get("roi"),
        )

    def update_modified_date(self) -> None:
        self.modified_date = datetime.utcnow()


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
            # Return most recent first, limit 5
            items.sort(key=lambda x: x.get("last_opened", ""), reverse=True)
            return items[:5]
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

