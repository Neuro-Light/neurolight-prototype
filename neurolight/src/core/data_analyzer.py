from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from core.experiment_manager import Experiment


class DataAnalyzer:
    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def calculate_statistics(self, data: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }

    def generate_plots(self, data: np.ndarray, plot_type: str = "hist"):
        fig, ax = plt.subplots()
        if plot_type == "hist":
            ax.hist(data.flatten(), bins=50)
            ax.set_title("Histogram")
        else:
            ax.plot(data)
        return fig

    def save_results_to_experiment(self, experiment: Experiment) -> None:
        experiment.analysis_results.setdefault("runs", []).append({
            "summary": "Placeholder analysis",
        })

    # Placeholders for future expansion
    def time_series_analysis(self, data: np.ndarray):
        return {}

    def correlation_analysis(self, data_a: np.ndarray, data_b: np.ndarray):
        return {}

