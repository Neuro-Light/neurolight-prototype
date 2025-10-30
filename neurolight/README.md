# Neurolight Prototype

Neurolight is a PySide6 desktop application for processing and analyzing large TIF image stacks with an experiment-centric workflow. It is modular, extensible, and designed for collaboration.

## Installation

1) Clone or download the project and open a terminal in the project root.
2) Create a virtual environment (Windows/macOS/Linux).
3) Install dependencies: pip install -r requirements.txt
4) Run the app: python src/main.py

## Project Structure

neurolight/
- README.md
- requirements.txt
- .gitignore
- src/
  - __init__.py
  - main.py
  - ui/
    - __init__.py
    - startup_dialog.py
    - main_window.py
    - image_viewer.py
    - analysis_panel.py
  - core/
    - __init__.py
    - experiment_manager.py
    - image_processor.py
    - gif_generator.py
    - data_analyzer.py
  - utils/
    - __init__.py
    - file_handler.py
- experiments/
  - .gitkeep
- assets/
  - icons/

### Module responsibilities
- core/experiment_manager.py: Create, load, save .nexp experiments; track recent experiments.
- utils/file_handler.py: Load/validate TIF stacks; random access to frames; tie stacks to experiments.
- core/image_processor.py: OpenCV preprocessing and processing history logging.
- core/gif_generator.py: Create/optimize GIFs from image stacks.
- core/data_analyzer.py: Basic stats and plotting; store results in experiment.
- ui/startup_dialog.py: New/Load experiment and recent list; new experiment dialog.
- ui/main_window.py: Menus, image/analysis panels, autosave hook.
- ui/image_viewer.py: TIF display, navigation, LRU cache, drag-and-drop.
- ui/analysis_panel.py: Placeholder tabs for future analysis.

## Experiment Workflow

An experiment is a JSON (.nexp) with metadata, image stack info, processing history, and analysis results. Save experiments in the experiments/ directory by default. Share experiments by providing the .nexp and the referenced image stack folder.

Format highlights (v1.0): name, description, principal_investigator, created_date, modified_date, image_stack (path, count, format, dimensions, bit_depth), processing.history, analysis.results, settings.

## Usage Guide

- On launch, the Startup Dialog appears.
- Start New Experiment: enter metadata and choose a save location; a .nexp is created.
- Load Existing Experiment: select a .nexp file; recent experiments are listed for quick access.
- Main Window: left is image navigation/viewer (drag-and-drop TIFs or a folder), right is analysis tabs (placeholders). Use Previous/Next and slider to navigate. Title shows experiment name. File menu includes Save/Save As/Close/Exit.

## Application Flow

Launch → Startup Dialog → New or Load → Main Window opens with experiment context → periodic autosave (configurable later).

## Recent Experiments

Stored at ~/.neurolight/recent_experiments.json with path, name, and last_opened. The Startup Dialog shows the last five.

## Architecture Principles

- Modularity and replaceable components
- Extensibility via clear interfaces
- Session-based actions tied to experiments
- Performance: lazy loading, background threads, progress indicators
- Error handling: user-friendly messages

## Testing

- Suggested: pytest
- Placeholders in tests/ to expand: experiment creation, save/load, and validation.

## Future Expansion (hooks in code)

- Versioning, collaboration, and cloud storage
- Experiment comparison tools and standardized export
- YOLOv8 detection, real-time pipelines, advanced stats
- Batch processing and custom filters
- Dark mode and improved styling
