# 🧠 Neurolight Prototype

A powerful PySide6 desktop application for processing and analyzing large TIF image stacks with scientific rigor. Built for neuroscientists and researchers who need reproducible, shareable experiment workflows.

---

## ✨ Features

- 🔬 **Experiment-Centric Workflow** – All work is organized into shareable experiment sessions
- 📸 **High-Volume Image Processing** – Handle 200+ TIF image stacks with ease
- 🎨 **Intuitive Interface** – Split-panel design with image navigation and analysis dashboard
- 🔄 **Processing Pipeline** – OpenCV integration with full history tracking
- 📊 **Scientific Analysis** – Built on NumPy, SciPy, and Matplotlib
- 💾 **Auto-Save** – Never lose your work with periodic session saving
- 🤝 **Collaboration Ready** – Share experiments as portable JSON files

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Install uv** (if not already installed)

   **Option A: Using pip (recommended)**

   ```bash
   Windows:
   pip install uv

   Mac:
   brew install uv
   ```

   **Option B: Official installer**

   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone or download the project**

   ```bash
   git clone https://github.com/Neuro-Light/neurolight-prototype
   cd neurolight-prototype
   ```

3. **Install dependencies and create virtual environment**

   ```bash
   uv sync
   ```

   This will automatically create a virtual environment (`.venv`) and install all runtime dependencies needed to run the application.

4. **Launch the application**

   **Recommended: Use `uv run`** (automatically uses the virtual environment):

   ```bash
   uv run neurolight
   ```

   This command works universally across Windows, macOS, and Linux. The `uv run` command automatically uses the virtual environment and runs the application.

   **Alternative: Manually activate the virtual environment**

   If you prefer to activate the venv yourself:

   **Windows:**

   ```bash
   .venv\Scripts\activate
   neurolight
   ```

   **macOS/Linux:**

   ```bash
   source .venv/bin/activate
   neurolight
   ```

### Development Tools (Optional)

For contributors and developers who want to run linting, type checking, or tests:

```bash
# Install development tools (ruff, mypy)
uv sync --group dev

# Install test tools (pytest, pytest-cov)
uv sync --extra test

# Or install everything at once
uv sync --group dev --extra test
```

**Note:** These tools are not required to run the application. They're only needed if you plan to:
- Contribute code (linting and type checking)
- Run the test suite
- Develop new features

For detailed information about development tools and how to use them, see [DEVELOPMENT.md](DEVELOPMENT.md).

---

## 📁 Project Structure

```
neurolight-prototype/
│
├── 📄 README.md
├── 📄 CONTRIBUTING.md
├── 📄 pyproject.toml
├── 📄 uv.lock
├── 📄 LICENSE-APACHE
├── 📄 LICENSE-MIT
├── 📄 .gitignore
│
├── 📂 src/
│   ├── __init__.py
│   ├── main.py                    # Application entry point
│   │
│   ├── 📂 ui/                     # User interface components
│   │   ├── __init__.py
│   │   ├── startup_dialog.py     # Experiment selection screen
│   │   ├── main_window.py        # Main application window
│   │   ├── image_viewer.py       # Image display & navigation
│   │   ├── analysis_panel.py     # Analysis dashboard
│   │   ├── alignment_dialog.py   # Image alignment interface
│   │   ├── alignment_progress_dialog.py  # Alignment progress tracking
│   │   ├── loading_dialog.py     # Loading indicators
│   │   ├── neuron_detection_widget.py    # Neuron detection UI
│   │   ├── neuron_trajectory_plot.py     # Trajectory visualization
│   │   └── roi_intensity_plot.py  # ROI intensity plotting
│   │
│   ├── 📂 core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── experiment_manager.py # Experiment session handling
│   │   ├── image_processor.py    # OpenCV processing pipeline
│   │   ├── gif_generator.py      # Animation export
│   │   ├── data_analyzer.py      # Statistical analysis
│   │   └── roi.py                # Region of Interest handling
│   │
│   └── 📂 utils/                  # Utilities
│       ├── __init__.py
│       └── file_handler.py       # TIF stack I/O
│
├── 📂 experiments/                # Default experiment storage
├── 📂 assets/
│   └── 📂 icons/
└── 📂 tests/                      # Unit tests
    ├── test_experiment_manager.py
    └── test_main_window_close_exit.py
```

### 🔧 Module Responsibilities

| Module                           | Purpose                                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| **experiment_manager.py**        | Create, load, save `.nexp` experiments; manage recent experiments list                   |
| **file_handler.py**              | Load/validate TIF stacks; provide random frame access; associate stacks with experiments |
| **image_processor.py**           | Apply OpenCV operations; maintain processing history for reproducibility                 |
| **gif_generator.py**             | Generate and optimize animated GIFs from image sequences                                 |
| **data_analyzer.py**             | Calculate statistics, generate plots; store results in experiment sessions               |
| **roi.py**                       | Region of Interest (ROI) data structures and operations                                  |
| **startup_dialog.py**            | Present new/load experiment options; show recent experiments                             |
| **main_window.py**               | Coordinate menus, panels, and auto-save functionality                                    |
| **image_viewer.py**              | Display TIFs with navigation controls; implement LRU caching; handle drag-and-drop       |
| **analysis_panel.py**            | Provide tabbed interface for future analysis tools                                       |
| **alignment_dialog.py**          | User interface for image alignment operations                                            |
| **alignment_progress_dialog.py** | Progress tracking and feedback for alignment processes                                   |
| **loading_dialog.py**            | Loading indicators and progress feedback for long-running operations                     |
| **neuron_detection_widget.py**   | UI components for neuron detection and tracking                                          |
| **neuron_trajectory_plot.py**    | Visualization of neuron trajectories over time                                           |
| **roi_intensity_plot.py**        | Plotting ROI intensity values across image sequences                                     |

---

## 🔬 Experiment Workflow

### What is an Experiment?

An **experiment** is a complete research session stored as a JSON file (`.nexp`) containing:

- 📋 Metadata (name, description, principal investigator, dates)
- 🖼️ Image stack information (path, dimensions, bit depth)
- ⚙️ Processing history (all operations and parameters)
- 📈 Analysis results (statistics, plots)
- 🎛️ Custom settings

### Experiment File Format (v1.0)

```json
{
  "version": "1.0",
  "experiment": {
    "name": "Cortical Response Study 001",
    "description": "Analysis of cortical neurons under stimulation",
    "principal_investigator": "Dr. Jane Smith",
    "created_date": "2025-10-30T10:30:00",
    "modified_date": "2025-10-30T14:45:00",
    "image_stack": {
      "path": "/path/to/images/",
      "file_list": ["image001.tif", "image002.tif"],
      "count": 200,
      "format": "tif",
      "dimensions": [1024, 1024],
      "bit_depth": 16
    },
    "processing": {
      "history": [...]
    },
    "analysis": {
      "results": {}
    },
    "settings": {}
  }
}
```

### Sharing Experiments

To collaborate with colleagues:

1. Export the `.nexp` file from your `experiments/` directory
2. Include the referenced image stack folder
3. Colleagues can load the experiment and reproduce your entire workflow

**Note:** Both the `.nexp` file and the image stack folder must be shared; the experiment file references image paths, so the folder structure should be preserved when sharing.

---

## 📖 Usage Guide

### Starting the Application

Run the application with:

```bash
uv run neurolight
```

**Launch Screen:**

1. Application opens to the **Startup Dialog**
2. Choose your path:
   - 🆕 **Start New Experiment** – Enter metadata and create a fresh session
   - 📂 **Load Existing Experiment** – Browse for an existing `.nexp` file
   - 🕒 **Recent Experiments** – Quick access to your last 5 experiments

### Working with Experiments

**Creating a New Experiment:**

- Provide experiment name (required)
- Add description and principal investigator
- Choose save location (defaults to `experiments/` directory)
- Click **Create** to begin

**Main Application Window:**

**Left Panel** (Image Navigation):

- Drag-and-drop TIF files or an entire folder
- Navigate frames with **Previous/Next** buttons
- Use the slider for quick jumping
- Frame counter displays current position

**Right Panel** (Analysis Dashboard):

- Tabbed interface with placeholders for:
  - 📊 Statistics
  - 📈 Graphs
  - 🎯 Detection (YOLOv8 integration planned)

**Menu Bar:**

- **File**: Save, Save As, Close Experiment, Open Image Stack, Export, Exit
- **Edit**: Experiment Settings (edit metadata)
- **Tools**: Generate GIF, Run Analysis (coming soon)
- **Help**: About

### Recent Experiments

Recent experiments are tracked in `~/.neurolight/recent_experiments.json` and display:

- Experiment name
- Last modified date
- Full file path

Double-click any recent experiment to load it instantly.

---

## 🏗️ Architecture

### Design Principles

- **🧩 Modularity** – Independent, replaceable components
- **🔌 Extensibility** – Clear interfaces for adding new features
- **💼 Session Management** – All actions tied to experiment context
- **⚡ Performance** – Lazy loading, background threads, progress feedback
- **🛡️ Error Handling** – Graceful failures with user-friendly messages

### Performance Features

- **Lazy Image Loading** – Images loaded on-demand, not all at once
- **LRU Cache** – Keeps ~20 recently viewed images in memory
- **Background Processing** – Long operations don't freeze the UI
- **Auto-Save** – Periodic background saves (configurable)

### Application Flow

```
Launch → Startup Dialog → Create/Load Experiment → Main Window → Auto-Save Loop
```

---

## 🧪 Testing

### Framework

We recommend **pytest** for unit and integration testing.

### Test Structure

```
tests/
├── test_experiment_manager.py
├── test_file_handler.py
├── test_image_processor.py
└── test_ui_components.py
```

### Running Tests

First, install test dependencies:

```bash
uv sync --extra test
```

Then run the tests:

```bash
uv run pytest tests/
```

Or if you've manually activated the virtual environment:

```bash
pytest tests/
```

---

## 🚧 Future Roadmap

### Planned Features

**Collaboration & Sharing:**

- 🔄 Experiment versioning and history
- ☁️ Cloud storage integration
- 🤝 Multi-user experiment comparison tools
- 📤 Export to standardized formats (HDF5, OME-TIFF)

**Advanced Analysis:**

- 🎯 YOLOv8 object detection integration
- ⚙️ Real-time processing pipelines
- 📊 Advanced statistical modeling (statsmodels)
- 🔬 Custom filter creation interface

**User Experience:**

- 🌙 Dark mode support
- 📦 Batch processing capabilities
- 🎨 Custom themes and layouts
- 📋 Experiment templates for common workflows

---

## 🤝 Contributing

Areas for improvement:

- Additional image processing algorithms
- New analysis visualizations
- UI/UX enhancements
- Documentation improvements
- Bug reports and feature requests

---

## 📝 License

MIT

---

## 🙏 Acknowledgments

Built with:

- [PySide6](https://doc.qt.io/qtforpython/) – Qt for Python
- [OpenCV](https://opencv.org/) – Computer vision library
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) – Scientific computing
- [Matplotlib](https://matplotlib.org/) – Plotting and visualization
- [YOLOv8](https://github.com/ultralytics/ultralytics) – Object detection (planned)

---

<div align="center">

**Made with 🧠 for neuroscience research**

</div>
