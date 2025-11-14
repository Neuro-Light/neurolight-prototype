import sys
from pathlib import Path

# Add src directory to Python path so imports work correctly
# This allows imports like "from core.experiment_manager" to work
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from ui.startup_dialog import StartupDialog
from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)

    # Show startup dialog (modal)
    startup = StartupDialog()
    result = startup.exec()

    if result != StartupDialog.Accepted or startup.experiment is None:
        return 0

    # Create main window with experiment context
    main_window = MainWindow(startup.experiment)
    # Carry over the .nexp path so autosaves and path updates persist
    try:
        main_window.current_experiment_path = startup.experiment_path  # type: ignore[attr-defined]
    except Exception:
        pass
    main_window.show()

    # Auto-save timer placeholder (configurable later)
    autosave_timer = QTimer()
    autosave_timer.setInterval(5 * 60 * 1000)
    autosave_timer.timeout.connect(main_window.autosave_experiment)
    autosave_timer.start()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
