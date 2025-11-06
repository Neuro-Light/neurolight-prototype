"""
Tests for MainWindow close and exit experiment functionality.

Tests verify that:
- "Close Experiment" navigates to home page (StartupDialog) while staying in application
- "Exit Experiment" terminates the entire application
- Both actions show appropriate confirmation dialogs
"""

import pytest
from unittest.mock import Mock, patch
from PySide6.QtWidgets import QApplication, QMessageBox, QDialog

from core.experiment_manager import Experiment
from ui.main_window import MainWindow


@pytest.fixture
def app():
    """Create QApplication instance for tests."""
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


@pytest.fixture
def sample_experiment():
    """Create a sample experiment for testing."""
    return Experiment(
        name="Test Experiment",
        description="Test description",
        principal_investigator="Test PI",
    )


@pytest.fixture
def main_window(app, sample_experiment):
    """Create a MainWindow instance for testing."""
    # Patch the layout initialization to avoid creating actual widgets
    with patch("ui.main_window.MainWindow._init_layout"):
        window = MainWindow(sample_experiment)
        # Initialize menu manually since we're skipping layout
        window._init_menu()
        return window


class TestCloseExperiment:
    """Tests for Close Experiment functionality."""

    def test_close_experiment_shows_confirmation_dialog(self, main_window):
        """Test that close experiment shows a confirmation dialog."""
        with patch.object(QMessageBox, "question") as mock_question:
            mock_question.return_value = QMessageBox.No  # User cancels
            main_window._close_experiment()

            # Verify confirmation dialog was shown
            mock_question.assert_called_once()
            call_args = mock_question.call_args
            assert call_args[0][0] == main_window  # parent widget
            assert call_args[0][1] == "Close Experiment"  # title
            assert "return to the home page" in call_args[0][2].lower()  # message

    def test_close_experiment_cancelled_does_nothing(self, main_window):
        """Test that cancelling close experiment does nothing."""
        with (
            patch.object(QMessageBox, "question") as mock_question,
            patch.object(main_window, "hide") as mock_hide,
            patch("ui.main_window.StartupDialog") as mock_startup,
        ):
            mock_question.return_value = QMessageBox.No

            main_window._close_experiment()

            # Verify window was not hidden and startup dialog was not shown
            mock_hide.assert_not_called()
            mock_startup.assert_not_called()

    def test_close_experiment_shows_startup_dialog_when_confirmed(self, main_window):
        """Test that close experiment shows StartupDialog when confirmed."""
        mock_startup_dialog = Mock()
        mock_startup_dialog.exec.return_value = QDialog.Rejected  # User cancels startup
        mock_startup_dialog.experiment = None

        with (
            patch.object(QMessageBox, "question") as mock_question,
            patch.object(main_window, "hide") as mock_hide,
            patch(
                "ui.main_window.StartupDialog", return_value=mock_startup_dialog
            ) as mock_startup_cls,
            patch.object(QApplication, "quit") as mock_quit,
        ):
            mock_question.return_value = QMessageBox.Yes

            main_window._close_experiment()

            # Verify window was hidden and startup dialog was shown
            mock_hide.assert_called_once()
            mock_startup_cls.assert_called_once()
            mock_startup_dialog.exec.assert_called_once()
            # Since user cancelled startup dialog, app should quit
            mock_quit.assert_called_once()

    def test_close_experiment_loads_new_experiment(
        self, main_window, sample_experiment
    ):
        """Test that close experiment loads a new experiment when selected."""
        new_experiment = Experiment(
            name="New Experiment",
            description="New description",
            principal_investigator="New PI",
        )

        mock_startup_dialog = Mock()
        mock_startup_dialog.exec.return_value = QDialog.Accepted
        mock_startup_dialog.experiment = new_experiment
        mock_startup_dialog.experiment_path = "/path/to/new/experiment.nexp"

        # Setup mocks for viewer and analysis components
        main_window.viewer.image_label = Mock()
        main_window.viewer.filename_label = Mock()
        main_window.viewer.slider = Mock()
        main_window.viewer.current_roi = None
        main_window.viewer.roi_selection_mode = False
        main_window.viewer.roi_start_point = None
        main_window.viewer.roi_end_point = None

        with (
            patch.object(QMessageBox, "question") as mock_question,
            patch.object(main_window, "show") as mock_show,
            patch("ui.main_window.StartupDialog", return_value=mock_startup_dialog),
            patch.object(main_window, "setWindowTitle") as mock_set_title,
            patch.object(main_window.stack_handler, "files", create=True),
            patch("ui.main_window._LRUCache") as mock_lru_cache,
            patch.object(
                main_window.analysis.roi_plot_widget, "clear_plot"
            ) as mock_clear_plot,
            patch("ui.main_window.DataAnalyzer"),
            patch("ui.main_window.QTimer") as mock_timer,
        ):
            mock_question.return_value = QMessageBox.Yes
            mock_lru_cache_instance = Mock()
            mock_lru_cache.return_value = mock_lru_cache_instance
            mock_timer.singleShot = Mock()

            main_window._close_experiment()

            # Verify experiment was updated
            assert main_window.experiment == new_experiment
            assert main_window.current_experiment_path == "/path/to/new/experiment.nexp"
            mock_set_title.assert_called_with("Neurolight - New Experiment")
            mock_clear_plot.assert_called_once()
            mock_show.assert_called_once()


class TestExitExperiment:
    """Tests for Exit Experiment functionality."""

    def test_exit_experiment_shows_confirmation_dialog(self, main_window):
        """Test that exit experiment shows a confirmation dialog."""
        with patch.object(QMessageBox, "question") as mock_question:
            mock_question.return_value = QMessageBox.No  # User cancels
            main_window._exit_experiment()

            # Verify confirmation dialog was shown
            mock_question.assert_called_once()
            call_args = mock_question.call_args
            assert call_args[0][0] == main_window  # parent widget
            assert call_args[0][1] == "Exit Experiment"  # title
            assert "exit the application" in call_args[0][2].lower()  # message

    def test_exit_experiment_cancelled_does_not_quit(self, main_window):
        """Test that cancelling exit experiment does not quit the app."""
        with (
            patch.object(QMessageBox, "question") as mock_question,
            patch.object(QApplication, "quit") as mock_quit,
        ):
            mock_question.return_value = QMessageBox.No

            main_window._exit_experiment()

            # Verify app was not quit
            mock_quit.assert_not_called()

    def test_exit_experiment_quits_when_confirmed(self, main_window):
        """Test that exit experiment quits the application when confirmed."""
        with (
            patch.object(QMessageBox, "question") as mock_question,
            patch.object(QApplication, "quit") as mock_quit,
        ):
            mock_question.return_value = QMessageBox.Yes

            main_window._exit_experiment()

            # Verify app was quit
            mock_quit.assert_called_once()


class TestMenuActions:
    """Tests for menu action connections."""

    def test_close_action_connected(self, main_window):
        """Test that Close Experiment action is properly connected."""
        # Find the close action in the menu
        file_menu = main_window.menuBar().actions()[0].menu()
        actions = file_menu.actions()

        close_action = None
        for action in actions:
            if action.text() == "Close Experiment":
                close_action = action
                break

        assert close_action is not None, "Close Experiment action not found in menu"
        assert close_action.isEnabled()

    def test_exit_action_connected(self, main_window):
        """Test that Exit Experiment action is properly connected."""
        # Find the exit action in the menu
        file_menu = main_window.menuBar().actions()[0].menu()
        actions = file_menu.actions()

        exit_action = None
        for action in actions:
            if action.text() == "Exit Experiment":
                exit_action = action
                break

        assert exit_action is not None, "Exit Experiment action not found in menu"
        assert exit_action.isEnabled()

    def test_action_labels_are_distinct(self, main_window):
        """Test that Close and Exit actions have distinct labels."""
        file_menu = main_window.menuBar().actions()[0].menu()
        actions = file_menu.actions()

        action_texts = [action.text() for action in actions]

        assert "Close Experiment" in action_texts
        assert "Exit Experiment" in action_texts
        assert action_texts.count("Close Experiment") == 1
        assert action_texts.count("Exit Experiment") == 1
