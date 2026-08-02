import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from PyQt6 import QtWidgets

    existing = QtWidgets.QApplication.instance()
    if existing is not None:
        yield existing
    else:
        instance = QtWidgets.QApplication([])
        yield instance
        instance.quit()


def test_gui_imports():
    from dfp_formula_project import gui

    assert hasattr(gui, "Ui_MainWindow")


def test_main_window_constructs(app):
    from dfp_formula_project.gui import Ui_MainWindow

    window = Ui_MainWindow()

    assert window.windowTitle() == "MainWindow"
    assert window.start_btn.text() == "START"
    assert window.funkcjaLabel_2.text() == "funkcja celu"
    assert window.centralwidget is not None
