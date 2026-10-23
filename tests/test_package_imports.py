import importlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_modules_import():
    eeg_analysis = importlib.import_module("eeg_analysis")
    forced_alignment = importlib.import_module("forced_alignment")
    package_data = importlib.import_module("package_data")

    assert hasattr(eeg_analysis, "describe_project")
    assert hasattr(forced_alignment, "describe_project")
    assert hasattr(package_data, "describe_project")


def test_main_script_runs():
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Hello from eeg-envelope-coherence-imageneuro!" in result.stdout
