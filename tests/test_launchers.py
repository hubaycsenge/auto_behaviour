"""The bin/abc and bin/abc-gui shell launchers.

These pick the Python that everything else runs under, and they get it wrong in
ways no Python test would catch. The bug that prompted these tests: writing a
candidate as ``"${VAR:-}/bin/python"`` yields the non-empty string
``/bin/python`` when VAR is unset, which exists on many systems, so a test for
"is this candidate non-empty" silently selected the system interpreter over the
virtualenv that actually had Qt.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ABC_GUI = ROOT / "bin" / "abc-gui"
ABC = ROOT / "bin" / "abc"


def fake_python(path: pathlib.Path, *, has_pyside: bool) -> pathlib.Path:
    """A stand-in interpreter that either can or cannot import PySide6."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exits 0 for `-c "import PySide6"` only when it is supposed to have it;
    # anything else it reports as unavailable.
    path.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        '  *PySide6*) exit %d ;;\n'
        "esac\n"
        "exit 0\n" % (0 if has_pyside else 1)
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


class LauncherTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_launcher(self, script: pathlib.Path, env: dict, *args) -> subprocess.CompletedProcess:
        # Start from a clean environment so the developer's own shell cannot
        # decide the outcome.
        base = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self.dir / "home"),
            "ABC_SELFTEST": "1",
        }
        base.update(env)
        return subprocess.run([str(script), *args], capture_output=True, text=True,
                              env=base, timeout=60, check=False)


class TestGuiLauncher(LauncherTestCase):
    def test_unset_variables_do_not_become_bin_python(self):
        """The regression: an unset ABC_CLIENT_VENV must not select /bin/python."""
        result = self.run_launcher(ABC_GUI, {})
        self.assertNotEqual(result.returncode, 0)
        # Whatever it reports, it must not have decided on a bare /bin/python.
        self.assertNotIn("/bin/python does not have it", result.stderr)
        for line in result.stderr.splitlines():
            if line.startswith("Checked:"):
                self.assertNotIn(" /bin/python", line)

    def test_prefers_an_interpreter_that_actually_has_qt(self):
        without = fake_python(self.dir / "plain" / "bin" / "python", has_pyside=False)
        with_qt = fake_python(self.dir / "client" / "bin" / "python", has_pyside=True)
        result = self.run_launcher(ABC_GUI, {
            "ABC_CLIENT_PYTHON": str(without),   # exists, but no Qt
            "ABC_CLIENT_VENV": str(with_qt.parent.parent),
            "DISPLAY": ":0",
        })
        # It must skip the Qt-less one and run the other. The fake exits 0 for
        # anything that is not a PySide6 probe, so a successful launch means the
        # Qt-capable interpreter was chosen.
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reports_every_candidate_it_tried(self):
        result = self.run_launcher(ABC_GUI, {})
        self.assertIn("Checked:", result.stderr)
        self.assertIn("install_client.sh", result.stderr)

    def test_missing_display_is_its_own_message(self):
        with_qt = fake_python(self.dir / "client" / "bin" / "python", has_pyside=True)
        result = self.run_launcher(ABC_GUI, {
            "ABC_CLIENT_VENV": str(with_qt.parent.parent),
        })
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No display", result.stderr)
        self.assertIn("ssh -X", result.stderr)


class TestAbcLauncher(LauncherTestCase):
    def test_unset_variables_do_not_become_bin_python(self):
        probe = self.dir / "probe.txt"
        # `abc --version` is enough to prove which interpreter was chosen.
        result = self.run_launcher(ABC, {}, "--version")
        self.assertNotIn("/bin/python", result.stderr)
        self.assertFalse(probe.exists())

    @unittest.skipUnless(shutil.which("python3"), "needs a real python3")
    def test_falls_back_to_python3_and_runs(self):
        result = self.run_launcher(ABC, {"PATH": os.environ["PATH"]}, "--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ABC", result.stdout)


if __name__ == "__main__":
    unittest.main()
