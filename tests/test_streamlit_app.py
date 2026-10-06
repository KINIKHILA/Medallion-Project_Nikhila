import unittest
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from streamlit.testing.v1 import AppTest


class StreamlitAppTest(unittest.TestCase):
    def test_app_starts(self):
        app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=15)

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Medallion Pipeline")
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(app.button[0].label, "Start flow")
        self.assertTrue(app.button[0].disabled)
        self.assertEqual(len(app.text_area), 0)

    def test_app_imports_from_outside_repository(self):
        app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
        with TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-c", "import sys; from streamlit.testing.v1 import AppTest; app = AppTest.from_file(sys.argv[1]).run(timeout=15); assert not app.exception, app.exception", str(app_path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=25,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()