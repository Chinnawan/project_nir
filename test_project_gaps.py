"""Offline checks for the new feature variant and its report."""
import ast
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.consent import ConsentRequiredError, require_consent
from src.kleynen_feature_report import report
from src.audit_recorded import audit


class FeatureReportTests(unittest.TestCase):
    def test_capa_run_requires_a_nonempty_local_consent_record(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConsentRequiredError):
                require_consent()
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "permission.txt")
            with patch.dict(os.environ, {"CAPA_CONSENT_FILE": path}):
                with self.assertRaises(ConsentRequiredError):
                    require_consent()
                with open(path, "w", encoding="utf-8") as f:
                    f.write("")
                with self.assertRaises(ConsentRequiredError):
                    require_consent()
                with open(path, "w", encoding="utf-8") as f:
                    f.write("Example written authorization for testing this gate")
                require_consent()

    def test_consent_cli_stops_before_data_work_without_permission(self):
        env = {k: v for k, v in os.environ.items() if k != "CAPA_CONSENT_FILE"}
        run = subprocess.run([sys.executable, "-m", "src.consent"], env=env,
                             capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Obtain written consent", run.stderr)
        self.assertNotIn("Traceback", run.stderr)

    def test_existing_recorded_results_agree(self):
        self.assertEqual(audit(os.path.join("outputs", "dl")), [])

    def test_separate_rows_and_denominators(self):
        def result(accuracy):
            return {
                "erode_px": 5,
                "band_sets": {
                    "450/750/800 (paper)": {
                        "A_paper": {
                            "paper_subset_280_246": {
                                "sound_pct": 86.1, "defective_pct": 78.0, "accuracy": accuracy
                            },
                            "groups": {"stem+calyx": {"pct": 25.6}},
                        }
                    }
                },
            }

        old, new = result(0.823), result(0.8)
        old.pop("erode_px")
        old["method"] = {"fruit_mask": "Otsu + erosion 5px"}
        new["method"] = {"feature_variant": "paper_described_only"}
        text = report(old, new)
        self.assertIn("n=526", text)
        self.assertIn("82.3%", text)
        self.assertIn("80.0%", text)
        self.assertIn("NOT an exact", text)

    def test_config_root_override(self):
        env = {**os.environ, "CAPA_PROJECT_ROOT": "/tmp/capa_data"}
        output = subprocess.check_output(
            [sys.executable, "-c", "from src.config import PROJECT_ROOT; print(PROJECT_ROOT)"],
            env=env, text=True,
        )
        self.assertEqual(output.strip(), os.path.abspath("/tmp/capa_data"))

    def test_preprocessing_does_not_remove_existing_data(self):
        with open("preprocess_pipeline.py", encoding="utf-8") as f:
            module = ast.parse(f.read())
        method = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "setup_directories")
        ns = {"os": os}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "preprocess_pipeline.py", "exec"), ns)
        with tempfile.TemporaryDirectory() as root:
            out = os.path.join(root, "Processed_Dataset")
            os.mkdir(out)
            marker = os.path.join(out, "keep.txt")
            with open(marker, "w", encoding="utf-8") as f:
                f.write("keep")
            ns["OUT_DIR"] = out
            with self.assertRaises(FileExistsError):
                ns["setup_directories"]()
            self.assertTrue(os.path.isfile(marker))


try:
    import cv2
    import numpy as np
    from src.kleynen import fruit_features
except ImportError:
    fruit_features = None


@unittest.skipIf(fruit_features is None, "OpenCV, NumPy and sklearn required")
class FruitFeaturesTests(unittest.TestCase):
    def test_features_only_is_prefix_of_original(self):
        vals = np.array([[0, 10], [100, 200], [200, 100], [50, 100]], dtype=np.uint8)
        rec = {"idx": np.arange(4), "vals": vals, "shape": (2, 2)}
        lut = np.zeros(64 ** 2)
        full, full_mask = fruit_features(rec, [0, 1], lut)
        base, base_mask = fruit_features(rec, [0, 1], lut, include_region_features=False)
        self.assertEqual(len(full), 15)
        self.assertEqual(len(base), 13)
        np.testing.assert_array_equal(full[:13], base)
        np.testing.assert_array_equal(full_mask, base_mask)


if __name__ == "__main__":
    unittest.main()
