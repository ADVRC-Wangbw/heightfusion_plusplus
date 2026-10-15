import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RefineResultsCliTest(unittest.TestCase):
    def test_smooths_linked_detections_and_preserves_frame_payload(self):
        """A removed temporal smoother would leave the middle box at x=1.5."""
        source = {
            "metadata": {"model": "HeightFusion++"},
            "frames": [
                {"frame_id": "000001", "detections": [
                    {"label": "car", "translation": [0.0, 0.0, 0.0],
                     "size": [4.0, 2.0, 1.5], "yaw": 0.0, "score": 1.0},
                ], "source": "first"},
                {"frame_id": "000002", "detections": [
                    {"label": "car", "translation": [1.5, 0.0, 0.0],
                     "size": [4.0, 2.0, 1.5], "yaw": 0.0, "score": 1.0},
                ], "source": "middle"},
                {"frame_id": "000003", "detections": [
                    {"label": "car", "translation": [2.0, 0.0, 0.0],
                     "size": [4.0, 2.0, 1.5], "yaw": 0.0, "score": 1.0},
                ], "source": "last"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = f"{directory}/results.pkl"
            output_path = f"{directory}/refined_results.pkl"
            with open(input_path, "wb") as stream:
                pickle.dump(source, stream)

            subprocess.run(
                [sys.executable, "refine_results.py", "--input", input_path, "--output", output_path,
                 "--window", "3", "--association-distance", "3"],
                check=True,
                cwd=PROJECT_ROOT,
            )

            with open(output_path, "rb") as stream:
                refined = pickle.load(stream)

        self.assertEqual(refined["metadata"], {"model": "HeightFusion++"})
        self.assertEqual([frame["source"] for frame in refined["frames"]], ["first", "middle", "last"])
        self.assertEqual(refined["frames"][1]["detections"][0]["translation"][0], 7 / 6)


if __name__ == "__main__":
    unittest.main()
