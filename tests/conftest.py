import os
import sys
from pathlib import Path

# Random backbone weights: the tests check the plumbing and run offline.
os.environ.setdefault("DEFECTS_BACKBONE_WEIGHTS", "none")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FIXTURE = Path(__file__).parent / "data" / "mvtec"
