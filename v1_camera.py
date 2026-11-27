from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from main import launch

if __name__ == "__main__":
    launch(1)
