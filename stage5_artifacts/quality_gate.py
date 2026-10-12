"""ML quality gate: exit code 1 (pipeline FAIL) if the model is below the thresholds."""
import json
import os
import sys
from pathlib import Path

MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.75"))
MIN_F1 = float(os.getenv("MIN_F1", "0.65"))


def main():
    m = json.loads((Path(__file__).resolve().parent / "metrics.json").read_text())
    print(f"Accuracy: {m['accuracy']} (required >= {MIN_ACCURACY})")
    print(f"F1 score: {m['f1']} (required >= {MIN_F1})")
    if m["accuracy"] < MIN_ACCURACY or m["f1"] < MIN_F1:
        print("QUALITY GATE: FAIL")
        sys.exit(1)
    print("QUALITY GATE: PASS")


if __name__ == "__main__":
    main()
