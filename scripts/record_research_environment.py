from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    names = ["numpy", "opencv-python", "opencv-contrib-python", "opencv-python-headless", "torch", "torchvision", "ultralytics", "transformers", "onnx", "onnxruntime", "PySide6", "pytest", "scikit-learn", "scipy", "matplotlib"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    payload = {"python": platform.python_version(), "system": platform.system(), "architecture": platform.machine(), "packages": versions}
    output = root / "docs/results/research_environment.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
