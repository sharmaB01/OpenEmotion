import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config():
    config_path = Path(__file__).resolve().parents[1] / "src/openemotion/config.py"
    spec = importlib.util.spec_from_file_location("openemotion_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CLASS_NAMES, module.IMG_SIZE


def main():
    parser = argparse.ArgumentParser(description="Create a reproducibility manifest")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", default="evaluation/artifact_manifest.json", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(args.model)

    class_names, image_size = load_config()
    manifest = {
        "model_path": str(args.model),
        "model_sha256": sha256(args.model),
        "class_order": class_names,
        "image_size": image_size,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "notes": args.notes,
    }
    if args.metrics and args.metrics.exists():
        manifest["metrics"] = json.loads(args.metrics.read_text(encoding="utf-8"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved reproducibility manifest to {args.output}")


if __name__ == "__main__":
    main()
