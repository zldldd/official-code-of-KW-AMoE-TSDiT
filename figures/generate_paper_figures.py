"""Plot unmodified model inference images from a provenance manifest."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


REQUIRED_FIELDS = {
    "method", "view", "timestep", "checkpoint", "condition_hash",
    "noise_seed", "path",
}


def load_manifest(path: Path):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("The manifest must be a non-empty JSON list")
    root = path.parent
    for index, record in enumerate(records):
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(f"Record {index} misses {sorted(missing)}")
        image_path = Path(record["path"])
        if not image_path.is_absolute():
            image_path = root / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Inference image not found: {image_path}")
        record["path"] = image_path
    return records


def plot_multiview(records, output_dir: Path, timestep=None):
    if timestep is None:
        timestep = min(record["timestep"] for record in records)
    selected = [record for record in records if record["timestep"] == timestep]
    if not selected:
        raise ValueError(f"No inference records for timestep {timestep}")

    methods = list(dict.fromkeys(record["method"] for record in selected))
    views = list(dict.fromkeys(record["view"] for record in selected))
    lookup = {(record["method"], record["view"]): record for record in selected}
    missing = [(method, view) for method in methods for view in views
               if (method, view) not in lookup]
    if missing:
        raise ValueError(f"Incomplete method/view grid: {missing}")

    fig, axes = plt.subplots(
        len(methods), len(views),
        figsize=(max(3, 3 * len(views)), max(2, 2.2 * len(methods))),
        squeeze=False,
    )
    for row, method in enumerate(methods):
        for col, view in enumerate(views):
            ax = axes[row][col]
            record = lookup[(method, view)]
            with Image.open(record["path"]) as image:
                ax.imshow(image.convert("RGB"))
            ax.axis("off")
            if row == 0:
                ax.set_title(str(view))
            if col == 0:
                ax.text(
                    -0.08, 0.5, method, transform=ax.transAxes,
                    rotation=90, va="center", ha="right",
                )
    fig.suptitle(f"Real inference outputs at t={timestep}")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"multiview_t{timestep}.png"
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestep", type=int, default=None)
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    destination = plot_multiview(records, args.output_dir, args.timestep)
    print(destination)


if __name__ == "__main__":
    main()
