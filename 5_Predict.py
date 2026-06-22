import argparse
import json
import os
from pathlib import Path

from evaluation.utils import predict_test_set


def collect_images(path):
    """Collect image paths from a file, directory, or single-image path."""
    image_path = Path(path)

    if image_path.is_file() and image_path.suffix.lower() == ".txt":
        return [
            line.strip()
            for line in image_path.read_text().splitlines()
            if line.strip()
        ]

    if image_path.is_file():
        return [str(image_path)]

    if image_path.is_dir():
        images = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            images.extend(str(p) for p in image_path.glob(pattern))
            images.extend(str(p) for p in image_path.glob(pattern.upper()))
        return sorted(set(images))

    raise FileNotFoundError(f"Could not find images at: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Mask-HybridGNet inference on new images."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the prepared dataset directory (must contain config.json and graph matrices)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default=None,
        help="Path to hyperparameters.json (default: same folder as the checkpoint)",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Folder of images, a single image path, or a .txt file listing image paths. "
             "If omitted, uses {dataset}/test.txt and {dataset}/images/",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory where segmentations (and optional landmarks) will be saved",
    )
    parser.add_argument(
        "--landmarks",
        action="store_true",
        help="Also export predicted organ landmarks as JSON files",
    )
    parser.add_argument(
        "--representation",
        choices=["independent", "unified"],
        default=None,
        help="Override the graph representation stored in hyperparameters.json",
    )

    args = parser.parse_args()

    hyperparameters_path = args.hyperparameters
    if hyperparameters_path is None:
        hyperparameters_path = Path(args.checkpoint).parent / "hyperparameters.json"

    with open(hyperparameters_path) as f:
        parameters = json.load(f)

    if args.representation is not None:
        parameters["representation"] = args.representation

    image_list = collect_images(args.images) if args.images else None
    if image_list is not None:
        print(f"Running inference on {len(image_list)} image(s).")
    else:
        print(f"Using images listed in {args.dataset}/test.txt")

    os.makedirs(args.output, exist_ok=True)

    predict_test_set(
        args.dataset,
        args.checkpoint,
        parameters,
        args.output,
        image_list=image_list,
        independent=False,
        landmarks=args.landmarks,
    )

    print(f"\nDone. Results saved to {args.output}")
