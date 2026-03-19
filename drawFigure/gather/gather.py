import argparse
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(input_dir: Path, exclude_path: Optional[Path] = None):
	"""Collect image files only from input_dir (non-recursive)."""
	exclude_resolved = exclude_path.resolve() if exclude_path is not None else None
	return sorted(
		[
			p
			for p in input_dir.iterdir()
			if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
			and (exclude_resolved is None or p.resolve() != exclude_resolved)
		]
	)


def merge_images(input_dir: Path, output_file: Path, dpi: int = 200):
	image_paths = collect_images(input_dir, exclude_path=output_file)
	if not image_paths:
		raise FileNotFoundError(f"No images found in {input_dir}")

	n_images = len(image_paths)
	n_cols = math.ceil(math.sqrt(n_images))
	n_rows = math.ceil(n_images / n_cols)

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.2))

	if hasattr(axes, "ravel"):
		axes = axes.ravel()
	else:
		axes = [axes]

	for idx, img_path in enumerate(image_paths):
		img = plt.imread(img_path)
		axes[idx].imshow(img)
		axes[idx].axis("off")

	for idx in range(n_images, len(axes)):
		axes[idx].axis("off")

	# Keep subplot gaps compact so the collage is visually dense.
	fig.subplots_adjust(
		left=0.01,
		right=0.99,
		bottom=0.01,
		top=0.99,
		wspace=0.05,
		hspace=0.12,
	)

	output_file.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
	plt.close(fig)


def parse_args():
	base_dir = Path(__file__).resolve().parents[1]
	default_input = base_dir / "analyse"
	default_output = default_input / "merged.png"

	parser = argparse.ArgumentParser(
		description="Merge all images in analyse directory (non-recursive) into one figure."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=default_input,
		help="Directory containing images to merge (non-recursive).",
	)
	parser.add_argument(
		"--output-file",
		type=Path,
		default=default_output,
		help="Output image file path.",
	)
	parser.add_argument("--dpi", type=int, default=200, help="Output DPI.")
	return parser.parse_args()


def main():
	args = parse_args()
	merge_images(args.input_dir, args.output_file, dpi=args.dpi)
	print(f"Saved merged image to: {args.output_file}")


if __name__ == "__main__":
	main()
