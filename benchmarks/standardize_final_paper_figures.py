"""Create standardized copies of the final manuscript figures.

The script intentionally leaves the original ``figures`` directory untouched.
It normalizes only the figures referenced by the final manuscript by
compositing transparency onto white, applying a tight crop plus common padding,
resizing to standard single-/double-column raster widths, and writing the
results with consistent 300 dpi metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "reports" / "Effect_of_eccentricity_for_GB_in_LISA-final"
SOURCE_DIR = PAPER_DIR / "figures"
OUTPUT_DIR = PAPER_DIR / "std_figures"

COLUMN_WIDTH_PX = 1540
TEXT_WIDTH_PX = 2200
PADDING_PX = 36
DPI = (300, 300)


@dataclass(frozen=True)
class FigureSpec:
    filename: str
    width_class: str

    @property
    def target_width(self) -> int:
        if self.width_class == "column":
            return COLUMN_WIDTH_PX
        if self.width_class == "text":
            return TEXT_WIDTH_PX
        raise ValueError(f"unknown width class: {self.width_class}")


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec("paper_time_waveform_then_fft_e0_e01_e03.png", "column"),
    FigureSpec("paper_periastron_timeseries_envelope_ratio_1pn_peri_vs_newtonian_e06_m14_m14_356days.png", "text"),
    FigureSpec("wide_peters_mathews_relevance_curves.png", "text"),
    FigureSpec("lisainstrument_welch_psd_detail.png", "column"),
    FigureSpec("source_corrections_summary.png", "text"),
    FigureSpec("full_model_vs_fixed_circular.png", "column"),
    FigureSpec("wide_fdot_tdi2_mismatch_vs_eccentricity_by_f0_no_e0_threshold.png", "column"),
    FigureSpec("paper_jaxgb_comprehensive_comparison.png", "text"),
)


def flatten_on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def content_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    array = np.asarray(image, dtype=np.uint8)
    non_white = np.any(array < 250, axis=2)
    if not np.any(non_white):
        return (0, 0, image.width, image.height)
    ys, xs = np.nonzero(non_white)
    left = max(int(xs.min()) - PADDING_PX, 0)
    upper = max(int(ys.min()) - PADDING_PX, 0)
    right = min(int(xs.max()) + PADDING_PX + 1, image.width)
    lower = min(int(ys.max()) + PADDING_PX + 1, image.height)
    return (left, upper, right, lower)


def standardize(source: Path, target: Path, target_width: int) -> None:
    image = flatten_on_white(Image.open(source))
    cropped = image.crop(content_bbox(image))
    scale = target_width / cropped.width
    target_height = max(1, int(round(cropped.height * scale)))
    resized = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    resized.save(target, dpi=DPI, optimize=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in FIGURES:
        source = SOURCE_DIR / spec.filename
        target = OUTPUT_DIR / spec.filename
        if not source.exists():
            raise FileNotFoundError(source)
        standardize(source, target, spec.target_width)
        print(f"{spec.width_class:6s} {source.name} -> {target.relative_to(PAPER_DIR)}")


if __name__ == "__main__":
    main()
