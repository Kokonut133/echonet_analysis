"""Generate a human-readable overview of the raw EchoNext dataset.

Reads the metadata CSV and memory-maps each `.npy` array in the dataset
directory (waveforms, tabular features) to summarize shapes, dtypes, sizes,
missingness, and per-column statistics — without loading full waveform
arrays into memory.

Input:  data/<DATASET_SUBDIR>/*.csv, *.npy  (see data/project_config.json)
Output: reports/data_overview.md            (markdown report)
        figures/ecg_examples_grid.png       (4 records x 12 leads, 2.5s window)
        figures/metadata_overview_grid.png  (label + numeric distributions)

Usage:
  python scripts/1_explore/generate_overview.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import (
    DATASET_SUBDIR,
    LEAD_NAMES,
    N_LEADS,
    SAMPLE_RATE_HZ,
    TARGET_LABELS,
)

FILE_DESCRIPTIONS = {
    "metadata": "Main metadata table: record IDs, demographics, ECG measurements, echo-derived labels.",
    "waveform": "12-lead ECG waveform array, one record per row.",
    "tabular": "Precomputed tabular ECG feature matrix (rate/interval measurements) for baseline models.",
}


@dataclass(frozen=True)
class OverviewPaths:
    dataset_dir: Path
    reports_dir: Path
    figures_dir: Path

    def ensure_output_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def describe_file(name: str) -> str:
    name = name.lower()
    if "metadata" in name:
        return FILE_DESCRIPTIONS["metadata"]
    if "waveform" in name:
        return FILE_DESCRIPTIONS["waveform"]
    if "tabular" in name:
        return FILE_DESCRIPTIONS["tabular"]
    return "Dataset file."


def list_data_files(dataset_dir: Path) -> list[Path]:
    return sorted(p for p in dataset_dir.iterdir() if p.suffix in (".csv", ".npy"))


def find_metadata_csv(dataset_dir: Path) -> Path | None:
    csvs = [p for p in list_data_files(dataset_dir) if p.suffix == ".csv"]
    metadata = [p for p in csvs if "metadata" in p.name.lower()]
    return metadata[0] if metadata else (csvs[0] if csvs else None)


def find_waveform_file(dataset_dir: Path, split_preference: tuple[str, ...] = ("train", "val", "test", "no_split")) -> Path | None:
    waveforms = [p for p in list_data_files(dataset_dir) if p.suffix == ".npy" and "waveform" in p.name.lower()]
    for split_name in split_preference:
        for path in waveforms:
            if split_name in path.name.lower():
                return path
    return waveforms[0] if waveforms else None


def build_file_overview_table(dataset_dir: Path) -> str:
    rows = []
    for path in list_data_files(dataset_dir):
        if path.suffix == ".csv":
            header = pd.read_csv(path, nrows=0)
            n_rows = sum(1 for _ in open(path)) - 1
            shape = f"{n_rows} x {header.shape[1]}"
            dtype, size = "", human_size(path.stat().st_size)
        else:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = " x ".join(str(d) for d in arr.shape)
            dtype, size = str(arr.dtype), human_size(arr.size * arr.dtype.itemsize)
        rows.append({
            "file": path.name,
            "description": describe_file(path.name),
            "shape": shape,
            "dtype": dtype,
            "size": size,
        })
    return pd.DataFrame(rows).to_markdown(index=False)


def summarize_metadata(metadata_path: Path) -> dict[str, str]:
    df = pd.read_csv(metadata_path)

    missing = (df.isna().mean() * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False).rename("missing_%").to_frame()
    missing_md = missing.to_markdown() if not missing.empty else "_No missing values._"

    numeric_cols = ["age_at_ecg", "ventricular_rate", "atrial_rate", "pr_interval", "qrs_duration", "qt_corrected"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    numeric_md = df[numeric_cols].describe().T.round(2).to_markdown()

    split_md = df["split"].value_counts().rename("n_records").to_frame().to_markdown() if "split" in df.columns else ""

    targets = [t for t in TARGET_LABELS if t in df.columns]
    prevalence = pd.DataFrame({
        "target": targets,
        "n_positive": [int(df[t].sum()) for t in targets],
        "prevalence_%": [round(100 * df[t].mean(), 1) for t in targets],
    }).sort_values("prevalence_%", ascending=False)
    prevalence_md = prevalence.to_markdown(index=False)

    return {
        "n_rows": f"{len(df):,}",
        "n_cols": str(df.shape[1]),
        "missing_md": missing_md,
        "numeric_md": numeric_md,
        "split_md": split_md,
        "prevalence_md": prevalence_md,
    }


def summarize_array_files(dataset_dir: Path) -> str:
    sections = []
    for path in list_data_files(dataset_dir):
        if path.suffix != ".npy":
            continue
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        sample = np.asarray(arr[: min(200, arr.shape[0])]).ravel()
        sample = sample[np.isfinite(sample)]
        stats = pd.Series(sample).describe().round(3).rename("value").to_frame().to_markdown()
        sections.append(
            f"### `{path.name}`\n"
            f"Shape: `{tuple(arr.shape)}`  Dtype: `{arr.dtype}`  Size: `{human_size(arr.size * arr.dtype.itemsize)}`\n\n"
            f"Sampled statistics (first {sample.size:,} finite values):\n\n{stats}\n"
        )
    return "\n".join(sections)


def normalize_one_ecg(record: np.ndarray) -> np.ndarray:
    """Return one ECG record as a (time, leads) array."""
    array = np.squeeze(np.asarray(record))
    if array.shape[-1] == N_LEADS:
        return array if array.shape[1] == N_LEADS else array.reshape(-1, N_LEADS)
    if array.shape[0] == N_LEADS:
        return array.T
    raise ValueError(f"Could not infer 12-lead axis from ECG record shape {array.shape}")


def save_ecg_examples_grid(
    paths: OverviewPaths,
    n_examples: int = 4,
    window_seconds: float = 2.5,
    seed: int = 42,
) -> str | None:
    """4 records x 12 leads, shared y-scale per record (row), fixed short time window."""
    waveform_path = find_waveform_file(paths.dataset_dir)
    if waveform_path is None:
        return None

    array = np.load(waveform_path, mmap_mode="r", allow_pickle=False)
    n_records = array.shape[0]
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(n_records, size=min(n_examples, n_records), replace=False).tolist())

    window = int(window_seconds * SAMPLE_RATE_HZ)
    time_axis = np.arange(window) / SAMPLE_RATE_HZ

    fig, axes = plt.subplots(len(indices), N_LEADS, figsize=(22, 2.1 * len(indices)), sharex=True)
    axes = np.atleast_2d(axes)

    for row, record_index in enumerate(indices):
        ecg = normalize_one_ecg(array[record_index])[:window]
        y_min, y_max = float(np.min(ecg)), float(np.max(ecg))
        pad = 0.1 * (y_max - y_min + 1e-6)

        for col in range(N_LEADS):
            axis = axes[row, col]
            axis.plot(time_axis, ecg[:, col], linewidth=0.6, color="tab:blue")
            axis.set_ylim(y_min - pad, y_max + pad)
            axis.tick_params(labelsize=5)
            if row == 0:
                axis.set_title(LEAD_NAMES[col], fontsize=8)
            if col == 0:
                axis.set_ylabel(f"rec #{record_index}", fontsize=7)
            if row == len(indices) - 1:
                axis.set_xlabel("s", fontsize=6)
            else:
                axis.tick_params(labelbottom=False)

    fig.suptitle(
        f"Representative 12-lead ECG examples ({window_seconds:.1f}s window, shared y-scale per record)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output_path = paths.figures_dir / "ecg_examples_grid.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path.name


def save_metadata_overview_grid(paths: OverviewPaths, metadata_path: Path) -> str | None:
    df = pd.read_csv(metadata_path)
    label_col = "shd_moderate_or_greater_flag" if "shd_moderate_or_greater_flag" in df.columns else None
    numeric_cols = [c for c in ["age_at_ecg", "ventricular_rate", "atrial_rate", "pr_interval", "qrs_duration", "qt_corrected"] if c in df.columns]

    panels = ([("bar", label_col)] if label_col else []) + [("hist", c) for c in numeric_cols]
    if not panels:
        return None

    n_cols = 3
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), layout="constrained")
    flat_axes = np.atleast_1d(axes).ravel()

    for axis, (kind, column) in zip(flat_axes, panels):
        if kind == "bar":
            df[column].value_counts(dropna=False).plot(kind="bar", ax=axis)
            axis.set_title(f"Label: {column}", fontsize=10)
        else:
            axis.hist(df[column].dropna(), bins=40)
            axis.set_title(column, fontsize=10)
        axis.tick_params(labelsize=8)

    for axis in flat_axes[len(panels):]:
        axis.axis("off")

    fig.suptitle("Metadata overview", fontsize=13)
    output_path = paths.figures_dir / "metadata_overview_grid.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path.name


def build_markdown_report(dataset_dir: Path, metadata_path: Path, figure_names: list[str]) -> str:
    file_table = build_file_overview_table(dataset_dir)
    meta = summarize_metadata(metadata_path)
    array_sections = summarize_array_files(dataset_dir)
    figures_section = "\n".join(f"- `{name}`" for name in figure_names) or "_No figures generated._"

    return f"""# EchoNext Data Overview

Generated by `scripts/1_explore/generate_overview.py`. Summarizes the raw dataset files, metadata columns, and target label prevalence.

## Dataset files

{file_table}

## Figures

{figures_section}

## Metadata

{meta['n_rows']} rows x {meta['n_cols']} columns.

### Records per split

{meta['split_md']}

### Missing values

{meta['missing_md']}

### Numeric feature summary

{meta['numeric_md']}

### Target label prevalence

{meta['prevalence_md']}

## Waveform / tabular array files

{array_sections}
"""


def run_overview(paths: OverviewPaths) -> None:
    paths.ensure_output_dirs()
    print(f"Dataset directory: {paths.dataset_dir.name}")

    metadata_path = find_metadata_csv(paths.dataset_dir)
    if metadata_path is None:
        raise FileNotFoundError(f"No metadata CSV found in {paths.dataset_dir}")

    print("Generating figures...")
    figure_names = []
    for save_fn in (lambda: save_ecg_examples_grid(paths), lambda: save_metadata_overview_grid(paths, metadata_path)):
        name = save_fn()
        if name:
            figure_names.append(name)

    print("Building markdown report...")
    report = build_markdown_report(paths.dataset_dir, metadata_path, figure_names)

    output_path = paths.reports_dir / "data_overview.md"
    output_path.write_text(report, encoding="utf-8")

    print(f"\nWrote: {output_path}")
    if figure_names:
        print(f"Wrote figures: {', '.join(figure_names)}")


def main(dataset_dir: Path, reports_dir: Path, figures_dir: Path) -> None:
    paths = OverviewPaths(dataset_dir=dataset_dir, reports_dir=reports_dir, figures_dir=figures_dir)
    run_overview(paths)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    main(
        dataset_dir=project_root / "data" / DATASET_SUBDIR,
        reports_dir=project_root / "reports",
        figures_dir=project_root / "figures",
    )
