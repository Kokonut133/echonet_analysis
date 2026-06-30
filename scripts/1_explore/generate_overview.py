from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass(frozen=True)
class OverviewPaths:
    dataset_dir: Path
    reports_dir: Path
    figures_dir: Path

    def ensure_output_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TableSection:
    body_rows: list[list[object]]
    row_labels: list[str] | None = None


def build_paths(dataset_dir: Path, reports_dir: Path, figures_dir: Path) -> OverviewPaths:
    return OverviewPaths(dataset_dir=dataset_dir, reports_dir=reports_dir, figures_dir=figures_dir)


def cleanup_previous_outputs(
    paths: OverviewPaths,
    report_output_names: tuple[str, ...] = ("data_overview.json", "data_overview.md"),
) -> None:
    for report_name in report_output_names:
        report_path = paths.reports_dir / report_name

        if report_path.is_file():
            report_path.unlink()

    if not paths.figures_dir.is_dir():
        return

    for figure_path in paths.figures_dir.iterdir():
        if not figure_path.is_file():
            continue

        if is_previous_figure(figure_path):
            figure_path.unlink()


def is_previous_figure(
    path: Path,
    figure_output_names: tuple[str, ...] = ("ecg_examples_grid.png", "metadata_overview_grid.png"),
) -> bool:
    name = path.name.lower()

    if name in figure_output_names:
        return True

    if name.endswith("_examples_grid.png"):
        return True

    if name.startswith("metadata_distribution_"):
        return True

    return name == "likely_label_distribution.png"


def list_data_files(dataset_dir: Path) -> list[Path]:
    data_files: list[Path] = []

    for entry in sorted(dataset_dir.iterdir()):
        if entry.is_file() and entry.suffix in {".csv", ".npy"}:
            data_files.append(entry)

    return data_files


def find_metadata_csv(dataset_dir: Path) -> Path | None:
    csv_files = [path for path in list_data_files(dataset_dir) if path.suffix == ".csv"]
    metadata_files = [path for path in csv_files if "metadata" in path.name.lower()]

    if metadata_files:
        return metadata_files[0]

    if csv_files:
        return csv_files[0]

    return None


def find_waveform_file(dataset_dir: Path) -> Path | None:
    waveform_files = [
        path
        for path in list_data_files(dataset_dir)
        if path.suffix == ".npy" and "waveform" in path.name.lower()
    ]

    if not waveform_files:
        return None

    split_preference = ("no_split", "train", "val", "test")

    for split_name in split_preference:
        for path in sorted(waveform_files):
            if split_name in path.name.lower():
                return path

    return sorted(waveform_files)[0]


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def describe_csv_file(name: str) -> str:
    if "metadata" in name:
        return "Main metadata table with record IDs, demographics, ECG measurements, and labels."

    return "CSV table with row-wise metadata or annotations."


def describe_npy_file(name: str) -> str:
    if "waveform" in name:
        split_descriptions = {
            "train": "Training split ECG waveforms.",
            "val": "Validation split ECG waveforms.",
            "test": "Test split ECG waveforms.",
            "no_split": "Full unsplit ECG waveform array.",
        }

        for split_name, description in split_descriptions.items():
            if split_name in name:
                return description

        return "ECG waveform array."

    if "tabular" in name:
        return "Precomputed tabular feature matrix for baseline models."

    return "NumPy binary array."


def describe_dataset_file(path: Path) -> str:
    name = path.name.lower()

    if name.endswith(".csv"):
        return describe_csv_file(name)

    if name.endswith(".npy"):
        return describe_npy_file(name)

    if name.endswith(".md"):
        return "Dataset documentation."

    if name.endswith(".txt"):
        return "License, checksum, or text metadata."

    return "Dataset file."


def round_number(value: float, decimal_places: int = 3) -> float:
    return round(float(value), decimal_places)


def format_number(value: object, decimal_places: int = 3) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        return f"{round_number(value, decimal_places):.{decimal_places}f}"

    return str(value)


def format_cell(value: object, max_chars: int = 24, decimal_places: int = 3) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, np.integer, float, np.floating, bool)):
        text = format_number(value, decimal_places)
    else:
        text = str(value)

    text = text.replace("\n", " ").strip()

    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."

    return text


def select_display_columns(columns: list[str], max_cols: int = 12) -> tuple[list[str], bool]:
    if len(columns) > max_cols:
        return columns[:max_cols], True

    return list(columns), False


def format_matrix_row_cells(
    row_values: list[object],
    row_label: str | None,
    max_cell_chars: int,
    decimal_places: int,
) -> list[str]:
    cells: list[str] = []

    if row_label is not None:
        cells.append(format_cell(row_label, max_cell_chars, decimal_places))

    for value in row_values:
        cells.append(format_cell(value, max_cell_chars, decimal_places))

    return cells


def measure_matrix_column_widths(header_cells: list[str], formatted_rows: list[list[str]]) -> list[int]:
    widths: list[int] = []

    for column_index in range(len(header_cells)):
        column_width = len(header_cells[column_index])

        for row in formatted_rows:
            column_width = max(column_width, len(row[column_index]))

        widths.append(column_width)

    return widths


def join_matrix_lines(
    header_cells: list[str],
    formatted_rows: list[list[str]],
    column_widths: list[int],
    cell_gap: str = " ",
) -> str:
    header_line = cell_gap.join(cell.ljust(width) for cell, width in zip(header_cells, column_widths))
    divider = cell_gap.join("-" * width for width in column_widths)
    body_lines = [
        cell_gap.join(cell.ljust(width) for cell, width in zip(row_cells, column_widths))
        for row_cells in formatted_rows
    ]
    lines = [header_line, divider]
    lines.extend(body_lines)

    return "\n".join(lines)


def format_aligned_matrix(
    column_headers: list[str],
    body_rows: list[list[object]],
    row_labels: list[str] | None = None,
    max_cell_chars: int = 24,
    column_widths: list[int] | None = None,
    decimal_places: int = 3,
    cell_gap: str = " ",
) -> str:
    if not column_headers:
        return "(empty table)"

    if row_labels is not None and len(row_labels) != len(body_rows):
        raise ValueError("row_labels length must match body_rows length")

    has_row_labels = row_labels is not None
    row_label_header = format_cell("", max_cell_chars, decimal_places) if has_row_labels else None
    header_cells = ([row_label_header] if row_label_header is not None else []) + [
        format_cell(column, max_cell_chars, decimal_places) for column in column_headers
    ]

    formatted_rows: list[list[str]] = []

    for row_index, row_values in enumerate(body_rows):
        row_label = row_labels[row_index] if has_row_labels else None
        formatted_rows.append(format_matrix_row_cells(row_values, row_label, max_cell_chars, decimal_places))

    if column_widths is None:
        column_widths = measure_matrix_column_widths(header_cells, formatted_rows)

    return join_matrix_lines(header_cells, formatted_rows, column_widths, cell_gap=cell_gap)


def compute_shared_column_widths(
    column_headers: list[str],
    sections: list[TableSection],
    max_cell_chars: int = 24,
    include_row_label: bool = False,
    decimal_places: int = 3,
) -> list[int]:
    header_cells = [format_cell(column, max_cell_chars, decimal_places) for column in column_headers]
    widths = [len(header) for header in header_cells]
    row_label_width = 0

    for section in sections:
        for row_index, row_values in enumerate(section.body_rows):
            if section.row_labels:
                label_cell = format_cell(section.row_labels[row_index], max_cell_chars, decimal_places)
                row_label_width = max(row_label_width, len(label_cell))

            for column_index, value in enumerate(row_values):
                cell_text = format_cell(value, max_cell_chars, decimal_places)
                widths[column_index] = max(widths[column_index], len(cell_text))

    if include_row_label:
        return [row_label_width] + widths

    return widths


def append_column_truncation_note(table: str, shown_columns: int, total_columns: int) -> str:
    if shown_columns >= total_columns:
        return table

    return table + f"\n\n(showing first {shown_columns} of {total_columns} columns)"


def format_excel_table(
    dataframe: pd.DataFrame,
    max_rows: int = 5,
    max_cols: int = 12,
    max_cell_chars: int = 24,
    display_columns: list[str] | None = None,
    decimal_places: int = 3,
    cell_gap: str = " ",
) -> str:
    if dataframe.empty:
        return "(empty table)"

    if display_columns is None:
        display_columns, _ = select_display_columns(list(dataframe.columns), max_cols)

    preview = dataframe.loc[:, display_columns].head(max_rows)
    body_rows = [list(row) for _, row in preview.iterrows()]
    table = format_aligned_matrix(
        display_columns,
        body_rows,
        max_cell_chars=max_cell_chars,
        decimal_places=decimal_places,
        cell_gap=cell_gap,
    )

    return append_column_truncation_note(table, len(display_columns), dataframe.shape[1])


def format_key_value_table(mapping: dict[str, object]) -> str:
    if not mapping:
        return "(none)"

    body_rows = [[value] for value in mapping.values()]

    return format_aligned_matrix(
        ["value"],
        body_rows,
        row_labels=list(mapping.keys()),
        max_cell_chars=40,
    )


def build_preview_body_rows(
    dataframe: pd.DataFrame,
    display_columns: list[str],
    preview_rows: int = 5,
) -> list[list[object]]:
    dtype_row = [str(dataframe[column].dtype) for column in display_columns]
    data_rows = [list(row) for _, row in dataframe[display_columns].head(preview_rows).iterrows()]
    body_rows = [dtype_row]
    body_rows.extend(data_rows)

    return body_rows


def build_head_preview_table(
    dataframe: pd.DataFrame,
    display_columns: list[str],
    column_widths: list[int] | None = None,
    preview_rows: int = 5,
) -> str:
    body_rows = build_preview_body_rows(dataframe, display_columns, preview_rows)
    table = format_aligned_matrix(display_columns, body_rows, max_cell_chars=40, column_widths=column_widths)

    return append_column_truncation_note(table, len(display_columns), dataframe.shape[1])


def build_numeric_summary_section(dataframe: pd.DataFrame, display_columns: list[str]) -> TableSection | None:
    numeric = dataframe.select_dtypes(include="number")

    if numeric.empty:
        return None

    stat_names = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    summary = numeric.describe().T
    body_rows = build_stat_rows(summary, display_columns, stat_names)

    return TableSection(body_rows=body_rows, row_labels=stat_names)


def build_stat_rows(summary: pd.DataFrame, display_columns: list[str], stat_names: list[str]) -> list[list[object]]:
    body_rows: list[list[object]] = []

    for stat_name in stat_names:
        row_values: list[object] = []

        for column in display_columns:
            if column in summary.index:
                row_values.append(round_number(summary.loc[column, stat_name]))
            else:
                row_values.append("")

        body_rows.append(row_values)

    return body_rows


def build_missingness_section(
    dataframe: pd.DataFrame,
    display_columns: list[str],
    decimal_places: int = 3,
) -> TableSection:
    missing_percent = (dataframe.isna().mean() * 100).round(decimal_places)
    row_values = [missing_percent[column] for column in display_columns]

    return TableSection(body_rows=[row_values], row_labels=["missing_%"])


def build_label_values_text(values: np.ndarray) -> tuple[object, object]:
    if len(values) > 5:
        return "", ""

    value_text = ", ".join(str(value) for value in values[:10])

    return len(values), value_text


def build_label_candidate_section(dataframe: pd.DataFrame, display_columns: list[str]) -> TableSection | None:
    unique_counts: list[object] = []
    value_lists: list[object] = []

    for column in display_columns:
        values = dataframe[column].dropna().unique()
        unique_count, value_text = build_label_values_text(values)
        unique_counts.append(unique_count)
        value_lists.append(value_text)

    if not any(unique_counts):
        return None

    return TableSection(body_rows=[unique_counts, value_lists], row_labels=["unique_count", "values"])


def render_table_section(display_columns: list[str], section: TableSection, column_widths: list[int]) -> str:
    return format_aligned_matrix(
        display_columns,
        section.body_rows,
        row_labels=section.row_labels,
        max_cell_chars=40,
        column_widths=column_widths,
    )


def merge_table_sections(
    missing_section: TableSection,
    label_section: TableSection | None,
    numeric_section: TableSection | None,
) -> TableSection:
    row_labels = list(missing_section.row_labels or [])
    body_rows = list(missing_section.body_rows)

    for extra_section in (label_section, numeric_section):
        if extra_section is None:
            continue

        if extra_section.row_labels:
            row_labels.extend(extra_section.row_labels)

        body_rows.extend(extra_section.body_rows)

    return TableSection(body_rows=body_rows, row_labels=row_labels)


def collect_csv_analysis_sections(
    dataframe: pd.DataFrame,
    display_columns: list[str],
) -> tuple[TableSection, TableSection, TableSection | None, TableSection | None]:
    head_section = TableSection(body_rows=build_preview_body_rows(dataframe, display_columns))
    missing_section = build_missingness_section(dataframe, display_columns)
    label_section = build_label_candidate_section(dataframe, display_columns)
    numeric_section = build_numeric_summary_section(dataframe, display_columns)

    return head_section, missing_section, label_section, numeric_section


def build_width_sections(
    head_section: TableSection,
    missing_section: TableSection,
    label_section: TableSection | None,
    numeric_section: TableSection | None,
) -> list[TableSection]:
    sections = [head_section, missing_section]

    if label_section is not None:
        sections.append(label_section)

    if numeric_section is not None:
        sections.append(numeric_section)

    return sections


def build_csv_column_metadata_tables(dataframe: pd.DataFrame) -> tuple[str, str]:
    display_columns, _ = select_display_columns(list(dataframe.columns))
    head_section, missing_section, label_section, numeric_section = collect_csv_analysis_sections(
        dataframe,
        display_columns,
    )
    shared_widths = compute_shared_column_widths(
        display_columns,
        build_width_sections(head_section, missing_section, label_section, numeric_section),
        max_cell_chars=40,
        include_row_label=True,
    )
    head_preview = build_head_preview_table(dataframe, display_columns, column_widths=shared_widths[1:])
    combined_section = merge_table_sections(missing_section, label_section, numeric_section)
    column_metadata = render_table_section(display_columns, combined_section, shared_widths)

    return head_preview, column_metadata


def find_label_column(dataframe: pd.DataFrame) -> str:
    label_terms = ("shd", "structural", "label", "target", "outcome", "disease")

    for column in dataframe.columns:
        values = dataframe[column].dropna().unique()

        if len(values) > 5:
            continue

        if any(term in str(column).lower() for term in label_terms):
            return str(column)

    for column in dataframe.columns:
        if dataframe[column].dropna().nunique() <= 5:
            return str(column)

    return ""


def summarize_csv(path: Path) -> dict:
    dataframe = pd.read_csv(path)
    head_preview, column_metadata = build_csv_column_metadata_tables(dataframe)

    return {
        "filename": path.name,
        "kind": "csv",
        "description": describe_dataset_file(path),
        "shape": list(dataframe.shape),
        "columns": list(dataframe.columns),
        "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        "head_preview": head_preview,
        "column_metadata": column_metadata,
    }


def preview_npy_1d(array: np.ndarray, max_rows: int) -> str:
    preview = pd.DataFrame({"index": range(min(max_rows, array.shape[0])), "value": array[:max_rows]})

    return format_excel_table(preview, max_rows=max_rows, max_cols=2)


def preview_npy_2d(array: np.ndarray, max_rows: int, max_cols: int) -> str:
    row_count = min(max_rows, array.shape[0])
    col_count = min(max_cols, array.shape[1])
    preview = pd.DataFrame(
        array[:row_count, :col_count],
        columns=[f"feature_{index:03d}" for index in range(col_count)],
    )
    preview.insert(0, "row", range(row_count))
    table = format_excel_table(preview, max_rows=row_count, max_cols=col_count + 1)

    return append_column_truncation_note(table, max_cols, array.shape[1])


def build_record_preview_row(record: np.ndarray, index: int) -> dict[str, object]:
    finite = record[np.isfinite(record)] if np.issubdtype(record.dtype, np.number) else np.array([])

    if not finite.size:
        return {"record": index, "shape": str(list(record.shape)), "min": "", "max": "", "mean": "", "std": ""}

    return {
        "record": index,
        "shape": str(list(record.shape)),
        "min": round_number(float(np.min(finite))),
        "max": round_number(float(np.max(finite))),
        "mean": round_number(float(np.mean(finite))),
        "std": round_number(float(np.std(finite))),
    }


def preview_npy_3d(array: np.ndarray, max_rows: int) -> str:
    records = []

    for index in range(min(max_rows, array.shape[0])):
        record = np.asarray(array[index])
        records.append(build_record_preview_row(record, index))

    return format_excel_table(pd.DataFrame(records), max_rows=max_rows, max_cols=6)


def build_npy_preview_table(array: np.ndarray, max_rows: int = 5, max_cols: int = 12) -> str:
    if array.ndim == 1:
        return preview_npy_1d(array, max_rows)

    if array.ndim == 2:
        return preview_npy_2d(array, max_rows, max_cols)

    if array.ndim >= 3:
        return preview_npy_3d(array, max_rows)

    return "No preview available."


def draw_array_sample(array: np.ndarray) -> np.ndarray:
    if array.ndim >= 1:
        sample_count = min(64, array.shape[0])
        indices = np.linspace(0, array.shape[0] - 1, sample_count).astype(int)

        return np.asarray(array[indices]).ravel()

    return np.asarray(array).ravel()


def build_finite_stats(finite: np.ndarray) -> dict[str, object]:
    return {
        "sampled_values": int(finite.size),
        "min": round_number(float(np.min(finite))),
        "max": round_number(float(np.max(finite))),
        "mean": round_number(float(np.mean(finite))),
        "std": round_number(float(np.std(finite))),
        "p01": round_number(float(np.percentile(finite, 1))),
        "p50": round_number(float(np.percentile(finite, 50))),
        "p99": round_number(float(np.percentile(finite, 99))),
    }


def sample_numeric_stats(array: np.ndarray) -> dict:
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return {}

    sample = draw_array_sample(array)
    finite = sample[np.isfinite(sample)]

    if finite.size == 0:
        return {}

    return build_finite_stats(finite)


def summarize_npy(path: Path, waveform_preview_rows: int = 10) -> dict:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    preview_rows = waveform_preview_rows if "waveform" in path.name.lower() else 5

    return {
        "filename": path.name,
        "kind": "npy",
        "description": describe_dataset_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": human_size(array.size * array.dtype.itemsize),
        "ndim": array.ndim,
        "preview": build_npy_preview_table(array, max_rows=preview_rows),
        "sample_stats": sample_numeric_stats(array),
    }


def summarize_dataset_files(paths: OverviewPaths) -> list[dict]:
    files = list_data_files(paths.dataset_dir)

    if not files:
        raise FileNotFoundError(f"No CSV or NPY files found in {paths.dataset_dir}")

    summaries: list[dict] = []

    for path in files:
        print(f"Summarizing {path.name}...")

        if path.suffix == ".csv":
            summaries.append(summarize_csv(path))
        elif path.suffix == ".npy":
            summaries.append(summarize_npy(path))

    return summaries


def normalize_one_ecg(record: np.ndarray) -> np.ndarray:
    array = np.squeeze(np.asarray(record))

    if array.ndim != 2:
        raise ValueError(f"Expected one ECG record as 2D array, got shape {array.shape}")

    if array.shape[1] == 12:
        return array

    if array.shape[0] == 12:
        return array.T

    raise ValueError(f"Could not infer 12-lead axis from ECG record shape {array.shape}")


def sample_record_indices(
    record_count: int,
    sample_count: int | None = None,
    max_figure_rows: int = 10,
    lead_rows_per_example: int = 2,
    seed: int = 42,
) -> list[int]:
    del seed

    if sample_count is None:
        sample_count = max_figure_rows // lead_rows_per_example

    draw_count = min(sample_count, record_count)

    if draw_count <= 0:
        return []

    if draw_count == record_count:
        return list(range(record_count))

    indices = np.linspace(0, record_count - 1, draw_count).astype(int)

    return sorted({int(index) for index in indices})


def trim_ecg_signal(
    ecg: np.ndarray,
    sample_rate_hz: int = 250,
    max_seconds: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    max_samples = min(ecg.shape[0], int(max_seconds * sample_rate_hz))
    trimmed = ecg[:max_samples]
    time_axis = np.arange(trimmed.shape[0]) / sample_rate_hz

    return trimmed, time_axis


def plot_leads_on_axes(
    ecg: np.ndarray,
    axes: np.ndarray,
    show_xlabel: bool = False,
    ylabel: str = "amp",
    sample_rate_hz: int = 250,
    max_seconds: float = 10.0,
) -> None:
    trimmed_ecg, time_axis = trim_ecg_signal(ecg, sample_rate_hz, max_seconds)
    grid = np.atleast_2d(axes)
    lead_rows, lead_cols = grid.shape
    last_row = lead_rows - 1

    for lead_index in range(12):
        row = lead_index // lead_cols
        col = lead_index % lead_cols
        axis = grid[row, col]
        axis.plot(time_axis, trimmed_ecg[:, lead_index], linewidth=0.7)
        axis.set_title(LEADS[lead_index], fontsize=7)
        axis.tick_params(labelsize=6)

        if col == 0:
            axis.set_ylabel(ylabel, fontsize=6)

        if row != last_row or not show_xlabel:
            axis.tick_params(labelbottom=False)

    if show_xlabel:
        for col in range(lead_cols):
            grid[last_row, col].set_xlabel("time [s]", fontsize=7)


def create_ecg_examples_figure(
    example_indices: list[int],
    waveform_name: str,
    lead_rows_per_example: int = 2,
    lead_cols_per_example: int = 6,
) -> tuple[plt.Figure, np.ndarray]:
    total_rows = len(example_indices) * lead_rows_per_example
    figure, axes = plt.subplots(
        total_rows,
        lead_cols_per_example,
        figsize=(14, 0.9 * total_rows),
        squeeze=False,
    )
    index_text = ", ".join(str(index) for index in example_indices)
    figure.suptitle(
        f"Representative ECG examples from {waveform_name} (indices: {index_text})",
        fontsize=11,
    )

    return figure, axes


def plot_ecg_examples_on_figure(
    figure: plt.Figure,
    axes: np.ndarray,
    array: np.ndarray,
    example_indices: list[int],
    lead_rows_per_example: int = 2,
) -> None:
    last_example_index = len(example_indices) - 1

    for example_number, record_index in enumerate(example_indices):
        row_start = example_number * lead_rows_per_example
        lead_axes = axes[row_start : row_start + lead_rows_per_example, :]
        ecg = normalize_one_ecg(array[record_index])
        plot_leads_on_axes(
            ecg,
            lead_axes,
            show_xlabel=example_number == last_example_index,
            ylabel=f"#{record_index}\namp",
        )

    figure.subplots_adjust(hspace=0.35, wspace=0.06, top=0.96)


def save_waveform_examples_grid(paths: OverviewPaths) -> str | None:
    waveform_path = find_waveform_file(paths.dataset_dir)

    if waveform_path is None:
        return None

    array = np.load(waveform_path, mmap_mode="r", allow_pickle=False)

    if array.ndim not in (3, 4):
        return None

    example_indices = sample_record_indices(array.shape[0])

    if not example_indices:
        return None

    figure, axes = create_ecg_examples_figure(example_indices, waveform_path.name)
    plot_ecg_examples_on_figure(figure, axes, array, example_indices)

    output_path = paths.figures_dir / "ecg_examples_grid.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    return output_path.name


def select_numeric_histogram_columns(dataframe: pd.DataFrame, max_columns: int = 8) -> list[str]:
    numeric_columns = list(dataframe.select_dtypes(include="number").columns)
    preferred_terms = ("age", "rate", "pr", "qrs", "qt")
    preferred = [
        column
        for column in numeric_columns
        if any(term in str(column).lower() for term in preferred_terms)
    ]

    if preferred:
        return preferred[:max_columns]

    return numeric_columns[:max_columns]


def build_metadata_panels(dataframe: pd.DataFrame) -> list[tuple[str, str]]:
    label_column = find_label_column(dataframe)
    histogram_columns = [
        column
        for column in select_numeric_histogram_columns(dataframe)
        if dataframe[column].notna().any()
    ]
    panels: list[tuple[str, str]] = []

    if label_column:
        panels.append(("bar", label_column))

    for column in histogram_columns:
        panels.append(("hist", column))

    return panels


def draw_metadata_panel(axis: plt.Axes, dataframe: pd.DataFrame, plot_kind: str, column: str) -> None:
    if plot_kind == "bar":
        dataframe[column].value_counts(dropna=False).plot(kind="bar", ax=axis)
        axis.set_title(f"Label: {column}", fontsize=10)
    else:
        axis.hist(dataframe[column].dropna(), bins=40)
        axis.set_title(column, fontsize=10)

    axis.tick_params(labelsize=8)


def save_metadata_overview_grid(paths: OverviewPaths, metadata_path: Path) -> str | None:
    dataframe = pd.read_csv(metadata_path)
    panels = build_metadata_panels(dataframe)

    if not panels:
        return None

    column_count = 3
    row_count = math.ceil(len(panels) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5 * column_count, 3.5 * row_count),
        layout="constrained",
    )
    flat_axes = np.atleast_1d(axes).ravel()

    for axis, (plot_kind, column) in zip(flat_axes, panels):
        draw_metadata_panel(axis, dataframe, plot_kind, column)

    for axis in flat_axes[len(panels) :]:
        axis.axis("off")

    figure.suptitle("Metadata overview", fontsize=13)

    output_path = paths.figures_dir / "metadata_overview_grid.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return output_path.name


def generate_figures(paths: OverviewPaths) -> list[str]:
    figure_paths: list[str] = []

    try:
        saved_path = save_waveform_examples_grid(paths)

        if saved_path:
            figure_paths.append(saved_path)
    except Exception as error:
        print(f"Could not plot waveform examples: {error}")

    metadata_path = find_metadata_csv(paths.dataset_dir)

    if metadata_path is not None:
        saved_path = save_metadata_overview_grid(paths, metadata_path)

        if saved_path:
            figure_paths.append(saved_path)

    return figure_paths


def render_csv_detail_section(summary: dict) -> str:
    return f"""### Top 5 rows (dtype row below headers)
{summary["head_preview"]}

### Overview
- Rows: {summary["shape"][0]}
- Columns: {summary["shape"][1]}
- Main table for record IDs, metadata, labels, and split alignment.

### Column metadata (missing %, labels, numeric summary)
{summary["column_metadata"]}
"""


def describe_npy_role(filename: str) -> str:
    if "waveform" in filename:
        return textwrap.dedent(
            """
            - ECG waveform data: one record per row, 12 leads over time.
            - Use for signal visualization and waveform models.
            """
        ).strip()

    if "tabular" in filename:
        return textwrap.dedent(
            """
            - Dense numeric feature matrix for baseline models.
            - Feature names may live in dataset documentation rather than the array file.
            """
        ).strip()

    return "- NumPy array; inspect shape and docs to infer role."


def describe_npy_shape(summary: dict) -> str:
    shape = summary.get("shape", [])

    if summary.get("ndim") == 3 and len(shape) == 3:
        return (
            f"- 3D array with {shape[0]} records.\n"
            "- Likely `(records, time, leads)` or `(records, leads, time)`."
        )

    if summary.get("ndim") == 2 and len(shape) == 2:
        return f"- 2D array: {shape[0]} records × {shape[1]} features."

    return ""


def render_npy_detail_section(summary: dict) -> str:
    filename = summary["filename"].lower()
    role_notes = describe_npy_role(filename)
    shape_notes = describe_npy_shape(summary)
    stats_table = format_key_value_table(summary.get("sample_stats", {}))

    return f"""**Dtype:** `{summary.get("dtype", "")}`
**Approx. size:** `{summary.get("size", "")}`

### Preview
{summary["preview"]}

### Overview
{role_notes}
{shape_notes}

### Sampled numeric statistics
{stats_table}
"""


def render_file_detail_section(summary: dict) -> str:
    header = f"""---
## `{summary["filename"]}`

**Description:** {summary["description"]}
**Kind:** `{summary["kind"]}`
**Shape:** `{summary.get("shape", "")}`
"""

    if summary["kind"] == "csv":
        return header + render_csv_detail_section(summary)

    return header + render_npy_detail_section(summary)


def build_overview_table(summaries: list[dict]) -> str:
    overview_rows = [
        {
            "file": summary["filename"],
            "kind": summary["kind"],
            "description": summary["description"],
            "shape": " × ".join(str(value) for value in summary.get("shape", [])),
            "dtype": summary.get("dtype", ""),
            "size": summary.get("size", ""),
        }
        for summary in summaries
    ]

    return format_excel_table(pd.DataFrame(overview_rows), max_rows=100, max_cols=6, max_cell_chars=40)


def build_figures_section(figure_paths: list[str]) -> str:
    if figure_paths:
        return "\n".join(f"- `{path}`" for path in figure_paths)

    return "_No figures generated._"


def build_markdown_report(summaries: list[dict], figure_paths: list[str]) -> str:
    overview_table = build_overview_table(summaries)
    figures_section = build_figures_section(figure_paths)
    detail_sections = "\n".join(render_file_detail_section(summary) for summary in summaries)

    return f"""# EchoNext Human-Readable Data Overview

Generated by `scripts/generate_overview.py`.

Quick inventory, Excel-style previews, array summaries, and grid figures on one page per chart.

## 1. File overview
{overview_table}

## 2. Generated figures
{figures_section}

## 3. Files in detail
{detail_sections}
"""


def write_reports(paths: OverviewPaths, summaries: list[dict], figure_paths: list[str]) -> tuple[Path, Path]:
    json_path = paths.reports_dir / "data_overview.json"
    markdown_path = paths.reports_dir / "data_overview.md"
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(build_markdown_report(summaries, figure_paths), encoding="utf-8")

    return json_path, markdown_path


def run_overview(paths: OverviewPaths) -> None:
    cleanup_previous_outputs(paths)
    paths.ensure_output_dirs()
    print(f"Dataset directory: {paths.dataset_dir.name}")

    summaries = summarize_dataset_files(paths)

    print("Generating grid figures...")
    figure_paths = generate_figures(paths)

    json_path, markdown_path = write_reports(paths, summaries, figure_paths)

    print(f"\nWrote: {json_path.name}")
    print(f"Wrote: {markdown_path.name}")

    if figure_paths:
        print(f"Wrote figures: {', '.join(figure_paths)}")


def main(dataset_dir: Path, reports_dir: Path, figures_dir: Path) -> None:
    paths = build_paths(dataset_dir, reports_dir, figures_dir)
    run_overview(paths)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / (
        "data/echonext-a-dataset-for-detecting-echocardiogram-confirmed-structural-heart-disease-from-ecgs-1.1.0"
    )
    output_reports_dir = project_root / "reports"
    output_figures_dir = project_root / "figures"

    main(input_dir, output_reports_dir, output_figures_dir)
