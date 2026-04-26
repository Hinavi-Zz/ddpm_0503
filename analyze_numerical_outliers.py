from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "numerical_outliers"


def modified_z_scores(values: np.ndarray, median: float, mad: float) -> np.ndarray:
    if mad == 0 or not np.isfinite(mad):
        return np.full(values.shape, np.nan, dtype=np.float64)
    return 0.6745 * (values - median) / mad


def summarize_values(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {
            "n_valid": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "max": np.nan,
            "iqr": np.nan,
            "iqr_lower_fence": np.nan,
            "iqr_upper_fence": np.nan,
            "iqr_outlier_count": np.nan,
            "iqr_outlier_rate": np.nan,
            "extreme_iqr_outlier_count": np.nan,
            "extreme_iqr_outlier_rate": np.nan,
            "mad": np.nan,
            "mad_outlier_count": np.nan,
            "mad_outlier_rate": np.nan,
            "max_abs_modified_z": np.nan,
            "tail_spread_ratio": np.nan,
        }

    q01, q05, q25, median, q75, q95, q99 = np.quantile(
        values,
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
    )
    iqr = q75 - q25
    mad = float(np.median(np.abs(values - median)))

    iqr_lower = np.nan
    iqr_upper = np.nan
    iqr_count = np.nan
    iqr_rate = np.nan
    extreme_count = np.nan
    extreme_rate = np.nan
    tail_spread_ratio = np.nan
    if iqr > 0:
        iqr_lower = q25 - 1.5 * iqr
        iqr_upper = q75 + 1.5 * iqr
        iqr_mask = (values < iqr_lower) | (values > iqr_upper)
        iqr_count = int(np.sum(iqr_mask))
        iqr_rate = iqr_count / n

        extreme_lower = q25 - 3.0 * iqr
        extreme_upper = q75 + 3.0 * iqr
        extreme_mask = (values < extreme_lower) | (values > extreme_upper)
        extreme_count = int(np.sum(extreme_mask))
        extreme_rate = extreme_count / n
        tail_spread_ratio = (q99 - q01) / iqr

    modified_z = modified_z_scores(values, median, mad)
    mad_mask = np.abs(modified_z) > 3.5
    mad_count = int(np.sum(mad_mask)) if np.isfinite(modified_z).any() else np.nan
    mad_rate = mad_count / n if np.isfinite(mad_count) else np.nan
    max_abs_modified_z = (
        float(np.nanmax(np.abs(modified_z)))
        if np.isfinite(modified_z).any()
        else np.nan
    )

    return {
        "n_valid": n,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(values)),
        "q01": float(q01),
        "q05": float(q05),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q95": float(q95),
        "q99": float(q99),
        "max": float(np.max(values)),
        "iqr": float(iqr),
        "iqr_lower_fence": float(iqr_lower) if np.isfinite(iqr_lower) else np.nan,
        "iqr_upper_fence": float(iqr_upper) if np.isfinite(iqr_upper) else np.nan,
        "iqr_outlier_count": iqr_count,
        "iqr_outlier_rate": iqr_rate,
        "extreme_iqr_outlier_count": extreme_count,
        "extreme_iqr_outlier_rate": extreme_rate,
        "mad": mad,
        "mad_outlier_count": mad_count,
        "mad_outlier_rate": mad_rate,
        "max_abs_modified_z": max_abs_modified_z,
        "tail_spread_ratio": tail_spread_ratio,
    }


def numerical_column_table(role: str, preprocessed_dir: Path) -> pd.DataFrame:
    columns = pd.read_csv(preprocessed_dir / f"{role}_columns.csv")
    value_rows = columns[
        (columns["source_type"] == "numerical")
        & (columns["meaning"] == "raw_value")
    ].copy()
    flag_rows = columns[
        (columns["source_type"] == "numerical")
        & (columns["meaning"] == "flag")
    ][["source_variable", "column_index"]].rename(
        columns={"column_index": "flag_column_index"}
    )
    merged = value_rows.merge(flag_rows, on="source_variable", how="left")
    return merged.sort_values("column_index").reset_index(drop=True)


def analyze_role(role: str, preprocessed_dir: Path) -> pd.DataFrame:
    columns = numerical_column_table(role, preprocessed_dir)
    array = np.load(preprocessed_dir / f"{role}_array.npy", mmap_mode="r")
    rows: list[dict[str, object]] = []

    for row in columns.itertuples(index=False):
        value_index = int(row.column_index)
        flag_index = int(row.flag_column_index)
        valid_mask = array[:, flag_index] == 0
        values = np.asarray(array[valid_mask, value_index], dtype=np.float64)
        summary = summarize_values(values)
        rows.append(
            {
                "role": role,
                "variable": row.source_variable,
                "value_column": row.column_name,
                "value_column_index": value_index,
                "flag_column_index": flag_index,
                "n_total": int(array.shape[0]),
                "n_flag_or_missing": int(array.shape[0] - np.sum(valid_mask)),
                "flag_or_missing_rate": float(1.0 - np.mean(valid_mask)),
                **summary,
            }
        )

    return pd.DataFrame(rows)


def analyze(preprocessed_dir: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.concat(
        [
            analyze_role("source", preprocessed_dir),
            analyze_role("target", preprocessed_dir),
        ],
        ignore_index=True,
    )
    result = result.sort_values(
        ["iqr_outlier_rate", "extreme_iqr_outlier_rate", "n_valid"],
        ascending=[False, False, False],
        na_position="last",
    )
    result.to_csv(
        output_dir / "numerical_outlier_summary.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    result.head(50).to_csv(
        output_dir / "top50_by_iqr_outlier_rate.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    result.sort_values(
        ["extreme_iqr_outlier_rate", "iqr_outlier_rate", "n_valid"],
        ascending=[False, False, False],
        na_position="last",
    ).head(50).to_csv(
        output_dir / "top50_by_extreme_iqr_outlier_rate.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    result.sort_values(
        ["mad_outlier_rate", "iqr_outlier_rate", "n_valid"],
        ascending=[False, False, False],
        na_position="last",
    ).head(50).to_csv(
        output_dir / "top50_by_mad_outlier_rate.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    result.sort_values(
        ["tail_spread_ratio", "iqr_outlier_rate", "n_valid"],
        ascending=[False, False, False],
        na_position="last",
    ).head(50).to_csv(
        output_dir / "top50_by_tail_spread_ratio.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze outlier severity for numerical source and target variables."
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=PREPROCESSED_DIR,
        help="Directory containing source/target arrays and column metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where outlier summary CSV files will be written.",
    )
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args.preprocessed_dir, args.output_dir)
    display_columns = [
        "role",
        "variable",
        "n_valid",
        "flag_or_missing_rate",
        "iqr_outlier_rate",
        "extreme_iqr_outlier_rate",
        "mad_outlier_rate",
        "tail_spread_ratio",
        "min",
        "q25",
        "median",
        "q75",
        "max",
    ]
    print(f"Analyzed numerical variables: {len(result)}")
    print(f"Saved results to: {args.output_dir}")
    print()
    print(f"Top {args.top} by IQR outlier rate:")
    with pd.option_context("display.max_rows", args.top, "display.width", 180):
        print(result[display_columns].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
