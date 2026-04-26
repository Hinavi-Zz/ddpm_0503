from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "target_numerical_correlations"


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered))
    if denominator == 0:
        return np.nan
    return float(np.sum(x_centered * y_centered) / denominator)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return pearson_corr(x_rank, y_rank)


def load_target_numerical_data(preprocessed_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    columns_path = preprocessed_dir / "target_columns.csv"
    target_path = preprocessed_dir / "target_array.npy"

    columns = pd.read_csv(columns_path)
    numerical = columns[
        (columns["source_type"] == "numerical")
        & (columns["target_type"] == "continuous")
    ].copy()
    flag_indices = (
        columns[
            (columns["source_type"] == "numerical")
            & (columns["meaning"] == "flag")
        ][["source_variable", "column_index"]]
        .rename(columns={"column_index": "flag_column_index"})
    )
    numerical = numerical.merge(flag_indices, on="source_variable", how="left")
    numerical = numerical.sort_values("column_index").reset_index(drop=True)

    target = np.load(target_path, mmap_mode="r")
    return numerical, target


def analyze(preprocessed_dir: Path, output_dir: Path, min_complete_cases: int) -> pd.DataFrame:
    numerical, target = load_target_numerical_data(preprocessed_dir)
    rows: list[dict[str, object]] = []

    for left, right in combinations(numerical.itertuples(index=False), 2):
        left_index = int(left.column_index)
        right_index = int(right.column_index)
        left_flag_index = int(left.flag_column_index)
        right_flag_index = int(right.flag_column_index)
        complete_mask = (
            (target[:, left_flag_index] == 0)
            & (target[:, right_flag_index] == 0)
        )
        n_complete = int(np.sum(complete_mask))

        row = {
            "left_variable": left.source_variable,
            "right_variable": right.source_variable,
            "left_column": left.column_name,
            "right_column": right.column_name,
            "left_index": left_index,
            "right_index": right_index,
            "left_flag_index": left_flag_index,
            "right_flag_index": right_flag_index,
            "n_complete_cases": n_complete,
            "pearson_r": np.nan,
            "abs_pearson_r": np.nan,
            "spearman_r": np.nan,
            "abs_spearman_r": np.nan,
        }

        if n_complete >= min_complete_cases:
            x = np.asarray(target[complete_mask, left_index], dtype=np.float64)
            y = np.asarray(target[complete_mask, right_index], dtype=np.float64)
            pearson = pearson_corr(x, y)
            spearman = spearman_corr(x, y)
            row.update(
                {
                    "pearson_r": pearson,
                    "abs_pearson_r": abs(pearson) if np.isfinite(pearson) else np.nan,
                    "spearman_r": spearman,
                    "abs_spearman_r": abs(spearman) if np.isfinite(spearman) else np.nan,
                }
            )

        rows.append(row)

    correlations = pd.DataFrame(rows).sort_values(
        ["abs_pearson_r", "n_complete_cases"],
        ascending=[False, False],
        na_position="last",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(
        output_dir / "target_numerical_pairwise_correlations.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    correlations.head(50).to_csv(
        output_dir / "target_numerical_top50_by_abs_pearson.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    correlations.sort_values(
        ["abs_spearman_r", "n_complete_cases"],
        ascending=[False, False],
        na_position="last",
    ).head(50).to_csv(
        output_dir / "target_numerical_top50_by_abs_spearman.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )

    return correlations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pairwise correlations among numerical target variables."
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=PREPROCESSED_DIR,
        help="Directory containing target_array.npy and target_columns.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where correlation CSV files will be written.",
    )
    parser.add_argument(
        "--min-complete-cases",
        type=int,
        default=3,
        help="Minimum complete-case count needed to compute a pairwise correlation.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top absolute Pearson correlations to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    numerical, _ = load_target_numerical_data(args.preprocessed_dir)
    correlations = analyze(args.preprocessed_dir, args.output_dir, args.min_complete_cases)
    valid = correlations["pearson_r"].notna()

    print(f"Numerical target variables: {len(numerical)}")
    print(f"Pair comparisons: {len(correlations)}")
    print(f"Computed Pearson correlations: {int(valid.sum())}")
    print(f"Saved results to: {args.output_dir}")
    print()
    print(f"Top {args.top} pairs by absolute Pearson r:")
    display_columns = [
        "left_variable",
        "right_variable",
        "n_complete_cases",
        "pearson_r",
        "spearman_r",
    ]
    top = correlations.loc[valid, display_columns].head(args.top).copy()
    with pd.option_context("display.max_rows", args.top, "display.width", 120):
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
