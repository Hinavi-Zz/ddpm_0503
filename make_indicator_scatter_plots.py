from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "indicator_scatter_2013_plus"
EXTRACTED_CSV = OUTPUT_DIR / "indicator_values_2013_plus.csv"
SUMMARY_CSV = OUTPUT_DIR / "pairwise_summary_2013_plus.csv"

START_YEAR = 2013
END_YEAR = 2024
METRICS = ["HE_glu", "HE_Uglu", "HE_HbA1c", "HE_Ualb"]
UGLU_CLASSES = set(range(6))

LABELS = {
    "HE_glu": "HE_glu (fasting glucose, mg/dL)",
    "HE_Uglu": "HE_Uglu (urine glucose, 0-5)",
    "HE_HbA1c": "HE_HbA1c (%)",
    "HE_Ualb": "HE_Ualb (urine albumin, ug/mL)",
}


def infer_year(path: Path) -> int:
    match = re.search(r"hn(\d{2})_", path.name.lower())
    if not match:
        raise ValueError(f"Could not infer year from filename: {path}")
    year = int(match.group(1))
    return 1900 + year if year >= 90 else 2000 + year


def expected_dataset_paths() -> list[Path]:
    paths = []
    for year in range(START_YEAR, END_YEAR + 1):
        path = DATASET_DIR / f"hn{year % 100:02d}_all.sas7bdat"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")
        paths.append(path)
    return paths


def extract_values() -> None:
    try:
        import pandas as pd
        import pyreadstat
    except ImportError as exc:
        raise SystemExit(
            "Extraction requires pandas and pyreadstat. In this workspace, run: "
            "py -3.10 make_indicator_scatter_plots.py --extract"
        ) from exc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for path in expected_dataset_paths():
        file_year = infer_year(path)
        _, metadata = pyreadstat.read_sas7bdat(str(path), metadataonly=True)
        available = set(metadata.column_names)
        usecols = [column for column in ["year", *METRICS] if column in available]
        missing_metrics = [metric for metric in METRICS if metric not in available]

        df, _ = pyreadstat.read_sas7bdat(str(path), usecols=usecols)
        if "year" not in df.columns:
            df["year"] = file_year
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(file_year).astype(int)

        for metric in METRICS:
            if metric not in df.columns:
                df[metric] = pd.NA
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

        df = df.loc[df["year"] >= START_YEAR, ["year", *METRICS]]
        frames.append(df)
        suffix = f", missing metrics: {', '.join(missing_metrics)}" if missing_metrics else ""
        print(f"Loaded {path.name}: {len(df):,} rows{suffix}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(EXTRACTED_CSV, index=False, encoding="utf-8-sig")
    print(f"Wrote {EXTRACTED_CSV} ({len(combined):,} rows)")


def valid_pair(df, x: str, y: str):
    import numpy as np
    import pandas as pd

    pair = df[["year", x, y]].copy()
    pair[x] = pd.to_numeric(pair[x], errors="coerce")
    pair[y] = pd.to_numeric(pair[y], errors="coerce")
    mask = np.isfinite(pair[x].to_numpy(dtype=float)) & np.isfinite(pair[y].to_numpy(dtype=float))

    if x == "HE_Uglu":
        mask &= pair[x].isin(UGLU_CLASSES).to_numpy()
    if y == "HE_Uglu":
        mask &= pair[y].isin(UGLU_CLASSES).to_numpy()

    return pair.loc[mask].reset_index(drop=True)


def axis_values(pair, metric: str, rng):
    values = pair[metric].to_numpy(dtype=float)
    if metric == "HE_Uglu":
        return values + rng.uniform(-0.08, 0.08, size=len(values))
    return values


def plot_pairs() -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Plotting requires pandas, numpy, and matplotlib. In this workspace, run: "
            "py -3.11 make_indicator_scatter_plots.py --plot"
        ) from exc

    if not EXTRACTED_CSV.exists():
        raise FileNotFoundError(
            f"Missing extracted data: {EXTRACTED_CSV}. Run --extract first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(EXTRACTED_CSV)
    rng = np.random.default_rng(20260426)
    summary_rows = []

    plt.style.use("default")
    for x, y in combinations(METRICS, 2):
        pair = valid_pair(df, x, y)
        if pair.empty:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = pair[[x, y]].corr(method="pearson").iloc[0, 1]
            spearman = pair[[x, y]].corr(method="spearman").iloc[0, 1]

        summary_rows.append(
            {
                "x": x,
                "y": y,
                "n_complete_cases": len(pair),
                "pearson_r": pearson,
                "spearman_r": spearman,
                "year_min": int(pair["year"].min()) if len(pair) else "",
                "year_max": int(pair["year"].max()) if len(pair) else "",
            }
        )

        fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
        ax.scatter(
            axis_values(pair, x, rng),
            axis_values(pair, y, rng),
            s=8,
            alpha=0.22,
            linewidths=0,
            color="#2563eb",
            rasterized=True,
        )
        ax.set_xlabel(LABELS[x])
        ax.set_ylabel(LABELS[y])
        ax.set_title(
            f"{x} vs {y} | {START_YEAR}-{END_YEAR} | complete cases n={len(pair):,}",
            fontsize=10,
        )
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)

        if x == "HE_Uglu":
            ax.set_xticks(range(6))
            ax.set_xlim(-0.5, 5.5)
        if y == "HE_Uglu":
            ax.set_yticks(range(6))
            ax.set_ylim(-0.5, 5.5)

        fig.tight_layout()
        output_path = OUTPUT_DIR / f"scatter_{x}_vs_{y}_2013_plus.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {output_path}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"Wrote {SUMMARY_CSV}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pairwise scatter plots for selected KNHANES indicators from 2013 onward."
    )
    parser.add_argument("--extract", action="store_true", help="Extract selected columns from SAS files.")
    parser.add_argument("--plot", action="store_true", help="Create pairwise scatter PNG files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_extract = args.extract or not args.plot
    run_plot = args.plot or not args.extract

    if run_extract:
        extract_values()
    if run_plot:
        plot_pairs()


if __name__ == "__main__":
    main()
