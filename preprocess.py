from pathlib import Path
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd


CAT_MISSING_VALUE = -1
DEFAULT_SPLIT_RATIOS = (0.9, 0.09, 0.01)
DEFAULT_SPLIT_SEED = 42
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class PreprocessedTable:
    num: np.ndarray
    num_mask: np.ndarray
    cat: np.ndarray
    cat_mask: np.ndarray
    num_features: list[str]
    cat_features: list[str]
    cat_cardinalities: list[int]


@dataclass(frozen=True)
class NumericPercentileNormalization:
    sorted_values: list[np.ndarray]

    @property
    def train_counts(self) -> list[int]:
        return [int(len(values)) for values in self.sorted_values]


def load_sas7bdat(path: str | Path) -> pd.DataFrame:
    """Load a SAS7BDAT file and return it as a pandas DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SAS7BDAT file not found: {path}")

    try:
        import pyreadstat
    except ImportError:
        return pd.read_sas(path, format="sas7bdat", encoding="utf-8")

    last_error = None
    for encoding in [None, "utf-8", "cp949", "euc-kr"]:
        try:
            kwargs = {"encoding": encoding} if encoding else {}
            df, _ = pyreadstat.read_sas7bdat(str(path), **kwargs)
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to load SAS7BDAT file: {path}") from last_error


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load metadata CSV and return it as a pandas DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    metadata = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    required_columns = ["variable_name", "type", "classes", "flags", "bottom", "top"]
    missing_columns = [col for col in required_columns if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata file is missing columns: {missing_columns}")

    return metadata


def load_variable_list(path: str | Path) -> list[str]:
    """Load a newline-delimited variable list."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Variable list file not found: {path}")

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def load_skip_years(path: str | Path) -> dict[str, set[int]]:
    """Load skip-year rules as {column_name: {year, ...}}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Skip years file not found: {path}")

    skip_years = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    required_columns = ["column", "years"]
    missing_columns = [col for col in required_columns if col not in skip_years.columns]
    if missing_columns:
        raise ValueError(f"Skip years file is missing columns: {missing_columns}")

    rules: dict[str, set[int]] = {}
    for row in skip_years.itertuples(index=False):
        column = str(row.column).strip()
        if not column:
            continue

        years = {
            int(year.strip())
            for year in str(row.years).split(",")
            if year.strip()
        }
        rules[column] = years

    return rules


def load_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load config files: pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def filter_metadata(metadata: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Keep metadata rows for variables in the given order."""
    if "variable_name" not in metadata.columns:
        raise ValueError("Metadata is missing column: variable_name")

    selected_variables = set(variables)
    metadata_variables = set(metadata["variable_name"])
    duplicates = sorted(
        set(metadata.loc[metadata["variable_name"].duplicated(), "variable_name"])
        & selected_variables
    )
    if duplicates:
        raise ValueError(f"Metadata contains duplicate selected variables: {duplicates}")

    missing = [variable for variable in variables if variable not in metadata_variables]
    if missing:
        raise ValueError(f"Metadata is missing selected variables: {missing}")

    variable_index = {variable: index for index, variable in enumerate(variables)}
    filtered = metadata[metadata["variable_name"].isin(variable_index)].copy()
    filtered["_variable_order"] = filtered["variable_name"].map(variable_index)
    filtered = filtered.sort_values("_variable_order").drop(columns="_variable_order")
    return filtered.reset_index(drop=True)


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    deduplicated = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        deduplicated.append(value)
        seen.add(value)
    return deduplicated


def select_metadata(metadata: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Select variables as one table, without source/target roles."""
    variables_path = config.get("variables_path")
    variable_paths = config.get("variable_paths")
    if variables_path and variable_paths:
        raise ValueError("Use only one of variables_path or variable_paths.")

    variables: list[str] = []
    if variables_path:
        variables.extend(load_variable_list(variables_path))
    elif variable_paths:
        if isinstance(variable_paths, (str, Path)):
            variable_paths = [variable_paths]
        for path in variable_paths:
            variables.extend(load_variable_list(path))
    else:
        for legacy_key in ("source_variables_path", "target_variables_path"):
            if path := config.get(legacy_key):
                variables.extend(load_variable_list(path))

    if not variables:
        return metadata

    return filter_metadata(metadata, _deduplicate_preserve_order(variables))


def infer_dataset_year(path: str | Path) -> int:
    """Infer dataset year from names like hn13_all.sas7bdat."""
    match = re.search(r"hn(\d{2})_", Path(path).name.lower())
    if not match:
        raise ValueError(f"Could not infer dataset year from file name: {path}")

    year = int(match.group(1))
    return 1900 + year if year >= 90 else 2000 + year


def load_datasets(
    dataset_paths: list[str | Path],
    skip_years_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load dataset paths and blank values for columns skipped in each dataset year."""
    if not dataset_paths:
        raise ValueError("dataset_paths must not be empty.")

    skip_years = load_skip_years(skip_years_path) if skip_years_path else {}

    frames: list[pd.DataFrame] = []
    for path in dataset_paths:
        year = infer_dataset_year(path)
        df = load_sas7bdat(path)
        skip_columns = [
            column
            for column, years in skip_years.items()
            if year in years and column in df.columns
        ]
        if skip_columns:
            df[skip_columns] = df[skip_columns].astype("object")
            df.loc[:, skip_columns] = pd.NA

        frames.append(df)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
            category=FutureWarning,
        )
        return pd.concat(frames, ignore_index=True, sort=False)


def _split_codes(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _merge_codes(*code_groups: object) -> list[str]:
    merged = []
    seen = set()
    for code_group in code_groups:
        for value in _split_codes(code_group):
            normalized = _normalize_code(value)
            if normalized is None or normalized in seen:
                continue
            merged.append(value)
            seen.add(normalized)
    return merged


def _to_decimal(value: object) -> Decimal | None:
    if pd.isna(value) or str(value).strip() == "":
        return None

    try:
        return Decimal(str(value).strip())
    except InvalidOperation:
        return None


def _format_decimal(value: Decimal | None) -> str:
    if value is None:
        return ""
    if value == value.to_integral_value():
        return str(value.to_integral_value())
    return format(value.normalize(), "f").rstrip("0").rstrip(".")


def _normalize_code(value: object) -> tuple[str, str] | None:
    if pd.isna(value) or str(value).strip() == "":
        return None

    value = str(value).strip()
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return ("str", value)

    return ("num", _format_decimal(parsed))


def _preprocess_categorical_value_and_mask(
    data: pd.Series,
    classes: list[object],
    flags: list[object],
) -> tuple[pd.Series, pd.Series]:
    """Encode categorical data as class indices and return a valid mask."""
    class_codes = [_normalize_code(value) for value in classes]
    flag_codes = {
        code
        for value in flags
        if (code := _normalize_code(value)) is not None
    }
    if any(code is None for code in class_codes):
        raise ValueError("classes must not contain missing or blank values.")
    if len(set(class_codes)) != len(class_codes):
        raise ValueError("classes must not contain duplicate values.")
    if set(class_codes) & flag_codes:
        raise ValueError("classes and flags must not overlap for categorical preprocessing.")

    class_to_index = {code: index for index, code in enumerate(class_codes)}
    normalized = data.map(_normalize_code)
    class_indices = normalized.map(class_to_index)
    missing_or_flag_mask = normalized.isna() | normalized.isin(flag_codes)
    invalid_class_mask = ~missing_or_flag_mask & class_indices.isna()
    masked = missing_or_flag_mask | invalid_class_mask

    valid_mask = ~masked
    encoded = class_indices.fillna(CAT_MISSING_VALUE).astype(np.int64)
    encoded.loc[masked] = CAT_MISSING_VALUE

    return (
        encoded,
        valid_mask.astype(np.int64),
    )


def _preprocess_numerical_value_and_mask(
    data: pd.Series,
    flags: list[object],
) -> tuple[pd.Series, pd.Series]:
    """Preprocess numerical data into one continuous value and a valid mask."""
    numeric = pd.to_numeric(data, errors="coerce")
    normalized = data.map(_normalize_code)
    missing_mask = normalized.isna()

    flag_codes = {
        code
        for flag in flags
        if (code := _normalize_code(flag)) is not None
    }
    flag_mask = normalized.isin(flag_codes) if flag_codes else pd.Series(False, index=data.index)
    nonnumeric_mask = numeric.isna() & ~missing_mask & ~flag_mask
    masked = missing_mask | flag_mask | nonnumeric_mask

    valid_mask = ~masked
    continuous = numeric.astype("float32")
    continuous.loc[~valid_mask] = 0.0

    return (
        continuous,
        valid_mask.astype(np.int64),
    )


def _make_missing_series(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="object")


def _stack_columns(columns: list[np.ndarray], n_rows: int, dtype: np.dtype) -> np.ndarray:
    if not columns:
        return np.empty((n_rows, 0), dtype=dtype)

    return np.column_stack(columns).astype(dtype, copy=False)


def preprocess_dataset(
    dataset: pd.DataFrame,
    metadata: pd.DataFrame,
    role: str = "dataset",
) -> PreprocessedTable:
    """Preprocess selected variables into separate numerical and categorical arrays."""
    required_columns = ["variable_name", "type", "classes", "flags", "bottom", "top"]
    missing_columns = [col for col in required_columns if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {missing_columns}")

    n_rows = len(dataset.index)
    num_values: list[np.ndarray] = []
    num_masks: list[np.ndarray] = []
    cat_values: list[np.ndarray] = []
    cat_masks: list[np.ndarray] = []
    num_features: list[str] = []
    cat_features: list[str] = []
    cat_cardinalities: list[int] = []

    for row in metadata.itertuples(index=False):
        name = str(row.variable_name)
        variable_type = str(row.type)
        if variable_type == "pass":
            continue

        data = dataset[name] if name in dataset.columns else _make_missing_series(dataset.index)
        if variable_type == "categorical":
            classes = _split_codes(row.classes)
            if not classes:
                raise ValueError(f"Categorical variable has no classes: {role}.{name}")

            encoded, mask = _preprocess_categorical_value_and_mask(
                data,
                classes,
                _split_codes(row.flags),
            )
            cat_values.append(encoded.to_numpy(dtype=np.int64))
            cat_masks.append(mask.to_numpy(dtype=np.int64))
            cat_features.append(name)
            cat_cardinalities.append(len(classes))
        elif variable_type == "numerical":
            continuous, mask = _preprocess_numerical_value_and_mask(
                data,
                _split_codes(row.flags),
            )
            num_values.append(continuous.to_numpy(dtype=np.float32))
            num_masks.append(mask.to_numpy(dtype=np.int64))
            num_features.append(name)
        else:
            raise ValueError(f"Unsupported metadata type for {role}.{name}: {variable_type}")

    return PreprocessedTable(
        num=_stack_columns(num_values, n_rows, np.dtype("float32")),
        num_mask=_stack_columns(num_masks, n_rows, np.dtype("int64")),
        cat=_stack_columns(cat_values, n_rows, np.dtype("int64")),
        cat_mask=_stack_columns(cat_masks, n_rows, np.dtype("int64")),
        num_features=num_features,
        cat_features=cat_features,
        cat_cardinalities=cat_cardinalities,
    )


def make_split_indices(
    n_rows: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, np.ndarray]:
    """Shuffle row indices and split them into train/val/test."""
    if n_rows < 0:
        raise ValueError("n_rows must not be negative.")
    if len(ratios) != 3:
        raise ValueError("ratios must contain train, val, and test ratios.")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0: {ratios}")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_rows)
    train_size = int(n_rows * ratios[0])
    val_size = int(n_rows * ratios[1])
    test_size = n_rows - train_size - val_size

    return {
        "train": shuffled[:train_size],
        "val": shuffled[train_size : train_size + val_size],
        "test": shuffled[train_size + val_size : train_size + val_size + test_size],
    }


def fit_num_normalization(
    num: np.ndarray,
    num_mask: np.ndarray,
    train_indices: np.ndarray,
) -> NumericPercentileNormalization:
    """Fit empirical percentile distributions from train valid values only."""
    n_features = num.shape[1]
    sorted_values: list[np.ndarray] = []
    if n_features == 0:
        return NumericPercentileNormalization(sorted_values=sorted_values)
    if len(train_indices) == 0:
        return NumericPercentileNormalization(
            sorted_values=[np.empty(0, dtype=np.float64) for _ in range(n_features)]
        )

    train_num = num[train_indices]
    train_mask = num_mask[train_indices].astype(bool, copy=False)
    for feature_index in range(n_features):
        valid = train_mask[:, feature_index]
        if not valid.any():
            sorted_values.append(np.empty(0, dtype=np.float64))
            continue

        values = train_num[valid, feature_index].astype(np.float64, copy=False)
        values = values[np.isfinite(values)]
        sorted_values.append(np.sort(values))

    return NumericPercentileNormalization(sorted_values=sorted_values)


def apply_num_normalization(
    num: np.ndarray,
    num_mask: np.ndarray,
    normalization: NumericPercentileNormalization,
) -> np.ndarray:
    """Apply percentile rank normalization and keep masked numerical values at 0.0."""
    if num.shape[1] == 0:
        return num.astype(np.float32, copy=True)

    normalized = np.zeros(num.shape, dtype=np.float32)
    valid_mask = num_mask.astype(bool, copy=False)
    for feature_index, train_values in enumerate(normalization.sorted_values):
        valid = valid_mask[:, feature_index]
        if not valid.any() or len(train_values) == 0:
            continue

        values = num[valid, feature_index].astype(np.float64, copy=False)
        lower = np.searchsorted(train_values, values, side="left")
        upper = np.searchsorted(train_values, values, side="right")
        percentile = (lower + upper) / (2.0 * len(train_values))
        normalized[valid, feature_index] = np.clip(percentile, 0.0, 1.0).astype(np.float32)

    return normalized


def normalize_preprocessed_table(
    table: PreprocessedTable,
    train_indices: np.ndarray,
) -> tuple[PreprocessedTable, NumericPercentileNormalization]:
    normalization = fit_num_normalization(table.num, table.num_mask, train_indices)
    normalized = PreprocessedTable(
        num=apply_num_normalization(table.num, table.num_mask, normalization),
        num_mask=table.num_mask,
        cat=table.cat,
        cat_mask=table.cat_mask,
        num_features=table.num_features,
        cat_features=table.cat_features,
        cat_cardinalities=table.cat_cardinalities,
    )
    return normalized, normalization


def _save_split_array(
    output_dir: Path,
    prefix: str,
    array: np.ndarray,
    split_indices: dict[str, np.ndarray],
) -> None:
    for split_name in SPLIT_NAMES:
        np.save(output_dir / f"{prefix}_{split_name}.npy", array[split_indices[split_name]])


def _remove_stale_preprocessed_files(output_dir: Path) -> None:
    stale_names = [
        "source_array.npy",
        "source_mask.npy",
        "source_columns.csv",
        "source_mask_columns.csv",
        "source_quantile_lookup.csv",
        "source_raw_value_summary.csv",
        "target_array.npy",
        "target_mask.npy",
        "target_columns.csv",
        "target_mask_columns.csv",
        "target_quantile_lookup.csv",
        "target_raw_value_summary.csv",
        "num_percentile_lookup.npz",
    ]
    legacy_split_prefixes = (
        "source_num",
        "source_num_mask",
        "source_cat",
        "source_cat_mask",
        "target_num",
        "target_num_mask",
        "target_cat",
        "target_cat_mask",
    )
    stale_names.extend(
        f"{prefix}_{split_name}.npy"
        for prefix in legacy_split_prefixes
        for split_name in SPLIT_NAMES
    )
    for name in stale_names:
        path = output_dir / name
        if path.exists():
            path.unlink()


def build_info(
    table: PreprocessedTable,
    split_indices: dict[str, np.ndarray],
    num_normalization: NumericPercentileNormalization,
) -> dict[str, object]:
    return {
        "name": "knhanes",
        "id": "knhanes",
        "task_type": "tabular_generation",
        "num_normalization": "percentile",
        "num_normalization_fit": "train",
        "num_normalization_valid_mask_only": True,
        "num_percentile_range": [0.0, 1.0],
        "num_percentile_tie_strategy": "average_rank",
        "num_percentile_lookup_file": "num_percentile_lookup.npz",
        "num_percentile_lookup_key_format": "num_{feature_index}",
        "num_masked_value": 0.0,
        "train_size": int(len(split_indices["train"])),
        "val_size": int(len(split_indices["val"])),
        "test_size": int(len(split_indices["test"])),
        "n_num_features": len(table.num_features),
        "n_cat_features": len(table.cat_features),
        "num_features": table.num_features,
        "cat_features": table.cat_features,
        "cat_cardinalities": table.cat_cardinalities,
        "num_percentile_train_counts": num_normalization.train_counts,
    }


def _save_num_percentile_lookup(
    output_dir: Path,
    num_normalization: NumericPercentileNormalization,
) -> None:
    np.savez_compressed(
        output_dir / "num_percentile_lookup.npz",
        **{
            f"num_{feature_index}": values.astype(np.float32, copy=False)
            for feature_index, values in enumerate(num_normalization.sorted_values)
        },
    )


def save_preprocessed(
    processed: PreprocessedTable,
    output_dir: str | Path,
    split_indices: dict[str, np.ndarray],
    num_normalization: NumericPercentileNormalization,
) -> None:
    """Save preprocessed arrays and dataset info."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_preprocessed_files(output_dir)

    _save_split_array(output_dir, "num", processed.num, split_indices)
    _save_split_array(output_dir, "num_mask", processed.num_mask, split_indices)
    _save_split_array(output_dir, "cat", processed.cat, split_indices)
    _save_split_array(output_dir, "cat_mask", processed.cat_mask, split_indices)
    _save_num_percentile_lookup(output_dir, num_normalization)

    info = build_info(
        processed,
        split_indices,
        num_normalization,
    )
    with open(output_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_dataset(dataset: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Validate dataset values against categorical and numerical metadata rules."""
    required_columns = ["variable_name", "type", "classes", "flags", "bottom", "top"]
    missing_columns = [col for col in required_columns if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {missing_columns}")

    categorical_report = []
    categorical = metadata[metadata["type"] == "categorical"]
    for row in categorical.itertuples(index=False):
        name = row.variable_name
        allowed = {
            normalized
            for value in _merge_codes(row.classes, row.flags)
            if (normalized := _normalize_code(value)) is not None
        }

        if name not in dataset.columns:
            categorical_report.append(
                {
                    "variable_name": name,
                    "status": "missing_variable",
                    "classes": ",".join(_split_codes(row.classes)),
                    "observed_values": "",
                    "invalid_values": "",
                    "invalid_count": "",
                    "observed_count": "",
                }
            )
            continue

        observed = {}
        invalid = {}
        for value, count in dataset[name].dropna().value_counts().items():
            normalized = _normalize_code(value)
            if normalized is None:
                continue
            observed[normalized] = observed.get(normalized, 0) + int(count)
            if normalized not in allowed:
                invalid[normalized] = invalid.get(normalized, 0) + int(count)

        invalid_items = sorted(invalid.items(), key=lambda item: (-item[1], item[0][1]))
        observed_items = sorted(observed.items(), key=lambda item: item[0][1])
        categorical_report.append(
            {
                "variable_name": name,
                "status": "ok" if not invalid else "invalid_values",
                "classes": ",".join(_split_codes(row.classes)),
                "observed_values": ";".join(
                    f"{value[1]}({count})" for value, count in observed_items
                ),
                "invalid_values": ";".join(
                    f"{value[1]}({count})" for value, count in invalid_items
                ),
                "invalid_count": sum(invalid.values()),
                "observed_count": sum(observed.values()),
            }
        )

    numerical_report = []
    numerical = metadata[metadata["type"] == "numerical"]
    for row in numerical.itertuples(index=False):
        name = row.variable_name
        bottom = _to_decimal(row.bottom)
        top = _to_decimal(row.top)
        flags = {
            parsed
            for value in _split_codes(row.flags)
            if (parsed := _to_decimal(value)) is not None
        }

        if name not in dataset.columns:
            numerical_report.append(
                {
                    "variable_name": name,
                    "status": "missing_variable",
                    "bottom": _format_decimal(bottom),
                    "top": _format_decimal(top),
                    "flags": ",".join(_split_codes(row.flags)),
                    "observed_min": "",
                    "observed_max": "",
                    "invalid_values": "",
                    "nonnumeric_values": "",
                    "invalid_count": "",
                    "observed_count": "",
                    "missing_count": "",
                }
            )
            continue

        observed_count = 0
        missing_count = 0
        invalid_values = {}
        nonnumeric_values = {}
        observed_min = None
        observed_max = None

        for value, count in dataset[name].value_counts(dropna=False).items():
            count = int(count)
            if pd.isna(value) or str(value).strip() == "":
                missing_count += count
                continue

            observed_count += count
            parsed = _to_decimal(value)
            if parsed is None:
                key = str(value).strip()
                nonnumeric_values[key] = nonnumeric_values.get(key, 0) + count
                continue

            if parsed not in flags:
                observed_min = parsed if observed_min is None else min(observed_min, parsed)
                observed_max = parsed if observed_max is None else max(observed_max, parsed)

            in_range = (bottom is None or parsed >= bottom) and (top is None or parsed <= top)
            if parsed not in flags and not in_range:
                key = _format_decimal(parsed)
                invalid_values[key] = invalid_values.get(key, 0) + count

        invalid_items = sorted(invalid_values.items(), key=lambda item: (-item[1], item[0]))
        nonnumeric_items = sorted(nonnumeric_values.items(), key=lambda item: (-item[1], item[0]))
        invalid_count = sum(invalid_values.values()) + sum(nonnumeric_values.values())
        numerical_report.append(
            {
                "variable_name": name,
                "status": "ok" if not invalid_count else "invalid_values",
                "bottom": _format_decimal(bottom),
                "top": _format_decimal(top),
                "flags": ",".join(_split_codes(row.flags)),
                "observed_min": _format_decimal(observed_min),
                "observed_max": _format_decimal(observed_max),
                "invalid_values": ";".join(
                    f"{value}({count})" for value, count in invalid_items
                ),
                "nonnumeric_values": ";".join(
                    f"{value}({count})" for value, count in nonnumeric_items
                ),
                "invalid_count": invalid_count,
                "observed_count": observed_count,
                "missing_count": missing_count,
            }
        )

    return {
        "categorical": pd.DataFrame(categorical_report),
        "numerical": pd.DataFrame(numerical_report),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="preprocess_config.template.yaml",
        help="Path to preprocess config YAML.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run dataset validation and write validation reports.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_paths = [dataset["path"] for dataset in config["datasets"]]

    dataset = load_datasets(dataset_paths, config.get("skip_years_path"))
    metadata = load_metadata(config["metadata_path"])
    selected_metadata = select_metadata(metadata, config)

    print(f"Loaded dataset shape: {dataset.shape}")
    print(f"Selected variables: {len(selected_metadata)}")
    if args.validate:
        reports = validate_dataset(dataset, selected_metadata)

        reports["categorical"].to_csv(
            "categorical_validation_report.csv",
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )
        reports["numerical"].to_csv(
            "numerical_validation_report.csv",
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )

        for report_name, report in reports.items():
            print(f"{report_name}: {report['status'].value_counts().to_dict()}")

    processed = preprocess_dataset(dataset, selected_metadata)

    split_seed = int(config.get("split_seed", DEFAULT_SPLIT_SEED))
    split_indices = make_split_indices(len(dataset), DEFAULT_SPLIT_RATIOS, split_seed)
    processed, num_normalization = normalize_preprocessed_table(
        processed,
        split_indices["train"],
    )
    output_dir = config.get("output_dir", "preprocessed")
    save_preprocessed(
        processed,
        output_dir,
        split_indices,
        num_normalization,
    )

    print(f"Preprocessed num shape: {processed.num.shape}")
    print(f"Preprocessed cat shape: {processed.cat.shape}")
    print(
        "Split sizes: "
        f"train={len(split_indices['train'])}, "
        f"val={len(split_indices['val'])}, "
        f"test={len(split_indices['test'])}"
    )
    print(f"Saved preprocessed files to: {output_dir}")


if __name__ == "__main__":
    main()
