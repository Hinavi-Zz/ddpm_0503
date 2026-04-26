from pathlib import Path
from decimal import Decimal, InvalidOperation
import argparse
import re
import warnings

import numpy as np
import pandas as pd


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

    variable_index = {variable: index for index, variable in enumerate(variables)}
    filtered = metadata[metadata["variable_name"].isin(variable_index)].copy()
    filtered["_variable_order"] = filtered["variable_name"].map(variable_index)
    filtered = filtered.sort_values("_variable_order").drop(columns="_variable_order")
    return filtered.reset_index(drop=True)


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


def _preprocess_source_categorical(data: pd.Series, classes: list[object]) -> pd.DataFrame:
    """One-hot encode 1D source categorical data, using the last column for missing values."""
    class_codes = [_normalize_code(value) for value in classes]
    if any(code is None for code in class_codes):
        raise ValueError("classes must not contain missing or blank values.")
    if len(set(class_codes)) != len(class_codes):
        raise ValueError("classes must not contain duplicate values.")

    class_to_index = {code: index for index, code in enumerate(class_codes)}
    columns = [str(value) for value in classes] + ["missing"]
    normalized = data.map(_normalize_code)
    class_indices = normalized.map(class_to_index)
    missing_mask = normalized.isna()
    invalid_mask = class_indices.isna() & ~missing_mask
    missing_or_invalid_mask = missing_mask | invalid_mask

    encoded = np.zeros((len(data), len(classes) + 1), dtype=np.int8)
    valid_positions = np.flatnonzero(~missing_or_invalid_mask.to_numpy())
    encoded[valid_positions, class_indices.dropna().astype(int).to_numpy()] = 1
    encoded[np.flatnonzero(missing_or_invalid_mask.to_numpy()), len(classes)] = 1

    return pd.DataFrame(encoded, index=data.index, columns=columns)


def _summarize_raw_values(values: pd.Series) -> dict[str, object]:
    """Return summary metadata for raw continuous values."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "method": "raw_value",
            "n_values": 0,
            "n_unique_values": 0,
            "min_value": np.nan,
            "max_value": np.nan,
            "median_value": np.nan,
            "placeholder_value": 0.0,
        }

    return {
        "method": "raw_value",
        "n_values": int(numeric.shape[0]),
        "n_unique_values": int(numeric.nunique(dropna=True)),
        "min_value": float(numeric.min()),
        "max_value": float(numeric.max()),
        "median_value": float(numeric.median()),
        "placeholder_value": 0.0,
    }


def _preprocess_source_numerical(
    data: pd.Series,
    flags: list[object],
    bottom: object,
    top: object,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Preprocess 1D source numerical data into indicators and a continuous value."""
    numeric = pd.to_numeric(data, errors="coerce")
    missing_mask = numeric.isna()

    parsed_flags = [
        (_format_decimal(parsed), float(parsed))
        for flag in flags
        if (parsed := _to_decimal(flag)) is not None
    ]
    flag_values = [value for _, value in parsed_flags]
    flag_mask = numeric.isin(flag_values) if flag_values else pd.Series(False, index=data.index)
    bottom_value = float(bottom) if _to_decimal(bottom) is not None else None
    top_value = float(top) if _to_decimal(top) is not None else None
    below_bottom_mask = (
        (numeric <= bottom_value) & ~flag_mask & ~missing_mask
        if bottom_value is not None
        else pd.Series(False, index=data.index)
    )
    above_top_mask = (
        (numeric >= top_value) & ~flag_mask & ~missing_mask & ~below_bottom_mask
        if top_value is not None
        else pd.Series(False, index=data.index)
    )

    categorical_columns = (
        [f"flag_{label}" for label, _ in parsed_flags]
        + ["le_bottom", "ge_top", "missing"]
    )
    categorical = np.zeros((len(data), len(categorical_columns)), dtype=np.int8)

    for index, flag in enumerate(flag_values):
        categorical[:, index] = (numeric == flag).to_numpy(dtype=np.int8)
    categorical[:, len(flag_values)] = below_bottom_mask.to_numpy(dtype=np.int8)
    categorical[:, len(flag_values) + 1] = above_top_mask.to_numpy(dtype=np.int8)
    categorical[:, len(flag_values) + 2] = missing_mask.to_numpy(dtype=np.int8)

    valid_continuous_mask = ~missing_mask & ~flag_mask
    summary = _summarize_raw_values(numeric[valid_continuous_mask])
    continuous = numeric.astype("float32")
    continuous.loc[~valid_continuous_mask] = 0.0

    return (
        pd.DataFrame(categorical, index=data.index, columns=categorical_columns),
        continuous,
        summary,
    )


def preprocess_source_dataset(dataset: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Preprocess source variables described in metadata."""
    required_columns = ["variable_name", "type", "classes", "flags", "bottom", "top"]
    missing_columns = [col for col in required_columns if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {missing_columns}")

    processed_parts: list[pd.DataFrame] = []
    column_info: list[dict[str, object]] = []
    raw_value_summary: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        name = row.variable_name
        variable_type = row.type
        if name not in dataset.columns:
            continue

        data = dataset[name]
        if variable_type == "categorical":
            categorical = _preprocess_source_categorical(
                data,
                _merge_codes(row.classes, row.flags),
            )
            for column in categorical.columns:
                column_info.append(
                    {
                        "column_name": f"{name}__{column}",
                        "source_variable": name,
                        "source_type": "categorical",
                        "feature_type": "one_hot",
                        "meaning": column,
                    }
                )
            categorical = categorical.add_prefix(f"{name}__")
            processed_parts.append(categorical)
        elif variable_type == "numerical":
            categorical, continuous, summary = _preprocess_source_numerical(
                data,
                _split_codes(row.flags),
                row.bottom,
                row.top,
            )
            for column in categorical.columns:
                column_info.append(
                    {
                        "column_name": f"{name}__{column}",
                        "source_variable": name,
                        "source_type": "numerical",
                        "feature_type": "indicator",
                        "meaning": column,
                    }
                )
            categorical = categorical.add_prefix(f"{name}__")
            continuous = continuous.rename(f"{name}__value").to_frame()
            column_info.append(
                {
                    "column_name": f"{name}__value",
                    "source_variable": name,
                    "source_type": "numerical",
                    "feature_type": "continuous",
                    "meaning": "raw_value",
                }
            )
            raw_value_summary.append(
                {
                    "source_variable": name,
                    "column_name": f"{name}__value",
                    "role": "source",
                    **summary,
                }
            )
            processed_parts.extend([categorical, continuous])
        elif variable_type == "pass":
            continue

    if not processed_parts:
        return pd.DataFrame(index=dataset.index)

    processed = pd.concat(processed_parts, axis=1)
    for index, info in enumerate(column_info):
        info["column_index"] = index
    processed.attrs["column_info"] = pd.DataFrame(column_info)
    processed.attrs["raw_value_summary"] = pd.DataFrame(raw_value_summary)
    return processed


def _write_preprocess_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )


def save_preprocessed_source(processed_source: pd.DataFrame, output_dir: str | Path) -> None:
    """Save preprocessed source array and memo files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "source_array.npy", processed_source.to_numpy(dtype=np.float32))
    _write_preprocess_csv(
        processed_source.attrs.get("column_info", pd.DataFrame()),
        output_dir / "source_columns.csv",
    )
    _write_preprocess_csv(
        processed_source.attrs.get("raw_value_summary", pd.DataFrame()),
        output_dir / "source_raw_value_summary.csv",
    )
    for stale_quantile_lookup in [
        output_dir / "source_quantile_lookup.csv",
    ]:
        if stale_quantile_lookup.exists():
            stale_quantile_lookup.unlink()


def _preprocess_target_categorical(
    data: pd.Series,
    classes: list[object],
    flags: list[object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical targets and mask flags/missing rows."""
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
        raise ValueError("classes and flags must not overlap for target preprocessing.")

    class_to_index = {code: index for index, code in enumerate(class_codes)}
    columns = [str(value) for value in classes]
    normalized = data.map(_normalize_code)
    class_indices = normalized.map(class_to_index)
    missing_mask = normalized.isna()
    flag_mask = normalized.isin(flag_codes) if flag_codes else pd.Series(False, index=data.index)
    valid_mask = class_indices.notna() & ~missing_mask & ~flag_mask

    target = np.zeros((len(data), len(classes)), dtype=np.float32)
    mask = np.zeros((len(data), len(classes)), dtype=np.float32)
    valid_positions = np.flatnonzero(valid_mask.to_numpy())
    target[valid_positions, class_indices[valid_mask].astype(int).to_numpy()] = 1.0
    mask[valid_positions, :] = 1.0

    return (
        pd.DataFrame(target, index=data.index, columns=columns),
        pd.DataFrame(mask, index=data.index, columns=columns),
    )


def _preprocess_target_numerical(
    data: pd.Series,
    flags: list[object],
) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    """Keep numerical targets as raw values and mask flags/missing rows."""
    numeric = pd.to_numeric(data, errors="coerce")
    missing_mask = numeric.isna()

    flag_values = [
        float(parsed)
        for flag in flags
        if (parsed := _to_decimal(flag)) is not None
    ]
    flag_mask = numeric.isin(flag_values) if flag_values else pd.Series(False, index=data.index)
    train_mask = ~missing_mask & ~flag_mask

    summary = _summarize_raw_values(numeric[train_mask])
    target = numeric.astype("float32")
    target.loc[~train_mask] = 0.0

    return (
        target,
        train_mask.astype("float32"),
        summary,
    )


def preprocess_target_dataset(
    dataset: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess target variables and return target and gradient mask arrays."""
    required_columns = ["variable_name", "type", "classes", "flags", "bottom", "top"]
    missing_columns = [col for col in required_columns if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {missing_columns}")

    target_parts: list[pd.DataFrame] = []
    mask_parts: list[pd.DataFrame] = []
    column_info: list[dict[str, object]] = []
    raw_value_summary: list[dict[str, object]] = []

    for row in metadata.itertuples(index=False):
        name = row.variable_name
        variable_type = row.type
        if name not in dataset.columns:
            continue

        data = dataset[name]
        if variable_type == "categorical":
            target, mask = _preprocess_target_categorical(
                data,
                _split_codes(row.classes),
                _split_codes(row.flags),
            )
            for column in target.columns:
                column_info.append(
                    {
                        "column_name": f"{name}__{column}",
                        "source_variable": name,
                        "source_type": "categorical",
                        "target_type": "one_hot",
                        "meaning": column,
                        "mask_rule": "0 when value is flag, missing, or outside classes",
                    }
                )
            target_parts.append(target.add_prefix(f"{name}__"))
            mask_parts.append(mask.add_prefix(f"{name}__"))
        elif variable_type == "numerical":
            target, mask, summary = _preprocess_target_numerical(
                data,
                _split_codes(row.flags),
            )
            column_name = f"{name}__value"
            target_parts.append(target.rename(column_name).to_frame())
            mask_parts.append(mask.rename(column_name).to_frame())
            column_info.append(
                {
                    "column_name": column_name,
                    "source_variable": name,
                    "source_type": "numerical",
                    "target_type": "continuous",
                    "meaning": "raw_value",
                    "mask_rule": "0 when value is flag or missing; bottom/top do not affect mask",
                }
            )
            raw_value_summary.append(
                {
                    "source_variable": name,
                    "column_name": column_name,
                    "role": "target",
                    **summary,
                }
            )
        elif variable_type == "pass":
            continue

    if not target_parts:
        empty = pd.DataFrame(index=dataset.index)
        empty.attrs["column_info"] = pd.DataFrame()
        empty.attrs["raw_value_summary"] = pd.DataFrame()
        return empty, empty.copy()

    target = pd.concat(target_parts, axis=1)
    mask = pd.concat(mask_parts, axis=1)
    for index, info in enumerate(column_info):
        info["column_index"] = index
    target.attrs["column_info"] = pd.DataFrame(column_info)
    target.attrs["raw_value_summary"] = pd.DataFrame(raw_value_summary)
    return target, mask


def save_preprocessed_target(
    processed_target: pd.DataFrame,
    target_mask: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save preprocessed targets, masks, and memo files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "target_array.npy", processed_target.to_numpy(dtype=np.float32))
    np.save(output_dir / "target_mask.npy", target_mask.to_numpy(dtype=np.float32))
    _write_preprocess_csv(
        processed_target.attrs.get("column_info", pd.DataFrame()),
        output_dir / "target_columns.csv",
    )
    _write_preprocess_csv(
        processed_target.attrs.get("raw_value_summary", pd.DataFrame()),
        output_dir / "target_raw_value_summary.csv",
    )
    for stale_quantile_lookup in [
        output_dir / "target_quantile_lookup.csv",
    ]:
        if stale_quantile_lookup.exists():
            stale_quantile_lookup.unlink()


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
    source_metadata = metadata
    source_variables_path = config.get("source_variables_path")
    if source_variables_path:
        source_variables = load_variable_list(source_variables_path)
        source_metadata = filter_metadata(metadata, source_variables)

    target_metadata = None
    target_variables_path = config.get("target_variables_path")
    if target_variables_path:
        target_variables = load_variable_list(target_variables_path)
        target_metadata = filter_metadata(metadata, target_variables)

    print(f"Loaded dataset shape: {dataset.shape}")
    print(f"Selected source variables: {len(source_metadata)}")
    if target_metadata is not None:
        print(f"Selected target variables: {len(target_metadata)}")
    if args.validate:
        validation_metadata = source_metadata
        if target_metadata is not None:
            validation_metadata = pd.concat(
                [source_metadata, target_metadata],
                ignore_index=True,
            ).drop_duplicates("variable_name")
        reports = validate_dataset(dataset, validation_metadata)

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

    processed_source = preprocess_source_dataset(dataset, source_metadata)
    save_preprocessed_source(processed_source, "preprocessed")
    print(f"Preprocessed source shape: {processed_source.shape}")
    print("Saved preprocessed source files to: preprocessed")

    if target_metadata is not None:
        processed_target, target_mask = preprocess_target_dataset(dataset, target_metadata)
        save_preprocessed_target(processed_target, target_mask, "preprocessed")
        print(f"Preprocessed target shape: {processed_target.shape}")
        print(f"Target mask shape: {target_mask.shape}")
        print("Saved preprocessed target files to: preprocessed")


if __name__ == "__main__":
    main()
