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
) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode categorical data and return a variable-level valid mask."""
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
    columns = [str(value) for value in classes]
    normalized = data.map(_normalize_code)
    class_indices = normalized.map(class_to_index)
    missing_or_flag_mask = normalized.isna() | normalized.isin(flag_codes)
    invalid_class_mask = ~missing_or_flag_mask & class_indices.isna()
    masked = missing_or_flag_mask | invalid_class_mask

    encoded = np.zeros((len(data), len(classes)), dtype=np.int8)
    valid_mask = ~masked
    valid_positions = np.flatnonzero(valid_mask.to_numpy())
    encoded[valid_positions, class_indices[valid_mask].astype(int).to_numpy()] = 1

    return (
        pd.DataFrame(encoded, index=data.index, columns=columns),
        valid_mask.astype(np.int8),
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
        valid_mask.astype(np.int8),
    )


def _make_missing_series(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="object")


def _empty_preprocessed(index: pd.Index) -> pd.DataFrame:
    empty = pd.DataFrame(index=index)
    empty.attrs["column_info"] = pd.DataFrame()
    empty.attrs["mask_info"] = pd.DataFrame()
    return empty


def _make_value_mask_frame(values: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    mask_values = np.repeat(
        mask.to_numpy(dtype=np.int8)[:, None],
        values.shape[1],
        axis=1,
    )
    return pd.DataFrame(
        mask_values,
        index=values.index,
        columns=[f"{column}__mask" for column in values.columns],
    )


def _value_type_column(role: str) -> str:
    if role == "source":
        return "feature_type"
    if role == "target":
        return "target_type"
    raise ValueError(f"Unsupported role: {role}")


def preprocess_dataset(
    dataset: pd.DataFrame,
    metadata: pd.DataFrame,
    role: str,
) -> pd.DataFrame:
    """Preprocess selected variables into value columns plus value-shaped masks."""
    value_type_column = _value_type_column(role)
    required_columns = ["variable_name", "type", "classes", "flags", "bottom", "top"]
    missing_columns = [col for col in required_columns if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {missing_columns}")

    value_parts: list[pd.DataFrame] = []
    mask_parts: list[pd.DataFrame] = []
    column_info: list[dict[str, object]] = []
    mask_info: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        name = row.variable_name
        variable_type = row.type
        if variable_type == "pass":
            continue

        data = dataset[name] if name in dataset.columns else _make_missing_series(dataset.index)
        if variable_type == "categorical":
            values, mask = _preprocess_categorical_value_and_mask(
                data,
                _split_codes(row.classes),
                _split_codes(row.flags),
            )
            values = values.add_prefix(f"{name}__")
            for column in values.columns:
                rawdata = column.removeprefix(f"{name}__")
                column_info.append(
                    {
                        "column_name": column,
                        "source_variable": name,
                        "source_type": "categorical",
                        value_type_column: "one_hot",
                        "rawdata": rawdata,
                        "default_value": 0.0,
                    }
                )
            value_parts.append(values)
        elif variable_type == "numerical":
            continuous, mask = _preprocess_numerical_value_and_mask(
                data,
                _split_codes(row.flags),
            )
            values = continuous.rename(f"{name}__value").to_frame()
            value_column = f"{name}__value"
            column_info.append(
                {
                    "column_name": value_column,
                    "source_variable": name,
                    "source_type": "numerical",
                    value_type_column: "continuous",
                    "rawdata": "raw_value",
                    "default_value": 0.0,
                }
            )
            value_parts.append(values)
        else:
            raise ValueError(f"Unsupported metadata type for {name}: {variable_type}")

        value_mask = _make_value_mask_frame(values, mask)
        mask_parts.append(value_mask)
        for mask_column in value_mask.columns:
            mask_info.append(
                {
                    "column_name": mask_column,
                    "source_variable": name,
                    "valid_rate": float(mask.mean()) if len(mask) else np.nan,
                }
            )

    if not value_parts:
        return _empty_preprocessed(dataset.index)

    processed = pd.concat(value_parts, axis=1)
    mask = pd.concat(mask_parts, axis=1)
    for index, info in enumerate(column_info):
        info["column_index"] = index
    for index, info in enumerate(mask_info):
        info["column_index"] = index
    processed.attrs["column_info"] = pd.DataFrame(column_info)
    processed.attrs["mask"] = mask
    processed.attrs["mask_info"] = pd.DataFrame(mask_info)
    return processed


def preprocess_source_dataset(dataset: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Preprocess source variables into value columns plus value-shaped masks."""
    return preprocess_dataset(dataset, metadata, "source")


def _write_preprocess_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )


def save_preprocessed_source(processed_source: pd.DataFrame, output_dir: str | Path) -> None:
    """Save preprocessed source value/mask arrays and memo files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_mask = processed_source.attrs.get(
        "mask",
        pd.DataFrame(index=processed_source.index),
    )

    np.save(output_dir / "source_array.npy", processed_source.to_numpy(dtype=np.float32))
    np.save(output_dir / "source_mask.npy", source_mask.to_numpy(dtype=np.float32))
    _write_preprocess_csv(
        processed_source.attrs.get("column_info", pd.DataFrame()),
        output_dir / "source_columns.csv",
    )
    _write_preprocess_csv(
        processed_source.attrs.get("mask_info", pd.DataFrame()),
        output_dir / "source_mask_columns.csv",
    )
    for stale_file in [
        output_dir / "source_quantile_lookup.csv",
        output_dir / "source_raw_value_summary.csv",
    ]:
        if stale_file.exists():
            stale_file.unlink()


def preprocess_target_dataset(
    dataset: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Preprocess target variables into value columns plus value-shaped masks."""
    return preprocess_dataset(dataset, metadata, "target")


def save_preprocessed_target(
    processed_target: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save preprocessed target value/mask arrays and memo files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_mask = processed_target.attrs.get(
        "mask",
        pd.DataFrame(index=processed_target.index),
    )

    np.save(output_dir / "target_array.npy", processed_target.to_numpy(dtype=np.float32))
    np.save(output_dir / "target_mask.npy", target_mask.to_numpy(dtype=np.float32))
    _write_preprocess_csv(
        processed_target.attrs.get("column_info", pd.DataFrame()),
        output_dir / "target_columns.csv",
    )
    _write_preprocess_csv(
        processed_target.attrs.get("mask_info", pd.DataFrame()),
        output_dir / "target_mask_columns.csv",
    )
    for stale_file in [
        output_dir / "target_quantile_lookup.csv",
        output_dir / "target_raw_value_summary.csv",
    ]:
        if stale_file.exists():
            stale_file.unlink()


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
        processed_target = preprocess_target_dataset(dataset, target_metadata)
        save_preprocessed_target(processed_target, "preprocessed")
        print(f"Preprocessed target shape: {processed_target.shape}")
        print("Saved preprocessed target files to: preprocessed")


if __name__ == "__main__":
    main()
