# KNHANES Preprocess

This project converts KNHANES raw SAS7BDAT files into value arrays and mask arrays that are easier to use for model training.

## 1. Environment Setup

Conda or Miniconda is recommended.

Create the environment from the project root:

```powershell
conda env create -f environment.yaml
conda activate knhanes-preprocess
```

If the environment already exists and `environment.yaml` has changed, update it with:

```powershell
conda env update -f environment.yaml --prune
conda activate knhanes-preprocess
```

`environment.yaml` includes Python 3.10, `numpy`, `pandas`, `pyyaml`, and `pyreadstat`. `pyreadstat` is important because reading SAS files through pandas alone can fail on encoding issues.

## 2. Download Raw Data

KNHANES raw data can be downloaded from the Korean public data portal:

https://www.data.go.kr/data/15076556/fileData.do#

Place the downloaded SAS7BDAT files in the `datasets/` directory at the project root.

The default config expects these files:

```text
datasets/hn13_all.sas7bdat
datasets/hn14_all.sas7bdat
datasets/hn15_all.sas7bdat
datasets/hn16_all.sas7bdat
datasets/hn17_all.sas7bdat
datasets/hn18_all.sas7bdat
datasets/hn19_all.sas7bdat
datasets/hn20_all.sas7bdat
datasets/hn21_all.sas7bdat
datasets/hn22_all.sas7bdat
datasets/hn23_all.sas7bdat
datasets/hn24_all.sas7bdat
```

## 3. Input Files

Preprocessing uses `preprocess_config.template.yaml` by default.

The main input files are:

- `metadata.csv`: variable type, categorical classes, and missing/flag code metadata.
- `source_variables.txt`: variables to include in the source array.
- `target_variables.txt`: variables to include in the target array.
- `skip_years.csv`: variable-year combinations that should be treated as missing.

## 4. Run Preprocessing

Run this command from the project root:

```powershell
python preprocess.py
```

To also create validation reports, run:

```powershell
python preprocess.py --validate
```

## 5. Output Files

Preprocessing writes outputs to the `preprocessed/` directory.

```text
preprocessed/source_array.npy
preprocessed/source_mask.npy
preprocessed/source_columns.csv
preprocessed/source_mask_columns.csv
preprocessed/target_array.npy
preprocessed/target_mask.npy
preprocessed/target_columns.csv
preprocessed/target_mask_columns.csv
```

`*_array.npy` files contain preprocessed values.

`*_mask.npy` files contain masks with the same shape as the corresponding value arrays.

`*_columns.csv` files describe value array columns.

`*_mask_columns.csv` files describe mask array columns.

## 6. Mask Rule

Mask values are `1` for valid values and `0` for missing values or metadata flag values.

Categorical variables are one-hot encoded. If the raw categorical value is valid, every one-hot column for that raw variable has mask `1`. If the raw value is missing or a metadata flag, every one-hot column for that raw variable has mask `0`.

Numerical variables are stored as one continuous value column. If the raw value is missing or a metadata flag, the value is filled with the default `0.0` and the mask is `0`.

## 7. Preprocessing Rules

Processing depends on the `type` column in `metadata.csv`.

- `categorical`: one-hot encoded using values listed in `classes`.
- `numerical`: stored as the raw numeric value in one continuous column.
- `pass`: excluded from the value and mask arrays.

Variable-year combinations listed in `skip_years.csv` are treated as missing, even if raw data values exist.

## 8. Check Results

After preprocessing, check output shapes with:

```powershell
@'
from pathlib import Path
import numpy as np

root = Path("preprocessed")
for name in ["source_array", "source_mask", "target_array", "target_mask"]:
    array = np.load(root / f"{name}.npy", mmap_mode="r")
    print(name, array.shape, array.dtype)
'@ | python -
```

The preprocessing is consistent when `source_array` and `source_mask` have the same shape, and `target_array` and `target_mask` have the same shape.
