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
- `output_dir`: output directory in the config.
- `split_seed`: random seed for reproducible shuffling before train/val/test split.

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
preprocessed/source_num_train.npy
preprocessed/source_num_val.npy
preprocessed/source_num_test.npy
preprocessed/source_num_mask_train.npy
preprocessed/source_num_mask_val.npy
preprocessed/source_num_mask_test.npy
preprocessed/source_cat_train.npy
preprocessed/source_cat_val.npy
preprocessed/source_cat_test.npy
preprocessed/source_cat_mask_train.npy
preprocessed/source_cat_mask_val.npy
preprocessed/source_cat_mask_test.npy
preprocessed/target_num_train.npy
preprocessed/target_num_val.npy
preprocessed/target_num_test.npy
preprocessed/target_num_mask_train.npy
preprocessed/target_num_mask_val.npy
preprocessed/target_num_mask_test.npy
preprocessed/target_cat_train.npy
preprocessed/target_cat_val.npy
preprocessed/target_cat_test.npy
preprocessed/target_cat_mask_train.npy
preprocessed/target_cat_mask_val.npy
preprocessed/target_cat_mask_test.npy
preprocessed/info.json
```

Rows are shuffled with `split_seed` from the config, then split into train/val/test with ratios `0.9`, `0.09`, and `0.01`.

Numerical arrays are z-score normalized and stored as `float32`.

Categorical arrays are stored as `int64`.

Mask arrays are stored as `int64` with values `1` for valid and `0` for masked.

## 6. Mask Rule

Mask values are `1` for valid values and `0` for missing values or metadata flag values.

Categorical variables are integer encoded using the order of `classes` in `metadata.csv`. For example, `classes` of `1,2,3` maps raw values `1`, `2`, and `3` to `0`, `1`, and `2`.

If a categorical value is missing, a metadata flag, or outside the configured classes, the encoded value is `-1` and the mask is `0`.

Numerical variables are stored as one continuous value column. If the raw value is missing or a metadata flag, the value is filled with the default `0.0` and the mask is `0`.

Numerical z-score constants are fit from the train split only, using rows where the corresponding mask is `1`. The same constants are applied to train, val, and test. Masked numerical values remain `0.0` after normalization.

## 7. Preprocessing Rules

Processing depends on the `type` column in `metadata.csv`.

- `categorical`: integer encoded using values listed in `classes`.
- `numerical`: stored as the raw numeric value in one continuous column.
- `pass`: excluded from the value and mask arrays.

Variable-year combinations listed in `skip_years.csv` are treated as missing, even if raw data values exist.

`info.json` contains dataset metadata, split sizes, feature names, categorical cardinalities, and numerical normalization constants in `source_num_means`, `source_num_stds`, `target_num_means`, and `target_num_stds`.

## 8. Check Results

After preprocessing, check output shapes with:

```powershell
@'
from pathlib import Path
import numpy as np

root = Path("preprocessed")
names = [
    "source_num",
    "source_num_mask",
    "source_cat",
    "source_cat_mask",
    "target_num",
    "target_num_mask",
    "target_cat",
    "target_cat_mask",
]
for split in ["train", "val", "test"]:
    for name in names:
        array = np.load(root / f"{name}_{split}.npy", mmap_mode="r")
        print(f"{name}_{split}", array.shape, array.dtype)
'@ | python -
```

The preprocessing is consistent when every value array and its matching mask array have the same shape for each split.
