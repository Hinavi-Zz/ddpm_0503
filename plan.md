# Quantile Transform Preprocessing Plan

## Goal

Replace the current mean/std z-score normalization for continuous numerical features with a quantile-based transform whose center is 0.

This applies to:

- input numerical continuous features
- output numerical target features

Categorical one-hot features, flag indicators, bottom/top indicators, missing indicators, and output masks should keep their current behavior.

## Current Behavior

`preprocess.py` currently normalizes continuous numerical values as:

```text
(value - mean) / std
```

The constants are saved to:

- `preprocessed/input_normalization_constants.csv`
- `preprocessed/output_normalization_constants.csv`

For input numerical features:

- flags and missing values are imputed with the mean
- bottom/top values are not masked and remain part of the continuous distribution
- extra indicator columns are generated for flags, bottom/top, and missing

For output numerical features:

- flags and missing values are masked out with `output_mask=0`
- masked values are filled with the mean before normalization
- bottom/top are not masked

## Target Behavior

Use an empirical quantile transform for each continuous numerical variable.

Proposed default transform:

```text
quantile = empirical_cdf(value)
normalized = quantile - 0.5
```

So the output range is approximately:

```text
[-0.5, 0.5]
```

and the median is centered near:

```text
0
```

The transform should be fitted separately for each variable using only trainable/valid continuous values:

- input numerical: values excluding flags and missing
- output numerical: values excluding flags and missing

Bottom/top values should remain included unless we decide otherwise, because the current output rule says top/bottom should not block gradient.

## Design Decisions To Confirm

1. Quantile output range

   Confirmed: `quantile - 0.5`, range `[0, 1] -> [-0.5, 0.5]`.

   Alternative: map to a normal distribution using inverse normal CDF. This gives unbounded values and may be harder to invert safely.

2. Tie handling

   Confirmed: identical source values map to the same quantile, using the average rank for that value.

3. Unknown or imputed values

   Confirmed:

   - input flags/missing are filled with the fitted median value before transform, so the transformed continuous value becomes `0`
   - output flags/missing are also filled to transformed `0`, while `output_mask=0` blocks their loss

4. Saved transform metadata

   Quantile transform needs more than mean/std. We need enough information to reproduce or invert the transform.

   Confirmed: save a per-variable quantile lookup table with source values and transformed values.

## Implementation Steps

1. Add quantile helper functions

   Add helpers near the existing numerical preprocessing code:

   - fit empirical quantile lookup from a numeric `Series`
   - transform numeric values using interpolation
   - return metadata needed for saving

2. Replace input numerical normalization

   Update `_preprocess_input_numerical`:

   - keep categorical indicator logic unchanged
   - fit quantile transform on non-flag, non-missing values
   - impute flag/missing continuous values to transformed `0`
   - return transformed continuous feature and quantile metadata

3. Replace output numerical normalization

   Update `_preprocess_output_numerical`:

   - keep mask logic unchanged
   - fit quantile transform on rows where `train_mask=True`
   - fill masked target values with transformed `0`
   - return transformed target and quantile metadata

4. Rename metadata meaning strings

   Update column metadata from:

   ```text
   z_score_normalized_value
   ```

   to something like:

   ```text
   quantile_centered_value
   ```

5. Update saved normalization constants

   The existing files can keep their names for compatibility:

   - `input_normalization_constants.csv`
   - `output_normalization_constants.csv`

   But their schema should change from mean/std to quantile metadata.

   Proposed columns:

   ```text
   variable,role,method,n_values,min_value,max_value,median_value
   ```

   Add separate lookup-table files:

   - `preprocessed/input_quantile_lookup.csv`
   - `preprocessed/output_quantile_lookup.csv`

   Proposed lookup columns:

   ```text
   variable,source_value,quantile,transformed_value
   ```

6. Update save functions

   Update:

   - `save_preprocessed_input`
   - `save_preprocessed_output`

   so they save both the summary constants and the lookup tables.

7. Re-run preprocessing

   Run:

   ```text
   python preprocess.py
   ```

   Confirm:

   - `input_array.npy` remains `float32`
   - `output_array.npy` remains `float32`
   - output mask shape is unchanged
   - new quantile lookup CSV files are created

8. Validate transformed values

   Check representative numerical columns:

   - transformed median is near `0`
   - transformed values are within roughly `[-0.5, 0.5]`
   - input missing/flag continuous values become `0`
   - output missing/flag continuous values become `0` and have mask `0`

9. Commit changes

   Commit code and metadata schema changes after verification.

## Files Expected To Change

- `preprocess.py`
- `plan.md`
- generated files under `preprocessed/`

Potential generated additions:

- `preprocessed/input_quantile_lookup.csv`
- `preprocessed/output_quantile_lookup.csv`

Note: lookup CSV files are generated locally and ignored by git because the input lookup can exceed GitHub's normal file-size limit.
