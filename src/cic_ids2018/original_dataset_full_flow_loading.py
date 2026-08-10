"""CIC-IDS2018 original dataset full-flow loading and splitting utilities.

Implements:
  - load_processed_files: reads the 7 cleaned Parquet files from full_flow_processed/
  - within_day_stratified_split: 70/15/15 stratified split per day, then concatenated
  - undersample_benign: caps Benign in training set at BENIGN_RATIO × total attack flows
  - get_class_weights: computes sklearn balanced class weights for model training

Splitting rationale (within-day rather than pooled random):
  Flows from the same capture day share network context (IP ranges, attacker machines,
  victim configuration). A pooled random split leaks session-level artifacts across
  train/test. Splitting within each day ensures no flow from the same session appears
  in both train and test, which is the closest approximation to real IDS deployment
  evaluation available in this dataset.

  Reference: Engelen et al. (2022). "Troubleshooting an Intrusion Detection Dataset:
  The CICIDS2017 Case Study." IEEE Security & Privacy.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


# Default paths (relative to repo root)
_REPO_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "data" / "CIC-IDS2018" / "parquet" / "full_flow_processed"
LABEL_MAPPING_PATH = PROCESSED_DIR / "label_mapping.json"

# Column names
LABEL_COL = "Label"
LABEL_ENCODED_COL = "label_encoded"


def load_label_mapping(path=None):
    """Load label_mapping.json. Returns the parsed dict."""
    p = Path(path) if path else LABEL_MAPPING_PATH
    with open(p) as f:
        return json.load(f)


def load_processed_files(data_dir=None, verbose=True):
    """
    Load all 7 processed Parquet files from full_flow_processed/.

    Parameters
    ----------
    data_dir : str or Path, optional
        Override the default directory (PROCESSED_DIR).
    verbose : bool
        Print a summary line per file.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are day names (e.g. "Friday-02-03-2018").
        Values are DataFrames with 42 feature columns + Label + label_encoded.
    """
    d = Path(data_dir) if data_dir else PROCESSED_DIR
    parquet_files = sorted(d.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {d}")

    day_dfs = {}
    for p in parquet_files:
        # Strip the CICFlowMeter export suffix to get a clean day name
        stem = p.stem
        day_name = stem.replace("_TrafficForML_CICFlowMeter", "")
        df = pd.read_parquet(p, engine="pyarrow")
        day_dfs[day_name] = df
        if verbose:
            n_attack = (df[LABEL_ENCODED_COL] != 0).sum()
            n_benign = (df[LABEL_ENCODED_COL] == 0).sum()
            print(f"  {day_name}: {len(df):>9,} rows  "
                  f"(Benign: {n_benign:>8,}, Attack: {n_attack:>8,})")

    if verbose:
        print(f"\nLoaded {len(day_dfs)} day files | "
              f"Total: {sum(len(v) for v in day_dfs.values()):,} rows")
    return day_dfs


def within_day_stratified_split(
    day_dfs,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    verbose=True,
):
    """
    Split each day's DataFrame 70/15/15 stratified by label_encoded, then
    concatenate the per-day splits into global train / val / test sets.

    Stratification is done within each day so that:
      1. No flow from the same capture session crosses split boundaries.
      2. All 11 label classes are present in all three splits (guaranteed by
         stratification — each day contributes proportionally to every class
         that exists in that day).

    Two-step split procedure per day:
      Step 1: StratifiedShuffleSplit → train_idx  vs  val_test_idx
      Step 2: StratifiedShuffleSplit → val_idx    vs  test_idx

    Parameters
    ----------
    day_dfs : dict[str, pd.DataFrame]
        Output of load_processed_files().
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0 (within floating-point tolerance).
    random_seed : int
    verbose : bool

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Each is a concatenation of per-day splits, reset index.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    train_frames, val_frames, test_frames = [], [], []

    for day_name, df in day_dfs.items():
        y = df[LABEL_ENCODED_COL].values

        # Step 1: carve out the val+test portion
        val_test_ratio = val_ratio + test_ratio
        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_test_ratio, random_state=random_seed
        )
        train_idx, val_test_idx = next(sss1.split(np.zeros(len(df)), y))

        df_train = df.iloc[train_idx].copy()
        df_val_test = df.iloc[val_test_idx].copy()

        # Step 2: split val_test evenly into val and test
        y_vt = df_val_test[LABEL_ENCODED_COL].values
        # test_size within val_test = test_ratio / (val_ratio + test_ratio)
        test_frac = test_ratio / val_test_ratio
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_frac, random_state=random_seed
        )
        val_idx, test_idx = next(sss2.split(np.zeros(len(df_val_test)), y_vt))

        df_val = df_val_test.iloc[val_idx].copy()
        df_test = df_val_test.iloc[test_idx].copy()

        train_frames.append(df_train)
        val_frames.append(df_val)
        test_frames.append(df_test)

        if verbose:
            print(f"  {day_name}: train={len(df_train):>8,}  "
                  f"val={len(df_val):>7,}  test={len(df_test):>7,}")

    train_df = pd.concat(train_frames, ignore_index=True)
    val_df = pd.concat(val_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    if verbose:
        total = len(train_df) + len(val_df) + len(test_df)
        print(f"\nGlobal totals — train: {len(train_df):,}  "
              f"val: {len(val_df):,}  test: {len(test_df):,}  "
              f"total: {total:,}")

    return train_df, val_df, test_df


def undersample_benign(df_train, benign_label=0, benign_ratio=3, random_seed=42):
    """
    Cap Benign rows in the training set at benign_ratio × total attack rows.

    Applied ONLY to the training split. Val and test retain the natural class
    distribution so that evaluation metrics reflect realistic deployment conditions.

    Rationale for undersampling rather than oversampling (SMOTE):
      Synthetic network flow interpolation can produce flows that do not correspond
      to any real traffic pattern. Attack classes like Infilteration and Bot have
      highly specific behavioral signatures; interpolating between them risks creating
      misleading synthetic samples. Undersampling Benign is lossless with respect to
      attack information and keeps training tractable.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training split (output of within_day_stratified_split).
    benign_label : int
        Integer encoding of the Benign class (default 0 per label_mapping.json).
    benign_ratio : float
        cap = min(n_attack × benign_ratio, n_benign_available).
        A ratio of 3 keeps Benign dominant but reduces imbalance from ~9:1 to 3:1.
    random_seed : int

    Returns
    -------
    pd.DataFrame  (shuffled, reset index)
    """
    benign_mask = df_train[LABEL_ENCODED_COL] == benign_label
    attack_mask = ~benign_mask

    n_attack = int(attack_mask.sum())
    n_benign_available = int(benign_mask.sum())
    cap = int(min(n_attack * benign_ratio, n_benign_available))

    df_benign_sampled = df_train[benign_mask].sample(n=cap, random_state=random_seed)
    df_attack = df_train[attack_mask]

    df_out = pd.concat([df_benign_sampled, df_attack], ignore_index=True)
    # Shuffle so batches during training don't see all Benign at the start
    df_out = df_out.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print(f"  Benign: {n_benign_available:>9,} → {cap:>9,} "
          f"(ratio {benign_ratio}× attack | cap={cap:,})")
    print(f"  Attack: {n_attack:>9,} (unchanged)")
    print(f"  Train size after undersampling: {len(df_out):,}")

    return df_out


def get_class_weights(y_train):
    """
    Compute sklearn balanced class weights for all classes in y_train.

    The 'balanced' mode sets weight[c] = n_samples / (n_classes × n_samples_c),
    which increases the loss contribution of minority classes proportionally.
    This handles the residual imbalance between attack classes that remains after
    Benign undersampling.

    Parameters
    ----------
    y_train : array-like of int
        Integer-encoded labels from the (undersampled) training split.

    Returns
    -------
    dict[int, float]
        Mapping from class integer → weight (e.g., {0: 0.42, 1: 3.17, ...}).
    """
    y = np.asarray(y_train)
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def get_feature_columns(label_mapping=None):
    """Return the 42 feature column names from label_mapping.json."""
    if label_mapping is None:
        label_mapping = load_label_mapping()
    return label_mapping["feature_columns"]
