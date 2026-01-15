# src/drift_report.py
import os
from pathlib import Path
import numpy as np
import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

TRAIN_PATH = os.environ.get("TRAIN_PATH", "data/Projet+Mise+en+prod+-+home-credit-default-risk/application_train.csv")
TEST_PATH  = os.environ.get("TEST_PATH",  "data/Projet+Mise+en+prod+-+home-credit-default-risk/application_test.csv")
OUT_DIR    = Path(os.environ.get("DRIFT_OUT_DIR", "reports"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = os.environ.get("TARGET_COL", "TARGET")
IGNORE_COLS = set(os.environ.get("IGNORE_COLS", "SK_ID_CURR").split(","))

N_REF = int(os.environ.get("N_REF", "50000"))
N_CUR = int(os.environ.get("N_CUR", "50000"))

def load_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_csv(path, nrows=nrows)

def prep(reference: pd.DataFrame, current: pd.DataFrame):
    reference = reference.drop(columns=[TARGET_COL], errors="ignore")
    current   = current.drop(columns=[TARGET_COL], errors="ignore")

    reference = reference.drop(columns=[c for c in reference.columns if c in IGNORE_COLS], errors="ignore")
    current   = current.drop(columns=[c for c in current.columns if c in IGNORE_COLS], errors="ignore")

    common_cols = sorted(list(set(reference.columns) & set(current.columns)))
    reference = reference[common_cols].copy()
    current   = current[common_cols].copy()

    reference = reference.replace({pd.NA: np.nan})
    current   = current.replace({pd.NA: np.nan})

    num_cols = reference.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in common_cols if c not in num_cols]

    definition = DataDefinition(
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
    )

    return reference, current, definition

def main():
    ref = load_csv(TRAIN_PATH, nrows=N_REF if N_REF > 0 else None)
    cur = load_csv(TEST_PATH,  nrows=N_CUR if N_CUR > 0 else None)

    ref, cur, definition = prep(ref, cur)

    ref_ds = Dataset.from_pandas(ref, data_definition=definition)
    cur_ds = Dataset.from_pandas(cur, data_definition=definition)

    report = Report(metrics=[DataDriftPreset()])

    # IMPORTANT: run() retourne un "snapshot"/evaluation result
    my_eval = report.run(current_data=cur_ds, reference_data=ref_ds)

    out_html = OUT_DIR / "data_drift.html"
    my_eval.save_html(str(out_html))
    print(f"[OK] Drift report sauvegardé: {out_html}")

if __name__ == "__main__":
    main()
