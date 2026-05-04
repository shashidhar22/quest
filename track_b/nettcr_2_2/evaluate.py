"""Evaluate a trained NetTCR-2.2 model on a TCRBench test CSV.

Loads the .h5 (Keras) or .tflite checkpoint, runs predictions, computes AUC
and AUC0.1 on the binder labels, writes metrics JSON.

Differs from the upstream predict.py in that we directly load the .h5 model
(via keras.models.load_model) and run a single forward pass on the entire
test set — avoids the per-peptide TFLite-interpreter dance which is only
needed for the per-peptide pretrained model.

CLI:
    python evaluate.py --test smoke_test_iid.csv \
        --model-path results/smoke/checkpoint/smoke.h5 \
        --out smoke_test_iid_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path("/home/ubuntu/quest/track_b/nettcr_2_2/repo")
RESULTS_DIR = Path("/home/ubuntu/quest/track_b/nettcr_2_2/results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True, help="Eval CSV (under results/).")
    p.add_argument("--model-path", required=True, help="Path to .h5 model (relative to results/).")
    p.add_argument("--out", required=True, help="Output JSON path (relative to results/).")
    args = p.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO_DIR / "src"))
    import keras_utils  # noqa: E402
    import tensorflow as tf  # noqa: E402
    from sklearn.metrics import roc_auc_score, accuracy_score  # noqa: E402

    test_path = RESULTS_DIR / args.test
    model_path = RESULTS_DIR / args.model_path
    out_path = RESULTS_DIR / args.out

    df = pd.read_csv(test_path).fillna("")

    encoding = keras_utils.blosum50_20aa
    a1_max, a2_max, a3_max = 7, 8, 22
    b1_max, b2_max, b3_max = 6, 7, 23
    pep_max = 12

    def enc(col, n):
        return np.float32(keras_utils.enc_list_bl_max_len(df[col], encoding, n) / 5)

    inputs = {
        "pep": enc("peptide", pep_max),
        "a1": enc("A1", a1_max), "a2": enc("A2", a2_max), "a3": enc("A3", a3_max),
        "b1": enc("B1", b1_max), "b2": enc("B2", b2_max), "b3": enc("B3", b3_max),
    }

    # Reproduce the auc_01 custom metric for load_model.
    def my_numpy_function(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred, max_fpr=0.1)
        except ValueError:
            return np.array([0.0])

    def auc_01(y_true, y_pred):
        return tf.numpy_function(my_numpy_function, [y_true, y_pred], tf.float64)

    from tensorflow import keras  # noqa: E402
    model = keras.models.load_model(str(model_path), custom_objects={"auc_01": auc_01})
    y_pred = model.predict(inputs, verbose=0).flatten()
    y_true = df["binder"].values.astype(int)
    pred = (y_pred > 0.5).astype(int)

    metrics = {
        "test": str(test_path.name),
        "n_rows": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
        "auc": float(roc_auc_score(y_true, y_pred)) if len(set(y_true)) > 1 else None,
        "auc_01": float(roc_auc_score(y_true, y_pred, max_fpr=0.1))
                  if len(set(y_true)) > 1 else None,
        "accuracy": float(accuracy_score(y_true, pred)),
        "mean_score_pos": float(y_pred[y_true == 1].mean()) if (y_true == 1).any() else None,
        "mean_score_neg": float(y_pred[y_true == 0].mean()) if (y_true == 0).any() else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
