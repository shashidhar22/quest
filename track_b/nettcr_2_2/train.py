"""Train NetTCR-2.2 on TCRBench AS data.

We use the upstream `nettcr_archs.CNN_CDR123_global_max` (pan-mode) or
`CNN_CDR123_global_max_two_step_pre_training` (pretrained-mode) architecture
classes directly, but drive training from our own loop. This avoids a known
issue with the upstream `train_nettcr_2_2_pan.py` / `_pretrained.py` scripts:
they call `pd.read_csv(...)` without `keep_default_na=False`, which turns
empty α-chain CDR cells into NaN and crashes the BLOSUM encoder.

CLI:
    python train.py --train smoke_train.csv --val smoke_val.csv \
        --outdir smoke --model-name smoke --epochs 2 --patience 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_DIR = Path("/home/ubuntu/quest/track_b/nettcr_2_2/repo")
RESULTS_DIR = Path("/home/ubuntu/quest/track_b/nettcr_2_2/results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Training CSV (under results/).")
    p.add_argument("--val", required=True, help="Validation CSV (under results/).")
    p.add_argument("--outdir", required=True, help="Output dir name (under results/).")
    p.add_argument("--model-name", default="model")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dropout-rate", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=15)
    args = p.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")

    # Ensure upstream modules importable.
    sys.path.insert(0, str(REPO_DIR / "src"))
    import keras_utils  # noqa: E402
    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402
    import tensorflow as tf  # noqa: E402
    from sklearn.metrics import roc_auc_score  # noqa: E402
    from tensorflow import keras  # noqa: E402

    from nettcr_archs import CNN_CDR123_global_max  # noqa: E402

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # IMPORTANT: keep_default_na=False so empty strings stay empty (not NaN).
    train_df = pd.read_csv(RESULTS_DIR / args.train, keep_default_na=False)
    val_df = pd.read_csv(RESULTS_DIR / args.val, keep_default_na=False)

    # NetTCR-2.2 sample weights (peptide-frequency-balanced).
    pep_counts = train_df["peptide"].value_counts()
    weight_dict = np.log2(train_df.shape[0] / pep_counts) / np.log2(len(pep_counts))
    weight_dict = weight_dict * (train_df.shape[0] / np.sum(weight_dict * pep_counts))
    train_df["sample_weight"] = train_df["peptide"].map(weight_dict).fillna(1.0)
    # val rows for peptides not in train get weight 1.0 (default).
    val_df["sample_weight"] = val_df["peptide"].map(weight_dict).fillna(1.0)

    encoding = keras_utils.blosum50_20aa
    a1_max, a2_max, a3_max = 7, 8, 22
    b1_max, b2_max, b3_max = 6, 7, 23
    pep_max = 12

    def enc(df, col, n):
        return np.float32(keras_utils.enc_list_bl_max_len(df[col], encoding, n) / 5)

    def make_inputs(df):
        return {
            "pep": enc(df, "peptide", pep_max),
            "a1": enc(df, "A1", a1_max), "a2": enc(df, "A2", a2_max), "a3": enc(df, "A3", a3_max),
            "b1": enc(df, "B1", b1_max), "b2": enc(df, "B2", b2_max), "b3": enc(df, "B3", b3_max),
        }

    x_train = make_inputs(train_df)
    y_train = train_df["binder"].values.astype(np.float32)
    w_train = train_df["sample_weight"].values.astype(np.float32)

    x_val = make_inputs(val_df)
    y_val = val_df["binder"].values.astype(np.float32)
    w_val = val_df["sample_weight"].values.astype(np.float32)

    def my_numpy_function(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred, max_fpr=0.1)
        except ValueError:
            return np.array([0.0])

    def auc_01(y_true, y_pred):
        return tf.numpy_function(my_numpy_function, [y_true, y_pred], tf.float64)

    model = CNN_CDR123_global_max(dropout_rate=args.dropout_rate, seed=args.seed)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[auc_01, "AUC"],
        weighted_metrics=[],
    )

    outdir = RESULTS_DIR / args.outdir
    (outdir / "checkpoint").mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / "checkpoint" / f"{args.model_name}.h5"

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc_01", mode="max", patience=args.patience),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path), monitor="val_auc_01", mode="max", save_best_only=True,
        ),
    ]

    model.fit(
        x=x_train, y=y_train, sample_weight=w_train,
        validation_data=(x_val, y_val, w_val),
        batch_size=args.batch_size, epochs=args.epochs,
        callbacks=callbacks, verbose=2, shuffle=True,
    )

    print(f"Trained model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
