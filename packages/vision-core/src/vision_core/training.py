"""The Keras training loop both packages run.

Each package had its own near-identical ``build_callbacks`` and ``train``.  The
differences that mattered - which model, which data, which hyper-parameters -
are arguments; the parts that were the same are here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import keras

from vision_core.cache import ensure_directories


def build_callbacks(
    checkpoint_dir: Path | str,
    log_dir: Path | str | None = None,
    model_name: str = "model",
    monitor: str = "val_accuracy",
    mode: str = "max",
) -> list[keras.callbacks.Callback]:
    """Checkpoint-the-best plus TensorBoard logging.

    Replaces both the tflearn ``DNN(checkpoint_path=..., tensorboard_dir=...)``
    arguments and the hand-rolled ``tf.train.Saver`` / ``latest_checkpoint``
    dance the TF1 script used, including its ``try/except tf.errors.OpError``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir) if log_dir is not None else checkpoint_dir
    ensure_directories(checkpoint_dir, log_dir)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / f"{model_name}.keras"),
            save_best_only=True,
            monitor=monitor,
            mode=mode,
        ),
        keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]


def train(
    model: keras.Model,
    dataset: dict,
    epochs: int,
    batch_size: int,
    callbacks: Sequence[keras.callbacks.Callback] | None = None,
    verbose: str | int = "auto",
) -> keras.callbacks.History:
    """Fit on the training split, monitoring the validation split.

    ``dataset`` is the dict both packages cache, keyed ``train_images`` /
    ``train_labels`` / ``valid_images`` / ``valid_labels``.

    The test split is deliberately not touched: choosing a snapshot on the
    test-set leaks it into model selection, which is what makes the final
    number meaningless.
    """
    for key in ("train_images", "train_labels", "valid_images", "valid_labels"):
        if key not in dataset:
            raise KeyError(f"dataset is missing {key!r}")

    return model.fit(
        dataset["train_images"],
        dataset["train_labels"],
        validation_data=(dataset["valid_images"], dataset["valid_labels"]),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=list(callbacks) if callbacks is not None else None,
        verbose=verbose,
    )


def restore_if_available(
    model: keras.Model,
    checkpoint_dir: Path | str,
    model_name: str = "model",
) -> keras.Model:
    """Load the saved model if there is a usable one, else keep the fresh weights."""
    checkpoint = Path(checkpoint_dir) / f"{model_name}.keras"

    if not checkpoint.exists():
        print("No checkpoint found. Using freshly initialized weights.")
        return model

    try:
        print("Trying to restore last checkpoint ...")
        restored = keras.models.load_model(checkpoint)
        print("Restored checkpoint from:", checkpoint)
        return restored
    except (OSError, ValueError) as error:
        # Only catch what means "this checkpoint is unusable", so that a real
        # bug is not silently swallowed here.
        print("Failed to restore checkpoint:", error)
        print("Using freshly initialized weights.")
        return model
