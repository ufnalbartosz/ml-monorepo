"""Training entrypoint for the AlexNet flower classifier.

This is the half of the old ``model.py`` that had side effects.  Keeping it in
its own module means importing the architecture no longer downloads 60 MB of
JPEGs and starts a 150-epoch run.  Every function here takes what it needs as
an argument, so a test can drive the whole pipeline with four random images.

Run it with::

    uv run python -m pure_alexnet.train --epochs 150
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import keras
import numpy as np

from pure_alexnet.dataset import DEFAULT_DATASET_PATH, DataSet, ensure_directories
from pure_alexnet.model import build_and_compile

DEFAULT_MODEL_NAME = "3_3"


def build_callbacks(
    checkpoint_dir: Path | str,
    log_dir: Path | str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[keras.callbacks.Callback]:
    """Checkpointing + TensorBoard, the tflearn ``DNN`` defaults in Keras form."""
    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    ensure_directories(checkpoint_dir, log_dir)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / f"{model_name}.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]


def train(
    model: keras.Model,
    data: dict[str, np.ndarray],
    epochs: int = 150,
    batch_size: int = 64,
    callbacks: Sequence[keras.callbacks.Callback] | None = None,
    verbose: str | int = "auto",
) -> keras.callbacks.History:
    """Fit ``model`` on the training split, monitoring the validation split."""
    return model.fit(
        data["train_images"],
        data["train_labels"],
        validation_data=(data["valid_images"], data["valid_labels"]),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=list(callbacks) if callbacks is not None else None,
        verbose=verbose,
    )


def accuracy(predictions: np.ndarray, one_hot_labels: np.ndarray) -> float:
    """Fraction of rows whose argmax agrees, on probabilities vs one-hot labels."""
    if len(predictions) == 0:
        raise ValueError("cannot compute accuracy over zero predictions")

    predicted = np.argmax(predictions, axis=1)
    expected = np.argmax(one_hot_labels, axis=1)

    return float(np.mean(predicted == expected))


def evaluate(model: keras.Model, data: dict[str, np.ndarray], batch_size: int = 64) -> float:
    """Test-set accuracy, computed the same way the tflearn script did."""
    predictions = model.predict(data["test_images"], batch_size=batch_size, verbose=0)
    return accuracy(predictions, data["test_labels"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dense-units", type=int, default=2048)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--save-dir", type=Path, default=Path("saves"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    data = DataSet(dataset_path=args.dataset_path).load()

    num_classes = data["train_labels"].shape[1]
    input_shape = data["train_images"].shape[1:]

    model = build_and_compile(
        input_shape=input_shape,
        num_classes=num_classes,
        dense_units=args.dense_units,
        learning_rate=args.learning_rate,
    )
    model.summary()

    train(
        model,
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=build_callbacks(args.checkpoint_dir, args.log_dir, args.model_name),
    )

    print(f"Test accuracy: {evaluate(model, data, batch_size=args.batch_size):.4f}")

    ensure_directories(args.save_dir)
    save_path = args.save_dir / f"{args.model_name}.keras"
    print(f"Saving model to {save_path} ...")
    model.save(save_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
