"""Training entrypoint for the CIFAR-100 subset models.

The TF1 version of this file was a script: importing it built placeholders,
built the graph twice under a reusing variable scope, opened a
``tf.Session``, restored a checkpoint, trained for 50 iterations and printed a
confusion matrix - all at module level, with the graph handles kept in globals
that the other functions closed over.

It is now a normal CLI over three interchangeable model builders::

    uv run python -m project_cnn.main --model dual-path --epochs 20
    uv run python -m project_cnn.main --model inception --epochs 20
    uv run python -m project_cnn.main --model alexnet --epochs 20
    uv run python -m project_cnn.main --help

Every step is a function that takes its inputs as arguments, so the pipeline
can be exercised on synthetic data in a test.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import keras
import numpy as np

from project_cnn.alexnet_model import build_alexnet
from project_cnn.dual_path import build_dual_path
from project_cnn.evaluate import print_test_accuracy, print_valid_accuracy
from project_cnn.inception import build_inception
from project_cnn.prepare_dataset import DEFAULT_DATASET_PATH, load_dataset

#: Model name -> builder. Each builder takes (input_shape, num_classes).
MODEL_BUILDERS: dict[str, Callable[..., keras.Model]] = {
    "dual-path": build_dual_path,
    "inception": build_inception,
    "alexnet": build_alexnet,
}

DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 64


def build_model(
    model_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    **kwargs,
) -> keras.Model:
    """Build one of the three architectures by name."""
    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError:
        known = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"Unknown model {model_name!r}; expected one of: {known}") from None

    return builder(input_shape=input_shape, num_classes=num_classes, **kwargs)


def compile_model(
    model: keras.Model,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> keras.Model:
    """Plain SGD, matching the TF1 ``GradientDescentOptimizer(1e-4)``."""
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_callbacks(save_dir: Path | str) -> list[keras.callbacks.Callback]:
    """Checkpointing and TensorBoard logging.

    Replaces the hand-rolled ``tf.train.Saver`` / ``latest_checkpoint`` dance,
    including the ``try/except tf.errors.OpError`` around the restore.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / "model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        keras.callbacks.TensorBoard(log_dir=str(save_dir)),
    ]


def restore_if_available(model: keras.Model, save_dir: Path | str) -> keras.Model:
    """Load the last checkpoint if there is a usable one, else keep the fresh weights."""
    checkpoint = Path(save_dir) / "model.keras"

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


def train(
    model: keras.Model,
    dataset: dict,
    epochs: int = 20,
    batch_size: int = DEFAULT_BATCH_SIZE,
    callbacks: Sequence[keras.callbacks.Callback] | None = None,
    verbose: str | int = "auto",
) -> keras.callbacks.History:
    """Fit on the training split, monitoring the validation split.

    The test-set stays untouched until the final measurement; picking a
    snapshot based on it would leak it into model selection.
    """
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on the CIFAR-100 subset.")
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), default="dual-path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--save-dir", type=Path, default=Path("logs"))
    parser.add_argument("--resume", action="store_true", help="restore the last checkpoint first")
    parser.add_argument("--show-example-errors", action="store_true")
    parser.add_argument("--show-confusion-matrix", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    np.set_printoptions(precision=3, suppress=True)

    dataset = load_dataset(args.dataset_path)

    input_shape = dataset["train_images"].shape[1:]
    num_classes = dataset["train_labels"].shape[1]

    model = build_model(args.model, input_shape=input_shape, num_classes=num_classes)
    if args.resume:
        model = restore_if_available(model, args.save_dir)
    compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    train(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=build_callbacks(args.save_dir),
    )

    print_valid_accuracy(model, dataset)
    print_test_accuracy(
        model,
        dataset,
        show_example_errors=args.show_example_errors,
        show_confusion_matrix=args.show_confusion_matrix,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
