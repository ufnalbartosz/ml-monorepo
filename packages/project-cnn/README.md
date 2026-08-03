# project-cnn

Three convolutional networks trained on a 20-class subset of CIFAR-100
(food containers, fruit and vegetables, household electrical devices,
household furniture), on Keras 3 / TensorFlow 2.

## Layout

| module | responsibility |
|---|---|
| `dual_path.py` | two-branch inception-ish net, formerly the Pretty Tensor graph in `main.py` |
| `inception.py` | two inception blocks with 1x1 / 3x3 / 5x5 / pool branches |
| `alexnet_model.py` | AlexNet at 32x32 |
| `layers.py` | `LocalResponseNormalization`, which Keras 3 does not ship |
| `tools.py` | image pre-processing (`PreProcessing` layer) and model introspection |
| `loader.py` | CIFAR-100 download, decode and class subsetting |
| `prepare_dataset.py` | train/validation/test splits and the on-disk cache |
| `evaluate.py` | batched prediction and accuracy reporting |
| `plot.py` | figures, written to PNG through the Agg backend |
| `main.py` | CLI |

## Running

```bash
uv run python -m project_cnn.main --model dual-path --epochs 20
uv run python -m project_cnn.main --model inception --epochs 20
uv run python -m project_cnn.main --model alexnet --epochs 20
uv run python -m project_cnn.main --help
```

The first run downloads CIFAR-100 into `data/CIFAR-100/` and caches the
prepared splits as `data/dataset.pickle`.

## Tests

```bash
uv run pytest packages/project-cnn
```

Nothing in the suite downloads anything: the CIFAR fixtures are built in
`tmp_path` and pickled the way Python 2 wrote the originals, which is what pins
down the byte-key handling in `loader.unpickle`.

## Notes on the port

`main.py` was a TF1 script: importing it created placeholders, built the graph
twice under `tf.variable_scope(reuse=...)` so the training and inference copies
would share weights, opened a `tf.Session`, restored a checkpoint through
`tf.train.Saver`, trained and printed a confusion matrix. `inception.py` and
`alexnet_model.py` did the equivalent through tflearn. Neither library runs on
TF2 - Pretty Tensor was archived in 2018 and tflearn stopped at TF1.

In the port, the "build the graph twice" trick is gone: `PreProcessing` and
`Dropout` read the `training` flag Keras threads through `__call__`, so one
model serves both paths. Checkpoint handling is `ModelCheckpoint`. The
introspection helpers take a `keras.Model` instead of reaching into
`tf.get_default_graph()` by tensor name - which, incidentally, silently assumed
every layer used ReLU.

Two Python 3 bugs were fixed on the way: `urllib.urlretrieve` (Python 2 only)
and the CIFAR-100 pickles, which were written by Python 2 and therefore hand
back `bytes` keys, so `data['data']` raised `KeyError`.
