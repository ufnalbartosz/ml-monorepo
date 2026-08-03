# pure-alexnet

AlexNet-style classifier for the Oxford 17-category flower data-set, on
Keras 3 / TensorFlow 2.

## Layout

| module | responsibility |
|---|---|
| `model.py` | the architecture - `build_alexnet()` returns a `keras.Model` |
| `dataset.py` | train/test/validation splits and the on-disk pickle cache |
| `oxflower17.py` | download, extract and decode the VGG archive |
| `train.py` | CLI: fit, evaluate, checkpoint, save |

## Running

```bash
uv run python -m pure_alexnet.train --epochs 150
uv run python -m pure_alexnet.train --help
```

The first run downloads ~60 MB from VGG into `17flowers/`, caches the prepared
splits as `17flowers/dataset.pickle`, and deletes the raw JPEGs afterwards.

## Tests

```bash
uv run pytest packages/pure-alexnet
```

The tests never touch the network and never build the full-size model: the
loader is injected, and `build_alexnet` takes `input_shape`/`dense_units` so a
structurally identical model can be built at 32x32 with 8-unit dense layers.

## Notes on the port

The tflearn version built the graph, downloaded the data, trained for 150
epochs and saved the result *at import time*, so importing the module was the
same as running the experiment - nothing could be tested in isolation. The port
separates architecture from data from training, and drops tflearn (unmaintained,
TF1-era) in favour of Keras 3. Layer names are unchanged.
