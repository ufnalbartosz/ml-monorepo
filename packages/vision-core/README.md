# vision-core

The parts `pure-alexnet` and `project-cnn` had in common.

The two packages train different models on different data-sets, so they stay
separate. But both fetched an archive over HTTPS, cached a prepared data-set as
a pickle, built the same two Keras callbacks, ran the same `fit` call and
computed accuracy — twice, in two copies that had already drifted apart (one
still called the Python 2 `urllib.urlretrieve`; the two accuracy functions
disagreed on whether they took probabilities or class numbers).

| module | what it holds | needs TF? |
|---|---|---|
| `cache.py` | `read_pickle`, `write_pickle`, `load_or_build`, `ensure_directories` | no |
| `archives.py` | `download`, `extract`, `download_and_extract` | no |
| `images.py` | `image_paths`, `load_image`, `load_images` | no |
| `labels.py` | `one_hot_encoded` | no |
| `metrics.py` | `predict_classes`, `classification_accuracy`, `accuracy_from_probabilities` | yes |
| `training.py` | `build_callbacks`, `train`, `restore_if_available` | yes |

Four of the six modules are pure Python and numpy, and importing them does not
pull in TensorFlow — which is why the data-side tests in both packages run in
milliseconds.

## Design notes

**Paths are always arguments.** Nothing here calls `os.getcwd()` or hard-codes
a relative name. That is the property that lets every caller's tests run
against `tmp_path`.

**Download is separate from extract.** Tests inject a stand-in for the first
and use the real second one against a fixture tarball, so extraction logic is
covered without a network.

**`train` refuses to touch the test split.** It reads `train_*` and `valid_*`
only, and raises a named `KeyError` if either is missing. Selecting a snapshot
on the test-set is what makes a final number meaningless, so the shared loop
cannot be asked to do it.

## Tests

```bash
uv run pytest packages/vision-core
```
