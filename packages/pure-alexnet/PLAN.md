# pure-alexnet: PyTorch port and the 2018 vs 2026 benchmark

This is the implementation plan for porting this package to PyTorch (it was written in
tflearn in 2017 and ported to Keras 3 in [PR #1](https://github.com/ufnalbartosz/ml-monorepo/pull/1))
and re-running the 2018 master's thesis experiment it comes from ("Klasyfikacja produktów
na podstawie obrazów z wykorzystaniem metod sztucznej inteligencji", Politechnika
Wrocławska, 2018) next to modern transfer learning and foundation models, with honest
statistics.

The plan is written so that each future session can pick up the next milestone and
implement it without re-deriving anything. Section 3 holds the facts established
while reviewing the thesis and the code on 2026-10-29. The thesis PDF is not in the
repository; everything needed from it is quoted here.

## Status

| Milestone | Title | Status |
|---|---|---|
| M1 | Skeleton and tooling | todo |
| M2 | Data (Oxford 17) | todo |
| M3 | AlexNet-2018 port and thesis reproduction | todo |
| M4 | Transfer-learning ladder on the official splits | todo |
| M5 | Normalization study | todo |
| M6 | Few-shot, calibration and error analysis | todo |
| M7 | Report notebook and results | todo |
| M8 | Oxford 102 | todo |
| M9 | Product angle (add a class, open set, retrieval) | todo |
| M10 | CI (optional) | todo |

## 0. How to work with this plan

- Implement the milestones in order. Do not start a milestone until the "Done when"
  of the previous one holds; some gates need numbers measured on the M4.
- Default: one pull request per milestone, titled `pure-alexnet M<n>: <title>`.
  Several milestones may share a PR if the user asks for it.
- In every PR: tick the finished checkboxes, update the Status table, and append
  measured numbers to the Results log (section 9). Do not change section 2
  (Decisions) without asking the user.
- There are two execution environments:
  - **Cloud sessions (Claude):** Linux, CPU only. The network policy may block the
    data and weight hosts (section 3.5). Write the code and offline tests here, on
    synthetic data. Never present numbers from synthetic data or mocks as results.
  - **The user's Apple M4:** runs the real experiments. Every milestone has a "Run on
    M4" block. Afterwards the user commits the small summaries in `results/` or pastes
    the printed summary into the next session.
- Conventions:
  - Python 3.12 (pinned by the root `.python-version`), uv workspace, ruff with the
    repository config, pytest with `uv run pytest`.
  - No import-time side effects. All randomness is seeded.
  - Every table and figure is generated from saved run outputs, never typed by hand.

## 1. Goal

1. Reproduce the thesis experiment faithfully (same model, data handling, optimizer
   and training length as the code that actually ran) and measure how much its single
   reported number depended on luck of the split and the seed.
2. Answer the thesis question properly: does the normalization layer matter (none /
   LRN / BN / GroupNorm / LayerNorm)?
3. Show what transfer learning (2018) and foundation models (2023–2025) give on the
   same official splits.
4. Cover overfitting, calibration, error analysis, a few-shot curve, a harder dataset
   (Oxford 102) and a product-style angle (adding a class without retraining, open-set
   rejection, retrieval).
5. Deliverables: the package, an executable report notebook (HTML), a generated
   `RESULTS.md`, and a "2018 vs 2026" table in the root README.

## 2. Decisions (agreed with the user on 2026-10-29)

| Topic | Decision |
|---|---|
| Location | Port in place in `packages/pure-alexnet` and keep the package name, replacing the Keras 3 implementation now on `main`. Keep its framework-independent parts: the loader, the split function and the offline test style (section 3.6). The tflearn code stays in git history (and in github.com/ufnalbartosz/pure_alexnet). |
| Framework | PyTorch with torchvision, timm and open_clip. Drop keras, tensorflow and tensorboard from this package once the PyTorch model replaces the Keras one (M3). `project-cnn` still uses Keras/TensorFlow, which is fine. |
| Target hardware | Base Apple M4, assume 16 GB unified memory, MPS backend. No ViT-L fine-tuning; ViT-L only for frozen features or zero-shot. |
| Notebook | Report layer only: a jupytext percent-format `.py`. The logic lives in the package. The notebook runs headless and exports HTML. |
| Scope | Everything: core plus all optional items (BioCLIP 2, Oxford 102, product angle, normalization study, few-shot, error analysis, statistics). CI is optional (M10). |
| Language | English for code, docs and the report. |
| Datasets | Oxford 17 (official splits plus the thesis protocol) and Oxford 102 (official splits). |
| DINOv3 | Not used by default because the weights are gated behind a license on Hugging Face. Optional if the user accepts the license. |
| `docs/modern-vision-guide.md` | This plan is the measured version of rungs 1–3 of the guide's §3 (fine-tuning, frozen features with a linear probe, zero-shot) on the official splits. Rung 4 (asking a VLM) is out of scope unless the user asks for it. |

## 3. Established facts (verified 2026-10-29)

### 3.1 What actually ran in 2017/2018

Sources: the thesis, and the git history of this package (commits `f17d5c1` to
`148f3e7`, which mirror github.com/ufnalbartosz/pure_alexnet, June to December 2017).

**Data.** Oxford 17 was loaded with
`tflearn.datasets.oxflower17.load_data(one_hot=True, resize_pics=(227, 227))`:
- PIL resize to exactly 227×227 with ANTIALIAS, so the aspect ratio is not kept;
- float32 scaled by 1/255, no mean/std normalization;
- shuffled without a seed.

**Split** (`dataset.py`). Per class, in tflearn's shuffled order: the first 16 images
go to test, the next 8 to validation and the remaining 56 to training, so
272 / 136 / 952 images. The thesis text describes 5-fold cross-validation, but the
reported numbers come from this fixed split.

**Model** (commit `148f3e7` and earlier). All padding is `same`; conv activations are
ReLU.

| # | Layer |
|---|---|
| 1 | conv 96 @ 11×11, stride 4 |
| 2 | max-pool 3×3, stride 2 |
| 3 | LRN |
| 4 | conv 256 @ 5×5 |
| 5 | max-pool 3×3, stride 2 |
| 6 | LRN |
| 7 | conv 384 @ 3×3 |
| 8 | conv 384 @ 3×3 |
| 9 | conv 256 @ 3×3 |
| 10 | max-pool 3×3, stride 2 |
| 11 | LRN |
| 12 | FC 4096, tanh |
| 13 | dropout 0.5 |
| 14 | FC 4096, tanh |
| 15 | dropout 0.5 |
| 16 | FC 17, softmax |

The Keras 3 `model.py` now on `main` is a port of the later variant from commit
`9404ea3` (conv1 96 @ 7×7 stride 1, 128-filter 3×3 convs, FC 2048 ReLU, no LRN), not
of this architecture. The faithful port starts from the table above.

**BN variant.** tflearn `batch_normalization` replaced each LRN, i.e. it came after
each max-pool. This code was never committed, so it has to be reconstructed from the
tflearn defaults below.

**tflearn defaults that matter** (tflearn 0.3.2 source; label sorting and LRN defaults also checked in 0.5.0):

| Component | Default |
|---|---|
| conv weights | `uniform_scaling` = U(±√(3/fan_in)), fan_in = k·k·C_in |
| FC weights | truncated normal, σ = 0.02 (truncated at 2σ) |
| biases | zeros |
| weight decay | none (`regularizer=None`; the `weight_decay` argument is then unused) |
| LRN | `tf.nn.lrn(depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)`, a window of 11 channels. The thesis text's k=2, n=5 is not what ran. |
| BN | eps 1e-5, moving-average decay 0.9, γ ~ N(1, 0.002), β = 0 |
| optimizer | `Momentum(lr=0.001, momentum=0.9)`, no LR decay |
| loss and batching | categorical cross-entropy, batch 64, reshuffled every epoch |

**Training length.** The thesis plots run to 4500 iterations, with validation every 250
iterations. The committed code says `n_epoch=150`, which is 2250 iterations at
batch 64. Use 4500 iterations at batch 64, as the thesis states.

**Shapes** (input 227, `same` padding):

| Stage | Shape |
|---|---|
| conv1 | 57×57×96 |
| pool1 | 29×29×96 |
| conv2 | 29×29×256 |
| pool2 | 15×15×256 |
| conv3 / conv4 / conv5 | 15×15×384 / 15×15×384 / 15×15×256 |
| pool3 | 8×8×256 |
| flatten / FC1 / FC2 / FC3 | 16384 / 4096 / 4096 / 17 |

**Parameters.**
- LRN or no normalization: 87,701,536 weights + 9,585 biases = **87,711,121**.
- BN, GN and LN variants: 1,216 more (γ and β), so **87,712,337**.
- The thesis's Table 3.1 says 9,035,104: conv1 is counted without its 3 input
  channels, and fc1 and fc2 come out 16× too small.
- Compute: ~1.41 GMAC (~2.8 GFLOP) forward per image, ~8.5 GFLOP per training image.

### 3.2 PyTorch equivalences for the faithful port

- **Padding.** TF `same` padding is symmetric for every layer here: conv1 pad 4,
  conv2 pad 2, conv3–5 pad 1, every max-pool pad 1. PyTorch's `MaxPool2d` pads with
  −inf, which matches TF. Assert the shapes from section 3.1 in a test.
- **LRN.** PyTorch's `LocalResponseNorm` divides α by `size`, so
  `tf.nn.lrn(depth_radius=r, bias=k, alpha=a, beta=b)` ≡
  `nn.LocalResponseNorm(size=2r+1, alpha=a*(2r+1), beta=b, k=k)`.
  - tflearn defaults → `LocalResponseNorm(11, 1.1e-3, 0.75, 1.0)` (variant `lrn_tflearn`).
  - AlexNet paper values (k=2, n=5, α=1e-4, β=0.75) → `LocalResponseNorm(5, 5e-4, 0.75, 2.0)`
    (variant `lrn_alexnet`).
  - Verify both against a NumPy reference implementation of `tf.nn.lrn` in a test.
    TensorFlow is not needed for that.
- **BN.** `nn.BatchNorm2d(C, eps=1e-5, momentum=0.1)` (PyTorch momentum = 1 − TF decay),
  γ initialized from N(1, 0.002).
- **Optimizer.** `torch.optim.SGD(lr=1e-3, momentum=0.9)` matches TF's
  `MomentumOptimizer` when the learning rate is constant.
- **Dropout.** Both frameworks use inverted dropout, so there is no scaling at test time.

### 3.3 Class names

The archive `17flowers.tgz` stores 80 consecutive images per class (block 0 is
`image_0001`–`image_0080`, and so on) in this order. `oxflower17.py` on `main` (from
PR #1) already uses this order, with label = block index.

| block | English | Polish (as in the thesis) | Latin |
|---|---|---|---|
| 0 | Daffodil | narcyz | *Narcissus* |
| 1 | Snowdrop | śnieżyczka | *Galanthus nivalis* |
| 2 | Lily of the valley | konwalia | *Convallaria majalis* |
| 3 | Bluebell | dzwonek* | *Hyacinthoides non-scripta* |
| 4 | Crocus | krokus | *Crocus* |
| 5 | Iris | irys | *Iris* |
| 6 | Tiger lily | lilia | *Lilium lancifolium* |
| 7 | Tulip | tulipan | *Tulipa* |
| 8 | Fritillary | szachownica | *Fritillaria meleagris* |
| 9 | Sunflower | słonecznik | *Helianthus annuus* |
| 10 | Daisy | stokrotka | *Bellis perennis* |
| 11 | Colt's foot | podbiał | *Tussilago farfara* |
| 12 | Dandelion | mniszek | *Taraxacum officinale* |
| 13 | Cowslip | pierwiosnek | *Primula veris* |
| 14 | Buttercup | jaskier | *Ranunculus* |
| 15 | Windflower | zawilec | *Anemone* |
| 16 | Pansy | fiołek* | *Viola × wittrockiana* |

\* These are the thesis's names. Botanically, bluebell is *hiacyntowiec* and pansy is
*bratek*.

- **Our loader:** keep that convention (label = block index).
- **Verification:**
  - automatically in M4: the CLIP zero-shot majority vote over each 80-image block
    must match the block's name;
  - by eye in M2: `image_0001` is a daffodil, `image_0081` a snowdrop, `image_1281` a
    pansy.
- **tflearn numbered classes differently.** It numbers them by the string-sorted
  directory names (`"0", "1", "10", …, "16", "2", …, "9"`).
  - The tflearn label → block mapping is
    `[0, 1, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 9]`.
  - The names in the thesis's confusion matrices (jaskier, podbiał, narcyz, …) assume a
    different order. Only label 4 (dandelion) is named correctly.
  - Keep a helper `tflearn_label_to_block`, with a test. Use it only when comparing
    with the thesis's tables.

### 3.4 What the thesis reported, and where the text and the code disagree

**Reported numbers** (reference only):

- **Validation, 136 images (Table 4.3).** Best LRN 0.713 (97/136); BN 0.708.
  - 0.708 cannot be written as k/136 and does not appear in Fig. 4.2, where the BN
    maximum is 0.691 (94/136). Treat the BN best as 0.691.
  - These are maxima over the validation curve of a single run, with no seeds.
- **Test, 272 images.** These are the confusion matrices, although their captions say
  "validation".

  | Source | Correct | Accuracy |
  |---|---|---|
  | Table 4.1, LRN | 204/272 | 75.0% |
  | Table 4.2, BN | 241/272 | 88.6% |
  | Fig. 4.5 (captioned LRN, plot title says BN) | 209/272 | 76.8% |
  | Fig. 4.6, BN | 186/272 | 68.4% |

  The figures and the tables differ in 60–86 of 289 cells.
- **Training curves.** Training accuracy reaches 100% (BN after ~500 iterations, LRN
  after ~2200). The LRN validation loss rises from 1.14 to 1.55 while the training
  loss goes to zero. The BN validation loss stays flat at ~1.13–1.2.

**Text vs code.** The port follows the code.

| # | Thesis text | What the code and data show | Handled in |
|---|---|---|---|
| 1 | input 256×256 | 227×227 squash resize | M2, M3 |
| 2 | 5-fold cross-validation | fixed 16/8/56 split | M3 (protocol × seeds), M4 (official splits) |
| 3 | 9,035,104 parameters | 87,711,121 | M3 test |
| 4 | LRN k=2, n=5 | tflearn defaults (bias 1, window 11) | M3 (both variants; default `lrn_tflearn`) |
| 5 | conv init N(0, 0.02) | conv `uniform_scaling`, FC truncated normal 0.02 | M3 |
| 6 | batch 128 in one place, 64 elsewhere | 64 | M3 |
| 7 | validation set used as the test set, best value of the curve reported | a separate 272-image test set existed; its results appear only as confusion matrices | M3 (both numbers, clearly labelled) |
| 8 | class names in Tables 4.1/4.2 | tflearn string-sort mapping, 16 of 17 names shifted | M2 helper, M7 report |
| 9 | "no overfitting" | 100% train vs ~70% validation, rising validation loss | M3, M6 (gap, calibration) |
| 10 | BN variant | code never committed | M3 (reconstructed from tflearn defaults) |

### 3.5 Environment facts

- **Workspace.**
  - uv, with Python 3.12 pinned in `.python-version`. Torch supports 3.12 on macOS
    arm64 and Linux.
  - `ruff.toml` applies the full rule set to `pure-alexnet` (no per-file ignores) and
    lists `pure_alexnet` as isort first-party.
  - `packages/pure-alexnet/tests` holds the offline tests of the Keras version. The
    loader and split tests are framework-independent and should keep passing through
    the port.
  - The current loader decodes in archive order, resizes with BILINEAR and does not
    shuffle, so its split is deterministic: the first 16 images of each block go to
    test. `DataSet` caches a pickle at `./17flowers/dataset.pickle`.
  - No CI yet.
- **Cloud sessions.**
  - CPU only (4 cores, ~15 GB RAM).
  - PyPI and GitHub are reachable. Blocked by default: `www.robots.ox.ac.uk`,
    `huggingface.co`, `download.pytorch.org`, `dl.fbaipublicfiles.com`. The user can
    allow them in the environment's network settings.
  - On Linux, PyPI's torch wheel pulls CUDA libraries (several GB). That is fine,
    only slow, so don't fight it.
  - With the hosts allowed, feature extraction, probes and zero-shot can also run on
    CPU here; AlexNet training cannot.
- **Apple M4** (base, 16 GB).
  - Use the MPS backend and set `PYTORCH_ENABLE_MPS_FALLBACK=1`, because some ops,
    LRN for example, may lack an MPS kernel.
  - Keep the dataset in memory as uint8 tensors (~270 MB for Oxford 17 at 256 px),
    use `num_workers=0`, and augment on the device.
  - Run long jobs under `caffeinate -i`.
  - MPS is not bit-deterministic, so use seeds and report mean ± std.
- **Estimated costs on a base M4.** These are estimates; replace them with the
  numbers M1's `bench` measures.

  | Job | Estimate |
  |---|---|
  | AlexNet-2018, 4500 it × 64 ≈ 288k image-steps ≈ 2.4 PFLOP | ~10–25 min per run |
  | early-stopped run on an official split | ~3–8 min |
  | DINOv2 ViT-B/14 features for 1360 images | 1–2 min |
  | CLIP zero-shot | seconds to minutes |

### 3.6 Prior art in this repository

[PR #1](https://github.com/ufnalbartosz/ml-monorepo/pull/1) ported `pure-alexnet` and
`project-cnn` to Keras 3 and is merged into `main`. It is the current implementation:

| File | What it holds |
|---|---|
| `model.py` | `build_alexnet()` returns a `keras.Model`; the late-2017 variant from section 3.1 |
| `dataset.py` | the pure split function `train_test_valid_split` (per class 20% test, 10% validation, i.e. 16/8/56) and `DataSet` with an injectable loader and a pickle cache |
| `oxflower17.py` | download, extract, decode; label = block index |
| `train.py` | CLI `python -m pure_alexnet.train` |
| `tests/` | offline tests on synthetic data |

Keep the loader, the split function and the test style. Replace `model.py`, `train.py`
and the pickle cache with the PyTorch modules (M2–M3).

`docs/modern-vision-guide.md`, also from PR #1, describes the same ladder in prose.
This plan is its concrete, measured version.

## 4. Target layout

```
packages/pure-alexnet/
  PLAN.md                       this file
  README.md                     exists (Keras); new CLI section in M1, rewritten in M3
  RESULTS.md                    generated by the report (M7)
  configs/*.toml                experiment grids
  notebooks/report.py           jupytext percent notebook (M7)
  results/                      small committed summaries (csv/json) generated from runs
  src/pure_alexnet/
    __main__.py, cli.py         python -m pure_alexnet <command>
    paths.py                    cache root: $PURE_ALEXNET_HOME or ~/.cache/pure_alexnet
    device.py                   cuda -> mps -> cpu, describe_device()
    config.py                   dataclasses, TOML loading, stable config hash
    runs.py                     run directories, skip-if-done, environment capture
    data/classes.py             section 3.3 table, tflearn_label_to_block
    data/oxford17.py            grown from oxflower17.py: download, sha256, extract, decode, uint8 cache
    data/oxford102.py           torchvision Flowers102 plus vendored class names
    data/splits.py              official splits, thesis protocol (dataset.py's split function), k-shot
    data/transforms.py          faithful / AlexNet+ / ImageNet-style / open_clip
    data/synthetic.py           tiny fake datasets for tests and quick mode
    models/alexnet2018.py       faithful port (norm and fc_activation switches)
    models/alexnet_plus.py      improved from-scratch variant
    models/torchvision_tl.py    ResNet-50 fine-tuning
    models/foundation.py        DINOv2, CLIP, SigLIP 2, BioCLIP 2: features and zero-shot
    train.py                    iteration-based loop, eval every N its, best-val checkpoint
    probe.py                    feature cache, linear probe, kNN
    evaluate.py                 metrics, bootstrap CI, McNemar, calibration
    explain.py                  Grad-CAM, error gallery
    product.py                  add a class, open set, retrieval
    report.py                   helpers the notebook calls
  tests/
```

Until M3 the Keras `model.py`, `train.py` and `dataset.py` stay in place, so `main`
keeps working after every milestone.

**Cache layout.** It lives outside the repository and is never committed:
- `<home>/data/...`
- `<home>/features/<model>/<dataset>-<preprocessing>.npz`
- `<home>/runs/<run_id>/{config.json, history.csv, metrics.json, predictions.npz, env.json, best.pt}`
- `<home>/reports/`

`run_id` is a readable name plus a short hash of the resolved config. An existing
`metrics.json` means the run is skipped unless `--force` is passed.

**CLI** (target):

| Command | What it does |
|---|---|
| `download --dataset {oxford17,oxford102}` | fetch and cache a dataset |
| `bench` | measure device throughput |
| `run --grid <name> [--force] [--dry-run]` | `--dry-run` lists the planned runs and the estimated time from `bench` |
| `report [--mode report\|quick\|full]` | build the report |
| `list` | summary of the cached runs |

## 5. Milestones

### M1: Skeleton and tooling

- [ ] `pyproject.toml`:
  - keep keras, tensorflow and tensorboard until M3, because the Keras model must keep
    working;
  - add torch, torchvision, timm, open_clip_torch, scikit-learn, scipy, numpy, pandas,
    matplotlib, pillow;
  - add a notebook extra or dependency group with jupytext, nbconvert and ipykernel;
  - then run `uv lock`.
- [ ] Leave the Keras `model.py`, `train.py` and `dataset.py` untouched until M3. The new
  modules live next to them, and the new CLI (`python -m pure_alexnet`) coexists with
  `python -m pure_alexnet.train`.
- [ ] Core modules:
  - `paths.py`;
  - `device.py`;
  - `config.py` (dataclasses, TOML through `tomllib`, a stable hash over the resolved
    config);
  - `runs.py`, which writes a run directory, skips finished runs and records `env.json`
    (git sha, torch version, device, OS version).
- [ ] CLI skeleton with argparse for `download`, `bench`, `run`, `report` and `list`.
- [ ] `bench`: times N training steps of AlexNet-2018 (batch 64, 227²) and N forward
  passes of ResNet-50 and a ViT-B/14, all with random weights and no download, on the
  selected device. It prints images/s and the estimated hours for the M3–M6 grids.
- [ ] `tests/conftest.py`:
  - a synthetic dataset fixture (17 classes × a few images in a temp dir);
  - `slow` and `network` markers, both excluded from the default run.
- [ ] Package `README.md`: add a section for the new CLI. The Keras section stays
  until M3.

**Tests:**
- device selection (monkeypatched);
- config hash stability;
- the run cache skips finished runs;
- `python -m pure_alexnet --help`.

**Run on M4:**

```bash
uv sync
```

```bash
uv run --package pure-alexnet python -m pure_alexnet bench
```

**Done when:**
- tests and ruff pass in a cloud session;
- the M4 `bench` numbers are recorded in section 9;
- the grid sizes in M3–M6 are adjusted so that each milestone's M4 run fits into one
  night (≤ 8 h): fewer seeds (minimum 2), capped iterations or, for M5 only, a 160 px
  input. Write down any change.

### M2: Data (Oxford 17)

- [ ] `data/oxford17.py`, grown from the existing `oxflower17.py`:
  - keep its label = block index convention, and keep its tests passing;
  - keep `pure_alexnet.oxflower17` importable (a re-export) until the Keras code goes
    in M3;
  - download `17flowers.tgz` and `datasplits.mat` from
    `https://www.robots.ox.ac.uk/~vgg/data/flowers/17/` and record their sha256 values;
  - extract the archive;
  - decode it once into a uint8 cache with two preprocessings: `tflearn` (squash
    resize to 227×227 with LANCZOS, as tflearn's ANTIALIAS did; the current loader
    uses BILINEAR) and `modern` (shorter side 256, kept for crops).
- [ ] `data/classes.py`: the section 3.3 table (English, Polish, Latin) and
  `tflearn_label_to_block`.
- [ ] `data/splits.py`:
  - **official splits** from `datasplits.mat`:
    - variables `trn1-3`, `val1-3`, `tst1-3` hold 1-based image numbers; convert them
      to 0-based;
    - assert 680/340/340 images per split, 40/20/20 per class, and no overlap;
    - if the file looks different, stop and re-check before going on;
  - **thesis protocol:** reuse `train_test_valid_split` (per class 16/8/56) after a
    seeded shuffle. The current code splits in archive order; the tflearn original
    split after an unseeded shuffle;
  - **k-shot subsampling** of a training split, seeded and per class.
- [ ] `data/transforms.py`:
  - faithful: just scale to [0, 1];
  - AlexNet+ train and eval;
  - ImageNet-style train (RandomResizedCrop 224, flip) and eval (resize 256, center
    crop 224);
  - the model's own transforms for open_clip.

  Augmentations run on device tensors.
- [ ] Exact-duplicate check (md5) across the archive, logged.

**Tests:**
- parsing a synthetic `.mat` written with `scipy.io.savemat`;
- split invariants for both protocols;
- class table invariants;
- `tflearn_label_to_block` reproduces the thesis mislabelling (label 4 is the only
  match).

**Run on M4:**

```bash
uv run --package pure-alexnet python -m pure_alexnet download --dataset oxford17
```

Then open `image_0001`, `image_0081` and `image_1281` to eyeball the block order.

**Done when:** the tests pass offline, and on the M4 the download, the split
assertions and the eyeball check succeed.

### M3: AlexNet-2018 port and thesis reproduction

- [ ] `models/alexnet2018.py`:
  - the section 3.1 architecture with tflearn initialization;
  - `norm` ∈ {`lrn_tflearn`, `lrn_alexnet`, `bn`, `gn`, `ln`, `none`}, where `ln` is
    GroupNorm with 1 group;
  - `fc_activation` ∈ {`tanh`, `relu`}, default `tanh`.
- [ ] `train.py`:
  - iteration-based loop, reshuffling every epoch;
  - evaluate on validation every 250 iterations and log train/val loss and accuracy
    to `history.csv`;
  - keep the best-validation checkpoint (highest val accuracy, ties broken by lower
    val loss);
  - evaluate the best and the last checkpoint on test and save their logits;
  - also record the thesis-style number (the maximum of the validation curve) and
    label it as biased.
- [ ] Grid `m3-thesis`:
  - thesis protocol split, and every seed draws its own split, as the original split
    was random;
  - `norm` ∈ {`lrn_tflearn`, `bn`};
  - seeds 0–4 (0–9 if the budget allows);
  - 4500 iterations, batch 64, SGD lr 1e-3, momentum 0.9.
- [ ] Retire the Keras implementation:
  - remove `model.py`, `train.py`, the pickle-based `DataSet`, the
    `pure_alexnet.oxflower17` re-export, and their Keras-specific tests;
  - drop keras, tensorflow and tensorboard from `pyproject.toml` and run `uv lock`;
  - update the root README (the `Running` section and the `pure-alexnet` row) and the
    package README.

**Tests:**
- per-layer shapes match section 3.1;
- parameter counts are 87,711,121 (LRN / none) and 87,712,337 (BN / GN / LN);
- LRN matches a NumPy `tf.nn.lrn` reference for both variants;
- one optimizer step lowers the loss on a fixed batch;
- the model can overfit a tiny synthetic set;
- a CLI run on synthetic data writes a complete run directory.

**Run on M4:**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i uv run --package pure-alexnet python -m pure_alexnet run --grid m3-thesis
```

**Done when:**
- section 9 has, per norm, the distribution (mean ± std, min–max) of best-validation
  accuracy, test accuracy at that checkpoint and the thesis-style maximum, next to the
  thesis values (0.713 / 0.691);
- if the mean validation accuracy falls outside ~0.55–0.85, stop and investigate the
  port before starting M4.

### M4: Transfer-learning ladder on the official splits

- [ ] `models/foundation.py`: frozen-backbone feature extraction, cached per (model,
  dataset, preprocessing). Models:

  | Model | Source |
  |---|---|
  | DINOv2 ViT-S/14 | timm `vit_small_patch14_dinov2.lvd142m`, `img_size=224` |
  | DINOv2 ViT-B/14 | timm `vit_base_patch14_dinov2.lvd142m`, `img_size=224` |
  | CLIP ViT-B/32 and ViT-L/14 | open_clip, `pretrained="openai"` |
  | SigLIP 2 ViT-B/16 | open_clip; resolve the exact tag with `open_clip.list_pretrained()` and pin it in the config |
  | BioCLIP 2 | open_clip `hf-hub:imageomics/bioclip-2` |
  | ResNet-50 penultimate layer | torchvision `IMAGENET1K_V1` |

- [ ] Zero-shot:
  - CLIP and SigLIP 2 with the prompt `"a photo of a {name}, a type of flower."` plus
    a small template ensemble;
  - BioCLIP 2 with common names and with Latin names (section 3.3).
- [ ] `probe.py`:
  - linear probe: standardize, then `LogisticRegression` with C ∈ `logspace(-3, 2, 8)`
    chosen on validation and refit on train;
  - kNN with cosine distance, k ∈ {1, 5, 10}, on the same features.
- [ ] `models/torchvision_tl.py`: ResNet-50 (`IMAGENET1K_V1`) fine-tuning.
  - Augmentation: RandomResizedCrop 224 and flip.
  - Optimizer and LR: SGD with momentum 0.9; head lr 1e-2 and backbone lr 1e-3, with a
    small sweep on the split-1 validation set; cosine schedule.
  - Length: at most 30 epochs, early stopping on validation loss with patience 5.
- [ ] `models/alexnet_plus.py`:
  - the AlexNet trunk with BN and a global-average-pooling head (no 16384→4096 FC);
  - augmentation, label smoothing 0.1, weight decay 5e-4, cosine LR, early stopping;
  - LR sweep {3e-3, 1e-2, 3e-2} on the split-1 validation set.
- [ ] Grid `m4-ladder`:
  - trained models (AlexNet-2018 LRN and BN with best-val checkpointing, AlexNet+,
    ResNet-50 fine-tuning): 3 official splits × 3 seeds;
  - probes and zero-shot are deterministic given the features: 3 splits each.
- [ ] Data checks on DINOv2-B embeddings:
  - near-duplicate pairs (cosine > 0.95, inspect them) within and across the official
    and thesis splits, reported;
  - automatic class-order check: the CLIP zero-shot majority vote over each 80-image
    block must equal the block's name, and the check fails loudly otherwise.

**Tests:**
- feature cache round trip;
- the linear probe reaches ≈100% on separable synthetic features;
- the zero-shot pipeline runs with a randomly initialized tiny open_clip model (no
  download);
- the block-order check logic works on synthetic predictions.

**Run on M4** (overnight):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i uv run --package pure-alexnet python -m pure_alexnet run --grid m4-ladder
```

**Done when:** section 9 has the test accuracy (mean ± std over splits × seeds) of every
ladder row, and the duplicate and block-order checks are reported.

### M5: Normalization study

- [ ] Grid `m5-norm`, on the AlexNet-2018 trunk with `norm` ∈ {`none`, `lrn_tflearn`,
  `bn`, `gn` (32 groups), `ln`}:
  - an LR sweep over {3e-4, 1e-3, 3e-3, 1e-2} per norm on split 1, seed 0, batch 64;
  - then the best LR per norm × batch size {16, 64} × 3 splits × 2 seeds;
  - early stopping with an iteration cap of 2000 to start with, adjusted by the M1
    gate.
- [ ] Record per run: test accuracy, NLL, ECE, iterations to the best validation
  checkpoint, wall-clock time.

**Done when:** section 9 has the table and the accuracy-vs-batch-size data, and the
conclusion stays within the confidence intervals.

### M6: Few-shot, calibration and error analysis

- [ ] Few-shot grid, with k ∈ {1, 2, 5, 10, 20, 40} images per class drawn from each
  official training split:
  - linear probes on all cached features, 3 subsample seeds;
  - AlexNet+ from scratch, 1 seed per split;
  - ResNet-50 fine-tuning for k ≥ 5, 1 seed;
  - zero-shot drawn as flat lines.
- [ ] Calibration: ECE (15 equal-width bins), reliability diagrams, temperature scaling
  fitted on validation, ECE before and after.
- [ ] Error analysis:
  - per-class accuracy and F1;
  - confusion matrices with the correct names;
  - a gallery of the most confident errors;
  - Grad-CAM for AlexNet conv5 and ResNet-50 layer4;
  - optional shortcut check, if segmentation trimaps are available on the VGG page:
    accuracy on foreground-only vs background-only images.

**Done when:** every artifact is produced from runs and the key numbers are in
section 9.

### M7: Report notebook and results

- [ ] `notebooks/report.py` in jupytext percent format. The mode comes from
  `PURE_ALEXNET_MODE`, set by the CLI:
  - `report` only reads runs;
  - `quick` runs a minimal grid end to end (1 split, 1 seed, short budgets) in
    minutes;
  - `full` runs every grid, skipping cached runs.
- [ ] Report sections:
  - setup and environment;
  - thesis reproduction: distributions vs the thesis's values;
  - the main "2018 vs 2026" table: model, year, trainable parameters, M4 time, test
    accuracy mean ± std, 95% bootstrap CI, ECE;
  - McNemar tests for the key pairs (section 6);
  - learning curves and the generalization gap;
  - normalization study;
  - few-shot curve;
  - calibration;
  - errors and Grad-CAM;
  - Oxford 102 (filled in M8);
  - product angle (filled in M9);
  - "thesis claims vs measured", using section 3.4.
- [ ] `python -m pure_alexnet report`:
  - executes the notebook headless (`jupytext --execute`, then `nbconvert --to html`);
  - writes the HTML to `<home>/reports/`;
  - regenerates `RESULTS.md` and `results/*.csv` in the package.
- [ ] Root `README.md`: update the `pure-alexnet` row and add the headline table.

**Tests:** the notebook runs in `quick` mode on synthetic runs (marked `slow`).

**Done when:** the HTML report and `RESULTS.md` have been generated on the M4, and
`RESULTS.md` and `results/*.csv` are committed.

### M8: Oxford 102

- [ ] `data/oxford102.py`:
  - load through `torchvision.datasets.Flowers102`, with the official splits
    train 1020 / val 1020 / test 6149;
  - vendor the class names from the Flowers102 section of the CLIP repository's
    `data/prompts.md`, with a source comment, since the official files have no names.
- [ ] Grid `m8-oxford102`:
  - zero-shot: CLIP, SigLIP 2, BioCLIP 2;
  - linear probes: DINOv2-S/B, CLIP-L, SigLIP 2, BioCLIP 2, ResNet-50;
  - ResNet-50 fine-tuning: 1–3 seeds;
  - AlexNet+ from scratch: 1 seed.

**Done when:** the table is in the report and in section 9.

### M9: Product angle

- [ ] Adding a class without retraining:
  - hold out one class at a time from probe training;
  - add it back through its k support embeddings (prototype or kNN, k ∈ {1, 5, 10});
  - report the accuracy on the new class and on the old classes.
- [ ] Open set:
  - hold out 3 random classes as unknown, repeated 3 times;
  - score with the max softmax and with the max cosine to the class prototypes;
  - report AUROC and the accuracy at 95% coverage of known classes.
- [ ] Retrieval: mAP@10 and recall@k with embeddings on the test split.

**Done when:** the section is in the report and in section 9.

### M10: CI (optional)

- [ ] GitHub Actions running ruff and the offline pytest suite on CPU for pushes and
  PRs. Torch from PyPI makes the job slow; cache the uv directory.

## 6. Statistics protocol

- **Per model:** accuracy for every (split, seed) run. Report mean ± std over all runs,
  plus the per-split means.
- **Metrics:** accuracy, macro-F1, NLL, ECE (15 bins on the max probability).
- **95% CI:** bootstrap over test images, 10,000 resamples. Pool the three official
  test splits at seed 0 (1020 predictions). Model differences use a paired bootstrap
  on the same images.
- **McNemar:** the exact binomial test on discordant pairs (`scipy.stats.binomtest`),
  per split at seed 0, with a Holm correction over this fixed list of comparisons:
  1. AlexNet-2018 LRN vs BN
  2. best AlexNet-2018 vs AlexNet+
  3. AlexNet+ vs ResNet-50 fine-tuning
  4. ResNet-50 fine-tuning vs DINOv2-B linear probe
  5. DINOv2-B vs CLIP-L linear probe
  6. best zero-shot vs best linear probe
- **No claims beyond the intervals.** "Better" requires a CI on the difference that
  excludes 0.

## 7. Overfitting protocol

- **Logging:** log train and validation loss and accuracy at every evaluation. Report
  the generalization gap (train − validation accuracy) at the selected checkpoint and
  the trend of the validation loss.
- **Selection:**
  - model selection and early stopping use only the validation set;
  - test is computed once, for the selected checkpoint;
  - the thesis-style maximum of the curve is reported only in the reproduction, labelled
    as biased.
- **Regularization:**
  - AlexNet-2018 stays faithful: dropout only, no weight decay, no augmentation;
  - AlexNet+ adds augmentation, weight decay, label smoothing, a GAP head and early
    stopping;
  - transfer learning: frozen probes, or fine-tuning with a low backbone LR and early
    stopping.
- **Overconfidence:** measure it with NLL and ECE. The rising validation loss in the
  thesis was exactly this.
- **Data hygiene:** near-duplicate check across splits, class-order check.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| MPS lacks a kernel (e.g. LRN) | `PYTORCH_ENABLE_MPS_FALLBACK=1`, or an own LRN built from `avg_pool` over channels |
| Model tags move (SigLIP 2, BioCLIP 2) | resolve them with `open_clip.list_pretrained()`, pin them in the config, record them in `env.json` |
| A dataset host is unavailable | keep sha256 sums and cached copies; tests never need the network |
| The budget is too big for a base M4 | M1 gate: fewer seeds (minimum 2), capped iterations, 160 px input in M5 only, and write it down |
| The thesis can't be reproduced exactly (unseeded tflearn shuffle, BN code never committed, original `dataset.pickle` lost) | compare distributions over seeds, not single numbers |
| Unified memory pressure on 16 GB | batch features in chunks, no ViT-L fine-tuning, free cached tensors between models |
| Keras/TensorFlow and torch share one environment until M3 (a bigger install) | accepted; M3 removes the Keras stack from this package |

## 9. Results log

Append-only. Fill it in during implementation.

| Date | Milestone | What | Value | Measured on |
|---|---|---|---|---|
| | | | | |
