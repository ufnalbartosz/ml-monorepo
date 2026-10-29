# From this repo to 2026 computer vision

A guide for going from what these packages do (AlexNet, hand-built inception
blocks, train-from-scratch on a few thousand images) to how image models are
actually built now — plus an honest answer on whether classical image
processing still matters.

Written mid-2026. Specific checkpoints move fast; the *shape* of the field
below has been stable for a couple of years now.

---

## 1. Where this repo sits

| | this repo | typical 2026 baseline |
|---|---|---|
| architecture | AlexNet (2012), GoogLeNet-style inception (2014) | ConvNeXt / ViT / hybrid, or no custom architecture at all |
| params | ~60M, most of it in two dense layers | 20–300M, almost none in dense layers |
| training | from scratch, SGD+momentum, fixed LR | fine-tune a pretrained backbone, AdamW + cosine, EMA |
| data | 1.4k–12k labelled images | pretrained on 100M–5B images, then 10–1000 labels |
| augmentation | flip, crop, hue/contrast jitter | RandAugment, mixup, CutMix, random erasing |
| label set | fixed at build time (17 flowers, 20 CIFAR classes) | open vocabulary — classes are text, decided at inference |
| normalization | local response normalization | LayerNorm / BatchNorm / RMSNorm |
| what you get | a classifier | features usable for classification, retrieval, detection, segmentation, captioning |

Nothing here is *wrong*. The models in this repo are historically the right
models. But almost every design choice in them was made under a constraint
that no longer applies.

---

## 2. The ten changes that actually matter

Roughly in the order they happened, because each one is a reaction to the
previous.

**1. Residual connections (ResNet, 2015).** The single most load-bearing idea
of the decade. `x + f(x)` makes gradients flow through 50, 100, 1000 layers.
Everything since — ViTs included — is built on residual blocks. This is the
first thing this repo is missing.

**2. Normalization and the training recipe.** BatchNorm (2015) made the LRN
layers in `alexnet_model.py` obsolete overnight. Then LayerNorm/GroupNorm for
when batch statistics are unreliable. But the deeper lesson landed later:
*"ResNet strikes back"* (2021) retrained a stock 2015 ResNet-50 with a modern
recipe — AdamW, cosine schedule, 300+ epochs, RandAugment, mixup/CutMix, label
smoothing, EMA — and took it from 76% to ~80% ImageNet top-1. Six years of
architecture search was worth less than the recipe. **If you change one thing
about how you train, change the recipe before the architecture.**

**3. Vision Transformers (ViT, 2020).** Cut the image into 16x16 patches, treat
each as a token, run a plain transformer. No convolutions, no locality prior.
The catch: it needs enormous data or aggressive augmentation, because it has to
*learn* the inductive biases a convnet gets for free. DeiT (2021) fixed the
data hunger with distillation and augmentation; Swin (2021) added back a
hierarchy with shifted local windows.

**4. Convnets modernised (ConvNeXt, 2022).** The rebuttal: take a ResNet,
apply every ViT-era design choice (7x7 depthwise convs, inverted bottleneck,
LayerNorm, GELU, fewer activations), and it matches ViTs. Conclusion the field
settled on: **architecture family matters much less than scale, data and
recipe.** Pick either; both work.

**5. Self-supervised pretraining (MAE 2021, DINOv2 2023, DINOv3 2025).**
Learn features from images with no labels at all — by reconstructing masked
patches, or by matching two augmented views. DINOv2/v3 features are good enough
that a *linear layer* on frozen features beats a fully fine-tuned supervised
convnet on many tasks, and they do dense correspondence and segmentation for
free. This is the change that most directly obsoletes what these packages do:
you no longer train a feature extractor on 12k images. You download one.

**6. Vision-language models (CLIP 2021, SigLIP/SigLIP 2 2023–2025).** Train on
image-caption pairs so images and text land in the same embedding space.
Consequence: **classification without training.** "Is this a daffodil or a
tulip?" becomes an embedding comparison against the text `"a photo of a
daffodil"`. `pure-alexnet` trains for 150 epochs to do something a SigLIP
checkpoint does zero-shot, usually better.

**7. Detection and segmentation went open-vocabulary.** DETR (2020) reframed
detection as set prediction and killed anchor boxes and NMS-heavy pipelines.
The YOLO line and RT-DETR own real-time detection. SAM / SAM 2 (2023–2024)
segment anything you point at, in images and video, with no class list.
Grounding-DINO + SAM = "segment every cup in this photo" from a text prompt.

**8. Diffusion and flow matching.** GANs → DDPM (2020) → latent diffusion
(2022, Stable Diffusion) → rectified flow / flow matching (2024, SD3, Flux).
Relevant even if you never generate art: these models are used for synthetic
training data, augmentation, inpainting-based editing, and super-resolution.

**9. Multimodal LLMs.** Qwen-VL, InternVL, and the frontier assistants take an
image and reason about it in text: OCR, charts, documents, spatial questions,
grounding. For a large class of "look at this image and tell me X" problems the
2026 answer is an API call, not a training run. Know where that line is —
latency, cost and privacy usually decide it, not accuracy.

**10. Geometry got learned too.** Depth Anything V2, learned optical flow
(RAFT and successors), learned keypoints/matchers (SuperPoint, LightGlue),
learned SfM front-ends. Monocular depth from a single image is now a solved-
enough problem to ship. This is the frontier where classical CV is genuinely
being displaced — see §4.

**Worth watching, not yet default:** state-space and linear-attention vision
backbones (VMamba and relatives) promise linear cost in sequence length; as of
now they are competitive but have not displaced ViT/ConvNeXt hybrids in
practice.

---

## 3. A concrete learning path

The fastest way to internalise the gap is to solve *this repo's own task* four
times, each time one rung up the ladder. Use the 17-flowers set — it is small,
it is already wired up in `pure_alexnet.dataset`, and it is exactly the regime
where the modern approach wins hardest.

**Rung 1 — modern recipe, same idea (a weekend).**
Fine-tune a pretrained `resnet50` or `convnext_tiny` from `timm` on
17-flowers. You will beat the from-scratch AlexNet in this repo by a wide
margin in a fraction of the compute. Lesson: pretraining > architecture.

**Rung 2 — frozen features + linear probe (an afternoon).**
Run DINOv2 over the 1360 images once, cache the embeddings, fit a logistic
regression with scikit-learn. No GPU training loop at all. Compare accuracy and
wall-clock against rung 1. Lesson: for small data, the backbone is the product.

**Rung 3 — zero-shot (an hour).**
Embed the 17 class names with SigLIP's text tower, embed the images with its
image tower, take the argmax cosine similarity. Zero training examples.
Lesson: for many problems, "collect a labelled dataset" was the wrong first
step.

**Rung 4 — ask a VLM.**
Send an image to a multimodal model and ask which of the 17 categories it is.
Note the cost and latency per image and where that stops being sensible.
Lesson: know the price of the easy answer.

Then, and only then, read the papers — you will have the questions they answer:

> ResNet → BatchNorm → ViT → DeiT → MAE → CLIP → ConvNeXt → DINOv2 → SAM

**Resources that are worth the time:**
- **CS231n** (Stanford) — still the best convnet fundamentals; the assignments
  are where backprop stops being magic.
- **fast.ai Practical Deep Learning** — top-down, gets you to working models fast.
- **Prince, *Understanding Deep Learning*** (free PDF) — the modern replacement
  for the 2016 Goodfellow book; good on transformers and diffusion.
- **Hugging Face Computer Vision course** — free, hands-on, current.
- **`timm` documentation and source** — the de-facto reference implementation of
  every vision backbone worth using. Reading `timm` is a curriculum by itself.
- **Papers With Code / the `lucidrains` GitHub repos** — minimal readable
  implementations of nearly every architecture.

**Tooling reality check.** This repo is now on Keras 3, which is a genuinely
good place to be — it runs on TensorFlow, PyTorch or JAX backends. But the
research and pretrained-model ecosystem lives in PyTorch: `timm`,
`transformers`, `diffusers`, `open_clip`, `ultralytics`, `segmentation_models`.
Learn enough PyTorch to read it, whatever you write in. When you want a
pretrained checkpoint at 2 a.m., it will be a `.safetensors` file on the Hugging
Face Hub with a PyTorch loading snippet next to it.

**Benchmarks to anchor on.** ImageNet top-1 is saturated and no longer
discriminative. Look at ImageNet-A / -R / -Sketch and ObjectNet for robustness,
VTAB for transfer, COCO for detection, and MMMU / DocVQA / ChartQA for
multimodal reasoning. Check a current leaderboard rather than trusting any
specific number in this document.

---

## 4. Does anyone still use classical image processing?

Yes — a great deal of it, every day, in production. But the division of labour
changed, and one specific technique you named really is obsolete. Taking these
separately, because the honest answer is different for each.

### Alive and completely standard

**Geometry and camera work.** Camera calibration (Zhang's checkerboard method),
lens undistortion, homographies, rectification, perspective warps, stereo
rectification. There is no learned replacement and nobody is looking for one.
Every robot, every AR headset, every multi-camera rig does this with the same
maths as in 2000.

**Fiducial markers.** ArUco, AprilTag, ChArUco. If you control the environment
and can put a printed tag in it, you get millimetre pose at hundreds of Hz for
free. Using a neural network here would be strictly worse.

**Industrial machine vision and metrology.** Controlled lighting, threshold,
connected components, contour extraction, sub-pixel edge fitting. When you need
to measure a machined part to ±10 µm and certify the result, a Canny edge with
sub-pixel interpolation beats a CNN — it is deterministic, explainable,
validated, and its failure modes are known. This is a large industry and it is
not going anywhere.

**Barcodes, QR, OCR preprocessing.** Classical detection and decoding,
adaptive thresholding, deskewing, binarisation.

**Morphology and post-processing.** Erode/dilate/open/close, connected
components, watershed, contour simplification. Constantly used to *clean up
neural network output* — a segmentation mask from SAM still gets morphological
post-processing before it drives anything real.

**Feature matching, SfM and SLAM.** This one is genuinely hybrid and worth
understanding properly. ORB and SIFT still ship in production systems
(COLMAP, ORB-SLAM3). Learned detectors and matchers (SuperPoint, LightGlue)
clearly win on wide baselines, illumination changes and textureless scenes, and
they are taking over the front end. But the back end — RANSAC, PnP, bundle
adjustment, epipolar geometry — is entirely classical and will remain so,
because it is optimisation over a known geometric model, not pattern
recognition.

**Optical flow.** Lucas-Kanade and Farnebäck are still used where "cheap and
good enough" wins; RAFT-style learned flow wins on accuracy and is the default
when you have the compute.

**The image signal processor in every camera.** Demosaicing, denoising, tone
mapping, lens shading correction. Increasingly learned on flagship phones,
still overwhelmingly classical everywhere else.

### Genuinely obsolete

**Haar cascades — yes, this one is dead for production use.** Viola-Jones
(2001) was a landmark: real-time face detection on a 2001 CPU, and for a decade
it was in every point-and-shoot camera. In 2026 it is essentially never the
right choice. It only handles near-frontal faces, degrades badly under pose,
lighting and occlusion, and has a false-positive rate that requires constant
threshold babysitting. Modern alternatives run *faster* on a phone CPU with far
better accuracy: MediaPipe BlazeFace, SCRFD, RetinaFace, or any small YOLO
variant. Haar cascades survive mainly as the default in OpenCV tutorials and as
a teaching device for boosting and integral images — both legitimate, neither a
reason to ship one.

**HOG + SVM** for pedestrian/object detection: superseded, same story.

**Hand-crafted descriptors for recognition and retrieval** — SIFT
bag-of-words, Fisher vectors, VLAD. Dead as a *classification* pipeline;
replaced entirely by learned embeddings. (SIFT itself lives on for *geometry*,
as above — the descriptor is fine, the classifier built on top of it is not.)

**Deformable part models.** Historically important, entirely superseded.

### The rule of thumb

Classical when the environment is **controlled** and the problem is
**geometric** — fixed lighting, known optics, measurement, calibration, pose
from markers. Learned when the problem is **semantic** and the world is
**open** — what is this, where is it, is it defective in a way nobody
enumerated.

Most real systems are both, and the seam is usually in the same file: classical
geometry to rectify and register, a network for semantics, classical morphology
to clean up the mask, classical maths to turn pixels into a decision.

One last argument for learning the classical material even if you never write a
Sobel filter by hand: it is what makes learned vision debuggable. Knowing what a
Gaussian pyramid is tells you what a feature pyramid network is doing. Knowing
what a Gabor filter looks like tells you why the first conv layer of every
trained network looks the way it does — including the one in this repo, which
you can now actually inspect via `project_cnn.tools.get_weights_variable` and
`project_cnn.plot.plot_conv_weights`.
