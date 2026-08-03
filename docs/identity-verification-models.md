# Identity verification and fraud detection: the 2026 state of the art

A defender's map of the field: what the models are, what actually breaks, how
it is measured, and what the law requires. Written mid-2026 — the model names
move, the structure below has been stable for a few years.

**The headline, up front:** face *recognition* is close to a solved problem for
cooperative capture. Top systems in NIST's evaluations reach false-non-match
rates well under 1% at a false-match rate of 1 in a million. If you are losing
to fraud in 2026, it is almost never because your matcher was inaccurate. It is
because someone put something in front of the camera that wasn't a person, or
never went through the camera at all. **Budget your effort accordingly:
recognition is a component you buy; attack detection is where the work is.**

---

## 1. Face recognition — the matcher

**The technique that defines the field is the margin-based softmax loss.** The
lineage runs FaceNet (triplet loss, 2015) → SphereFace → CosFace → **ArcFace**
(additive angular margin, 2019) → **AdaFace** and MagFace (2022, quality-adaptive
margins, which weight low-quality images differently rather than letting them
drag the embedding). ArcFace-style training is still the workhorse; AdaFace is
the usual choice when input quality varies, which in remote onboarding it always
does.

The output is a fixed-length embedding (typically 512-d); verification is a
cosine similarity against a threshold.

- **Backbones**: IResNet-100 remains a strong default; ViT-based face encoders
  are competitive and increasingly common.
- **Training data**: MS1M, Glint360K, WebFace260M. Note that several older
  academic face sets have been withdrawn over consent problems — check
  provenance before training on one, because it is both a legal and a
  reputational exposure.
- **Open source**: **InsightFace** is the reference implementation and is what
  most in-house systems start from.
- **Commercial**: Idemia, NEC, Paravision, Incode, iProov, Jumio, Veriff,
  Onfido/Entrust, Persona, Socure.

**The benchmark that matters is NIST.** The FRTE/FRVT programme (1:1
verification and 1:N identification) is the only evaluation with independent,
sequestered data and continuous vendor submissions. Treat a vendor's
self-reported LFW number as marketing; LFW has been saturated for a decade.
Academic sets still worth reading are IJB-C and TinyFace (low-resolution).

---

## 2. Presentation attack detection — the real battleground

PAD, or "liveness", answers: is the thing in front of the camera a live human,
or a photo, a screen, a printed mask, or a silicone mask?

**The standard is ISO/IEC 30107.** Part 3 defines the metrics, and you should
use its vocabulary because it is what auditors and certifiers use:

- **APCER** — attack presentation classification error rate: attacks accepted as
  genuine. Your security failure.
- **BPCER** — bona fide presentation classification error rate: real users
  rejected. Your conversion failure.

These trade off against each other. Any vendor quoting one without the other is
telling you nothing.

**Certification is meaningful here.** iBeta Level 1 and Level 2 testing (Level 2
uses higher-effort artefacts) against ISO 30107-3, and FIDO's Face Verification
certification, are independent labs actually attacking the system. This is one
of the few places in ML where a compliance badge tracks real capability.

**Passive vs active.** Active liveness asks the user to blink, turn their head,
or read digits. Passive uses a single image or short clip with no instruction.
The industry moved decisively toward passive — partly because challenge-response
hurts completion rates, and partly because generative video can now satisfy a
naive blink-or-turn challenge. Where active survives, it is in forms that are
hard to anticipate, such as randomised illumination sequences.

**Signals that carry real information:**

- **Depth** — structured light or time-of-flight. Face ID's approach. Defeats
  every flat attack by construction, but needs hardware you usually don't have
  on the web.
- **rPPG** (remote photoplethysmography) — recovering a pulse waveform from
  subtle skin colour changes across frames. Elegant, and hard to fake in a
  replay, but sensitive to compression and lighting.
- **Controlled active illumination** — the device emits a known light sequence
  and the system verifies the reflection is consistent with a 3D face in a real
  environment. This doubles as an injection defence, because the challenge is
  unpredictable and per-session.
- **Texture and frequency artefacts** — moiré from screens, print halftones,
  specular reflection patterns.

**Model families**: CDCN (central difference convolutions, which bake in a
gradient-like operator well suited to texture artefacts), DeepPixBiS
(pixel-wise supervision), and ViT-based detectors. The active research problem
is **domain generalisation** — SSDG, SSAN and successors — because PAD models
overfit their capture devices brutally.

> **The single most important evaluation caveat in this entire document:**
> intra-dataset PAD accuracy is meaningless. A model at 99% on OULU-NPU's
> internal protocol can collapse to near-chance on an unseen camera and attack
> type. Only cross-dataset and leave-one-attack-out protocols predict field
> performance. Datasets: SiW, OULU-NPU, CASIA-SURF, CelebA-Spoof, and the
> cross-dataset protocols built over them.

---

## 3. Injection attacks — where the threat actually moved

A presentation attack holds something up to a camera. An **injection attack**
skips the camera: a virtual camera driver, a rooted device, an emulator, or a
tampered client feeds frames straight into the video stream. PAD does not see
it, because from the model's point of view the frames are of a real person —
they just aren't of a real person *who is there now*.

This is where attacker effort concentrated as deepfake generation got cheap, and
where defence is least about the vision model:

- **Device and app attestation** — Play Integrity, App Attest, DeviceCheck.
  Establishes that your real client is running on a real, unmodified device.
- **SDK integrity and anti-tamper** — obfuscation, root/jailbreak detection,
  virtual-camera driver detection.
- **Per-session cryptographic challenge** bound into the capture — a nonce
  reflected in the illumination sequence or embedded in frame metadata, verified
  server-side. Makes pre-rendered video useless.
- **Server-side capture** — never trust a client-supplied image. Frames should
  reach your servers through a path the client cannot substitute.
- **Provenance** — C2PA content credentials, as an ecosystem-level answer.

**If you build one thing beyond a bought matcher, build this.** A certified PAD
vendor with no injection defence is a solved problem for a motivated attacker.

---

## 4. Deepfake and morph detection

**Morphing attack detection (MAD)** matters at document *issuance*: a morph
blends two faces so one passport matches two people. NIST runs a FATE MORPH
track. Differential MAD (comparing the document image against a live capture) is
substantially more reliable than single-image detection.

**Generated-face detection** is the weakest link in the stack, and you should
plan around that. Detectors trained on one family of generators degrade sharply
on unseen ones, and the retraining cycle is permanent. Frequency-domain
artefacts, which worked well against GANs, are much less reliable against
diffusion and flow-matching models.

**Do not architect a system whose security depends on a standalone deepfake
detector.** Use it as one weak signal, and put the structural defences — depth,
active illumination, attestation, injection detection, provenance — underneath.

---

## 5. Document verification

The other half of remote onboarding.

- **Classification and OCR** — identify the document type, then read it.
  MRZ parsing is classical and reliable. VLMs now handle the messy
  non-MRZ fields well, but validate their output against a schema; do not let a
  generative model be the sole authority on a date of birth.
- **Template and security-feature checks** — verify layout, fonts and print
  characteristics against the known specification for that document type and
  issuing year. Holograms and OVI need video, not a still.
- **Tampering detection** — copy-move and splicing detection, and print-scan
  ("recapture") detection. Error Level Analysis is popular and weak; treat it as
  a hint, never as evidence.
- **NFC chip reading (eMRTD)** — reading the ePassport/eID chip and verifying
  **passive authentication** against the issuing country's signing certificate.

> **NFC is the strongest signal available in document verification, by a wide
> margin.** It is a cryptographic proof from the issuing authority, not a visual
> inference. Where the document supports it and the device can read it, prefer
> it over anything a model can conclude from pixels. The EUDI wallet and
> eIDAS 2.0 push further in this direction: verified attributes rather than
> photographed documents.

- **Face match** — the document portrait against the live capture, using the
  §1 matcher. Note the domain gap: a printed, laminated, often decade-old
  portrait against a phone selfie is exactly the low-quality regime AdaFace-style
  quality-adaptive training exists for.

---

## 6. The non-vision half

Identity fraud detection is not primarily a computer vision problem, and a team
that treats it as one will be beaten by fraud that never touches the camera.

- **Device and network intelligence** — fingerprinting, proxy/VPN/hosting
  detection, emulator detection.
- **Velocity and graph features** — how many applications share this device,
  IP, phone number or address? **Graph analysis is what catches organised
  fraud rings**, and it catches them when each individual application looks
  perfectly clean.
- **Behavioural biometrics** — typing cadence, touch dynamics, form-fill
  patterns. Weak alone, useful in aggregate, and legally sensitive in some
  jurisdictions.
- **Consortium data** — shared signals across institutions.

**On models: gradient-boosted trees still win on tabular fraud data.** XGBoost,
LightGBM and CatBoost remain the honest default, and a deep model that beats
them on a real fraud table is rare enough to be worth double-checking for
leakage. Use GNNs for ring detection on the entity graph, and sequence models
for session behaviour. Extreme class imbalance is the defining difficulty:
optimise PR-AUC or recall at a fixed low FPR, never accuracy or ROC-AUC.

---

## 7. Measuring it honestly

- **Pick the operating point by cost, not accuracy.** Write down the cost of a
  false accept (fraud loss, regulatory exposure) and a false reject (lost
  customer, support call), and choose the threshold that minimises total cost.
  A system tuned to maximise accuracy is tuned for nothing anyone cares about.
- **DET curves, not single numbers.** Report FNMR at several fixed FMRs.
- **Measure demographic differentials, per subgroup, always.** NIST's
  demographic-effects work established that error rates vary materially across
  skin tone, age and sex, and that the variation differs by algorithm. An
  aggregate number conceals exactly the failure that will generate both harm
  and litigation. This is not an optional fairness add-on; in this domain it is
  a core quality metric.
- **Adversarial evaluation is not optional.** Run a red team, or hire a lab. The
  threat model changes faster than your test set does.
- **Report APCER and BPCER together**, at a stated operating point.

---

## 8. Legal and regulatory constraints

In this domain compliance shapes the architecture, so it belongs in the design
discussion, not after it. Confirm current status with counsel — several of these
are phasing in.

- **GDPR Article 9** — biometric data for identification is special-category;
  you need an explicit lawful basis, and a DPIA in practice.
- **EU AI Act** — remote biometric identification is heavily restricted, with
  some uses prohibited and others high-risk carrying conformity-assessment
  obligations. Obligations phase in through 2026–2027.
- **Illinois BIPA** — the one with teeth in the US: private right of action,
  statutory damages per violation, and a history of nine-figure settlements.
  Requires written consent and a published retention schedule. Texas CUBI and
  Washington's law are similar without the private right of action.
- **NIST SP 800-63** (Digital Identity Guidelines, revision 4 finalised 2025) —
  defines Identity Assurance Levels. If someone asks you to hit "IAL2", this is
  the document.
- **KYC/AML** — FinCEN CIP rules in the US, AMLD in the EU.
- **eIDAS 2.0 / EUDI wallet** — the EU's direction of travel is verified
  attributes from a wallet rather than photographs of documents.

**Two architectural consequences worth internalising:** store *templates*, not
images, wherever you can, and set a retention schedule you can actually enforce.
Look at template protection — cancelable biometrics, or matching over encrypted
templates — if you are holding biometrics at scale. A breach of a face database
is unlike a password breach: users cannot rotate their face.

---

## 9. Practical recommendation

**For almost every organisation: buy the biometric stack, build the risk
engine.**

PAD and injection detection are a permanent adversarial arms race requiring
dedicated red-team capability and lab certification. Vendors amortise that
across every customer. Your in-house model will be state of the art on the day
it ships and quietly obsolete within a year, and you will not know which day
that was.

What you should own is the **orchestration and risk layer**, because it encodes
your business's actual risk appetite and no vendor can:

1. **Never rely on a single signal.** Document + NFC + face match + PAD +
   injection detection + device + behavioural + graph. Any one of them fails;
   the combination is what holds.
2. **Risk-based step-up.** Low-risk sessions pass with light checks; anomalous
   ones escalate to NFC, or to a video call. Uniform friction is simultaneously
   too much for good users and too little for attackers.
3. **Keep humans in the loop for the contested middle.** Automate the confident
   accepts and the confident rejects; route the uncertain band to review, and
   feed those decisions back as training data.
4. **Log everything, with provenance.** You will need to reconstruct why a
   decision was made, months later, for a regulator or a court.
5. **Assume compromise and design for rotation.** Thresholds, model versions and
   vendors should all be swappable without re-architecting.

---

## Related reading in this repo

- [`modern-vision-guide.md`](modern-vision-guide.md) — how the CNNs in this
  workspace relate to current vision models. The face-recognition embeddings in
  §1 are the same idea as the metric-learning embeddings described there.
- [`model-development-pipeline.md`](model-development-pipeline.md) — the
  engineering around the model. The slice-based evaluation it recommends is what
  §7's demographic-differential requirement looks like in code.
