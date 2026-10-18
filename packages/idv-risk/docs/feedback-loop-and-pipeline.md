# Feedback loop and production pipeline

How `idv-risk` gets its labels, and what has to exist around it for those labels
to be worth training on.

The short version: **the model is the easy part.** A gradient-boosted tree on a
labelled table is a solved problem. Everything hard about a production fraud
system is upstream of it — where the labels come from, how long they take, how
biased they are, and whether the features you train on are the ones production
actually saw.

---

## Part 1: The feedback loop

### 1.1 Where labels come from

There is no single source of truth. There are six, with wildly different
latency, coverage and reliability:

| source | latency | covers | reliability |
|---|---|---|---|
| Manual review verdict | hours–days | step-up band only | high, but analyst-biased |
| Chargeback / dispute | 30–120 days | accepted only | high |
| Customer complaint ("that wasn't me") | days–months | accepted only | high |
| Consortium / bureau hit | days–weeks | partial | medium |
| Law enforcement or regulator notice | months | rare | very high |
| Investigation of a linked case | variable | graph-propagated | medium |

Two properties fall out of that table and drive the whole design:

- **Every high-reliability source only covers approvals.** Rejected applications
  never generate an outcome. That is the selection-bias problem in §1.3.
- **Labels arrive over months, not minutes.** That is the maturation problem in
  §1.2.

### 1.2 Labels ripen: the vintage problem

A cohort of applications looks cleaner than it is until its fraud has been
discovered. A 30-day-old cohort might have surfaced 40% of its eventual fraud;
at 90 days, 85%; at 180 days, 98%.

Three consequences:

1. **Only train on matured cohorts.** Define a performance window — 90 days is
   typical for onboarding fraud — and exclude anything younger. Your training
   data is therefore permanently ~90 days stale. This is not a flaw to engineer
   away; it is the shape of the problem, and it is precisely why rules exist to
   cover the fresh end.
2. **Track vintage curves.** Cumulative fraud discovered by days-since-application,
   per monthly cohort. Two things show up here that show up nowhere else: a
   cohort that is discovering fraud *faster* than its predecessors (an attack in
   progress) and one discovering it *slower* (your detection has degraded, or an
   attacker found a typology that surfaces late).
3. **Never use "current known status" as the label.** A fraud confirmed in March
   against a January application is future information. Every label row must
   carry `label_known_at`, and the training join must be
   *point-in-time correct*: as of the cutoff, what did we know?

```
label_store
  application_id
  label                 -- fraud | genuine | unknown
  label_source          -- chargeback | review | complaint | consortium | ...
  label_known_at        -- when WE learned it, not when the fraud happened
  typology              -- nullable; often only known for investigated cases
  confidence            -- confirmed | suspected
```

Append-only. A later, better label supersedes an earlier one; it does not
overwrite it, because reproducing last quarter's model requires last quarter's
beliefs.

### 1.3 The three biases

**Selection bias (reject inference).** You observe outcomes only for what you
approved. Retrain naively on approved-only data and the model learns that your
previous thresholds were correct — it cannot learn anything about the region it
never let through.

The mitigations, in descending order of how much they actually help:

- **Random bypass sample.** Approve a small random share of would-be-rejects
  and let them run. This is the *only* source of genuinely unbiased labels in
  the reject region. It costs real fraud losses, so it gets sized and budgeted
  like any other control: typically 0.5–2% of the reject band, priced at
  `sample_rate × reject_volume × false_accept_cost`. Cheap relative to the
  alternative, which is a model that slowly calcifies around its own past
  decisions.
- **Step-up outcomes.** The review band is the ambiguous region *and* it gets
  resolved, so it is the densest source of informative labels you have for free.
- **Classical reject inference** — parcelling, fuzzy augmentation, reweighting,
  extrapolation. Worth knowing, worth being sceptical about: these methods have
  no strong theoretical basis, are prone to overfitting, and recent work argues
  the improvements are often illusory. Treat them as a supplement to a bypass
  sample, never a replacement for one.

**Verification bias.** If the review queue is sorted by risk score, the analyst's
prior is contaminated by the model's opinion, and their verdict is not
independent evidence. Mitigations: blind a random slice of the queue to the
score, dual-review a sample, and track analyst-versus-eventual-truth agreement
as its own metric. An analyst who agrees with the model 99% of the time is not
adding information.

**Survivorship in the graph features.** `graph_ring_size` computed *today*
against *today's* graph is not what the model saw in January. Graph and velocity
features must be reconstructed as-of, or — much better — logged at scoring time.
See §2.3.

### 1.4 The loop

```
                    ┌──────────────────────────────────────────┐
                    │            scoring service               │
   application ────►│  collect signals → features → model →    │
                    │  calibrate → threshold → decision        │
                    └──────┬──────────────────┬────────────────┘
                           │                  │
             scoring log ◄─┘                  ▼
        (feature vector,           accept    step-up    reject
         score, model version,       │          │         │
         thresholds, as-of ts)       │          │         ├─► bypass sample (2%)
                           │         │          │         │      │
                           │         ▼          ▼         ▼      ▼
                           │   chargebacks   review    (no label ever)
                           │   complaints    verdict           │
                           │         │          │              │
                           │         └────┬─────┴──────────────┘
                           │              ▼
                           │        label store (append-only, point-in-time)
                           │              │
                           │              ▼
                           │       maturation filter (≥ 90 days)
                           │              │
                           └──────────────┤
                                          ▼
                              training set = logged features
                                             ⋈ matured labels
                                          ▼
                          train → calibrate → thresholds → evaluate
                                          ▼
                                   promotion gates (§2.5)
                                          ▼
                              shadow → challenger → champion
```

### 1.5 The fast lane

The loop above is measured in months. Attackers move in days. So there are two
speeds, and conflating them is a common failure:

- **Slow lane** — model retraining. Weeks to months. Handles drift, new signal
  sources, structural change.
- **Fast lane** — thresholds, rules, and blocklists. Hours. Handles the attack
  happening right now.

Plus one thing that turns a single label into immediate value: **graph
propagation on confirmation**. When one application is confirmed fraudulent,
re-score everything sharing a device, IP or PII root with it, straight away.
That is the cheapest detection in the entire system, and it needs no retraining
at all — it is why the graph features exist.

### 1.6 What to monitor

| signal | what it catches | typical alarm |
|---|---|---|
| PSI / KS per feature vs training reference | vendor recalibrated their score; upstream bug | PSI > 0.2 on any feature |
| Score distribution and band volumes | same, faster | step-up rate moves > 30% day over day |
| Vendor model versions | known change, not drift | any change → pin as an event |
| Observed fraud rate in the accept band | attackers found a gap | rises against a stable score distribution |
| Vintage curve slope by cohort | detection degrading | discovery slower than the last three cohorts |
| Calibration (ECE) on matured cohorts | the thresholds no longer mean what they say | ECE > 2× the value at promotion |
| Subgroup inequity rate | a differential opened up | > the gate value from §2.5 |

The one worth calling out: **a step-up band that doubles overnight is a vendor
incident about nine times out of ten, not a fraud wave.** Check the signal
distributions before you staff up the review team.

---

## Part 2: The pipeline

### 2.1 Two paths, one feature builder

```
OFFLINE (daily/weekly)                 ONLINE (per application)
──────────────────────                 ────────────────────────
warehouse: scoring log                 vendor calls (parallel, timeouts)
   ⋈ label store (point-in-time)              │
        │                                     ▼
        ▼                              features.build_features()  ◄── SAME CODE
  maturation filter                           │
        │                                     ▼
        ▼                              RiskModel.score()
  splits.time_grouped_split                   │
        │                                     ▼
        ▼                              decision + scoring log ──┐
  model.fit_classifier                                          │
  calibration.fit_calibrator                                    │
  decision.optimise_thresholds                                  │
        │                                                       │
        ▼                                                       │
  metrics + fairness gates                                      │
        │                                                       │
        ▼                                                       │
  artifact.RiskModel.save ──► registry ──► serving ◄────────────┘
```

`features.build_features` being the *same function* on both paths is the single
most important structural property. The tabular equivalent of the classic vision
resize-mismatch bug is a column order that differs, a category encoded two ways,
or a value imputed in training and left NaN in production — all silent, all
returning a plausible wrong number.

### 2.2 Serving

**Latency budget.** IDV is generous compared with payment authorisation: this is
a one-shot decision during an onboarding flow the user expects to take seconds.
A p99 of 200–300ms end-to-end is comfortable, which means you can afford
synchronous graph lookups rather than precomputed embeddings. XGBoost inference
itself is sub-millisecond for a 28-feature model — the budget is spent on vendor
calls and feature lookups, not the model.

**Deployment shape.** Embed the model in the decision service rather than
standing up a separate model server. One less network hop, one less thing to
version-skew, and the artifact bundle is already self-contained.

**Degradation policy — decide this explicitly, and test it.** Vendors time out.
For each signal:

- *Optional signals* (NFC, rPPG, active challenge) → NaN. XGBoost routes it. This
  is already how the schema is built.
- *Required signals* (PAD, face match, attestation) → `build_features` raises.
  That is correct for a coding error and wrong for a vendor outage. Production
  needs a documented choice per signal: fail to `step_up` (safe, costs friction)
  or degrade to NaN with a flag feature marking the degradation (cheaper, needs
  the model trained on degraded rows to be meaningful).

The safe default is **fail to step-up**, because a scoring request that cannot
see the PAD result is exactly the request an attacker would like you to guess on.

**Idempotency.** Key scoring by `application_id`. A retry must not increment
velocity counters twice — otherwise a flaky network turns a genuine applicant
into a ring member.

**Feature freshness.** Velocity and graph features need an online store with
real-time counters (Redis, DynamoDB, or a feature store's online path). The
offline path reconstructs the same values as-of the scoring timestamp — or
skips the reconstruction entirely by using the logged vector, which is why §2.3
matters.

### 2.3 Log the features, not just the inputs

**Log the exact feature vector that was scored**, alongside the score, the model
version, the thresholds in force, and the decision.

The alternative — recomputing features at training time from raw signals — is
how training/serving skew gets in. The vendor changed their scoring in March.
The graph looks different today than it did in January. A bug fix to the feature
builder silently changes historical rows. Logged vectors are immune to all
three: the training data is literally what production saw.

This is what feature-store practitioners mean by point-in-time correctness, and
logging at the inference service is the most reliable way to get it, because the
values are guaranteed identical to what the model consumed.

```
scoring_log
  application_id, scored_at
  model_version, artifact_fingerprint
  features        -- the full vector, as scored
  raw_score, calibrated_score
  accept_below, reject_at_or_above   -- thresholds in force
  decision
  degraded_signals[]                 -- which vendors timed out
  vendor_versions{}                  -- per-signal model version
```

`RiskModel.score()` currently returns scores and decisions; in production it
should emit this record as well. That is the main serving-side gap in the
package as it stands.

### 2.4 Training orchestration

Airflow, Dagster or Prefect — the tool matters far less than the DAG:

1. **Assemble cohort** — scoring log rows older than the maturation window.
2. **Join labels point-in-time** — `label_known_at <= cutoff`.
3. **Assert data quality** — row counts, fraud rate within expected band, no
   schema drift. Fail the run rather than train on a broken extract.
4. **Split** — `splits.time_grouped_split`, purging applicants, keeping devices.
5. **Train / calibrate / optimise thresholds** — `pipeline.train`.
6. **Evaluate** — `metrics.evaluation_summary`, including subgroup slices.
7. **Gate** — §2.5. Blocks promotion automatically.
8. **Register** — versioned artifact plus the config, data fingerprint and git
   SHA that `artifact.Provenance` already captures.

Weekly is a reasonable default. More often than the label maturation window
buys nothing: you would be retraining on almost identical data.

### 2.5 Promotion gates

A candidate model is promoted only if it passes *all* of these against the
incumbent, on the same untouched holdout, automatically:

| gate | rationale |
|---|---|
| PR-AUC ≥ champion − 0.01 | headline discrimination |
| recall @ 1% FPR ≥ champion | performance at the actual operating point |
| ECE ≤ 0.02 | thresholds must mean what they say |
| expected cost ≤ champion | the only gate the business cares about |
| per-typology recall ≥ champion − 0.05, **on every typology** | stops a candidate trading all its ring detection for a point of aggregate PR-AUC |
| inequity rate ≤ agreed ceiling | fairness is a blocking gate, not a report |
| regression set fully caught | frozen known-fraud cases that must never regress |

The per-typology gate is the one most often missing and most often needed:
aggregate metrics happily hide a model that got better at the easy 70% of fraud
and much worse at the expensive 5%.

### 2.6 Rollout

**Shadow** (1–2 weeks) — score in parallel, act on nothing, compare score and
band-volume distributions against champion. Catches feature-pipeline breakage
that offline evaluation cannot, because offline evaluation uses the offline
feature path.

**Challenger** (2–4 weeks) — route 5–10% of live traffic, measure *real*
outcomes. Note this takes a maturation window to read properly; interim signals
are review verdicts and step-up rates, not fraud rates.

**Champion** — full traffic, previous model retained and one config flag away.

**Kill switch** — revert to the previous bundle, or to rules-only. Test that it
works, on a schedule. An untested kill switch is a kill switch that does not
work.

### 2.7 Governance

- **Reconstructible decisions.** Every decision must be explainable months later
  from the scoring log: inputs, model version, thresholds, output. This is a
  regulatory requirement in most jurisdictions that matter, not a nice-to-have.
- **Model documentation per version.** If the system performs remote biometric
  identification, the EU AI Act's high-risk obligations bring conformity
  assessment and documentation duties with them.
- **Retention.** The scoring log holds *scores and features*, never images or
  biometric templates. Set a retention schedule for each and enforce it.
- **Fairness reporting on a cadence**, not only at promotion — a stable model on
  a shifting population can develop a differential without anything being
  redeployed.

---

## What the package would need

Honest gap list, roughly in value order:

1. **`scoring.py`** — emit the §2.3 record from `RiskModel.score()`, with model
   version, thresholds in force and degraded-signal flags.
2. **`gates.py`** — the §2.5 promotion gates as code. Highly testable, and the
   piece that most directly prevents a bad model reaching production.
3. **`outcomes.py`** — label schema, maturation filter, vintage curves, and the
   point-in-time join.
4. **`monitoring.py`** — PSI/KS drift against a stored training reference.
5. **Degradation policy** — per-signal timeout behaviour in the schema, plus
   tests that a required-signal outage produces `step_up` rather than an
   exception.
6. **Bypass sampling** — helper to select and account for the random
   reject-band holdout that §1.3 depends on.

Items 1 and 2 are the ones I would build first: without the scoring log there
is nothing trustworthy to retrain on, and without the gates there is nothing
stopping a worse model from shipping.
