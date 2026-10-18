# The Python data stack, mapped to this repo

A learning roadmap built around the familiar "Python + X = Y" framing, with one
change: every row is anchored to code that already exists here, so the next step
is opening a file rather than starting a tutorial project.

The list is a good map of the territory. Two caveats before the table. First,
several rows name a *library* where the real skill is a *discipline* — the hard
part of "Python + Time Series = Forecasting" is backtesting, not `statsmodels`.
Second, the single most valuable data skill is missing from it entirely; see
[What the list leaves out](#what-the-list-leaves-out).

---

## 1. Python + Pandas = Data Cleaning

**What it's really for.** Not just cleaning — pandas is the substrate every
other row sits on. Cleaning is maybe a fifth of it; the rest is reshaping,
joining, grouping and time-aware alignment.

**Already here.** `idv_risk/features.py` is a worked example of the thing that
actually matters in production pandas: one function, deterministic output, fixed
column order, explicit dtypes. `idv_risk/synthetic.py` builds a 28-column frame
with masked group assignment.

**Next step.** Method chaining and `.pipe()`; the `category` dtype (load-bearing
in `features.py`); `merge_asof` for point-in-time joins. Then learn where pandas
stops: past a few GB in memory, **Polars** or **DuckDB** are not optional extras,
they are the right answer. This workspace runs pandas 3.x, where copy-on-write is
the default — worth understanding, as it changes when a view aliases its parent.

## 2. Python + Matplotlib + Seaborn = Data Visualization

**What it's really for.** Matplotlib is the engine; seaborn is a statistical
grammar on top. Most people learn the `pyplot` state machine and then fight it
forever.

**Already here.** `project_cnn/plot.py` — confusion matrices, conv-kernel grids,
layer activations, and the `matplotlib.use("Agg")` call that makes it work on a
headless box. `dfp_formula_project/figure.py` draws optimizer contour plots.

**Next step.** Learn the object-oriented API (`fig, ax = plt.subplots()`) and
stop using `plt.` for anything but figure creation. Then build the plots this
repo currently computes as *numbers only*: `idv_risk/metrics.py` produces a
reliability table, PR-AUC and subgroup error rates — all three are far more
legible as a reliability diagram, a PR curve and a per-group error chart.

## 3. Python + Scikit-learn = Machine Learning

**What it's really for.** Less "the algorithms" than "the API contract". `fit`
/ `transform` / `predict`, `Pipeline`, `ColumnTransformer`, and the
cross-validation machinery are the lingua franca — used even when the model
underneath is XGBoost or PyTorch.

**Already here.** `idv_risk/calibration.py` uses `CalibratedClassifierCV` with
`FrozenEstimator`; `idv_risk/metrics.py` uses the metrics module throughout. The
model is XGBoost, but it is wrapped in scikit-learn's estimator interface — which
is exactly why the calibration wrapper composes with it at all.

**Next step.** `Pipeline` + `ColumnTransformer` so preprocessing travels with the
model instead of living in a separate function. Note the honest tension: this
repo deliberately does *not* do that, because the train/serve contract is easier
to test as one explicit function. Learn both, then choose.

## 4. Python + NumPy + Statistics = Statistical Analysis

**What it's really for.** NumPy is the array layer. The statistics is the part
that takes years and pays forever.

**Already here.** `packages/nadaraya-watson` is non-parametric kernel regression
with several kernel functions — genuine statistics, not an ML wrapper.
`idv_risk/metrics.py` uses Jeffreys smoothing `(k + ½)/(n + 1)` so a subgroup
with zero observed errors yields a finite rate rather than an infinite ratio;
that one line is applied Bayesian estimation.

**Next step.** Bootstrap confidence intervals — the metrics module reports point
estimates with no uncertainty, which is its biggest weakness. "PR-AUC 0.93"
means little without "± 0.02 over 1000 bootstrap resamples". Then: `scipy.stats`,
and enough hypothesis testing to run a champion-vs-challenger comparison honestly.

## 5. Python + Web Scraping + BeautifulSoup = Data Collection

**Be sceptical of this one.** It is the weakest row. BeautifulSoup parses HTML;
that is a narrow slice of data collection, and usually the last resort:

- An **API** beats scraping whenever one exists — structured, documented, stable.
- For bulk HTML, `lxml` or `selectolax` are an order of magnitude faster.
- Scraping carries real terms-of-service and copyright exposure. Check before,
  not after.

**Already here.** The honest version of data collection:
`pure_alexnet/oxflower17.py` and `project_cnn/loader.py` fetch archives over
HTTPS and unpack them — including `tar.extractall(..., filter="data")`, which
refuses entries that would escape the destination directory. Downloading data
safely is a more useful skill than parsing HTML.

**Next step.** `httpx` (async, modern), pagination and rate-limit handling,
retries with exponential backoff, and caching so a re-run does not re-fetch.

## 6. Python + Automation + Scheduling = Workflow Automation

**What it's really for.** Not "running a script on a timer" — that is `cron`, and
`cron` is fine until it isn't. An orchestrator buys you retries, backfills,
dependency graphs and lineage.

**Already here.** Nothing yet, and that is the gap.
`packages/idv-risk/docs/feedback-loop-and-pipeline.md` §2.4 specifies the DAG
that should exist: assemble matured cohort → point-in-time label join → quality
assertions → split → train → evaluate → gate → register.

**Next step.** Learn **one** orchestrator properly — Dagster if you like typed
assets and lineage, Airflow if you need the ecosystem, Prefect for the gentlest
start. The transferable skill is idempotency: a task that can be re-run on the
same inputs without double-counting.

## 7. Python + NLP + Text Data = Sentiment Analysis

**Update the framing.** "Sentiment analysis" is a 2015 problem statement. VADER
and NLTK are teaching tools now; in 2026 you would use a small fine-tuned
transformer or just ask an LLM, and the interesting text problems are extraction,
classification against a taxonomy, and retrieval.

**Relevance here.** Fraud has real text surfaces this repo does not touch yet:
chargeback narratives, analyst case notes, and scam-message detection — the last
being central to authorised-push-payment fraud, where every device and biometric
signal says "genuine" because the victim really is the user.

**Next step.** Hugging Face `transformers` and `sentence-transformers`. The
pattern worth learning: embed the text, then feed the embedding as features into
a model like the one in `idv_risk` — that composes with everything already built.

## 8. Python + Time Series + Forecasting = Predictive Analytics

**What it's really for.** The library is the easy part. The discipline is
**backtesting with expanding windows** and never letting the future leak
backwards.

**Already here.** Two things, and they are the same idea:
`idv_risk/splits.py` trains on the past and tests on the future, purging repeat
applicants. And the **vintage curves** in the feedback-loop doc §1.2 — cumulative
fraud discovered by days-since-application, per cohort — are a time-series
problem whose answer determines when a cohort is ripe enough to train on.

**Next step.** `statsmodels` (already a dependency, via nadaraya-watson) for the
classical models; then learn why a naive seasonal baseline beats most fancy
forecasters, and how to measure against it.

## 9. Python + API + JSON = Real Time Data Analysis

**What it's really for.** Serving. Getting a model out of a notebook and behind
an HTTP endpoint with a typed request contract.

**Already here.** `idv_risk/artifact.py` writes JSON metadata and uses XGBoost's
native JSON serialisation rather than pickle, specifically so a six-month-old
artifact still loads. The scoring-log schema in the feedback-loop doc §2.3 is a
JSON record contract waiting to be implemented.

**Next step.** **FastAPI + Pydantic** wrapping `RiskModel.score()`. This is the
most valuable single addition to the repo: a typed request model generated from
`idv_risk/schema.py`, a p99 latency budget, and the per-signal timeout
degradation policy the doc specifies. Pydantic validation and the `Signal`
dataclass are solving the same problem from two directions.

## 10. Python + Dashboard + Streamlit = Data Web Apps

**What it's really for.** Internal tools, fast. Streamlit is excellent for a
dashboard three analysts use and wrong for anything with real concurrency or
auth requirements.

**Already here.** Nothing — but two obvious fits:

- A **review queue** for the step-up band: application, risk score, contributing
  signals, accept/reject buttons writing back to the label store. That is the
  feedback loop's human half, and it does not exist yet.
- A **monitoring dashboard**: PSI drift per feature, band volumes over time,
  vintage curves, subgroup inequity rate.

**Next step.** Build the review queue. It forces you to confront what an analyst
needs to see to make a decision — which is a better interpretability exercise
than any SHAP tutorial.

---

## What the list leaves out

Five gaps, roughly in order of how much they matter:

1. **SQL.** The most valuable data skill there is, and absent entirely. Most
   real work starts with a query against a warehouse, and window functions are
   how point-in-time joins and velocity features actually get computed at scale.
2. **Testing and version control.** This repo has 457 tests across five packages
   precisely so changes can be made without fear. "Can I change this safely?" is
   the question that separates a script from a system.
3. **Data modelling and the warehouse layer.** dbt, dimensional modelling, slowly
   changing dimensions. Where the tables the list assumes you already have come
   from.
4. **Experiment tracking and model registry.** MLflow or equivalent. See
   `idv_risk/artifact.py` for the hand-rolled minimum: config, git SHA, data
   fingerprint and library versions travelling with the model.
5. **Packaging and deployment.** `uv`, lockfiles, Docker. This workspace is a uv
   workspace with five independently installable packages sharing one lockfile —
   that structure is itself the lesson.

## A suggested order

**Foundations** — SQL, pandas, NumPy, git and pytest. Nothing else works without
these.

**Analysis** — statistics, matplotlib/seaborn, time series *as a discipline*.

**Modelling** — scikit-learn's API, then XGBoost (see
`packages/idv-risk/README.md` for a model using most of its interesting surface),
then calibration and evaluation that goes beyond accuracy.

**Production** — FastAPI, one orchestrator, experiment tracking, monitoring.

The fastest way through any of these is to extend something that already runs.
Each section above names a specific gap in this repo; picking one and closing it
teaches more than a greenfield tutorial, because the surrounding code already
has tests that will tell you when you broke it.
