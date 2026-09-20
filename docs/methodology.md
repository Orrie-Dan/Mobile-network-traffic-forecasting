# Methodology

High-level methodological notes for the project. The three forecasting models are
**locked** as **SARIMA**, **LSTM**, and **TCN**. A naive persistence forecast is
used only as an evaluation baseline and is not one of the research models.

Planned stages include dataset access and preprocessing, exploratory analysis,
literature-informed model selection, one-step-ahead forecasting experiments,
evaluation (MAE, RMSE, MAPE, and runtime), and comparison across geographical
areas with different traffic characteristics.

Phase 5 (this document, later sections) defines the forecasting experiment and
prepares data. **No models are trained in Phase 5.**

---

# Data Loading and Memory Management

Phase 2 prototype (one representative day only). Measured values below come from
the local run summarised in `results/metrics/phase2_memory_report.json`.

## Why the full dataset is not loaded at once

The Milan SMS/Call/Internet release comprises **62** daily TXT files
(~**20.8 GB** uncompressed). A single day is already ~**323 MB** text /
~**4.84 million** rows. Loading all days as one in-memory table is unnecessary
for forecasting prep and exceeds practical laptop/Colab RAM budgets.

## Why chunked processing is necessary

Chunked reads bound peak RAM to approximately one chunk plus compact aggregates,
instead of the full raw day table. This is the pattern required before scaling
from one day to many days.

## Temporary extraction strategy

- Discover `dataverse_files*.zip` under a configurable `DATA_DIR` /
  `MILAN_DATA_DIR` (or pass explicit `zip_paths`).
- Locate the target member with `find_member`.
- Extract **only** that TXT into `tmp/phase2_extract/`.
- Delete the temporary TXT after the experiment (`remove_path`).
- Do **not** extract entire ZIP archives.

## Representative file

| Item | Value |
|------|--------|
| File | `sms-call-internet-mi-2013-11-01.txt` |
| Source ZIP | `dataverse_files.zip` |
| Uncompressed size | 322,874,887 bytes (~308 MiB / ~323 MB) |

## Baseline approach

- `pandas.read_csv` via `read_traffic_file(..., chunksize=None)`
- Tab separator, `header=None`, explicit column names
- Default dtype inference (int64 / float64 for all numeric columns)
- Full 8-column table retained in memory
- Process RSS measured with `psutil` before/after load; DataFrame
  `memory_usage(deep=True)` recorded; object then deleted

### Observed baseline (2013-11-01)

| Metric | Value |
|--------|--------|
| Raw rows | 4,842,625 |
| Load time | ~2.2 s |
| RSS before | ~83.4 MB |
| RSS after | ~377.0 MB |
| RSS delta | **~293.6 MB** |
| DataFrame memory | **~295.6 MB** |
| Inferred dtypes | `int64` × 3 + `float64` × 5 |

## Optimized approach

- Chunked reader: `chunksize=500_000`
- `usecols` limited to Internet-task columns:
  `square_id`, `time_interval`, `country_code`, `internet_traffic`
- Explicit dtypes (see below)
- Per-chunk aggregation to `(square_id, time_interval)`, then
  `merge_chunk_aggregates`
- Add UTC `timestamp` from Unix milliseconds
- Empty fields parsed as NaN; no imputation

### Chunk size

**500,000** rows per chunk (configurable). On this day that yields 10 chunks.

### Dtype decisions

| Column | Dtype | Rationale |
|--------|--------|-----------|
| `square_id` | `int32` | Grid IDs are integers in 1…10000; `int32` is sufficient |
| `time_interval` | `int64` | Unix time in milliseconds requires 64-bit integers |
| `country_code` | `int32` | Integer codes; values observed include multi-digit codes; no missing codes on the prototype day |
| `internet_traffic` | `float32` | Activity values; float32 vs float64 on a 100k-row probe: max abs error ~7.6×10⁻⁶, max rel error ~5.9×10⁻⁸ (within tolerance) |

SMS/call columns are omitted for the Internet forecasting path to reduce IO and RAM.

### Observed optimized (2013-11-01)

| Metric | Value |
|--------|--------|
| Rows read | 4,842,625 (matches baseline) |
| Runtime | ~2.9 s |
| RSS before | ~148.0 MB |
| RSS peak during chunks | ~204.8 MB |
| RSS peak delta | **~56.8 MB** |
| Aggregated DataFrame (`sum_all_countries`) | **~33.0 MB** |
| Aggregated rows | 1,439,982 |

## Memory measurement method

- Process RSS via `psutil.Process(...).memory_info().rss`
- DataFrame footprint via `DataFrame.memory_usage(deep=True).sum()`
- Baseline and optimized runs on the **same** extracted file
- Garbage collection requested between stages

## Memory reduction

Two comparable reductions (do not mix definitions):

| Comparison | Baseline | Optimized | Reduction |
|------------|----------|-----------|-----------|
| Retained table memory | Raw day DF **295.6 MB** | Aggregated Internet DF **33.0 MB** | **~88.8%** |
| Process RSS delta | Load delta **293.6 MB** | Chunk peak delta **56.8 MB** | **~80.7%** |

The DataFrame comparison reflects the forecasting-oriented output (square × time
Internet series) versus keeping every raw country-level row. The RSS comparison
reflects peak process growth during each approach.

## Correctness validation

Baseline vs optimized on 2013-11-01:

| Check | Result |
|-------|--------|
| Raw row count | Match (4,842,625) |
| Aggregated row counts (`sum_all` / `39_only`) | Match (1,439,982 / 1,439,981) |
| Unique squares / timestamps (raw) | 10,000 / 144 |
| Timestamp grid | All adjacent global unique timestamps = 600,000 ms |
| Total Internet relative difference | ~2.8×10⁻⁸ |
| Max abs diff on aggregated cells | ~5.8×10⁻⁴ (float32 vs float64) |
| Spot checks (square 1, first bins) | abs diffs ≲ 5×10⁻⁷ |
| Mismatched aggregation keys | 0 |

Conclusion: memory optimisations do not change analytical results beyond
expected float32 rounding.

## Country-code aggregation (summary)

See also `docs/dataset_description.md` → **Country Code Aggregation Decision**.

- **Recommended:** sum Internet traffic across all `country_code` values per
  `(square_id, time_interval)`.
- **Sensitivity:** `country_code == 39` only (~99.72% of Internet sum on this day).

## Storage considerations

- Raw ZIPs remain external; never commit them.
- Temporary extracts under `tmp/` (gitignored) must be deleted after use.
- Prototype output: `data/samples/processed_sample_day.csv`
  (~1.44M rows, ~60 MB, gitignored via `*.csv` / `data/samples/*`).
- Machine-readable metrics: `results/metrics/phase2_memory_report.json`.

## Limitations and trade-offs

- Results are for **one day**; country-code shares and memory figures may vary
  by day (holidays, events).
- float32 reduces RAM but introduces tiny numeric differences vs float64.
- Chunked aggregation needs a final merge step so square/time groups split
  across chunk boundaries are combined correctly.
- Optimized RSS “after” can rise once both strategy aggregates are retained;
  peak-during-chunks is the fairer processing footprint metric.
- Full 62-day processing is validated in Phase 3A (see below); reuse the same
  day-by-day chunked pattern for later targeted extractions.

## Implementation entry points

- `src/data/loading.py` — discovery, single-member extract, `read_traffic_file`,
  timestamps, memory helpers, `discover_unique_daily_members`
- `src/data/preprocessing.py` — country summaries, Internet aggregation,
  `build_square_traffic_summary` / `process_full_dataset`
- `notebooks/mobile_traffic_forecasting.ipynb` — Phase 2 and Phase 3A walkthrough

---

# Phase 3A — Full-Dataset Processing Strategy

Measured run: `results/metrics/phase3_dataset_summary.json` and
`results/metrics/phase3_processing_log.csv`.

## Why the raw dataset is processed sequentially

The complete release is **62** daily TXT members across **4** ZIP archives
(~20.8 GB uncompressed). Processing **one day at a time** (extract → chunk →
aggregate → delete temp → next day) keeps peak memory near a single-day
footprint and provides a recoverable per-day log.

## Why chunked loading is necessary

Each day has millions of country-level rows. Chunked `pandas.read_csv` with
`chunksize=500_000`, Internet-task columns only, and explicit dtypes reuses the
Phase 2 approach that cut retained table memory by ~89% on a prototype day.

## Why the full square × timestamp dataset is not materialised

A complete 10-minute panel for all squares would be on the order of
**62 × 10,000 × 144 ≈ 89.3 million** cells. Phase 3A only needs:

1. per-square totals (to rank areas), and
2. daily per-square totals (for temporal summaries),

so later Phase 3B can extract 10-minute series **only** for selected squares
(top 3, 4159, 4556).

## Why `sum_all_countries` is used

Primary aggregation remains **sum across all country codes** per
`(square_id, time_interval)`, matching the Phase 2 methodological decision and
the official schema (Internet activity is defined per country code within a
square/interval). Country 39 is not used as the primary filter.

## Why daily square totals are stored

`daily_square_internet_totals.csv` (~620k rows) supports calendar-level checks
and plots without re-reading raw ZIPs. It is far smaller than a full 10-minute
panel.

## Why targeted extraction will happen later

Phase 3B will re-scan ZIPs (or selected days) to build 10-minute Internet series
only for the squares required by the research question. That second pass avoids
storing tens of millions of unused square-time rows now.

## Missing-observation policy

Absent square × timestamp combinations were **not** filled with zeros.
`observation_count` records how many aggregated square-time bins contributed to
each total. Interpreting absence as zero vs missing is deferred to Phase 3B.

## Memory-management strategy

- Discover ZIPs via configurable `DATA_DIR` / `MILAN_DATA_DIR` (no personal Drive hard-code in library defaults)
- Extract one TXT member to `tmp/phase3a_extract/`
- Chunked read + per-chunk aggregation + day merge
- Update cumulative dicts / daily frames
- Delete temporary TXT immediately
- `gc.collect()` between days
- Track process RSS with `psutil`

### Observed Phase 3A run

| Metric | Value |
|--------|--------|
| Daily files processed | **62 / 62** |
| Date range | 2013-11-01 → 2014-01-01 |
| Raw rows | **319,896,289** |
| Aggregated square×time rows | **89,127,473** |
| Unique squares | **10,000** |
| Unique timestamps | **8,928** (= 62 × 144) |
| Country codes observed | **364** |
| Total Internet traffic (`sum_all_countries`) | **≈ 5.552894 × 10⁹** |
| Missing/invalid key-field rows | **0** |
| Processing time | **≈ 422 s (~7.0 min)** |
| Peak process RSS | **≈ 134 MB** |
| Top 3 squares | **5161, 5059, 5259** |

Validation highlights: continuous dates, no duplicate days, cumulative square
totals equal sum of daily totals (abs diff 0), all days `schema_ok`, all days
per-file unique timestamps on a 10-minute grid, and global unique timestamps
also consecutive at 600,000 ms.

## Limitations and trade-offs

- Summaries use float64 accumulation of float32-parsed Internet values; tiny
  numeric differences vs a pure float64 raw parse are possible but totals are
  internally consistent (daily sum ≡ square sum).
- Daily output has **619,724** rows (not a full 62×10,000 grid) because
  square-days with no contributing Internet observations are omitted — no
  zero-filling.
- Country-code share for code 39 was quantified on one prototype day in Phase 2;
  Phase 3A tracks the count of distinct codes (364) but does not recompute
  per-code Internet shares for every day.
- Peak RSS ~134 MB is for summary construction only; later 10-minute extraction
  for selected squares will add its own memory profile.

---

# Phase 3B — Targeted Extraction and Exploratory Analysis

## Why only five squares were extracted at 10-minute resolution

Phase 3A ranked all 10,000 squares. Forecasting comparisons need detailed
series only for:

- Top 3 by total Internet: **5161, 5059, 5259**
- Assignment-required: **4159, 4556**

Extracting 10-minute series for every square would recreate ~89M rows of storage
without helping the research question.

## Why the full 89-million-row panel was not loaded

Phase 3B reuses Phase 3A compact files for distribution/ranking EDA and performs
a **targeted second pass** over the ZIPs that keeps only the five square IDs
inside each chunk before aggregation.

## How targeted extraction works

`extract_target_square_timeseries()` in `src/data/preprocessing.py`:

1. Discovers the 62 unique daily members
2. Extracts one TXT at a time
3. Chunk-reads Internet-task columns with optimized dtypes
4. Filters to `TARGET_SQUARES` immediately
5. Aggregates with **`sum_all_countries`** to `(square_id, time_interval)`
6. Appends day results; deletes the temporary TXT
7. Writes `data/processed/target_square_internet_timeseries.csv` (+ parquet)

Measured extraction: **44,640** rows (5 × 8,928), ~327 s, peak RSS ~83 MB.

## Missingness handling

Missing square×timestamp bins were **not** filled with zeros. Against the full
10-minute grid spanning the series, each target square has **0** missing bins
(8,928 / 8,928 observed; 144 observations on every filename day).

## Aggregation method and temporal resolution

- Aggregation: `sum_all_countries`
- Resolution: 10 minutes
- Columns: `timestamp`, `date`, `square_id`, `internet_traffic`, `observation_count`

## Why EDA informs model selection

EDA quantified skewness across squares, area-specific means/CVs, strong
short-term and daily/weekly autocorrelation for 5161, hour/weekday seasonality,
and rare extreme spikes. These evidence items define a **model-characteristic
shortlist** (persistence, seasonality, nonlinearity/peaks, interpretable
baseline). Final three models are **not** selected in Phase 3B.

Key artefacts: `results/metrics/phase3b_*.json`,
`results/metrics/target_square_descriptive_statistics.csv`,
`results/figures/eda_*.png`, notebook section Phase 3B.

---

# Phase 5 — Forecasting Experiment Design and Data Preparation

This section specifies the one-step-ahead experiment for the locked models
(SARIMA, LSTM, TCN). It is a **design and data-preparation** stage. Models are
not fitted, hyperparameters are not tuned, and no model ranking is claimed.

Configuration is centralised in `src/config.py`. Preparation code lives in
`src/data/forecasting.py`. Metrics are in `src/evaluation/metrics.py`.

## Locked models and baseline

| Role | Name |
|------|------|
| Research model 1 | SARIMA |
| Research model 2 | LSTM |
| Research model 3 | TCN |
| Evaluation reference only | Naive persistence baseline |

The naive baseline is **not** treated as a fourth sequential research model. It
exists so later test errors can be compared with a transparent persistence rule.

Squares **4159** and **4556** remain in the Phase 3B EDA extract. They are **not**
substituted into the forecasting experiment. The three primary areas are the
Phase 3A highest-total-traffic squares: **5161**, **5059**, and **5259**.

## One-step-ahead definition

The target is `internet_traffic` at **10-minute** resolution.

One-step-ahead means: given observations available through time \(t\), predict
Internet traffic at \(t + 10\) minutes.

\[
\hat{y}_{t+1} = f(y_{\leq t})
\]

where one step equals 10 minutes. The forecast for a target timestamp must not
use that timestamp’s actual value or any later observation.

Worked boundary (UTC timestamps in the processed series):

- input window ends at **2013-12-15 23:50**
- target is **2013-12-16 00:00**

This is the first test target. It uses history that would have been available at
the end of 15 December, including the validation period.

Evaluation is one-step-ahead with **observed** lags (not recursive multi-step
feedback of predictions). Later test points may therefore include earlier
**actual** test observations in the input window, because those values are known
at time \(t\).

## Why chronological splitting is used

The observations are an ordered temporal process with strong lag-1
autocorrelation and daily/weekly structure (Phase 3B). A random shuffle of
timestamps would place future bins next to past bins in the training set and
would allow the scaler, SARIMA likelihood, and neural windows to see temporal
patterns that would not be available in a real forecast issued at time \(t\).
Random splitting is therefore inappropriate for this task.

Splits are contiguous UTC intervals on the `timestamp` column (not the filename
`date` column). Filename dates follow the Milan source-file calendar and are
offset by one hour from UTC midnight; forecasting uses the UTC clock already
stored on each row.

## Why 16–22 December is reserved for final testing

The assignment requires a final evaluation window of **2013-12-16 through
2013-12-22 inclusive**. That week is held out from fitting and from
model/hyperparameter selection.

Using the last available calendar week of November–December as test would mix
Christmas-week behaviour into the official test definition. Dates after
2013-12-22 exist in the extract (through 2014-01-01) but are **not** used in
this experiment, so the prescribed test week remains untouched.

## Chronological train / validation / test

Inclusive UTC bounds:

| Split | Start | End | Expected rows per square |
|-------|-------|-----|--------------------------|
| Train | 2013-11-01 00:00 | 2013-12-08 23:50 | 38 × 144 = **5472** |
| Validation | 2013-12-09 00:00 | 2013-12-15 23:50 | 7 × 144 = **1008** |
| Test | 2013-12-16 00:00 | 2013-12-22 23:50 | 7 × 144 = **1008** |

Training and validation lie strictly before the test period. Validation is the
week immediately preceding test, so configuration selection (later) uses a
recent, seasonally comparable window without touching the final week.

Rows before 2013-11-01 00:00 UTC (six bins on 2013-10-31 23:00–23:50) and rows
after 2013-12-22 23:50 UTC are excluded from the forecasting table. They are not
zero-filled and are not moved into another split.

## Data integrity and missingness

Phase 3B found complete 10-minute coverage for the five EDA squares. Phase 5
re-checks each forecasting square for:

- 10-minute frequency
- no duplicate timestamps
- no missing timestamps relative to the split grids
- no unexpected `square_id` values
- chronological order
- expected row counts
- no test timestamps in train or validation

If a check fails, preparation **stops**. Missing bins are reported; they are not
interpolated, forward-filled, or replaced with zeros. The policy matches the
earlier EDA decision that absence must not be silently treated as zero traffic.

The tidy output is a single file,
`data/processed/forecasting_target_squares.csv`, with columns `timestamp`,
`square_id`, `internet_traffic`, `split`. One file avoids duplicating the same
series three times.

## LSTM and TCN sequence construction

Neural models use a sliding window on the univariate series:

\[
X = [y_{t-L+1},\ldots,y_{t-1},y_t], \qquad \text{target}=y_{t+1}.
\]

The sample’s split is the split of the **target** timestamp. Inputs may come
from earlier splits so that the first validation and test targets still have a
full history of length \(L\). That is required for the 2013-12-15 23:50 →
2013-12-16 00:00 example. Building windows independently inside the test split
would drop the first \(L\) test points and would break that alignment.

Leakage rules enforced in code:

- a training target cannot have validation or test values in \(X\)
- a validation target cannot have test values in \(X\)
- \(X\) never includes \(y_{t+1}\) or later
- \(y_{t+1}\) is exactly 10 minutes after the last input

LSTM and TCN use the **same** \(L\) so that differences in later test error are
not confounded by different look-back lengths.

## Sequence-length selection

Candidates examined (no training):

| \(L\) | Horizon covered | Daily lag 144 in window | Weekly lag 1008 in window | Train samples / square | Relative LSTM unroll vs \(L=144\) |
|------:|-----------------|-------------------------|---------------------------|------------------------|-----------------------------------|
| 144 | 24 hours | yes | no | 5328 | 1× |
| 288 | 48 hours | yes | no | 5184 | 2× |
| 1008 | 7 days | yes | yes | 4464 | 7× |

**Primary choice: \(L = 144\)** for both LSTM and TCN.

Reasoning:

1. **EDA.** Lag-1 autocorrelation on square 5161 is ~0.99, so the most recent
   bins dominate a 10-minute forecast. Lag 144 (~0.88) is the main seasonal
   structure at this resolution. With \(L=144\), the first input is
   \(y_{t-143}\), which is the daily seasonal counterpart of the target
   \(y_{t+1}\).
2. **Forecasting objective.** The task is one-step-ahead, not long-horizon
   recursive forecasting. Extra history beyond one daily cycle adds weaker
   incremental information relative to lag 1 and lag 144.
3. **Computation.** LSTM unrolling cost scales with \(L\). \(L=1008\) is a
   seven-fold sequential depth versus \(L=144\), with only a modest change in
   the number of training windows. TCN compute also grows with input length.
   Window tensors remain small in all three cases (~3–17 MB float32 per square
   for training \(X\)), so memory is not the binding constraint; LSTM depth is.
4. **Temporal structure.** \(L=144\) captures short-term dependence and one
   daily cycle. It does **not** include last week’s same-time bin. That is a
   documented limitation, not a silent omission.
5. **Consistency.** The same \(L\) is used for LSTM and TCN. There is no
   methodological reason in this phase to give TCN a longer window than LSTM.

\(L=1008\) remains a possible later sensitivity check. It is not the primary
specification, because weekly SARIMA at \(s=1008\) is already rejected as
impractical (below), and because one-step-ahead LSTM training at length 1008 is
disproportionately expensive relative to the evidence for weekly lags.

## Normalization (LSTM and TCN)

Each square has its own scaler. The scaler is **fitted on training observations
only**, then applied to training, validation, and test values of that square.

This is a leakage-prevention measure. Fitting on the full series would let
validation/test extrema define the scale used during training. Min–max scaling
to \([0,1]\) is used because Internet activity is non-negative; the training
minimum and maximum of each square become the scale anchors
(`results/metrics/phase5_scaler_params.json`).

Validation or test values may fall outside \([0,1]\) after transform if they
exceed the training range. That is accepted; the test maximum is not used to
respan the scaler.

SARIMA is fitted on the original-scale univariate training series and does not
use this neural scaler.

**Predictions from LSTM/TCN must be inverse-transformed before MAE, RMSE, and
MAPE.** Metrics are defined on the original traffic scale so that the three
research models remain comparable to each other and to the naive baseline.

The forecasting CSV stores **unscaled** `internet_traffic`. Scaled arrays are
produced at train time from the train-only scaler.

## SARIMA data representation

SARIMA does not use the sliding-window matrix. It receives a single
chronological univariate series per square, at 10-minute frequency, restricted
to the training split for fitting. Validation is reserved for later order /
configuration selection. The test week remains unused until the final
evaluation.

### Seasonal period

Phase 3B ACF supports both a daily period (\(s=144\)) and a weekly period
(\(s=1008\)).

**Recommended SARIMA seasonal period: \(s = 144\).**

- Statistically, lag 144 matches the 24-hour cycle at this resolution.
- Practically, \(s=144\) is already large for classical SARIMA, but it is the
  smallest seasonal period that encodes daily seasonality without aggregating
  away the 10-minute step.
- \(s=1008\) is not used. Seasonal differencing of order 1008 discards a full
  week; the 38-day training window contains only about 5.4 weekly cycles, which
  is thin for weekly seasonal ARMA; and likelihood estimation at lag 1008 is
  computationally heavy.
- A **single** SARIMA model cannot carry two seasonal periods (144 and 1008)
  without leaving the standard SARIMA class. Dual-seasonal extensions are
  therefore out of scope for the locked model list.

Seasonal AR/MA orders will be kept small when fitting begins. No SARIMA model
is estimated in Phase 5.

## Naive persistence baseline

\[
\hat{y}_{t+1} = y_t
\]

Named **Naive persistence baseline**. It uses only the last observed 10-minute
value. Given lag-1 ACF near 0.99, it is a strong reference: any selected model
must beat persistence by enough to justify added complexity. It is prepared as
a function (`naive_persistence_forecast`) and is not counted among SARIMA,
LSTM, and TCN.

## Metrics and MAPE handling

MAE, RMSE, and MAPE will be computed on the original scale after any
inverse-transform. MAE and RMSE always use **every** test observation.

MAPE is unstable if actual traffic is zero or extremely small, because the
percentage denominator vanishes. Phase 5 inspected the three forecasting
squares over train, validation, and test:

- no NaNs
- no exact zeros
- no values below the MAPE numerical floor
- minima on the experiment window are well above 1 activity unit
  (square 5161 train min ≈ 93; 5059 ≈ 179; 5259 ≈ 115)

**MAPE rule:** the denominator is \(\max(|y_t|, \varepsilon)\) with
\(\varepsilon = 10^{-8}\). Observations are **not** dropped. The floor is a
numerical safeguard; on these series it does not change the value of MAPE
because all \(|y_t| \gg \varepsilon\). Diagnostic counts (`n_zero`,
`n_below_floor`) are stored so any later zero would be visible rather than
silently omitted.

## Leakage prevention (summary)

| Risk | Control |
|------|---------|
| Future timestamps in training | Chronological splits; test starts 2013-12-16 00:00 UTC |
| Test rows in train/validation tables | Integrity checks; overlapping timestamps raise an error |
| Scaler using future extrema | Per-square scaler fit on train only |
| Window including the target | \(X\) ends at \(t\); target is \(t+1\) |
| Train windows using validation/test | Enforced when building sequences |
| Random shuffle | Not used |
| Zero-fill creating fake observations | Not used; failures are reported |
| Fitting on the test week | Test unused until the final experiment |

## Outputs of Phase 5

- `data/processed/forecasting_target_squares.csv`
- `results/metrics/phase5_split_validation.csv`
- `results/metrics/phase5_sequence_length_analysis.csv`
- `results/metrics/phase5_zero_traffic_inspection.csv`
- `results/metrics/phase5_scaler_params.json`
- `results/metrics/phase5_experiment_config.json`
- `results/metrics/phase5_alignment_example.csv`

No model checkpoints are written. Raw ZIP/TXT files are not added to Git.

## What Phase 5 does not do

Phase 5 does not train SARIMA, LSTM, or TCN; does not tune hyperparameters;
does not generate final test predictions for the research models; and does not
produce a model-comparison table. Those steps follow only after this design has
been reviewed.

---

# Phase 6A — Model Implementation

Phase 6A implements the locked research models and the naive reference baseline.
It does **not** evaluate the reserved test week (2013-12-16 through 2013-12-22),
does not select hyperparameters on the test set, and does not rank models.

Implementation modules:

| Module | Role |
|--------|------|
| `src/models/naive.py` | Naive persistence baseline (reference only) |
| `src/models/sarima.py` | SARIMA (statsmodels SARIMAX) |
| `src/models/lstm.py` | LSTM (TensorFlow / Keras) |
| `src/models/tcn.py` | Causal dilated residual TCN (TensorFlow / Keras) |
| `src/models/training.py` | Shared neural training / inverse-transform helpers |
| `src/models/base.py` | Timing, alignment assertions, shape helpers |

Defaults live in `src/config.py`. Smoke tests:
`scripts/run_phase6a_smoke_tests.py` → `results/metrics/phase6a_smoke_tests.json`.

## Why three distinct forecasting approaches

The three research models represent different sequential-modelling families for the
same one-step-ahead task:

1. **SARIMA** — linear Gaussian state-space model with explicit daily seasonal
   structure at lag 144.
2. **LSTM** — nonlinear recurrent network that compresses a length-144 history
   through gated memory cells.
3. **TCN** — nonlinear convolutional network that covers the history with
   *causal* dilated filters and residual blocks, without recurrence.

Holding the target, squares, splits, sequence length, and (for neural models)
scaling fixed isolates family differences from data-protocol differences.

## Naive persistence baseline

\[
\hat{y}_{t+1} = y_t
\]

Implemented in `NaivePersistenceModel`. Predictions are aligned to the target
timestamp \(t+1\) and equal the observed value at \(t\). This is **not** one of
the three research models; it is only a transparent reference for later
evaluation phases.

## SARIMA

**Formulation.** Seasonal ARIMA on the univariate Internet-traffic series:

\[
\phi(B)\,\Phi(B^{s})\,(1-B)^{d}\,(1-B^{s})^{D}\, y_t
=
\theta(B)\,\Theta(B^{s})\,\varepsilon_t
\]

with seasonal period \(s = 144\) (24 hours at 10-minute resolution).

**Why \(s=144\).** Phase 3B/5 ACF shows a strong daily peak at lag 144. The
weekly lag 1008 is visible but impractical as a primary SARIMA seasonal period
(thin weekly sample in 38 training days; heavy seasonal state). Dual seasonality
is outside the standard single-period SARIMA class used here.

**Input scale.** Original (unscaled) traffic units. The Phase 5 MinMaxScaler is
**not** applied to SARIMA.

**Configuration.** Non-seasonal order \((p,d,q)\) and seasonal order
\((P,D,Q,144)\) are configurable. Defaults:

- \((p,d,q) = (1,0,1)\)
- \((P,D,Q,s) = (1,0,1,144)\)

A deliberately small candidate set is stored in `src/config.py` for later
**train/validation** selection (Phase 6B). The test week is not used for order
choice. Fitting uses statsmodels `SARIMAX.fit` (MLE, default method `lbfgs`).
Convergence failures raise a clean `RuntimeError` rather than returning a silent
partial fit.

## LSTM

**Architecture (default).**

```text
Input (144, 1)
  → LSTM(32)
  → Dropout(0.1)
  → Dense(1)
```

**Sequence length.** \(L = 144\), matching Phase 5. Input tensors have shape
`(samples, 144, 1)`; the output is one scalar next-step prediction per sample.

**Learning.** Nonlinear recurrent sequence modelling of short-term and daily
structure present in the window. Training uses MSE loss and Adam.

**Scaling.** Train-only MinMaxScaler from Phase 5. The LSTM **does not refit**
the scaler. Predictions are inverse-transformed to original traffic units before
any later metric calculation.

**Defaults.** 32 units, 1 layer, dropout 0.1, learning rate \(10^{-3}\), batch
size 64, up to 20 epochs, early stopping on **validation** loss only (patience
5). Random seed 42 where practical.

## TCN

**Architecture (default).** A genuine Temporal Convolutional Network, not a
plain CNN:

```text
Input (144, 1)
  → Causal dilated residual block (dilation 1)
  → Causal dilated residual block (dilation 2)
  → Causal dilated residual block (dilation 4)
  → Causal dilated residual block (dilation 8)
  → Last-timestep representation
  → Dense(1)
```

Each residual block uses left-padded causal convolutions (no future frames),
ReLU, spatial dropout, and a residual / projection path.

**Why this counts as a TCN.** Causality, dilation, and residual connections are
the defining temporal-convolution design choices used here. Dilations
\((1,2,4,8)\) with kernel size 3 yield a receptive field of
\(1 + (3-1)(1+2+4+8) = 31\) steps inside the shared 144-step input window.
Extending the dilation schedule further is possible later; the Phase 6A default
stays compact for a fair comparison rather than maximum capacity.

**Sequence length and scaling.** Same as LSTM: \(L=144\), train-only MinMax
scaling, inverse-transform before original-scale metrics.

**Defaults.** 32 filters, kernel size 3, dilations (1,2,4,8), dropout 0.1,
learning rate \(10^{-3}\), batch size 64, up to 20 epochs, early stopping on
validation loss only.

## Fairness between LSTM and TCN

Shared experimental protocol:

- same target (`internet_traffic`)
- same squares (5161, 5059, 5259)
- same train / validation / test periods (unchanged from Phase 5)
- same sequence length \(L=144\)
- same train-only MinMax normalisation
- same one-step-ahead horizon (10 minutes)

Architectural differences (recurrent vs causal-dilated residual) are intentional
family differences, not unequal data access.

## Leakage safeguards in the implementations

- Fit methods can assert that training / validation target timestamps do not
  enter the reserved test period (`assert_no_test_in_fit_window`).
- Neural early stopping monitors validation loss only; test loss is never an
  early-stopping signal.
- Scalers are injected from Phase 5 and are not refit inside model code.
- Sequence construction reuses Phase 5 utilities; one-step alignment
  (including 2013-12-15 23:50 → 2013-12-16 00:00) is asserted in smoke tests.
- Timing records fit and predict wall-clock separately for SARIMA, LSTM, and TCN.

## What Phase 6A does not do

Phase 6A does not run final Dec 16–22 evaluation, does not produce comparison
tables, and does not claim that any model performs better than another. Full
training on the complete training split and validation-based configuration
selection are left to subsequent phases.

---

# Phase 6C — Final Test Evaluation

Phase 6C consumes the **locked** configurations from Phase 6B, refits each
model under the Phase 5 protocol, and scores the reserved test week
(**2013-12-16 through 2013-12-22**) once.

It does **not** search hyperparameters on the test set. Cross-family ranking on
test metrics is for reporting only; configuration choice already happened on
validation in Phase 6B.

## Inputs

| Artefact | Role |
|----------|------|
| `data/processed/forecasting_target_squares.csv` | Phase 5 forecasting frame |
| `results/metrics/phase6b_configurations.json` (or Drive `locked_configs.json`) | Locked SARIMA / LSTM / TCN configs |

Default locked picks from the completed Phase 6B Colab run (also in
`PHASE6C_DEFAULT_LOCKED`):

| Family | ID | Configuration |
|--------|----|---------------|
| SARIMA | A | \((1,0,1)\times(1,0,1,144)\) |
| LSTM | B | 64 units, 1 layer, dropout 0.1, lr \(10^{-3}\) |
| TCN | D | filters 32, kernel 5, dilations \((1,2,4,8,16)\), lr \(5\cdot10^{-4}\) |

## Fitting protocol

| Model | Fit | Predict |
|-------|-----|---------|
| SARIMA | Full training split (5472 bins), original scale, `SARIMA_MAXITER` | One-step via Kalman `extend` through validation + test; score test only |
| LSTM / TCN | Train sequences; early stopping on validation loss; train-only MinMax | One-step test windows with observed history; inverse-transform before metrics |
| Naive | Persistence reference | \(\hat y_{t+1}=y_t\) on test (observed lags) |

## Outputs

Written under `results/metrics/`:

- `phase6c_square_results.csv` — per-square MAE / RMSE / MAPE / nMAE / nRMSE
- `phase6c_summary.csv` — means across squares
- `phase6c_comparison.json` — locked configs + ranking
- `phase6c_test_report.md` — human-readable report
- `phase6c_predictions/` — per-model per-square prediction CSVs

Colab notebook: `notebooks/phase6c_final_evaluation.ipynb`  
Local runner: `python scripts/run_phase6c_evaluation.py`

## What Phase 6C does not do

Phase 6C does not re-open the Phase 6B candidate set, does not tune on the test
week, and does not change the locked architectures. Any model ranking is a
**post-hoc** comparison of already-locked configs on held-out data.

## Observed Colab results (Dec 16–22)

Extracted from the completed Colab run into [`docs/phase6c_results.md`](phase6c_results.md)
and `results/metrics/phase6c_*`.

**Ranking by mean normalised MAE (nMAE = MAE / train-mean traffic):**

| Model | mean MAE | mean RMSE | mean MAPE | mean nMAE |
|-------|----------|-----------|-----------|-----------|
| **SARIMA-A** | **73.3** | **105.7** | **7.98%** | **0.0528** |
| Naive persistence | 83.3 | 119.7 | 8.47% | 0.0601 |
| LSTM-B | 91.3 | 126.0 | 12.16% | 0.0654 |
| TCN-D | 181.7 | 264.7 | 17.80% | 0.1279 |

Interpretation (brief):

- **SARIMA** is the only research model that clearly beats Naive on the held-out
  week, consistent with strong daily seasonality (\(s=144\)) seen in Phase 3B/5.
- **LSTM** is competitive but worse than Naive in aggregate — useful as a
  nonlinear baseline, not the preferred forecaster here.
- **TCN** is weakest, especially on square 5161 (nMAE ≈ 0.19). The locked
  dilation schedule has receptive field 249 steps while the input window is
  \(L=144\), so much of the theoretical RF is unused; Christmas-week traffic
  may also hurt a model that underfit the daily cycle on validation.
- Per-square ordering is stable for SARIMA (best on all three squares); TCN
  variance across squares is large.

Full-train SARIMA fits took on the order of 20–28 minutes per square
(`maxiter=50`) in the Colab run.

