# Formative findings — sequential models for Milan traffic forecasting

## Research question

How do different sequential models compare for one-step-ahead mobile network
traffic forecasting, and how does their performance vary across geographical
areas with different traffic characteristics?

## Experiment snapshot

| Item | Choice |
|------|--------|
| Target | `internet_traffic`, 10-minute bins |
| Squares | 5161, 5059, 5259 (Phase 3A highest-total-traffic) |
| Horizon | One-step-ahead with observed lags |
| Train / val / test | Nov 1–Dec 8 / Dec 9–15 / **Dec 16–22** (held out until Phase 6C) |
| Models | SARIMA, LSTM, TCN (+ Naive persistence reference) |
| Selection | Phase 6B on validation → locked **SARIMA-A**, **LSTM-B**, **TCN-D** |
| Final score | Phase 6C full-train (SARIMA) / train+val early-stop (neural) → test once |

Details: `docs/methodology.md`, `docs/phase6c_results.md`.

## Main result

On the reserved test week, **SARIMA-A** is the best research model by mean
normalised MAE and is the only research model that beats Naive:

| Model | mean nMAE | vs Naive |
|-------|-----------|----------|
| SARIMA-A | **0.0528** | better |
| Naive | 0.0601 | — |
| LSTM-B | 0.0654 | worse |
| TCN-D | 0.1279 | much worse |

Absolute errors (mean over squares): SARIMA MAE ≈ 73, Naive ≈ 83, LSTM ≈ 91,
TCN ≈ 182 (original traffic units).

## Variation across squares

- **SARIMA** wins on every square (5161, 5059, 5259) with nMAE in a tight band
  (~0.052–0.054).
- **Naive** is a strong baseline everywhere (nMAE ~0.058–0.062).
- **LSTM** is close to Naive on 5259 but clearly worse on 5161 (higher MAPE).
- **TCN** fails badly on **5161** (nMAE 0.191) and is merely mediocre on 5059/5259.
  Geographical “difficulty” shows up most for the convolutional model.

So performance **does** vary by area, but the **ranking of SARIMA ≻ Naive ≻ LSTM ≻ TCN**
is stable in aggregate; the main square-specific story is TCN’s collapse on 5161.

## Why these outcomes (discussion points)

1. **Daily seasonality** at lag 144 is strong (Phase 3B/5 ACF). Classical SARIMA
   with \(s=144\) encodes that structure directly; full-train refit in 6C (unlike
   the 7-day / `maxiter=15` selection fit in 6B) recovers a competitive filter.
2. **Naive persistence** exploits lag-1 autocorrelation. Beating it is non-trivial
   at 10-minute resolution; SARIMA’s edge is modest but consistent.
3. **LSTM** can model nonlinearity but, with \(L=144\) and a small locked
   architecture, did not beat persistence on this Christmas-adjacent week.
4. **TCN** locked config has RF = 249 > SEQ_LEN = 144, so part of the dilation
   stack cannot be used effectively from the input window. Combined with weaker
   validation behaviour in 6B, poor test general is expected rather than surprising.

## Limitations

- Single hold-out week (includes pre-Christmas behaviour); no rolling re-test.
- One-step-ahead only; no multi-step recursive evaluation.
- Three high-traffic squares only — not a city-wide sample.
- Neural results are seed-sensitive; reported run uses seed 42.
- SARIMA full-train MLE is expensive (~1+ hour total for three squares).

## Artefacts

| Path | Content |
|------|---------|
| `docs/phase6c_results.md` | Tables committed for the report |
| `results/metrics/phase6c_*` | Local extract from Colab (may be gitignored) |
| Drive `milan_traffic/results/phase6c/` | Plots + CSVs from the Colab run |
| `notebooks/phase6b_candidate_selection.ipynb` | Validation locking |
| `notebooks/phase6c_final_evaluation.ipynb` | Final test evaluation |

## Suggested next writing steps

- Expand literature links (why SARIMA / LSTM / TCN were chosen).
- Insert 2–3 test-week plots from Drive (one square × SARIMA vs Naive).
- Short conclusion answering the RQ in two sentences.
