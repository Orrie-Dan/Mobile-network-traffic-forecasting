# Milan Telecommunications Dataset

## 1. Dataset Source

This project uses **Telecommunications – SMS, Call, Internet – MI**, published by Telecom Italia on [Harvard Dataverse](https://doi.org/10.7910/DVN/EGZHFV).

The dataset measures telecommunication activity over the city of Milan, derived from anonymized Call Detail Records (CDRs) on Telecom Italia’s cellular network. Activity values are temporally aggregated into **10-minute** intervals and spatially aggregated onto the Milan grid (approx. 235 × 235 m squares; 10,000 cells).

**Official documentation (not bundled in the local ZIP downloads):**

| Resource | Role |
|----------|------|
| [Harvard Dataverse DOI: 10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV) | Dataset landing page; points to full column description |
| [Barlacchi et al., *Scientific Data* (2015)](https://doi.org/10.1038/sdata.2015.55) | Peer-reviewed description of fields, aggregation, and CDR types |

**Local availability:** The four ZIP archives inspected for Phase 1 contain **only** daily `.txt` data files. No README, codebook, or metadata files were present inside the ZIPs or under this repository’s `data/` tree. Schema meanings below therefore cite the external Dataverse / *Scientific Data* documentation, cross-checked against small local samples.

---

## 2. File Organization

### ZIP batches inspected

Four archives were located on the local machine (Downloads). Together they form the complete Milan SMS/Call/Internet daily series used for this project:

| ZIP archive | Compressed size | Uncompressed (declared in ZIP) | TXT files |
|-------------|-----------------|--------------------------------|-----------|
| `dataverse_files.zip` | ~0.89 GB | ~3.46 GB | 10 |
| `dataverse_files (1).zip` | ~2.18 GB | ~8.47 GB | 25 |
| `dataverse_files (2).zip` | ~0.96 GB | ~3.70 GB | 12 |
| `dataverse_files (4).zip` | ~1.33 GB | ~5.18 GB | 15 |
| **Total** | **~5.37 GB** | **~20.8 GB** | **62** |

Notes:

- There is **no** `dataverse_files (3).zip` in the inspected Downloads folder; the unnumbered `dataverse_files.zip` plus batches `(1)`, `(2)`, and `(4)` already cover **62 unique** daily files with **no duplicate filenames**.
- Project notes cite ~19.4 GB for the full dataset; local ZIP central-directory totals sum to ~20.8 GB uncompressed. Treat ~19–21 GB as the practical full-extract footprint. **Do not commit these archives to Git.**

### Number of data files

**62** tab-separated text files (one calendar day each). Matches the expected full Milan activity release.

### File naming pattern

```text
sms-call-internet-mi-YYYY-MM-DD.txt
```

Examples: `sms-call-internet-mi-2013-11-01.txt`, `sms-call-internet-mi-2014-01-01.txt`.

### Approximate date coverage

| Item | Value |
|------|--------|
| Earliest filename date | 2013-11-01 |
| Latest filename date | 2014-01-01 |
| Unique dates | 62 |
| Gaps in the filename calendar | **None** (continuous day sequence) |

ZIP → date span (by filename):

- `dataverse_files.zip`: 2013-11-01 … 2013-11-10  
- `dataverse_files (4).zip`: 2013-11-11 … 2013-11-25  
- `dataverse_files (1).zip`: 2013-11-26 … 2013-12-20  
- `dataverse_files (2).zip`: 2013-12-21 … 2014-01-01  

### File size observations

- Individual daily TXT files are typically on the order of **~280–380 MB uncompressed** inside the ZIP.
- Compressed member sizes are roughly **~72–95 MB** each.
- Files must be streamed or partially read; loading many full days into memory is not appropriate for Phase 1.

### Inspection paths used

```text
C:\Users\user\Downloads\dataverse_files.zip
C:\Users\user\Downloads\dataverse_files (1).zip
C:\Users\user\Downloads\dataverse_files (2).zip
C:\Users\user\Downloads\dataverse_files (4).zip
```

Colab / Drive setups must point `src.data.loading` helpers at the corresponding external paths; defaults in code match this local layout only.

---

## 3. Raw Data Format

| Property | Finding | Confidence |
|----------|---------|------------|
| Delimiter | Tab (`\t`) | **Confirmed** on samples from three files / three ZIPs |
| Headers | **None** (headerless) | **Confirmed** — first rows are numeric fields, not names |
| Number of columns | **8** per row | **Confirmed** on all sampled lines |
| Encoding | UTF-8-compatible ASCII numeric text | **Confirmed** for sampled bytes |
| Row unit | One record per (square, time interval, country code) | **Inferred** from documentation + sample structure |

### Example structure (schematic)

```text
<square_id>\t<time_ms>\t<country_code>\t<sms_in>\t<sms_out>\t<call_in>\t<call_out>\t<internet>
```

Empty fields appear as consecutive tabs (missing activity measures). Trailing empty Internet fields leave a final empty column before end-of-line.

Representative files sampled (first kilobytes only; no full-file load):

1. `sms-call-internet-mi-2013-11-01.txt` (`dataverse_files.zip`)
2. `sms-call-internet-mi-2013-11-15.txt` (`dataverse_files (4).zip`)
3. `sms-call-internet-mi-2014-01-01.txt` (`dataverse_files (2).zip`)

---

## 4. Column Schema

Official field definitions come from Barlacchi et al. (*Scientific Data*, 2015) / Dataverse. **File column order** is not printed as an index table in every secondary summary; the order below is the standard order used for these TSV files and is **strongly consistent** with local samples (e.g. column 3 dominated by `39`, Italy’s country calling code).

| Raw Column | Proposed Name | Data Type (observed) | Meaning | Confidence/Evidence |
|------------|---------------|----------------------|---------|---------------------|
| 1 | `square_id` | integer | Geographical square ID on the Milan GRID | **High / verified meaning** (docs) + **confirmed** integer IDs in samples (`1`, `10`, `100`, …) |
| 2 | `time_interval` | integer (large) | Start of the 10-minute interval as Unix time in **milliseconds**; end = start + 600,000 ms | **High / verified meaning** (docs) + **confirmed** convertible timestamps in samples |
| 3 | `country_code` | integer | Phone country code for the activity nation (roaming context) | **High / verified meaning** (docs) + **strong sample support** (`39` most frequent; also `0`, `33`, `41`, …) |
| 4 | `sms_in` | float or empty | Activity proportional to received SMSs in the square/interval for that country code | **Verified meaning** (docs); type/missingness **confirmed** in samples |
| 5 | `sms_out` | float or empty | Activity proportional to sent SMSs | **Verified meaning** (docs); type/missingness **confirmed** in samples |
| 6 | `call_in` | float or empty | Activity proportional to received calls | **Verified meaning** (docs); type/missingness **confirmed** in samples |
| 7 | `call_out` | float or empty | Activity proportional to issued calls | **Verified meaning** (docs); type/missingness **confirmed** in samples |
| 8 | `internet_traffic` | float or empty | Internet traffic activity (Internet-related CDRs in the square/interval for that country code) | **Verified meaning** (docs); presence as 8th numeric field **confirmed** in samples |

### Separation of fact vs inference

- **Verified (documentation):** Meanings of square id, time interval (+10 min end), country code, SMS-in/out, call-in/out, and Internet traffic activity; 10-minute aggregation; CDR basis.
- **Confirmed (local inspection):** 8 tab-separated headerless columns; dtypes as above; empty strings for missing activity cells; timestamps decode as ms epoch; 600,000 ms spacing on inspected squares.
- **Strongly supported inference:** Column **order** in the TSV matches the table above (country code in column 3; Internet in column 8). Secondary sources (e.g. community loaders citing the same schema) agree. The *Scientific Data* prose lists the same fields; local values (especially `39` in column 3) align with that ordering.
- **Unknown / not yet measured locally:** Exact presence of all 10,000 squares every day; full-file null rates; whether country code `0` has a documented special meaning beyond “observed in data.”

---

## 5. Timestamp Analysis

| Check | Result |
|-------|--------|
| Format | Integer Unix epoch **milliseconds** |
| Example | `1383260400000` → `2013-10-31 23:00:00+00:00` (UTC) |
| Relation to filename date | Filename date is calendar day in **local (CET/CEST) context**; UTC midnight-adjacent timestamps are expected (Italy UTC+1 in November–December) |
| Interval | Adjacent unique timestamps for a given square are **600,000 ms = 10 minutes** |
| Consistency with 10-minute data | **Yes** — matches official documentation and sample diffs |

Additional sample notes:

- For square `1` on `2013-11-01`, 144 unique timestamps were observed in the sampled head of the file, with **all** adjacent diffs equal to 600,000 ms (a full day of 10-minute bins).
- On `2014-01-01`, unique timestamps across the sampled head also stepped by 600,000 ms overall. For some squares, the early-file sample may not yet include every bin (file is ordered by square id), so sparse per-square coverage in a byte-limited peek is not evidence of irregular sampling.

---

## 6. Missing Values

| Aspect | Finding |
|--------|---------|
| Representation | Empty fields (consecutive tabs); **not** `NA` / `NaN` / `null` text tokens in the inspected samples |
| Columns always populated (samples) | Columns 1–3 (`square_id`, `time_interval`, `country_code`) |
| Columns with empties (samples) | Columns 4–8 (SMS, call, and Internet activity) |
| Pattern | Missingness looks **activity-specific**: a row may have only some of SMS/call/Internet filled for a given country code. Country code `0` rows often have many empty activity fields; country `39` rows more often have denser Internet/SMS/call values |

**No imputation, dropping, or filling was performed.** Full-dataset missingness rates are **not** estimated here.

---

## 7. Schema Consistency

Compared representatives:

| File | ZIP | Columns | Delimiter | Header | Timestamp style | Missing style |
|------|-----|---------|-----------|--------|-----------------|---------------|
| `…-2013-11-01.txt` | `dataverse_files.zip` | 8 | tab | none | ms epoch | empty fields |
| `…-2013-11-15.txt` | `dataverse_files (4).zip` | 8 | tab | none | ms epoch | empty fields |
| `…-2014-01-01.txt` | `dataverse_files (2).zip` | 8 | tab | none | ms epoch | empty fields |

**Conclusion from samples:** Column count, delimiter, headerlessness, timestamp format, and missing-value representation are **consistent** across the inspected files. No schema drift was detected in Phase 1 peeks. Unaudited full-file row counts / rare malformed lines remain a residual risk for later pipeline validation (streaming checks), not a Phase 1 contradiction.

---

## 8. Internet Traffic Variable

**Status: identified with documentation-backed confidence.**

- **Column:** raw column **8** (1-based) / index **7** (0-based)  
- **Proposed name:** `internet_traffic`  
- **Official meaning:** Internet traffic activity — Internet-related CDRs generated inside a given square during the time interval, associated with the nation identified by `country_code` (Barlacchi et al., 2015; Dataverse description).  
- **CDR note (docs):** An Internet CDR is generated when a connection starts or ends; additional CDRs may be generated if a connection lasts >15 minutes or transfers >5 MB. Shared activity values are scaled by a Telecom Italia constant (true counts are obfuscated).

**Why this is safe to use for forecasting design (with one caveat):**

1. Official schema names an Internet traffic activity field among the eight activity-file attributes.  
2. Local files have exactly eight columns; community loaders place Internet last.  
3. Sample values in column 8 are non-negative floats when present, often larger for country code `39`, consistent with an Internet activity measure.

**Caveat before preprocessing:** Forecasting usually needs **one series per square and time**. Raw rows are further split by `country_code`. Phase 2 must decide how to aggregate Internet across country codes (commonly sum over codes, or retain only `39`) — that aggregation rule is **not** chosen in Phase 1.

---

## 9. Key Findings for the Next Phase

### What we know

- Complete local coverage appears available as **4 ZIPs → 62 daily TXT files**, naming `sms-call-internet-mi-YYYY-MM-DD.txt`, dates **2013-11-01 through 2014-01-01**.
- Raw format is stable: **headerless, tab-separated, 8 columns**.
- Schema aligns with the official Milan telecommunications activity description; **Internet traffic is column 8**.
- Timestamps are **ms Unix time** on a **10-minute** grid.
- Missing activity cells are **empty strings**; do not assume zeros without an explicit rule.

### What remains uncertain / deferred

- Whether country-code Internet shares remain ~99.7% on days other than the Phase 2 prototype (especially holidays).
- How to treat absent square×timestamp combinations (zero vs missing) — deferred to Phase 3B.
- 10-minute series for selected squares (top 3, 4159, 4556) — targeted extraction in Phase 3B.
- No local copy of the *Scientific Data* PDF inside the repo; cite DOI externally.

### Phase status

- **Phase 1:** schema verification — complete.
- **Phase 2:** efficient loading / memory management on one day — complete.
- **Phase 3A:** full 62-day compact summaries — complete (see below).
- **Phase 3B:** targeted extraction + EDA for five squares — complete.
- **Next:** model shortlist review / selection (not started).

---

## Phase 3A — Complete-Dataset Processing Findings

Full sequential processing was executed against all unique daily members discovered
in the four Dataverse ZIP archives. Metrics: `results/metrics/phase3_dataset_summary.json`.

| Finding | Measured value |
|---------|----------------|
| ZIP archives | 4 |
| Daily files processed | 62 / 62 |
| Date range | 2013-11-01 → 2014-01-01 |
| Dates continuous / duplicates | continuous; no duplicate dates |
| Raw rows processed | 319,896,289 |
| Aggregated square×time rows | 89,127,473 |
| Unique squares | 10,000 |
| Unique timestamps | 8,928 (10-minute grid throughout) |
| Country codes observed | 364 |
| Aggregation method | `sum_all_countries` |
| Total Internet traffic | ≈ 5.552894×10⁹ |
| Missing/invalid key-field rows | 0 |
| Processing time | ≈ 422 s |
| Peak process RSS | ≈ 134 MB |
| Daily square-total rows written | 619,724 |

### Top 10 squares by total Internet traffic

| Rank | square_id | total_internet_traffic | observation_count |
|------|-----------|------------------------|-------------------|
| 1 | 5161 | ≈ 1.2740×10⁷ | 8928 |
| 2 | 5059 | ≈ 1.1171×10⁷ | 8928 |
| 3 | 5259 | ≈ 1.0486×10⁷ | 8928 |
| 4 | 5061 | ≈ 9.584×10⁶ | 8928 |
| 5 | 5258 | ≈ 8.707×10⁶ | 8928 |
| 6 | 5159 | ≈ 8.704×10⁶ | 8928 |
| 7 | 6064 | ≈ 8.675×10⁶ | 8928 |
| 8 | 4855 | ≈ 8.491×10⁶ | 8928 |
| 9 | 4856 | ≈ 8.230×10⁶ | 8928 |
| 10 | 5262 | ≈ 8.163×10⁶ | 8928 |

**Top 3 for Phase 3B (by total Internet):** 5161, 5059, 5259.

### Outputs produced

- `data/processed/square_internet_totals.csv` (10,000 rows, sorted by traffic desc)
- `data/processed/daily_square_internet_totals.csv`
- `results/metrics/phase3_processing_log.csv` (62 rows)
- `results/metrics/phase3_dataset_summary.json`

### Validation summary

- All 62 files processed; no ZIP/file failures.
- Cumulative per-square totals equal the sum of daily totals (abs diff 0).
- Per-day schema checks passed; per-day unique timestamps on 10-minute spacing.
- Global unique timestamps also form a continuous 10-minute sequence (8,928 bins).
- No silent dropping of malformed key-field rows (count = 0).
- Missing square×timestamp combinations were **not** zero-filled.

---

## Country Code Aggregation Decision

### Evidence (prototype day: 2013-11-01)

Measured with the Phase 2 chunked reader on `sms-call-internet-mi-2013-11-01.txt`:

| Finding | Value |
|---------|--------|
| Unique `country_code` values | 246 |
| Missing `country_code` | 0 |
| Internet traffic share for code **39** (Italy) | **~99.72%** |
| Internet traffic share for code **0** | ~2.5×10⁻⁷ (negligible) |
| Combined share of all non-39 codes | **~0.28%** |
| Top non-39 codes by Internet sum | 33, 46, 49, 44, 41, … |

Official documentation (Barlacchi et al., 2015; Dataverse): each activity field, including Internet traffic, is associated with the phone **country code** of the users generating CDRs in that square and time interval. Rows are therefore square × time × country, not already collapsed to square × time.

### Strategies compared

| Strategy | Definition | Prototype Internet sum share vs total |
|----------|------------|----------------------------------------|
| A. `sum_all_countries` | Sum `internet_traffic` over all country codes per `(square_id, time_interval)` | 100% (reference) |
| B. `country_39_only` | Keep `country_code == 39`, then aggregate | ~99.72% of A |

Aggregated shapes were nearly identical (1,439,982 vs 1,439,981 square–time rows).

### Preliminary recommendation

**Use `sum_all_countries` as the default forecasting target construction.**

Justification:

1. **Schema-aligned:** total Internet activity in a geographic cell over 10 minutes is the sum of the documented per-country Internet measures.
2. **Non-39 traffic is real:** ~0.28% is small but not zero; dropping it is a modelling choice, not a schema requirement.
3. **Not chosen because totals are larger:** the recommendation follows the definition of cell-level activity, not maximising the numeric sum.
4. **Sensitivity:** `country_39_only` remains useful for robustness checks because it is nearly identical on this day.

This decision was applied as the **primary aggregation** for the completed Phase 3A
full-dataset summaries (`sum_all_countries`). Country 39 remains available later as
a sensitivity option only.

---

## Phase 3B — Targeted Extraction and EDA Findings

### Extracted series

| Item | Value |
|------|--------|
| Target squares | 5161, 5059, 5259, 4159, 4556 |
| Rows | 44,640 (= 5 × 8,928) |
| Rows per square | 8,928 each |
| Timestamp range (UTC) | 2013-10-31 23:00 → 2014-01-01 22:50 |
| First-two-weeks window (filename dates) | 2013-11-01 → 2013-11-14 |
| Aggregation | `sum_all_countries` |
| Zero-filled missing bins | No |
| Output | `data/processed/target_square_internet_timeseries.csv` (+ parquet) |

### Missingness (target squares)

Against the complete 10-minute grid: **0 missing timestamps** for every target square
(min/max/avg observations per filename day = **144**).

### Descriptive statistics (10-minute Internet traffic)

| square_id | mean | median | std | CV | min | max |
|-----------|------|--------|-----|----|-----|-----|
| 5161 | ≈1427 | ≈859 | ≈1382 | ≈0.97 | ≈56 | ≈8044 |
| 5059 | ≈1251 | ≈901 | ≈961 | ≈0.77 | ≈109 | ≈4666 |
| 5259 | ≈1174 | ≈654 | ≈1103 | ≈0.94 | ≈73 | ≈4442 |
| 4556 | ≈512 | ≈501 | ≈248 | ≈0.48 | ≈80 | ≈1869 |
| 4159 | ≈275 | ≈206 | ≈181 | ≈0.66 | ≈47 | ≈941 |

### Square 5161 ACF (selected lags)

| Lag | ACF |
|-----|-----|
| 10 minutes (1) | ≈0.987 |
| 1 hour (6) | ≈0.939 |
| 6 hours (36) | ≈0.010 |
| 12 hours (72) | ≈−0.683 |
| 1 day (144) | ≈0.878 |
| 1 week (1008) | ≈0.838 |

### Seasonality (5161, Europe/Rome)

- Peak hour ≈ **16**; low hour ≈ **4**
- Peak weekday ≈ **Saturday**; low weekday ≈ **Wednesday**
  (observed averages; causal interpretation not claimed)

### Notable unusual period (5161)

Robust spike flagged near **2013-12-27 15:30 UTC** (Internet ≈ 3999; robust-z ≈ 6.2).
Cause unknown from the dataset alone.

### Artefacts

Figures under `results/figures/eda_*.png`; metrics under
`results/metrics/phase3b_*.json` and `target_square_descriptive_statistics.csv`.

---

## Verification Summary

### Confirmed

- Four local ZIP archives; **62** unique daily TXT members; continuous filename dates 2013-11-01 … 2014-01-01.  
- No README/codebook inside the ZIPs.  
- Files are headerless and tab-separated with **8** fields.  
- Columns 1–3 always populated in samples; 4–8 may be empty.  
- Column 2 values decode as Unix timestamps in milliseconds; observed spacing **600,000 ms**.  
- Schema meanings for the eight fields are documented by Dataverse / Barlacchi et al. (2015).

### Inferred

- TSV **column order** is square → time → country → SMS-in → SMS-out → call-in → call-out → Internet (strongly supported by docs + sample country codes).  
- Filename dates reflect local calendar days; UTC timestamps near 23:00 previous day are expected under CET.  
- Empty activity cells mean “no recorded activity measure for that field,” not necessarily numeric zero (treatment still to be decided).

### Still Unknown

- Whether country-code Internet shares remain ~99.7% on other days (especially holidays such as 2014-01-01).  
- How absent bins should be treated for squares **outside** the five targets (targets show 0% missing on the full grid).  
- Presence/location of ZIP batch `(3)` elsewhere (not required; four archives already yielded 62 unique days).
- Final three forecasting models (deferred until after Phase 3B review).