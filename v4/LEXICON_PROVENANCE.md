# Lexicon Provenance Audit

**Date:** 2026-08-07  
**Question:** Can the pre-corpus-validation “seed” lexicon be recovered exactly from repository provenance, so that lexicon-expansion ablation A3 can be computed deterministically?

## Artefacts inspected (read-only)

| Path | Type | What was examined |
|------|------|-------------------|
| `V1/annotation_scheme_v3.xlsx` | spreadsheet | Sheet `Tier1_Shortlist` — column D header = “Tier 2 lexicon (tools & synonyms — extend during pilot)” |
| `v2/annotation_scheme_v3.xlsx` | spreadsheet | Same structure as V1 (identical sheet names) |
| `v3/manual_work/annotation_scheme_v3.xlsx` | spreadsheet | Same `Tier1_Shortlist` header, identical to V1/v2 |
| `V1/filter_corpus.py`, `v2/filter_corpus.py`, `v3/manual_work/filter_corpus.py` | python | No lexicon definitions — only corpus title filters |
| `v3/base-model/config.py` | python | First committed Python `LEXICONS` (13-category, corpus-validated) |
| `v4/config.py` | python | Frozen copy of the above (`v3-13cat-frozen`) |
| `git log --all -p -- "*/config.py"` | git history | Only two commits introduce `LEXICONS`: `dd7c0b7` (initial v3) and `4207363` (frozen v4 copy) — no earlier seed commit |
| `V1/pilot_annotation_completed.xlsx`, `v2/gold_standard_annotation_workbook.xlsx`, `v3/manual_work/gold_standard_annotation_workbook_v2.xlsx` | spreadsheets | No lexicon lists |
| `docs/RESEARCH_PLAN_v1_5.6_PRO.md` | doc | Mentions “seed lists from ESCO/O*NET, extended during corpus validation” but does not enumerate the seed |

## Evidence

- `v3/base-model/config.py` docstring: “Seed lists from ESCO/O*NET, **extended during corpus validation and annotation**.”
- `V1/annotation_scheme_v3.xlsx` / `Tier1_Shortlist` column D header: “Tier 2 lexicon (tools & synonyms — **extend during pilot**)” — this sheet is the only historical lexicon artefact.
- `Tier1_Shortlist` rows (V1) contain 13 rows but **category set does not match the frozen 13**:

  | V1 Tier1_Shortlist category label | Maps to `v4` `CATEGORIES` key | Frozen `CATEGORIES` present in V1? |
  |---|---|---|
  | programming | `programming` | yes |
  | database querying (SQL) | `sql` | yes |
  | data visualisation / BI tools | `visualisation_bi` | yes |
  | reporting / dashboarding | `reporting` | yes |
  | spreadsheets (Excel) | `excel` | yes |
  | statistical analysis | `statistics` | yes |
  | machine learning | `machine_learning` | yes |
  | data cleaning / preparation | `data_cleaning` | yes |
  | data engineering / ETL | `etl` | yes |
  | data modelling | `data_modelling` | yes |
  | cloud platforms | `cloud` | yes |
  | **big data** | — (merged/removed) | **no — category eliminated before v3** |
  | **business intelligence / analytics** | — (maps vaguely to `stakeholder_comm`/`reporting`) | **no — not a 1:1 mapping** |
  | — | `stakeholder_comm` | **missing in V1 (0 terms)** |
  | — | `ethics_governance` | **missing in V1 (0 terms)** |

- Term counts (case-insensitive distinct terms):

  | Category | V1 shortlist terms | Frozen `v4` terms | Naïve “added” (frozen – V1) would include |
  |---|---|---|---|
  | programming | 11 | 16 | would imply +8 / –3 (`go`,`r`,`ruby` removed) |
  | sql | 14 | 23 | +10 / –1 (`oracle` → `oracle database`) |
  | visualisation_bi | 10 | 15 | +5 |
  | reporting | 7 | 12 | +6 / –1 (`dashboards` removed) |
  | excel | 7 | 12 | +5 |
  | statistics | 10 | 19 | +9 |
  | machine_learning | 9 | 16 | +8 / –1 (`ai` → `artificial intelligence`) |
  | data_cleaning | 6 | 11 | +5 |
  | etl | 12 | 21 | +9 |
  | data_modelling | 7 | 12 | +5 |
  | cloud | 10 | 15 | +7 / –2 (`cloud`,`microsoft azure` removed) |
  | stakeholder_comm | 0 | 14 | +14 (no seed) |
  | ethics_governance | 0 | 6 | +6 (no seed) |

  (Full term listings compared in `python3` interactive inspection; see commit history for reproducibility.)

- No committed Python file prior to `v3/base-model/config.py` contains a lexicon. V1/v2 have no `config.py`. The only lexicon artefact before `v3` is the spreadsheet column described above.

## Analysis

- The `Tier1_Shortlist` header **explicitly marks** the listed terms as pre-expansion (“extend during pilot”), so it is a plausible *candidate* seed.
- However it is **not** a reproducible seed for the frozen 13-category system because:
  1. Two frozen categories (`stakeholder_comm`, `ethics_governance`) have **zero terms** in the shortlist — they were introduced/renamed after V1.
  2. Two shortlist categories (`big data`, `business intelligence / analytics`) were **eliminated/merged** and do not map 1:1 to any frozen key without invented judgement.
  3. Even for retained categories, seed terms were altered non-additively (e.g. `oracle` → `oracle database`, `cloud` (bare) removed, `ai` → `artificial intelligence`), so “added terms = frozen – seed” would conflate additions with deliberate lexical cleaning.
  4. The shortlist’s `Sources` column cites `O*NET DS`, `ESCO`, etc., but with truncated lists and notes like “Python/R missing from both O*NET exports (truncated hot-technology list)”, indicating it is a **summarised** extract, not a verbatim ESCO/O*NET export that could be re-derived today.

- No other artefact (git commit, spreadsheet sheet, doc) isolates an earlier Python `LEXICONS` that maps 1:1 to the frozen `CATEGORIES` keys and can be diffed deterministically.
- Going to ESCO/O*NET today to create a new list and pretending it was the original would violate the “do not fabricate” rule.

## Verdict

**AUTHENTIC SEED LEXICON NOT RECOVERABLE** as an exact, reproducible pre-corpus-validation lexicon for the frozen 13-category `v3-13cat-frozen` system.

- **Evidence inspected:** listed above (V1/v2/v3 annotation schemes, filter scripts, config history, research-plan docs).
- **Closest historical artefact:** `V1/annotation_scheme_v3.xlsx` / `Tier1_Shortlist` column D — but categories and term sets are not congruent with the frozen 13, so a deterministic `seed → final` diff is not computable without invented mappings.
- **Authentic seed commit/path:** none exists for the 13-category Python `LEXICONS`.
- **Deterministic diff:** not computable.

## Consequence for ablation

- Ablation stage **A3 — lexicon expansion is NOT IDENTIFIABLE FROM AVAILABLE PROVENANCE** and will be recorded as `NOT IDENTIFIABLE` / omitted, not fabricated.
- The ablation will follow **CASE B** (reduced sequence): A0 (naive substring, final frozen lexicon), A1 (whole-word), A2 (whole-word + negative patterns), A3 omitted, A4 (nested threshold tuning, unweighted), A5 (nested IDF weighting).
- A4/A5 designations are **kept** (not relabelled) so the audit trail is honest; A2→A4 transition is explicitly noted as combining **no lexicon change** (same final frozen lexicon).

## Reproducibility note

Inspection commands used (read-only) are logged in the commit that introduces this file’s companion code. Re-running the comparison is:

```bash
PYTHONPATH=. python3 -c "
import openpyxl, sys; sys.path.insert(0,'.')
from v4.config import LEXICONS
wb=openpyxl.load_workbook('V1/annotation_scheme_v3.xlsx', data_only=True)
ws=wb['Tier1_Shortlist']
# then compare as in audit script
"
```
