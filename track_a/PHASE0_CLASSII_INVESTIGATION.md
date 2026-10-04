# Phase 0 — Class-II AS splits investigation

**Date**: 2026-05-04
**Question**: AS-Class-II only has `as_{trb,paired,tra}_ii_eval.parquet` on disk. Is this design or a generation-pipeline gap?

**Verdict**: **(a) Design.** AS-Class-II is intentionally eval-only — no `_train`, `_val`, or `_test_<condition>` files exist for class II by spec. This is consistent across the docs, the build script's code path, and the on-disk inventory.

## Evidence trail

### 1. `docs/BENCHMARK_SPLITS.md` task-taxonomy table (line 54)

```
| AS  | Given TCR+MHC, predict peptide (retrieval) |
|     | Class I:  TRB-I, TRA-I, Paired-I — full splits |
|     | Class II: TRB-II, TRA-II, Paired-II — eval-only |
```

(Same table, same wording, in `docs/wiki/benchmark_v2/BENCHMARK_SPLITS.md`.)

### 2. `docs/BENCHMARK_SPLITS.md` § 8 "Files emitted" (line 210)

The AS task explicitly emits:
- 8 Class-I partitions per benchmark (`train, val, test_iid, test_novel_tcr, test_novel_pep, test_novel_allele, test_level4, test_mixed`)
- exactly **one** Class-II file per benchmark: `as_trb_ii_eval.parquet`, `as_tra_ii_eval.parquet`, `as_paired_ii_eval.parquet`

### 3. `scripts/data_processing/build_benchmark_splits.py` — class-II code path

```python
# line 594
def _as_class_ii_one(name: str, subset_keys: List[str]) -> Dict[str, int]:
    print(f"[as] === {name} (eval-only) ===", flush=True)
    ...
    out = OUT_ROOT / f"{name}_eval.parquet"
    pq.write_table(table, str(out), compression="zstd")
```

The function reads enriched class-II rows, attaches a `row_id`, and writes a single eval parquet. There is no clustering, no novelty-condition labeling, and no train/val/test_iid/test_novel_* stratification. Compare with `_as_class_i_one` (line 470+) which does the full 8-way partition.

### 4. On-disk inventory matches

`benchmark_v2/splits/split_summary.json` lists exactly:
- `as_paired_ii_eval.parquet` — 151,632 rows
- `as_trb_ii_eval.parquet`    — 56,076 rows
- `as_tra_ii_eval.parquet`    — 40,806 rows

No additional class-II files. `docs/wiki/benchmark_v2/inventory.json` reports the same canonical counts.

### 5. Why the design

AS-Class-II rows likely were too small or too unbalanced to support the 8-way novelty stratification used for class I (clustering ratios + held-out alleles assume `>=50` examples per allele etc., per BENCHMARK_SPLITS.md §4 step 6). Class II is therefore positioned as a transfer/zero-shot eval against the class-I-trained model.

## Consequences for the launch matrix

For every "retrained" baseline that has a class-II row in the user's matrix:
- **Train on class-I data** (e.g., `as_trb_i_train.parquet`)
- **Evaluate on class-II eval** (`as_trb_ii_eval.parquet`)
- Report a single line of metrics for class II — no per-condition fan-out (no IID / novel_tcr / novel_pep / novel_allele / level4 columns to evaluate against)

The user-prompt's "4 conditions for Class II" expectation should be downgraded to **1 condition (`eval`)** for class II in the per-task results JSON.

For the original/zero-shot arms (TULIP-original, optionally PanPep-style baselines), class II is identical: forward-pass on the eval file.

The class-II eval file is also flat — there's no embedded `condition` column to stratify post-hoc (`_as_class_ii_one` doesn't add one). If we want a finer-grained class-II breakdown for the paper, we would need to re-cluster the eval rows post-hoc; out of scope for the 75h window.

## Schema note

Class-II eval files share most of the enriched-parquet schema (TARGET_COLUMNS minus the train-time `tcr_cluster`, `pep_cluster`, `condition` columns). `track_a/shared/load_splits.py` will distinguish class I (8 conditions) from class II (1 condition = `eval`) and return a normalized DataFrame either way.

## Cross-reference

Same eval-only treatment applies to **MR Class II** (`mr_tra_ii_eval.parquet`, `mr_paired_ii_eval.parquet`) — see line 1054 of build_benchmark_splits.py: `[mr]   wrote {out.name}: {n:,} rows (eval-only)`. (`mr_trb_ii` is allowed a 2-condition split if there are ≥5 alleles with ≥20 examples; otherwise it falls back to eval-only.) Out of scope for Track A.

## Action items implied for Track A

1. `track_a/shared/load_splits.py` exposes both `load_as_class_i(task, condition)` and `load_as_class_ii(task)` helpers; the latter returns a single DataFrame.
2. Per-baseline `evaluate.py`: class I emits 5 condition entries per task; class II emits 1 entry (`eval`) per task.
3. `results_schema.py` accepts a `conditions` dict where class II uses `{"eval": {...}}` and class I uses `{"iid": {...}, "novel_tcr": {...}, "novel_pep": {...}, "novel_allele": {...}, "level4": {...}}`.
4. The user's per-task output JSON spec is preserved; the class-II case just has fewer keys.

**No regeneration of class-II splits is needed or desirable.** Proceed with current data.
