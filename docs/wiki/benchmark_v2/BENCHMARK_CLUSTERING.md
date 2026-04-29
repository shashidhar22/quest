# Benchmark Clustering

Clustering artifacts that feed `benchmark_v2` split generation. Every CDR3 and
peptide in the corpus is mapped to a cluster id so that train/test splits can
be stratified on cluster membership and leakage across splits is prevented.

Artifacts live at `data/benchmark_v2/clusters/` and are produced by
`scripts/data_processing/build_benchmark_clusters.py`.

---

## 1. Purpose & downstream consumers

Benchmark splits for TCR-pMHC modeling must avoid leaking similar sequences
across train / val / test. The community convention is to cluster the
distinguishing short sequence (TCR CDR3 or peptide) at a sequence-identity
threshold and treat each cluster as an atomic unit — all members go to the
same split.

Downstream split scripts load a `*_clusters.tsv` (two columns: `sequence`,
`cluster_id`), join it against the per-row CDR3 or peptide column, and use
`cluster_id` as the stratification key.

---

## 2. Source dataset

`data/deduplicated_again/exploded_deduped_enriched/`

- Hive-partitioned by `subset_key` and `order_key`
- **31 subset_keys, 639 parquet files, 45 GB total**
- Schema (20 columns): `tra_full, trb_full, peptide, mhc_one, mhc_two,
  tra_cdr1-3, trb_cdr1-3, sequence, subset_key, order_key,
  mhc_{one,two}_{pocket,contact,pocket_contact}`
- CDR3 columns contain nulls on rows where the chain is absent
- **`mhc_one` / `mhc_two` hold full ~250 AA protein sequences, not HLA allele
  names.** The allele-name columns (`mhc_one_allele`, `mhc_two_allele`) exist
  only upstream at `data/standardized_again/` and are dropped during
  deduplication. See §7 for how to recover allele names if needed.

---

## 3. Scope definitions

A `subset_key` records which molecular components were present in the original
row — e.g., `trb_peptide_mhc_one` means "TRB chain + peptide + Class I MHC."
Clustering is computed over four scopes:

| Scope | Predicate on subset_key | # subset_keys | Purpose |
|-------|--------------------------|----------------|---------|
| `interaction` | contains `tra` or `trb` **and** `peptide` | 12 | TCR-peptide binding records — the core of contrastive / cross-encoder / seq2seq training |
| `restriction` | contains `tra` or `trb` **and** (`mhc_one` or `mhc_two`) | 18 | Broader set of TCR-MHC restriction records (with or without peptide) |
| `pm` | contains `peptide` **and** (`mhc_one` or `mhc_two`) | 12 | Peptide-MHC context set (pMHC modeling inputs) |
| `all` | contains the relevant component at all | 30+ | Foundation clusters used for MLM / TCR-only pretraining scopes |

`interaction` is **not** necessarily a subset of `restriction`: interaction
includes peptide-only TCR rows (e.g., `tra_peptide`) that do not carry an MHC
component. The two scopes overlap heavily but differ at their edges. See the
per-row `subset_key` in source data for exact membership.

---

## 4. Output artifacts

All files land in `data/benchmark_v2/clusters/`.

### Unique sequence files (Task 1 output — one sequence per line)

| File | Contents | Typical size |
|------|----------|--------------|
| `unique_trb_cdr3_interaction.txt` | unique TRB CDR3s from `interaction` rows | ~10⁵–10⁶ |
| `unique_trb_cdr3_restriction.txt` | unique TRB CDR3s from `restriction` rows | ~10⁵–10⁶ |
| `unique_trb_cdr3_all.txt` | every unique TRB CDR3 in the corpus | ~10⁸ |
| `unique_tra_cdr3_interaction.txt` | unique TRA CDR3s from `interaction` rows | ~10⁵ |
| `unique_tra_cdr3_restriction.txt` | unique TRA CDR3s from `restriction` rows | ~10⁵ |
| `unique_tra_cdr3_all.txt` | every unique TRA CDR3 in the corpus | ~10⁷ |
| `unique_peptides_interaction.txt` | unique peptides from `interaction` rows | ~10⁴ |
| `unique_peptides_pm.txt` | unique peptides from `pm` rows | ~10⁶ |
| `unique_mhc_one_alleles.txt` | unique **Class I MHC protein sequences** | ~10⁴ |
| `unique_mhc_two_alleles.txt` | unique **Class II MHC protein sequences** | ~10³ |
| `sequence_counts.json` | integer count per file above | — |

The `_alleles` files are named per the original task spec, but their
**contents are unique protein sequences**, not HLA identifiers.

### Cluster TSV files (Tasks 2 & 3 output)

All cluster TSVs share the format `sequence\tcluster_id` (peptide TSVs use
`peptide\tcluster_id`). `cluster_id` is the representative sequence string of
each cluster.

| File | Method | Identity | Scope |
|------|--------|----------|-------|
| `trb_cdr3_interaction_clusters.tsv` | MMseqs2 easy-cluster | 90% | interaction |
| `tra_cdr3_interaction_clusters.tsv` | MMseqs2 easy-cluster | 90% | interaction |
| `trb_cdr3_restriction_clusters.tsv` | MMseqs2 easy-cluster | 90% | restriction |
| `tra_cdr3_restriction_clusters.tsv` | MMseqs2 easy-cluster | 90% | restriction |
| `trb_cdr3_all_clusters.tsv` | MMseqs2 easy-cluster | 90% | all — **multi-hour background job** |
| `tra_cdr3_all_clusters.tsv` | MMseqs2 easy-cluster | 90% | all |
| `peptide_clusters_interaction.tsv` | rapidfuzz Levenshtein + single-linkage DSU | `d ≤ ⌈0.2·min(len)⌉` | interaction |
| `peptide_clusters_pm.tsv` | **MMseqs2 easy-cluster** (see note below) | 80% | pm |
| `peptide_clusters_mmseqs2.tsv` | MMseqs2 easy-cluster | 80% | interaction (appendix comparison) |

**Note on PM peptides**: the PM scope in the current corpus contains ~4.7M
unique peptides (dominated by NetMHCpan training data and IEDB pMHC
records). Full all-pairs Levenshtein at that scale is ~11 trillion pairs
and infeasible even with rapidfuzz + 64 cores. The PM clusters are
therefore produced by MMseqs2 at 80% identity rather than by edit
distance. Interaction peptides remain edit-distance clustered and the
appendix file (``peptide_clusters_mmseqs2.tsv``) preserves the ED vs
MMseqs2 methods comparison at interaction scale.

### JSON inventories (Task 4 output)

- `mhc_one_alleles.json` — `{ "count": N, "alleles": [...sorted Class I protein sequences...] }`
- `mhc_two_alleles.json` — same for Class II

### Summary

- `clustering_summary.json` — counts, cluster size distributions, top-10
  largest clusters, peptide-method agreement stats, timings, TRB-foundation
  status
- `timings.json` — wall-clock seconds per task
- `peptide_clustering_agreement.json` — Levenshtein vs MMseqs2 agreement on
  interaction peptides

---

## 5. Clustering methods & rationale

### TCR CDR3: MMseqs2 easy-cluster, 90% identity, 80% coverage, `--cluster-mode 1`

```bash
mmseqs easy-cluster input.fasta out_prefix tmp \
  --min-seq-id 0.9 -c 0.8 --cluster-mode 1 --threads 64
```

- **90% identity**: CDR3s are short (10–20 AA). Small substitutions often
  change specificity, so a conservative threshold is appropriate. 90% is the
  de-facto convention for TCR specificity splits in the literature.
- **80% coverage**: MMseqs2 requires mutual coverage of alignment span; 80%
  avoids clustering completely unrelated short fragments.
- **`--cluster-mode 1`**: connected-component (single-linkage) clustering —
  more inclusive than greedy set-cover (mode 0); consistent with the pairwise
  similarity interpretation used downstream.
- **`easy-cluster` vs multi-step `cluster`**: `easy-cluster` is a single
  command that wraps `createdb → cluster → createtsv` with sensible defaults
  and produces the `{prefix}_cluster.tsv` directly. There is no accuracy
  difference from the step-wise path in `scripts/data_processing/create_tcr_specificity_dataset.py`.

### Peptide (interaction): rapidfuzz Levenshtein + single-linkage DSU

```python
from rapidfuzz.distance import Levenshtein
threshold = math.ceil(0.2 * min(len(a), len(b)))
d = Levenshtein.distance(a, b, score_cutoff=threshold)
if d <= threshold:
    union_find.union(i, j)
```

- **Why not MMseqs2 for peptides?** A reviewer flagged that MMseqs2's k-mer
  seeding can miss near-duplicate short peptides (8–11 AA) because short
  sequences do not generate enough k-mer hits to pass the seeding threshold.
  All-pairs Levenshtein avoids that blind spot.
- **Length-relative threshold `⌈0.2 × min(len)⌉`**: scales with peptide
  length — ≤2 substitutions on a 9-mer, ≤3 on a 15-mer, ≤4 on a 20-mer.
  Captures sequence-level near-duplicates without over-merging epitopes with
  shared anchor residues.
- **Single-linkage connected components**: A ~ B ~ C clusters A, B, C even if
  A ≢ C. This is transitive closure — matches the intended use of cluster as
  a split stratum (no member is isolated from its near-neighbors across
  splits).
- **Parallelization**: 64-worker `multiprocessing.Pool`, outer loop chunked
  by starting index. rapidfuzz's C-extension plus the `score_cutoff`
  short-circuit bail out of most pairs quickly.

### Peptide (PM): MMseqs2 (fallback for scale)

PM peptides are clustered with MMseqs2 `easy-cluster --min-seq-id 0.8 -c
0.8 --cluster-mode 1` because the ~4.7M unique PM peptides make full
all-pairs Levenshtein infeasible. 80% identity is a standard peptide
clustering threshold (used in e.g. pMHC eval split construction). The
deviation from ED-only is the scale of PM, not a methodological
preference — a future PR could implement k-mer-indexed or
length-bucketed ED to bring PM back under edit-distance clustering.

### Peptide method comparison (appendix)

`peptide_clusters_mmseqs2.tsv` runs MMseqs2 `easy-cluster --min-seq-id 0.8`
on the interaction peptides only. `peptide_clustering_agreement.json` reports
how often the two methods place pairs in the same / different clusters.

---

## 6. Scope boundary: CDR3 clustering ≠ full-chain clustering

The CDR3 clusters are **keyed on the CDR3 string**. They project onto any
longer representation of the same TCR (`tra_full`, `trb_full`,
CDR1+CDR2+CDR3 concatenations) via a simple join on the CDR3 column — every
row carrying that CDR3 inherits the cluster id.

**However, they do not equal full-chain identity clustering.** Two TCRs with
identical CDR3 but different V/J regions collapse into one CDR3 cluster; two
TCRs with nearly identical full chains but a single substitution in CDR3 sit
in different clusters. CDR3-level is the standard for specificity-leak
prevention in TCR benchmarks, which is why it is the scope of this work.

If you need full-protein-identity splits in addition (e.g., to prevent
leakage at the full chain level), add a separate clustering job on
`tra_full` / `trb_full` directly. It is **not** in scope here.

---

## 7. MHC representation

The `mhc_one` / `mhc_two` columns in `exploded_deduped_enriched` hold full
~250 AA protein sequences — not HLA allele names. Allele names are present
upstream at `data/standardized_again/` (see `mhc_one_allele`,
`mhc_two_allele`) and were dropped during deduplication because the
benchmark pipeline consumes sequences downstream (pocket / contact /
pseudosequence extraction all operate on the protein).

If you need allele-level IDs:

- `data/deduplicated_again/mhc_pseudo_lookup.json` keys protein sequence →
  pseudosequence components; not a direct name lookup but narrows candidates.
- `data/standardized_again/imgthla/` preserves `(allele_name, sequence)`
  pairs; reverse-lookup `sequence → allele_name` is exact at 4-digit
  resolution.
- `scripts/data_processing/standardize/imgthla.py` is the authoritative
  parser for the IMGT HLA FASTA (`hla_prot.fasta`).

The output files `unique_mhc_one_alleles.txt` / `unique_mhc_two_alleles.txt`
and `mhc_{one,two}_alleles.json` contain **protein sequences**, one per line
or element, sorted lexicographically.

---

## 8. Infrastructure

- **Host**: x2gd.16xlarge (aarch64, 64 vCPU, 995 GB RAM) — Graviton2
- **Ephemeral scratch**: two NVMe drives (1.7 TB each) mounted separately at
  `/scratch` and `/scratch2`. MMseqs2 temp dirs and intermediate FASTAs live
  on `/scratch`; the large TRB foundation clustering writes its temp DB to
  `/scratch2` to isolate it from the foreground jobs.
- **Persistent outputs**: `/home/ubuntu/quest/data/benchmark_v2/clusters/`
  on `/data` (6.5 TB free)
- **Mount setup** (one-time, needs sudo):
  ```bash
  sudo mkfs.xfs -f /dev/nvme1n1
  sudo mkfs.xfs -f /dev/nvme2n1
  sudo mkdir -p /scratch /scratch2
  sudo mount /dev/nvme1n1 /scratch
  sudo mount /dev/nvme2n1 /scratch2
  sudo chown ubuntu:ubuntu /scratch /scratch2
  ```

### Expected wall-clock

| Task | Expected time |
|------|---------------|
| Collect unique sequences (45 GB scan) | 10–30 min |
| CDR3 interaction/restriction clustering (×4) | <15 min each |
| Peptide edit-distance clustering | 10–30 min |
| Peptide MMseqs2 appendix | <5 min |
| TRA CDR3 all | 30–90 min |
| **TRB CDR3 all (foundation)** | **many hours — background** |

The TRB foundation job is launched in a detached shell; the driver continues
with other tasks and reports its status in the summary.

---

## 9. Reproducibility

### Versions (pinned via conda/pip at integration time)

- MMseqs2 18.8cc5c (bioconda)
- rapidfuzz 3.14.5 (pip, has aarch64 wheels)
- Python 3.13 via miniforge
- pyarrow (repo env)

### Commands

```bash
# Full pipeline
python scripts/data_processing/build_benchmark_clusters.py --task all

# Or per-task
python scripts/data_processing/build_benchmark_clusters.py --task collect
python scripts/data_processing/build_benchmark_clusters.py --task tcr_cluster
python scripts/data_processing/build_benchmark_clusters.py --task peptide_cluster
python scripts/data_processing/build_benchmark_clusters.py --task allele_inventory
python scripts/data_processing/build_benchmark_clusters.py --task summary
```

`tcr_cluster` launches the TRB-all clustering in the background, then runs
the foreground jobs. Re-running `tcr_cluster` skips re-launch if
`logs/trb_all.pid` already exists.

`summary` detects whether TRB-all has finished (via `_cluster.tsv` presence
and process-liveness check) and, if done, converts the raw MMseqs2 output
into the final `trb_cdr3_all_clusters.tsv`.

---

## 10. Consuming the artifacts

Downstream split generation joins a per-row CDR3 / peptide column against
the cluster TSV and stratifies on `cluster_id`:

```python
import pandas as pd
import pyarrow.dataset as ds

# Load CDR3 clusters
clusters = pd.read_csv(
    "data/benchmark_v2/clusters/trb_cdr3_interaction_clusters.tsv",
    sep="\t",
)
cdr3_to_cluster = dict(zip(clusters["sequence"], clusters["cluster_id"]))

# Per-row cluster id
rows = ds.dataset(
    "data/deduplicated_again/exploded_deduped_enriched/"
    "subset_key=trb_peptide_mhc_one",
    format="parquet",
).to_table(columns=["trb_cdr3", "peptide"]).to_pandas()
rows["trb_cluster"] = rows["trb_cdr3"].map(cdr3_to_cluster)

# Stratified GroupKFold on trb_cluster -> split definitions
```

For peptide clusters use `peptide` as the join column and
`peptide_clusters_interaction.tsv` (or `_pm.tsv`) as the source.

---

## 11. Related scripts

- `scripts/data_processing/create_tcr_specificity_dataset.py` — earlier
  MMseqs2 wrapper (`mmseqs cluster` rather than `easy-cluster`). Kept as
  reference; `build_benchmark_clusters.py` is the active driver.
- `scripts/data_processing/create_tcr_contrastive_dataset.py` — contrastive
  dataset build that also clusters at 90% identity.
- `scripts/mil/data/create_global_clusters.py` — unrelated embedding-based
  (Leiden / k-means) clustering for MIL; different pipeline.
