# Corpus repair step 10 — make the builder meet the sampling contract, then a small build

Opened 2026-11-10. Not a stage. The contract is frozen in ACS
`quest-cpt/audits/2026-11-09/repair/contract/sampling-contract.json` (`cpt-sampling-contract-v1`);
the step text and its amendments are in `corpus-repair-plan.md` beside it. Lightweight plan; no
box until step 6 below.

## Status 2026-11-10

Steps 1-5 done; step 6 done on the laptop, box half open; step 7 done (independent verifier and
Codex review, eight gaps found and fixed). Results: ACS `repair-status.md`, "Step 10, laptop
half". The first build exposed an error in the contract's residue budget; the contract is now
revision 2. Remaining: the box build against the real single-receptor partitions.

## Goal

A builder that produces either contract version from the same inputs and refuses when it cannot
meet a target, plus one small deterministic end-to-end build that proves it. Today it can only
sample rows per partition, labels MHC class from the allele name, and has never run under the
three new policies.

## Approach

- **One definition of an example.** The stratum rules, the example hash and the cap-then-count
  selection exist only in the audit scripts (`size-contracts.py`). Move them into a small module,
  `quest/data/contract_sampling.py`, as SQL builders with Python twins, and have the audit scripts
  import it. The builder and the sizing then cannot drift.
- **A third distribution, `--distribution contract --sampling-contract <json> --contract-version
  <name>`.** Proportional and balanced stay byte-identical. In contract mode the order of
  operations changes, because targets are per split: eligibility and isolation, then routing, then
  example selection, then one order variant per example. Routing needs no sampling first: receptor
  votes are hashes of the row's own strings and the peptide vote is one small join.
- **Background rows.** Single-receptor rows (1.42B) are routed by the same receptor hash and taken
  as the N lowest `md5(seed | axis | cdr3)` per split, without replacement. This fixes the rule
  the contract left open; it is written back into the contract's notes, not into its numbers.
- **MHC class from the eligibility table** under `reference-aware-v1`: the builder's `mhc_class`
  column, the manifest's `mhc_class_distribution` and verifier V7. Legacy policy keeps the
  name-based label so past builds still reproduce.
- **Paths as arguments.** `--source-root`, `--clusters-dir`, `--split-dir` with the box paths as
  defaults, so the small build runs on the laptop.

Tradeoff taken: the contract mode lives in the existing builder rather than a second script.
That keeps one verifier and one manifest, at the cost of a longer `main()`. Not in scope:
`token_exact`, pair-level isolation of the three unisolated benchmark parts, any change to the
contract's numbers.

## Steps

1. `contract_sampling.py` + tests: stratum assignment, example hash (NULL-safe), cap-then-count,
   order-variant choice; SQL and Python twins agree on fixtures. Audit scripts import it and
   `check-sizing.py` still passes 201 checks on unchanged output.
2. Builder path arguments; a legacy-policy dry run over fixtures produces the same targets as
   before the change.
3. `mhc_class` from the eligibility table, in builder, manifest and V7; fixtures cover a 365-aa
   class-I chain with no allele name and a name/sequence conflict.
4. Contract mode in the builder, both versions; refusal tests for a target above supply, a
   contract whose policy or hashes differ from the build's flags, and a missing stratum.
5. Verifier: a contract check (every stratum count in every split equals its target; background
   and residue totals within the stated tolerances; no example twice in a split).
6. Small build, laptop first: all 41.1M MHC-, peptide- and paired rows are local; single-receptor
   rows come from a bounded fixed subset. Run twice with the same seed and once with shard order
   reversed. Then the same on a box against the real single-receptor partitions.
7. One combined review of the whole change, plus `verifier` and `codex-reviewer` on the sampling
   and isolation paths (data boundary), before any full rebuild is priced.

## Verification

- Steps 1-5: the new tests fail before each change and pass after; the existing 202 pass
  throughout.
- Step 6: V2, V5, V10 and the new contract check pass; logical record hashes are identical
  across the two same-seed runs and the reversed-order run; an interrupted build leaves no
  completion marker; per-split CDR3 length profiles agree; the built enriched training split has
  exactly 12,284 paired TCR + peptide + class-I examples and no peptide above 1,000 in the three
  capped strata.
- What failure looks like: any count off by one, a hash that differs between runs, or a build
  that completes while short of a target.
