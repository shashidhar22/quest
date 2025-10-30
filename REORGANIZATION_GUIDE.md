# Repository Reorganization - Quick Start

## Overview
Your QUEST repository has been prepared for reorganization. The new structure will organize files by purpose, separate outputs from source code, and make the codebase more maintainable.

## What Will Change

### Before (Current State)
```
quest/
├── 30+ files in root (scripts, outputs, tests mixed together)
├── scripts/ (only has Ray-related scripts)
├── parsers/
├── quest/
└── tests/
```

### After (New Structure)
```
quest/
├── scripts/
│   ├── training/          # Ray training scripts
│   ├── inference/         # Inference scripts
│   ├── data_processing/   # Data parsing/processing
│   ├── analysis/          # Analysis scripts
│   └── utils/             # Shell scripts
├── tests/
│   └── integration/       # Integration tests
├── outputs/               # All generated files
│   ├── reports/
│   ├── csv/
│   └── logs/
├── data/                  # Data files
├── parsers/               # (unchanged)
├── quest/                 # (unchanged)
├── setup.py               # NEW: Package installation
└── ORGANIZATION.md        # NEW: Documentation
```

## Files That Will Be Moved

### Scripts → `scripts/inference/`
- `run_inference.py`
- `inference_examples.py`
- `inference_with_dataset.py`
- `quick_inference.py`

### Scripts → `scripts/analysis/`
- `analyze_duplication_parallel.py`
- `calculate_perplexity.py`
- `evaluate_masked_predictions.py`
- `export_embeddings.py`
- `best_worst_by_molecule.py`
- `show_best_worst.py`
- `summarize_data.py`
- `summarize_source_data.py`
- `create_source_mapping.py`

### Scripts → `scripts/data_processing/`
- `reformat_data.py`
- `run_parser_production.py`
- `run_parser_test.py`
- `reparse_databases.py`

### Tests → `tests/integration/`
- `test_tcrconvert_integration.py`
- `test_tcr_stitching.py`
- `test_stitchr_direct.py`
- `test_stitcher_diagnostic.py`

### Output Files → `outputs/`
- All `.csv` files → `outputs/csv/`
- All `.txt` reports → `outputs/reports/`
- All `.log` files → `outputs/logs/`

## How to Run the Reorganization

### Option 1: Automatic (Recommended)
```bash
cd /home/ubuntu/quest
./run_reorganization.sh
```

This will run all steps automatically with confirmation.

### Option 2: Manual (Step by Step)
```bash
# Step 1: Create directories and move files
python3 reorganize_repo.py

# Step 2: Update import statements
python3 update_imports.py

# Step 3: Update .gitignore
python3 update_gitignore.py
```

## After Reorganization

### 1. Test Your Scripts
```bash
# Test an inference script
python scripts/inference/quick_inference.py --help

# Test a data processing script
python scripts/data_processing/run_parser_test.py --help
```

### 2. Install the Package
```bash
pip install -e .
```

This creates command-line tools:
- `quest-train`
- `quest-finetune`
- `quest-evaluate`
- `quest-inference`

### 3. Commit Changes
```bash
# Review what changed
git status

# Stage all changes
git add -A

# Commit
git commit -m "Reorganize repository structure for better maintainability"
```

## Benefits

1. **Clearer organization**: Scripts grouped by purpose
2. **Cleaner root**: Only essential files in root directory
3. **Better .gitignore**: Outputs excluded from version control
4. **Proper package**: Can install with `pip install -e .`
5. **Entry points**: Command-line tools for common tasks
6. **Easier navigation**: Find files by their function
7. **Better for collaboration**: Clear structure for new contributors

## Rollback

If you need to undo the changes:
```bash
git reset --hard HEAD
git clean -fd
```

## Questions?

- Check `ORGANIZATION.md` for detailed documentation
- The reorganization scripts are in `.reorganization_scripts/` for reference
- All original files are moved, not deleted

## Current Branch

You're on branch: `verify_inputs`

Consider creating a new branch for this reorganization:
```bash
git checkout -b reorganize-structure
./run_reorganization.sh
# Test everything
git add -A && git commit -m "Reorganize repository structure"
```

Then you can merge it when ready or keep testing in isolation.
