# Reorganization Checklist

## Pre-Reorganization

- [ ] **Backup your work**: Commit current changes or create a backup
  ```bash
  git status
  git add -A
  git commit -m "Checkpoint before reorganization"
  # Or create a backup branch
  git checkout -b backup-before-reorg
  git checkout verify_inputs
  ```

- [ ] **Create a new branch** (recommended):
  ```bash
  git checkout -b reorganize-structure
  ```

- [ ] **Review what will be moved**: Check `REORGANIZATION_GUIDE.md`

## Running the Reorganization

- [ ] **Run the master script**:
  ```bash
  ./run_reorganization.sh
  ```
  
  Or manually:
  - [ ] `python3 reorganize_repo.py`
  - [ ] `python3 update_imports.py`
  - [ ] `python3 update_gitignore.py`

- [ ] **Verify the reorganization**:
  ```bash
  python3 verify_reorganization.py
  ```

## Post-Reorganization Testing

- [ ] **Test inference scripts**:
  ```bash
  python scripts/inference/quick_inference.py --help
  python scripts/inference/inference_examples.py --help
  ```

- [ ] **Test data processing scripts**:
  ```bash
  python scripts/data_processing/run_parser_test.py --help
  ```

- [ ] **Test analysis scripts**:
  ```bash
  python scripts/analysis/calculate_perplexity.py --help
  ```

- [ ] **Test training scripts**:
  ```bash
  python scripts/training/ray_train.py --help
  ```

- [ ] **Check that parsers still work**:
  ```bash
  python -c "from parsers import utils; print('✓ Parsers import OK')"
  ```

- [ ] **Check that quest package still works**:
  ```bash
  python -c "from quest import config; print('✓ Quest imports OK')"
  ```

## Package Installation

- [ ] **Install the package**:
  ```bash
  pip install -e .
  ```

- [ ] **Test entry points** (if main functions exist):
  ```bash
  quest-train --help
  quest-inference --help
  ```

## Git Integration

- [ ] **Review changes**:
  ```bash
  git status
  git diff HEAD
  ```

- [ ] **Check what's ignored**:
  ```bash
  git status --ignored
  ```

- [ ] **Stage the changes**:
  ```bash
  git add -A
  ```

- [ ] **Commit the reorganization**:
  ```bash
  git commit -m "Reorganize repository structure

  - Move scripts to organized subdirectories (training, inference, analysis, data_processing)
  - Move output files to outputs/ directory
  - Move integration tests to tests/integration/
  - Add setup.py for package installation
  - Update .gitignore to exclude outputs and generated files
  - Add ORGANIZATION.md documentation
  "
  ```

## Cleanup

- [ ] **Remove reorganization scripts** (optional):
  ```bash
  rm -rf .reorganization_scripts/
  git add -A
  git commit -m "Remove reorganization scripts"
  ```

- [ ] **Update documentation**: Add notes about new structure to README if needed

- [ ] **Inform collaborators**: If working with others, let them know about the structure change

## Rollback (If Needed)

If something goes wrong:

```bash
# Undo all changes
git reset --hard HEAD
git clean -fd

# Or switch back to backup branch
git checkout backup-before-reorg
```

## Notes

- Output files (CSV, logs, reports) are now in `outputs/` and gitignored
- Scripts are organized by purpose in `scripts/` subdirectories
- Tests are in `tests/` with integration tests in `tests/integration/`
- The package can be installed with `pip install -e .`
- Import statements have been updated automatically

## Status

**Started**: _______________

**Completed**: _______________

**Issues encountered**: 

_____________________________________

_____________________________________

_____________________________________

**Resolution**:

_____________________________________

_____________________________________

_____________________________________
