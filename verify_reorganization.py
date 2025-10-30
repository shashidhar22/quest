#!/usr/bin/env python3
"""
Verify that the reorganization was successful.
This script checks that files are in their expected locations.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent

# Expected structure after reorganization
EXPECTED_STRUCTURE = {
    'directories': [
        'scripts/training',
        'scripts/inference',
        'scripts/data_processing',
        'scripts/analysis',
        'scripts/utils',
        'tests/integration',
        'outputs/reports',
        'outputs/csv',
        'outputs/logs',
        'data',
        'quest',
        'parsers',
        'config',
        'env',
    ],
    'files': {
        'root': [
            'README.md',
            'setup.py',
            'requirements.txt',
            'ORGANIZATION.md',
            '.gitignore',
        ],
        'scripts/training': [
            'ray_train.py',
            'ray_fine_tune.py',
            'ray_evaluator.py',
        ],
        'scripts/inference': [
            'run_inference.py',
            'inference_examples.py',
            'inference_with_dataset.py',
            'quick_inference.py',
        ],
        'scripts/data_processing': [
            'ray_datawriter.py',
            'reformat_data.py',
            'run_parser_production.py',
            'run_parser_test.py',
        ],
        'scripts/analysis': [
            'analyze_duplication_parallel.py',
            'calculate_perplexity.py',
            'evaluate_masked_predictions.py',
            'export_embeddings.py',
        ],
    }
}

def check_directories():
    """Check that all expected directories exist."""
    print("Checking directories...")
    missing = []
    for dir_path in EXPECTED_STRUCTURE['directories']:
        full_path = REPO_ROOT / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
            missing.append(dir_path)
    return missing

def check_files():
    """Check that expected files are in their new locations."""
    print("\nChecking files...")
    missing = []
    for dir_name, files in EXPECTED_STRUCTURE['files'].items():
        if dir_name == 'root':
            dir_path = REPO_ROOT
        else:
            dir_path = REPO_ROOT / dir_name
        
        for filename in files:
            file_path = dir_path / filename
            if file_path.exists():
                print(f"  ✓ {dir_name}/{filename}")
            else:
                print(f"  ✗ {dir_name}/{filename} (missing)")
                missing.append(f"{dir_name}/{filename}")
    return missing

def check_old_locations():
    """Check that files are no longer in old locations (root)."""
    print("\nChecking old locations...")
    
    old_script_files = [
        'run_inference.py',
        'inference_examples.py',
        'quick_inference.py',
        'analyze_duplication_parallel.py',
        'calculate_perplexity.py',
        'reformat_data.py',
        'run_parser_production.py',
    ]
    
    remaining = []
    for filename in old_script_files:
        file_path = REPO_ROOT / filename
        if file_path.exists():
            print(f"  ⚠ {filename} still in root directory")
            remaining.append(filename)
        else:
            print(f"  ✓ {filename} (moved)")
    
    return remaining

def check_gitignore():
    """Check that .gitignore has been updated."""
    print("\nChecking .gitignore...")
    gitignore_path = REPO_ROOT / '.gitignore'
    
    if not gitignore_path.exists():
        print("  ✗ .gitignore not found")
        return False
    
    with open(gitignore_path, 'r') as f:
        content = f.read()
    
    required_entries = ['outputs/', '*.log', '__pycache__/']
    missing = []
    
    for entry in required_entries:
        if entry in content:
            print(f"  ✓ Contains '{entry}'")
        else:
            print(f"  ✗ Missing '{entry}'")
            missing.append(entry)
    
    return len(missing) == 0

def check_setup_py():
    """Check that setup.py exists and is valid."""
    print("\nChecking setup.py...")
    setup_path = REPO_ROOT / 'setup.py'
    
    if not setup_path.exists():
        print("  ✗ setup.py not found")
        return False
    
    with open(setup_path, 'r') as f:
        content = f.read()
    
    required_items = ['setuptools', 'name="quest"', 'packages=']
    all_present = True
    
    for item in required_items:
        if item in content:
            print(f"  ✓ Contains '{item}'")
        else:
            print(f"  ✗ Missing '{item}'")
            all_present = False
    
    return all_present

def main():
    """Main verification routine."""
    print("=" * 60)
    print("QUEST Repository Reorganization Verification")
    print("=" * 60)
    print()
    
    # Run all checks
    missing_dirs = check_directories()
    missing_files = check_files()
    remaining_files = check_old_locations()
    gitignore_ok = check_gitignore()
    setup_ok = check_setup_py()
    
    # Summary
    print()
    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    issues = []
    
    if missing_dirs:
        issues.append(f"Missing {len(missing_dirs)} directories")
    
    if missing_files:
        issues.append(f"Missing {len(missing_files)} files")
    
    if remaining_files:
        issues.append(f"{len(remaining_files)} files still in old locations")
    
    if not gitignore_ok:
        issues.append(".gitignore not properly updated")
    
    if not setup_ok:
        issues.append("setup.py not properly configured")
    
    if issues:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  • {issue}")
        print("\nThe reorganization may not be complete.")
        print("Run the reorganization scripts again or check manually.")
        sys.exit(1)
    else:
        print("\n✓ All checks passed!")
        print("\nThe repository has been successfully reorganized.")
        print("\nNext steps:")
        print("  1. Test your scripts")
        print("  2. Run: pip install -e .")
        print("  3. Commit the changes")
        sys.exit(0)

if __name__ == '__main__':
    main()
