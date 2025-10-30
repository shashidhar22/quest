#!/usr/bin/env python3
"""
Update import statements after reorganization.
This script fixes imports in moved files to reflect their new locations.
"""

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent

# Import pattern replacements
IMPORT_UPDATES = {
    # Files that moved from root to scripts/inference
    'scripts/inference': {
        'from quest': 'from quest',  # Keep quest imports as-is
        'from parsers': 'from parsers',  # Keep parsers imports as-is
        'import sys': 'import sys',  # Keep standard imports
    },
    
    # Files that moved from root to scripts/analysis
    'scripts/analysis': {
        'from quest': 'from quest',
        'from parsers': 'from parsers',
    },
    
    # Files that moved from root to scripts/data_processing
    'scripts/data_processing': {
        'from quest': 'from quest',
        'from parsers': 'from parsers',
    },
    
    # Files in scripts/training (already in scripts)
    'scripts/training': {
        'from quest': 'from quest',
    },
}

def update_sys_path_statements(file_path: Path, script_dir: str):
    """Update sys.path statements to account for new directory depth."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern for sys.path.append with parent directory
    # Common patterns:
    # sys.path.append(os.path.dirname(__file__))
    # sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    # sys.path.append(str(Path(__file__).parent))
    
    # Determine how many levels up we need to go
    depth = len(Path(script_dir).parts)
    
    # Replace patterns that add current directory to sys.path
    patterns_to_fix = [
        (r"sys\.path\.append\(os\.path\.dirname\(__file__\)\)",
         f"sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))"),
        
        (r"sys\.path\.append\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)",
         f"sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))"),
        
        (r"sys\.path\.append\(str\(Path\(__file__\)\.parent\)\)",
         f"sys.path.append(str(Path(__file__).parent.parent.parent))"),
    ]
    
    for pattern, replacement in patterns_to_fix:
        if 'scripts/training' in str(file_path):
            # Scripts in training already had scripts/ prefix, so need 2 levels up
            replacement = replacement.replace('.parent.parent.parent', '.parent.parent')
            replacement = replacement.replace('dirname(os.path.dirname(os.path.dirname', 'dirname(os.path.dirname')
        
        content = re.sub(pattern, replacement, content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def update_imports_in_file(file_path: Path):
    """Update import statements in a single file."""
    if not file_path.is_file() or file_path.suffix != '.py':
        return False
    
    # Determine which script directory this file is in
    script_dir = None
    for dir_name in ['scripts/inference', 'scripts/analysis', 'scripts/data_processing', 'scripts/training']:
        if dir_name in str(file_path):
            script_dir = dir_name
            break
    
    if not script_dir:
        return False
    
    # Update sys.path statements
    changed = update_sys_path_statements(file_path, script_dir)
    
    if changed:
        print(f"✓ Updated imports in: {file_path.relative_to(REPO_ROOT)}")
        return True
    return False

def scan_and_update():
    """Scan all moved files and update their imports."""
    updated_count = 0
    
    directories_to_scan = [
        'scripts/inference',
        'scripts/analysis',
        'scripts/data_processing',
        'scripts/training',
    ]
    
    for dir_name in directories_to_scan:
        dir_path = REPO_ROOT / dir_name
        if dir_path.exists():
            for py_file in dir_path.glob('*.py'):
                if update_imports_in_file(py_file):
                    updated_count += 1
    
    return updated_count

def main():
    """Main routine."""
    print("=" * 60)
    print("Updating Import Statements")
    print("=" * 60)
    print()
    
    updated = scan_and_update()
    
    print()
    print(f"✓ Updated {updated} files")
    print()
    print("Note: Please review the changes and test your scripts.")
    print("Some imports may need manual adjustment.")

if __name__ == '__main__':
    main()
