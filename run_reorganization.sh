#!/bin/bash
# Master script to reorganize the QUEST repository
# Run this script from the repository root

set -e  # Exit on error

echo "============================================================"
echo "QUEST Repository Reorganization"
echo "============================================================"
echo ""
echo "This script will:"
echo "  1. Create new directory structure"
echo "  2. Move files to appropriate locations"
echo "  3. Update import statements"
echo "  4. Update .gitignore"
echo "  5. Create documentation"
echo ""
echo "A backup is recommended before proceeding."
echo ""
read -p "Do you want to continue? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1/4: Running reorganization script..."
python3 reorganize_repo.py

echo ""
echo "Step 2/4: Updating import statements..."
python3 update_imports.py

echo ""
echo "Step 3/4: Updating .gitignore..."
python3 update_gitignore.py

echo ""
echo "Step 4/4: Cleaning up reorganization scripts..."
# Move reorganization scripts to a subdirectory
mkdir -p .reorganization_scripts
mv reorganize_repo.py .reorganization_scripts/
mv update_imports.py .reorganization_scripts/
mv update_gitignore.py .reorganization_scripts/
mv run_reorganization.sh .reorganization_scripts/

echo ""
echo "============================================================"
echo "✓ Reorganization Complete!"
echo "============================================================"
echo ""
echo "Summary:"
echo "  • Files have been moved to organized directories"
echo "  • Import statements have been updated"
echo "  • .gitignore has been updated"
echo "  • Documentation created: ORGANIZATION.md"
echo ""
echo "Next steps:"
echo "  1. Review the changes: git status"
echo "  2. Test your main scripts to ensure they work"
echo "  3. Install the package: pip install -e ."
echo "  4. Commit the changes: git add -A && git commit -m 'Reorganize repository structure'"
echo ""
echo "Reorganization scripts moved to: .reorganization_scripts/"
echo "(You can delete this directory after confirming everything works)"
echo ""
