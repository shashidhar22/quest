#!/bin/bash
# Quick test and production commands for finding RCC TCRs in dataset

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "RCC TCR Finder - Quick Start"
echo "=============================="
echo ""

# Test run (sample 100 files)
echo "TEST RUN (sample 100 files):"
echo "python scripts/analysis/find_rcc_tcrs_in_dataset.py \\"
echo "    --rcc-csv \$PROJECT_ROOT/data/processed/rcc_tcrs.csv \\"
echo "    --parsed-dir /mnt/ephemeral/parsed_output/seq \\"
echo "    --output-report \$PROJECT_ROOT/outputs/rcc_tcr_matches_test.txt \\"
echo "    --sample 100"
echo ""

# Full run
echo "FULL RUN (all 26,563 files):"
echo "python scripts/analysis/find_rcc_tcrs_in_dataset.py \\"
echo "    --rcc-csv \$PROJECT_ROOT/data/processed/rcc_tcrs.csv \\"
echo "    --parsed-dir /mnt/ephemeral/parsed_output/seq \\"
echo "    --output-report \$PROJECT_ROOT/outputs/rcc_tcr_matches_full.txt"
echo ""

# Ask user which to run
echo "Which would you like to run?"
echo "1) Test run (100 files, ~30 seconds)"
echo "2) Full run (all files, ~10-15 minutes)"
echo "3) Exit"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Running test..."
        python scripts/analysis/find_rcc_tcrs_in_dataset.py \
            --rcc-csv "$PROJECT_ROOT/data/processed/rcc_tcrs.csv" \
            --parsed-dir /mnt/ephemeral/parsed_output/seq \
            --output-report "$PROJECT_ROOT/outputs/rcc_tcr_matches_test.txt" \
            --output-csv "$PROJECT_ROOT/outputs/csv/rcc_tcr_matches_test.csv" \
            --sample 100
        ;;
    2)
        echo ""
        echo "Running full scan..."
        python scripts/analysis/find_rcc_tcrs_in_dataset.py \
            --rcc-csv "$PROJECT_ROOT/data/processed/rcc_tcrs.csv" \
            --parsed-dir /mnt/ephemeral/parsed_output/seq \
            --output-report "$PROJECT_ROOT/outputs/rcc_tcr_matches_full.txt" \
            --output-csv "$PROJECT_ROOT/outputs/csv/rcc_tcr_matches_full.csv"
        ;;
    3)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
