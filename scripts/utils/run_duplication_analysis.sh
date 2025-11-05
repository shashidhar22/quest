#!/bin/bash
# Run the improved duplication analysis with all features enabled

set -e

echo "========================================================================"
echo "TCR DUPLICATION ANALYSIS WITH IMPROVEMENTS"
echo "========================================================================"
echo ""
echo "Features:"
echo "  ✓ Total and unique counts reported"
echo "  ✓ All tra_trb combinations included"
echo "  ✓ Category and study-level breakdowns for 'other' folder"
echo ""

# Check if mapping file exists
if [ ! -f "source_file_mapping.json" ]; then
    echo "📋 Creating source file mapping..."
    python3 create_source_mapping.py
    echo ""
fi

# Run the analysis
echo "🚀 Starting duplication analysis..."
echo "   - Using 6 parallel workers"
echo "   - Processing MRI files"
echo "   - Source mapping: source_file_mapping.json"
echo ""

python3 analyze_duplication_parallel.py \
    --parsed-dir /mnt/ephemeral/parsed_output \
    --temp-dir /mnt/ephemeral \
    --output-report data_duplication_report_improved.txt \
    --output-csv file_duplication_stats_improved.csv \
    --parallel 10 \
    --mapping-file source_file_mapping.json

echo ""
echo "========================================================================"
echo "✓ Analysis Complete!"
echo "========================================================================"
echo ""
echo "Reports generated:"
echo "  📄 duplication_report_improved.txt   - Main report with all improvements"
echo "  📊 file_duplication_stats_improved.csv - Per-file statistics"
echo ""
echo "New sections in report:"
echo "  1. Total counts alongside unique counts"
echo "  2. All tra_trb combinations in dataset summary"
echo "  3. 'OTHER' breakdown by study category"
echo "  4. 'OTHER' breakdown by study ID (top 30)"
echo ""
