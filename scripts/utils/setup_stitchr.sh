#!/bin/bash
# Setup script for stitchr TCR full-length sequence generation
# This downloads the reference data needed for stitching human TCRs

echo "=========================================="
echo "Setting up stitchr for TCR stitching"
echo "=========================================="
echo ""

# Check if stitchr is installed
if ! command -v stitchr &> /dev/null; then
    echo "Error: stitchr not found"
    echo "Install with: pip install stitchr"
    exit 1
fi

echo "✓ stitchr is installed"
echo ""

# Check if stitchrdl is available
if ! command -v stitchrdl &> /dev/null; then
    echo "Error: stitchrdl not found"
    echo "This should have been installed with stitchr"
    echo "Try: pip install --upgrade stitchr"
    exit 1
fi

echo "✓ stitchrdl is available"
echo ""

# Download human TCR reference data
echo "Downloading human TCR reference data from IMGT..."
echo "This may take a few minutes..."
echo ""

stitchrdl -s HUMAN

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup complete!"
    echo "=========================================="
    echo ""
    echo "Reference data installed successfully."
    echo "You can now use TCR stitching in the parser."
    echo ""
    echo "To test the functionality, run:"
    echo "  python test_stitcher_diagnostic.py"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Setup failed"
    echo "=========================================="
    echo ""
    echo "There was an error downloading reference data."
    echo "Check your internet connection and try again."
    exit 1
fi
