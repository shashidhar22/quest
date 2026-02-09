#!/bin/bash
# TCRdb 2.0 Bulk Download Script
# Run this from your local terminal (not in Cowork sandbox)
# Usage: ./download_all.sh [category]
# Categories: cancer, autoimmunity, inflammation, viral, healthy, transplantation, all

BASE_URL="https://guolab.wchscu.cn/TCRdb2/download/download"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Function to download and extract
download_project() {
    local id=$1
    local folder=$2
    local zip_path="${SCRIPT_DIR}/${folder}/${id}.zip"
    local extract_path="${SCRIPT_DIR}/${folder}/${id}"

    mkdir -p "${SCRIPT_DIR}/${folder}"

    if [ -d "$extract_path" ] || [ -f "$zip_path" ]; then
        echo "  ⊘ Skipped: $id (already exists)"
        return
    fi

    echo "  ↓ Downloading: $id"
    if curl -s -o "$zip_path" "${BASE_URL}/${id}.zip"; then
        if [ -s "$zip_path" ]; then
            unzip -q -o "$zip_path" -d "$extract_path" 2>/dev/null
            echo "  ✓ Complete: $id"
        else
            rm -f "$zip_path"
            echo "  ✗ Failed: $id (empty file)"
        fi
    else
        echo "  ✗ Failed: $id (download error)"
    fi
    sleep 0.5  # Rate limiting
}

# Cancer datasets
CANCER_PROJECTS=(
    PRJNA330606 PRJNA506151 PRJNA321261 PRJNA301507 PRJNA316033
    PRJNA325416 PRJEB33490 PRJNA297261 PRJNA267461 PRJNA491656
    PRJNA516984 PRJNA391483 PRJNA356992 PRJNA422601 PRJNA544699
    PRJNA315543 PRJNA436233 PRJNA429872 PRJNA491658 PRJNA665539
    PRJNA767497 PRJNA734593 PRJDB9359 PRJNA672919 PRJNA646628
    PRJNA438509 PRJNA642967 PRJNA592172 PRJEB38454 PRJNA768494
    immunoSEQ01 immunoSEQ02 immunoSEQ09 immunoSEQ10 immunoSEQ12
    immunoSEQ20 immunoSEQ23 immunoSEQ24 immunoSEQ26 immunoSEQ32
    immunoSEQ92 immunoSEQ37 immunoSEQ39 immunoSEQ66 immunoSEQ140
    immunoSEQ97 immunoSEQ109 immunoSEQ129 immunoSEQ144 immunoSEQ74
)

# Autoimmunity datasets
AUTOIMMUNITY_PROJECTS=(
    PRJNA280417 PRJNA318495 PRJNA449605 PRJNA495603 PRJNA516296
    PRJNA634746 PRJDB10910 PRJEB33806
    immunoSEQ27 immunoSEQ33 immunoSEQ40 immunoSEQ50 immunoSEQ56
    immunoSEQ59 immunoSEQ77 immunoSEQ115 immunoSEQ143
)

# Inflammation datasets
INFLAMMATION_PROJECTS=(
    PRJNA393498 PRJEB27352 PRJNA385561 PRJNA659721
    immunoSEQ03 immunoSEQ08 immunoSEQ22 immunoSEQ36 immunoSEQ44
    immunoSEQ114
)

# Viral datasets
VIRAL_PROJECTS=(
    PRJNA427746 PRJNA359580 PRJNA473147 PRJNA533091 PRJNA633317
    PRJNA705196
    immunoSEQ04 immunoSEQ05 immunoSEQ06 immunoSEQ11 immunoSEQ18
    immunoSEQ19 immunoSEQ21 immunoSEQ28 immunoSEQ29 immunoSEQ30
    immunoSEQ31 immunoSEQ34 immunoSEQ54 immunoSEQ108 immunoSEQ60
)

# Healthy datasets
HEALTHY_PROJECTS=(
    PRJNA390125 PRJNA79519 PRJNA229070 PRJNA312766 PRJNA298417
    PRJEB31057 PRJEB31283 PRJEB37219 PRJNA395098
    immunoSEQ25 immunoSEQ41 immunoSEQ67 immunoSEQ38
)

# Transplantation datasets
TRANSPLANTATION_PROJECTS=(
    PRJNA312319 PRJEB37571 PRJNA493805 immunoSEQ41
)

download_category() {
    local category=$1
    shift
    local projects=("$@")

    echo ""
    echo "========================================"
    echo "Downloading ${#projects[@]} ${category} projects"
    echo "========================================"

    for project in "${projects[@]}"; do
        download_project "$project" "$category"
    done
}

# Main
echo "TCRdb 2.0 Bulk Downloader"
echo "========================="
echo "Output: $SCRIPT_DIR"
echo ""

case "${1:-all}" in
    cancer)
        download_category "cancer" "${CANCER_PROJECTS[@]}"
        ;;
    autoimmunity)
        download_category "autoimmunity" "${AUTOIMMUNITY_PROJECTS[@]}"
        ;;
    inflammation)
        download_category "inflammation" "${INFLAMMATION_PROJECTS[@]}"
        ;;
    viral)
        download_category "viral" "${VIRAL_PROJECTS[@]}"
        ;;
    healthy)
        download_category "healthy" "${HEALTHY_PROJECTS[@]}"
        ;;
    transplantation)
        download_category "transplantation" "${TRANSPLANTATION_PROJECTS[@]}"
        ;;
    all)
        download_category "cancer" "${CANCER_PROJECTS[@]}"
        download_category "autoimmunity" "${AUTOIMMUNITY_PROJECTS[@]}"
        download_category "inflammation" "${INFLAMMATION_PROJECTS[@]}"
        download_category "viral" "${VIRAL_PROJECTS[@]}"
        download_category "healthy" "${HEALTHY_PROJECTS[@]}"
        download_category "transplantation" "${TRANSPLANTATION_PROJECTS[@]}"
        ;;
    *)
        echo "Usage: $0 [cancer|autoimmunity|inflammation|viral|healthy|transplantation|all]"
        exit 1
        ;;
esac

echo ""
echo "Download complete!"

