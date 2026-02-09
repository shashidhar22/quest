#!/usr/bin/env python3
"""IEDB Peptide-MHC Deep Dive Analysis.

Loads MySQL-sourced binding tables into the existing IEDB SQLite database
(built by iedb_sqlite_summary.py), then produces cross-table pMHC analyses
as TSV files and a narrative summary.

Usage:
    python scripts/analysis/iedb_pmhc_deep_dive.py
    python scripts/analysis/iedb_pmhc_deep_dive.py --skip_mysql --skip_download
    python scripts/analysis/iedb_pmhc_deep_dive.py --db_path data/analysis/iedb/iedb.sqlite
"""

import argparse
import csv
import gzip
import os
import re
import sqlite3
import subprocess
import sys
import time
import zipfile
from datetime import date

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_PATH = "data/analysis/iedb/iedb.sqlite"
MYSQL_DUMP = "data/databases/IEDB/full_database/iedb_public.sql.gz"
MHC_LIGAND_URL = "https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_v3.zip"
OUTPUT_DIR = "data/analysis/iedb/pmhc"

# SQL fragments for coalescing curated/calculated CDR3 values
CDR3A = "COALESCE(NULLIF(chain_1_cdr3_curated, ''), NULLIF(chain_1_cdr3_calculated, ''))"
CDR3B = "COALESCE(NULLIF(chain_2_cdr3_curated, ''), NULLIF(chain_2_cdr3_calculated, ''))"
VGA = "COALESCE(NULLIF(chain_1_curated_v_gene, ''), NULLIF(chain_1_calculated_v_gene, ''))"
VGB = "COALESCE(NULLIF(chain_2_curated_v_gene, ''), NULLIF(chain_2_calculated_v_gene, ''))"
JGA = "COALESCE(NULLIF(chain_1_curated_j_gene, ''), NULLIF(chain_1_calculated_j_gene, ''))"
JGB = "COALESCE(NULLIF(chain_2_curated_j_gene, ''), NULLIF(chain_2_calculated_j_gene, ''))"


# ---------------------------------------------------------------------------
# Reusable helpers (from iedb_sqlite_summary.py)
# ---------------------------------------------------------------------------
def _table_exists(cur, table_name):
    """Check if a table exists in the SQLite database (including temp tables)."""
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    if cur.fetchone() is not None:
        return True
    # Also check temporary tables
    try:
        cur.execute("SELECT name FROM sqlite_temp_master WHERE type='table' AND name=?", (table_name,))
        return cur.fetchone() is not None
    except sqlite3.OperationalError:
        return False


def classify_mhc(allele):
    """Heuristic MHC class assignment from allele name."""
    a = allele.upper()
    if a == "HLA CLASS I":
        return "Class I"
    if a == "HLA CLASS II":
        return "Class II"
    if re.match(r"HLA-[ABCEFG]", a):
        return "Class I"
    if re.match(r"H-?2-?[KDL]", a):
        return "Class I"
    if re.match(r"HLA-D", a):
        return "Class II"
    if re.match(r"H-?2-?I", a):
        return "Class II"
    return "Unknown"


def _scalar(cur, sql, params=()):
    """Execute SQL and return the first column of the first row."""
    cur.execute(sql, params)
    row = cur.fetchone()
    return row[0] if row else 0


def parse_mysql_values(values_str):
    """Parse a MySQL VALUES clause into a list of row tuples."""
    rows = []
    i = 0
    n = len(values_str)

    while i < n:
        while i < n and values_str[i] != '(':
            i += 1
        if i >= n:
            break
        i += 1

        row = []
        while i < n:
            while i < n and values_str[i] in (' ', '\t'):
                i += 1
            if i >= n:
                break
            if values_str[i] == ')':
                i += 1
                break
            elif values_str[i] == ',':
                i += 1
                continue
            elif values_str[i] == "'":
                i += 1
                chars = []
                while i < n:
                    if values_str[i] == '\\' and i + 1 < n:
                        chars.append(values_str[i + 1])
                        i += 2
                    elif values_str[i] == "'":
                        i += 1
                        break
                    else:
                        chars.append(values_str[i])
                        i += 1
                row.append(''.join(chars))
            elif values_str[i:i + 4].upper() == 'NULL':
                row.append(None)
                i += 4
            else:
                start = i
                while i < n and values_str[i] not in (',', ')'):
                    i += 1
                row.append(values_str[start:i].strip())

        rows.append(tuple(row))

    return rows


def make_column_names(group_row, sub_row):
    """Combine 2-row CSV headers into clean snake_case column names."""
    names = []
    seen = {}
    for g, s in zip(group_row, sub_row):
        g = g.strip()
        s = s.strip()
        raw = f"{g}_{s}" if g else s
        col = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
        if col in seen:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
        else:
            seen[col] = 1
        names.append(col)
    return names


# ---------------------------------------------------------------------------
# MySQL dump loader (targeted tables only)
# ---------------------------------------------------------------------------
def load_mysql_dump(db_path, sql_gz_path):
    """Parse MySQL dump for binding-related tables and load into SQLite."""
    if not os.path.exists(sql_gz_path):
        print(f"  WARNING: MySQL dump not found at {sql_gz_path} — skipping")
        return

    print(f"\n{'='*60}")
    print(f"Parsing MySQL dump from {sql_gz_path}")
    print(f"{'='*60}")
    t0 = time.time()

    target_tables = {
        "mhc_bind": [
            "mhc_bind_id", "reference_id", "curated_epitope_id", "as_location",
            "as_type_id", "as_char_value", "as_num_value", "as_inequality",
            "as_comments", "mhc_allele_restriction_id", "mhc_allele_name", "complex_id",
        ],
        "mhc_elution": None,
        "mhc_allele_restriction": [
            "mhc_allele_restriction_id", "displayed_restriction", "synonyms", "includes",
            "restriction_level", "organism", "organism_ncbi_tax_id", "class", "haplotype",
            "locus", "serotype", "molecule", "chain_i_name", "chain_ii_name",
            "chain_i_source_org", "chain_ii_source_org", "chain_i_source_org_ncbi_tax_id",
            "chain_ii_source_org_ncbi_tax_id", "chain_i_mhc_gene_name", "chain_ii_mhc_gene_name",
            "chain_i_idb_acc", "chain_ii_idb_acc", "chain_i_length", "chain_ii_length",
            "comments",
        ],
        "assay_type": [
            "assay_type_id", "category", "assay_type", "response", "units", "obi_id", "class",
        ],
        "epitope": None,
        "curated_epitope": None,
    }

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    table_counts = {t: 0 for t in target_tables}

    try:
        insert_patterns = {}
        for table_name in target_tables:
            pattern = re.compile(
                r"INSERT INTO `" + re.escape(table_name) + r"`\s+(?:\([^)]*\)\s+)?VALUES\s+(.*)",
                re.IGNORECASE,
            )
            insert_patterns[table_name] = pattern

        created_tables = set()
        print(f"  Scanning for INSERT statements...")
        line_count = 0

        with gzip.open(sql_gz_path, 'rb') as f:
            try:
                for raw_line in f:
                    line_count += 1
                    if line_count % 1_000_000 == 0:
                        print(f"  ... scanned {line_count:,} lines", end="\r")
                    if b'INSERT INTO' not in raw_line:
                        continue

                    line = raw_line.decode('utf-8', errors='replace').rstrip('\n').rstrip('\r')

                    for table_name, pattern in insert_patterns.items():
                        m = pattern.match(line)
                        if not m:
                            continue

                        values_str = m.group(1).rstrip(';')
                        rows = parse_mysql_values(values_str)
                        if not rows:
                            continue

                        sqlite_table = f"mysql_{table_name}"
                        ncols_data = len(rows[0])

                        if sqlite_table not in created_tables:
                            col_names = target_tables[table_name]
                            if col_names is None:
                                col_names = [f"col_{i}" for i in range(ncols_data)]
                                target_tables[table_name] = col_names
                            while len(col_names) < ncols_data:
                                col_names.append(f"col_{len(col_names)}")
                            if len(col_names) > ncols_data:
                                col_names = col_names[:ncols_data]
                                target_tables[table_name] = col_names

                            col_defs = ", ".join(f'"{c}" TEXT' for c in col_names)
                            cur.execute(f"DROP TABLE IF EXISTS {sqlite_table}")
                            cur.execute(f"CREATE TABLE {sqlite_table} ({col_defs})")
                            created_tables.add(sqlite_table)

                        col_names = target_tables[table_name]
                        expected_ncols = len(col_names)
                        placeholders = ", ".join(["?"] * expected_ncols)
                        insert_sql = f"INSERT INTO {sqlite_table} VALUES ({placeholders})"

                        normalized = []
                        for row in rows:
                            if len(row) < expected_ncols:
                                row = row + (None,) * (expected_ncols - len(row))
                            elif len(row) > expected_ncols:
                                row = row[:expected_ncols]
                            normalized.append(tuple('' if v is None else v for v in row))

                        cur.executemany(insert_sql, normalized)
                        table_counts[table_name] += len(normalized)
                        break
            except (gzip.BadGzipFile, OSError) as e:
                print(f"\n  Note: gzip read ended at line {line_count:,} ({e})")
                print(f"  This is normal for multi-member gzip files — data parsed so far is usable.")

        conn.commit()

        print(f"\n  Scanned {line_count:,} lines in {time.time()-t0:.1f}s")
        print(f"  Tables loaded from MySQL dump:")
        for table_name, count in table_counts.items():
            sqlite_table = f"mysql_{table_name}"
            if count > 0:
                print(f"    {sqlite_table}: {count:,} rows")
            else:
                print(f"    {sqlite_table}: (not found in dump)")

        # Create indexes
        print("  Creating indexes...")
        if table_counts["mhc_bind"] > 0:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_mhcbind_epitope ON mysql_mhc_bind(curated_epitope_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_mhcbind_allele ON mysql_mhc_bind(mhc_allele_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_mhcbind_type ON mysql_mhc_bind(as_type_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_mhcbind_char ON mysql_mhc_bind(as_char_value)")
        if table_counts["mhc_elution"] > 0:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_mhcelut_col0 ON mysql_mhc_elution(col_0)")
        if table_counts["assay_type"] > 0:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_assaytype_id ON mysql_assay_type(assay_type_id)")
        if table_counts["epitope"] > 0:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_epitope_col0 ON mysql_epitope(col_0)")
        if table_counts["curated_epitope"] > 0:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_curatedepi_col0 ON mysql_curated_epitope(col_0)")
        if table_counts["mhc_allele_restriction"] > 0:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_mhcallele_id ON mysql_mhc_allele_restriction(mhc_allele_restriction_id)")

        conn.commit()
        print(f"  MySQL dump parsing complete ({time.time()-t0:.1f}s total)")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Ensure required tables
# ---------------------------------------------------------------------------
def ensure_mysql_tables(db_path, mysql_dump_path):
    """Load MySQL tables into SQLite if not already present."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    needed = ["mysql_mhc_bind", "mysql_assay_type", "mysql_curated_epitope",
              "mysql_mhc_allele_restriction"]
    missing = [t for t in needed if not _table_exists(cur, t)]
    conn.close()

    if not missing:
        print("  All required MySQL tables already present in SQLite.")
        return

    print(f"  Missing MySQL tables: {missing}")
    print(f"  Loading from MySQL dump...")
    load_mysql_dump(db_path, mysql_dump_path)


def ensure_mhc_ligand(iedb_dir, db_path):
    """Re-download and load mhc_ligand CSV if zip is 0 bytes."""
    zip_path = os.path.join(iedb_dir, "mhc_ligand", "mhc_ligand_full_v3.zip")
    csv_path = os.path.join(iedb_dir, "mhc_ligand", "mhc_ligand_full_v3.csv")

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        print(f"  MHC ligand CSV already exists ({os.path.getsize(csv_path)/1e6:.1f} MB)")
        return

    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        print(f"  Extracting existing zip...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(os.path.dirname(zip_path))
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                print(f"  Extracted {csv_path} ({os.path.getsize(csv_path)/1e6:.1f} MB)")
                return
        except zipfile.BadZipFile:
            print(f"  Bad zip file, will re-download...")

    print(f"  Downloading mhc_ligand from IEDB...")
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    try:
        result = subprocess.run(
            ["wget", "-q", "-O", zip_path, MHC_LIGAND_URL],
            timeout=600, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: Download failed (return code {result.returncode})")
            print(f"  stderr: {result.stderr[:200]}")
            return

        if os.path.getsize(zip_path) == 0:
            print(f"  WARNING: Downloaded zip is 0 bytes — IEDB may require browser download")
            return

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(os.path.dirname(zip_path))
        print(f"  Extracted {csv_path} ({os.path.getsize(csv_path)/1e6:.1f} MB)")
    except Exception as e:
        print(f"  WARNING: MHC ligand download/extract failed: {e}")


def _create_epitope_lookup(cur):
    """Create temp lookup table mapping curated_epitope_id -> epitope_name.

    Join chain: curated_epitope.col_0 (curated_epitope_id)
                -> curated_epitope.col_6 (epitope_id)
                -> mysql_epitope.col_0 (epitope_id)
                -> mysql_epitope.col_1 (epitope description = receptor.epitope_name)
    """
    if not _table_exists(cur, "mysql_curated_epitope"):
        return False
    has_epitope = _table_exists(cur, "mysql_epitope")

    cur.execute("DROP TABLE IF EXISTS _epitope_name_lookup")
    if has_epitope:
        # Join through mysql_epitope for accurate sequence-level names
        cur.execute("""
            CREATE TEMP TABLE _epitope_name_lookup AS
            SELECT DISTINCT ce.col_0 AS curated_epitope_id, e.col_1 AS epitope_name
            FROM mysql_curated_epitope ce
            JOIN mysql_epitope e ON ce.col_6 = e.col_0
            WHERE e.col_1 != ''
        """)
    else:
        # Fall back to curated_epitope.col_2 (curated name, less accurate)
        cur.execute("""
            CREATE TEMP TABLE _epitope_name_lookup AS
            SELECT DISTINCT col_0 AS curated_epitope_id, col_2 AS epitope_name
            FROM mysql_curated_epitope WHERE col_2 != ''
        """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_enl_id ON _epitope_name_lookup(curated_epitope_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_enl_name ON _epitope_name_lookup(epitope_name)")
    cnt = _scalar(cur, "SELECT COUNT(*) FROM _epitope_name_lookup")
    print(f"  Epitope name lookup: {cnt:,} mappings")
    return True


# ---------------------------------------------------------------------------
# TSV Export Functions
# ---------------------------------------------------------------------------

def export_binding_affinity_overview(cur, output_dir):
    """Export binding_affinity_overview.tsv — per (allele, assay_type) from mysql_mhc_bind."""
    path = os.path.join(output_dir, "binding_affinity_overview.tsv")

    if not _table_exists(cur, "mysql_mhc_bind"):
        print(f"  SKIP: mysql_mhc_bind not available")
        _write_empty_tsv(path, [
            "allele", "assay_type", "assay_category", "units",
            "total_measurements", "quantitative_count", "positive_count",
            "negative_count", "unique_epitope_ids",
        ])
        return

    has_assay_type = _table_exists(cur, "mysql_assay_type")

    if has_assay_type:
        rows = cur.execute("""
            SELECT
                mb.mhc_allele_name AS allele,
                COALESCE(at.assay_type, '') AS assay_type,
                COALESCE(at.category, '') AS assay_category,
                COALESCE(at.units, '') AS units,
                COUNT(*) AS total_measurements,
                SUM(CASE WHEN mb.as_num_value != '' THEN 1 ELSE 0 END) AS quantitative_count,
                SUM(CASE WHEN mb.as_char_value IN ('Positive', 'Positive-High',
                    'Positive-Intermediate', 'Positive-Low') THEN 1 ELSE 0 END) AS positive_count,
                SUM(CASE WHEN mb.as_char_value IN ('Negative') THEN 1 ELSE 0 END) AS negative_count,
                COUNT(DISTINCT mb.curated_epitope_id) AS unique_epitope_ids
            FROM mysql_mhc_bind mb
            LEFT JOIN mysql_assay_type at ON mb.as_type_id = at.assay_type_id
            WHERE mb.mhc_allele_name != ''
            GROUP BY mb.mhc_allele_name, at.assay_type
            ORDER BY total_measurements DESC
        """).fetchall()
    else:
        rows = cur.execute("""
            SELECT
                mb.mhc_allele_name AS allele,
                '' AS assay_type,
                '' AS assay_category,
                '' AS units,
                COUNT(*) AS total_measurements,
                SUM(CASE WHEN mb.as_num_value != '' THEN 1 ELSE 0 END) AS quantitative_count,
                SUM(CASE WHEN mb.as_char_value IN ('Positive', 'Positive-High',
                    'Positive-Intermediate', 'Positive-Low') THEN 1 ELSE 0 END) AS positive_count,
                SUM(CASE WHEN mb.as_char_value IN ('Negative') THEN 1 ELSE 0 END) AS negative_count,
                COUNT(DISTINCT mb.curated_epitope_id) AS unique_epitope_ids
            FROM mysql_mhc_bind mb
            WHERE mb.mhc_allele_name != ''
            GROUP BY mb.mhc_allele_name
            ORDER BY total_measurements DESC
        """).fetchall()

    headers = ["allele", "assay_type", "assay_category", "units",
               "total_measurements", "quantitative_count", "positive_count",
               "negative_count", "unique_epitope_ids"]
    _write_tsv(path, headers, rows)
    print(f"  -> {path} ({len(rows)} rows)")


def export_mhc_allele_metadata(cur, output_dir):
    """Export mhc_allele_metadata.tsv — from mysql_mhc_allele_restriction."""
    path = os.path.join(output_dir, "mhc_allele_metadata.tsv")

    if not _table_exists(cur, "mysql_mhc_allele_restriction"):
        print(f"  SKIP: mysql_mhc_allele_restriction not available")
        _write_empty_tsv(path, [
            "allele_restriction_id", "displayed_restriction", "class", "locus",
            "organism", "haplotype", "serotype", "molecule",
            "chain_i_name", "chain_ii_name", "chain_i_gene", "chain_ii_gene",
            "restriction_level",
        ])
        return

    rows = cur.execute("""
        SELECT
            mhc_allele_restriction_id,
            displayed_restriction,
            class,
            locus,
            organism,
            haplotype,
            serotype,
            molecule,
            chain_i_name,
            chain_ii_name,
            chain_i_mhc_gene_name,
            chain_ii_mhc_gene_name,
            restriction_level
        FROM mysql_mhc_allele_restriction
        ORDER BY displayed_restriction
    """).fetchall()

    headers = ["allele_restriction_id", "displayed_restriction", "class", "locus",
               "organism", "haplotype", "serotype", "molecule",
               "chain_i_name", "chain_ii_name", "chain_i_gene", "chain_ii_gene",
               "restriction_level"]
    _write_tsv(path, headers, rows)
    print(f"  -> {path} ({len(rows)} rows)")


def export_epitope_mhc_coverage(cur, output_dir, has_lookup=False):
    """Export epitope_mhc_coverage.tsv — per epitope, evidence from each source."""
    path = os.path.join(output_dir, "epitope_mhc_coverage.tsv")

    has_mhc_bind = _table_exists(cur, "mysql_mhc_bind")

    # Collect epitopes from receptor table
    receptor_stats = cur.execute(f"""
        SELECT
            epitope_name,
            COUNT(*) AS receptor_count,
            SUM(CASE WHEN {CDR3A} IS NOT NULL OR {CDR3B} IS NOT NULL THEN 1 ELSE 0 END) AS receptor_with_cdr3
        FROM receptor
        WHERE epitope_name != ''
        GROUP BY epitope_name
    """).fetchall()
    receptor_map = {r[0]: (r[1], r[2]) for r in receptor_stats}

    # Collect epitopes from tcell table
    tcell_stats = cur.execute("""
        SELECT
            epitope_name,
            COUNT(*) AS tcell_total,
            SUM(CASE WHEN assay_qualitative_measurement IN
                ('Positive', 'Positive-Low', 'Positive-High', 'Positive-Intermediate')
                THEN 1 ELSE 0 END) AS tcell_positive_count
        FROM tcell
        WHERE epitope_name != ''
        GROUP BY epitope_name
    """).fetchall()
    tcell_map = {r[0]: (r[1], r[2]) for r in tcell_stats}

    # Collect epitopes from mhc_bind via lookup table
    mhc_bind_map = {}
    if has_mhc_bind and has_lookup:
        bind_stats = cur.execute("""
            SELECT
                enl.epitope_name,
                SUM(CASE WHEN mb.as_char_value IN ('Positive', 'Positive-High',
                    'Positive-Intermediate', 'Positive-Low') THEN 1 ELSE 0 END) AS mhc_bind_positive,
                COUNT(*) AS mhc_bind_measurements
            FROM mysql_mhc_bind mb
            JOIN _epitope_name_lookup enl ON mb.curated_epitope_id = enl.curated_epitope_id
            GROUP BY enl.epitope_name
        """).fetchall()
        mhc_bind_map = {r[0]: (r[1], r[2]) for r in bind_stats}

    # Union of all epitope names
    all_epitopes = sorted(set(receptor_map) | set(tcell_map) | set(mhc_bind_map))

    rows = []
    for ep in all_epitopes:
        rec_count, rec_cdr3 = receptor_map.get(ep, (0, 0))
        tc_total, tc_pos = tcell_map.get(ep, (0, 0))
        mb_pos, mb_meas = mhc_bind_map.get(ep, (0, 0))

        sources = []
        if rec_count > 0:
            sources.append("receptor")
        if tc_total > 0:
            sources.append("tcell")
        if mb_meas > 0:
            sources.append("mhc_bind")

        rows.append((
            ep, rec_count, rec_cdr3, tc_total, tc_pos,
            mb_pos, mb_meas, ";".join(sources),
        ))

    headers = ["epitope_name", "receptor_count", "receptor_with_cdr3",
               "tcell_total", "tcell_positive_count",
               "mhc_bind_positive_count", "mhc_bind_measurements",
               "evidence_sources"]
    _write_tsv(path, headers, rows)
    print(f"  -> {path} ({len(rows)} rows)")


def export_tcell_vs_receptor_overlap(cur, output_dir):
    """Export tcell_vs_receptor_overlap.tsv — per epitope comparison."""
    path = os.path.join(output_dir, "tcell_vs_receptor_overlap.tsv")

    # Receptor stats per epitope
    receptor_stats = cur.execute(f"""
        SELECT
            epitope_name,
            COUNT(*) AS receptor_total,
            SUM(CASE WHEN {CDR3A} IS NOT NULL AND {CDR3B} IS NOT NULL
                THEN 1 ELSE 0 END) AS receptor_paired,
            COUNT(DISTINCT rma.mhc_allele) AS receptor_mhc_alleles
        FROM receptor r
        LEFT JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
        WHERE r.epitope_name != ''
        GROUP BY r.epitope_name
    """).fetchall()
    rec_map = {r[0]: (r[1], r[2], r[3]) for r in receptor_stats}

    # Tcell stats per epitope
    tcell_stats = cur.execute("""
        SELECT
            epitope_name,
            COUNT(*) AS tcell_total,
            SUM(CASE WHEN assay_qualitative_measurement IN
                ('Positive', 'Positive-Low', 'Positive-High', 'Positive-Intermediate')
                THEN 1 ELSE 0 END) AS tcell_positive,
            COUNT(DISTINCT mhc_restriction_name) AS tcell_mhc_alleles
        FROM tcell
        WHERE epitope_name != ''
        GROUP BY epitope_name
    """).fetchall()
    tc_map = {r[0]: (r[1], r[2], r[3]) for r in tcell_stats}

    all_epitopes = sorted(set(rec_map) | set(tc_map))

    rows = []
    for ep in all_epitopes:
        rec_total, rec_paired, rec_mhc = rec_map.get(ep, (0, 0, 0))
        tc_total, tc_pos, tc_mhc = tc_map.get(ep, (0, 0, 0))

        if rec_total > 0 and tc_total > 0:
            category = "both"
        elif rec_total > 0:
            category = "receptor_only"
        else:
            category = "tcell_only"

        rows.append((
            ep, rec_total, rec_paired, rec_mhc,
            tc_total, tc_pos, tc_mhc, category,
        ))

    headers = ["epitope_name", "receptor_total", "receptor_paired",
               "receptor_mhc_alleles", "tcell_total", "tcell_positive",
               "tcell_mhc_alleles", "overlap_category"]
    _write_tsv(path, headers, rows)
    print(f"  -> {path} ({len(rows)} rows)")


def export_data_source_gaps(cur, output_dir, has_lookup=False):
    """Export data_source_gaps.tsv — epitopes/alleles with partial evidence."""
    path = os.path.join(output_dir, "data_source_gaps.tsv")

    has_mhc_bind = _table_exists(cur, "mysql_mhc_bind")

    # Gather epitope sets from each source
    rec_epitopes = set(r[0] for r in cur.execute(
        "SELECT DISTINCT epitope_name FROM receptor WHERE epitope_name != ''"
    ).fetchall())

    tc_pos_epitopes = set(r[0] for r in cur.execute("""
        SELECT DISTINCT epitope_name FROM tcell
        WHERE epitope_name != '' AND assay_qualitative_measurement IN
            ('Positive', 'Positive-Low', 'Positive-High', 'Positive-Intermediate')
    """).fetchall())

    bind_epitopes = set()
    if has_mhc_bind and has_lookup:
        bind_epitopes = set(r[0] for r in cur.execute("""
            SELECT DISTINCT enl.epitope_name
            FROM mysql_mhc_bind mb
            JOIN _epitope_name_lookup enl ON mb.curated_epitope_id = enl.curated_epitope_id
        """).fetchall())

    # Gather allele sets
    rec_alleles = set(r[0] for r in cur.execute(
        "SELECT DISTINCT mhc_allele FROM receptor_mhc_alleles"
    ).fetchall())

    bind_alleles = set()
    if has_mhc_bind:
        bind_alleles = set(r[0] for r in cur.execute(
            "SELECT DISTINCT mhc_allele_name FROM mysql_mhc_bind WHERE mhc_allele_name != ''"
        ).fetchall())

    rows = []

    # Epitope gaps
    for ep in sorted(rec_epitopes - tc_pos_epitopes):
        rows.append(("epitope", ep, "in_receptor_not_tcell_positive",
                      "receptor", "tcell_positive", ""))
    for ep in sorted(tc_pos_epitopes - rec_epitopes):
        tc_count = _scalar(cur, """
            SELECT COUNT(*) FROM tcell WHERE epitope_name = ?
            AND assay_qualitative_measurement IN
                ('Positive', 'Positive-Low', 'Positive-High', 'Positive-Intermediate')
        """, (ep,))
        rows.append(("epitope", ep, "in_tcell_not_receptor",
                      "tcell_positive", "receptor", str(tc_count)))
    for ep in sorted(bind_epitopes - rec_epitopes):
        rows.append(("epitope", ep, "in_mhc_bind_not_receptor",
                      "mhc_bind", "receptor", ""))
    for ep in sorted(rec_epitopes - bind_epitopes):
        if has_mhc_bind:
            rows.append(("epitope", ep, "in_receptor_not_mhc_bind",
                          "receptor", "mhc_bind", ""))

    # Allele gaps
    for al in sorted(bind_alleles - rec_alleles):
        rows.append(("allele", al, "in_mhc_bind_not_receptor",
                      "mhc_bind", "receptor", ""))
    for al in sorted(rec_alleles - bind_alleles):
        if has_mhc_bind:
            rows.append(("allele", al, "in_receptor_not_mhc_bind",
                          "receptor", "mhc_bind", ""))

    headers = ["entity_type", "entity_name", "gap_type",
               "source_with_data", "source_missing_data", "count_in_source"]
    _write_tsv(path, headers, rows)
    print(f"  -> {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# TSV utility
# ---------------------------------------------------------------------------
def _write_tsv(path, headers, rows):
    """Write rows to a TSV file."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(headers)
        for row in rows:
            w.writerow(row)


def _write_empty_tsv(path, headers):
    """Write a TSV with headers only."""
    _write_tsv(path, headers, [])


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------
def write_pmhc_summary(cur, output_dir):
    """Write PMHC_SUMMARY.txt with 8 narrative sections."""
    lines = []

    lines.append("=" * 80)
    lines.append("              IEDB Peptide-MHC Deep Dive Summary")
    lines.append(f"              Generated: {date.today().isoformat()}")
    lines.append("=" * 80)
    lines.append("")

    # ---- 1. DATA INVENTORY ----
    lines.append("1. DATA INVENTORY")
    lines.append("-" * 80)
    lines.append("")

    # Row counts per table
    core_tables = ["receptor", "receptor_mhc_alleles", "tcell", "epitope"]
    mysql_tables = ["mysql_mhc_bind", "mysql_mhc_elution", "mysql_assay_type",
                    "mysql_epitope", "mysql_curated_epitope", "mysql_mhc_allele_restriction"]

    lines.append("Core tables (from CSV exports):")
    for t in core_tables:
        if _table_exists(cur, t):
            cnt = _scalar(cur, f'SELECT COUNT(*) FROM "{t}"')
            lines.append(f"  {t:<30s} {cnt:>12,} rows")
        else:
            lines.append(f"  {t:<30s} NOT LOADED")

    lines.append("")
    lines.append("MySQL-derived tables:")
    mysql_loaded = []
    mysql_missing = []
    for t in mysql_tables:
        if _table_exists(cur, t):
            cnt = _scalar(cur, f'SELECT COUNT(*) FROM "{t}"')
            lines.append(f"  {t:<35s} {cnt:>12,} rows")
            mysql_loaded.append(t)
        else:
            lines.append(f"  {t:<35s} NOT LOADED")
            mysql_missing.append(t)

    lines.append("")
    if mysql_missing:
        lines.append(f"Note: {len(mysql_missing)} MySQL tables not loaded: {', '.join(mysql_missing)}")
        lines.append("Run without --skip_mysql to parse the MySQL dump.")
    else:
        lines.append("All MySQL tables loaded successfully.")
    lines.append("")

    # ---- 2. EPITOPE LANDSCAPE ----
    lines.append("2. EPITOPE LANDSCAPE")
    lines.append("-" * 80)
    lines.append("")

    rec_epi = _scalar(cur, "SELECT COUNT(DISTINCT epitope_name) FROM receptor WHERE epitope_name != ''")
    tc_epi = _scalar(cur, "SELECT COUNT(DISTINCT epitope_name) FROM tcell WHERE epitope_name != ''")
    epi_epi = _scalar(cur, "SELECT COUNT(DISTINCT epitope_name) FROM epitope WHERE epitope_name != ''")

    lines.append(f"Unique epitopes across tables:")
    lines.append(f"  receptor table:  {rec_epi:>10,}")
    lines.append(f"  tcell table:     {tc_epi:>10,}")
    lines.append(f"  epitope table:   {epi_epi:>10,}")
    lines.append("")

    # Overlap analysis
    rec_set = set(r[0] for r in cur.execute(
        "SELECT DISTINCT epitope_name FROM receptor WHERE epitope_name != ''"
    ).fetchall())
    tc_set = set(r[0] for r in cur.execute(
        "SELECT DISTINCT epitope_name FROM tcell WHERE epitope_name != ''"
    ).fetchall())
    both_rec_tc = rec_set & tc_set
    lines.append(f"Epitope overlap:")
    lines.append(f"  In both receptor and tcell:    {len(both_rec_tc):>8,}")
    lines.append(f"  In receptor only:              {len(rec_set - tc_set):>8,}")
    lines.append(f"  In tcell only:                 {len(tc_set - rec_set):>8,}")
    lines.append("")

    # Top 15 epitopes by receptor + tcell count
    top_epi = cur.execute(f"""
        SELECT
            r.epitope_name,
            COUNT(*) AS rec_cnt,
            COALESCE(t.tc_cnt, 0) AS tc_cnt
        FROM receptor r
        LEFT JOIN (
            SELECT epitope_name, COUNT(*) AS tc_cnt FROM tcell
            WHERE epitope_name != '' GROUP BY epitope_name
        ) t ON r.epitope_name = t.epitope_name
        WHERE r.epitope_name != ''
        GROUP BY r.epitope_name
        ORDER BY rec_cnt + COALESCE(t.tc_cnt, 0) DESC LIMIT 15
    """).fetchall()
    lines.append(f"Top 15 epitopes by combined receptor + tcell count:")
    lines.append(f"  {'Epitope':<30s} {'Receptor':>10s} {'Tcell':>10s} {'Total':>10s}")
    for ep, rec, tc in top_epi:
        lines.append(f"  {ep:<30s} {rec:>10,} {tc:>10,} {rec+tc:>10,}")
    lines.append("")

    # Evidence source distribution from epitope_mhc_coverage.tsv
    cov_path = os.path.join(output_dir, "epitope_mhc_coverage.tsv")
    if os.path.exists(cov_path):
        source_counts = {}
        with open(cov_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                key = row["evidence_sources"]
                source_counts[key] = source_counts.get(key, 0) + 1
        lines.append(f"Evidence source distribution:")
        for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {src:<40s} {cnt:>8,} epitopes")
        lines.append("")

    # ---- 3. MHC ALLELE LANDSCAPE ----
    lines.append("3. MHC ALLELE LANDSCAPE")
    lines.append("-" * 80)
    lines.append("")

    rec_alleles_cnt = _scalar(cur, "SELECT COUNT(DISTINCT mhc_allele) FROM receptor_mhc_alleles")
    tc_alleles_cnt = _scalar(cur,
        "SELECT COUNT(DISTINCT mhc_restriction_name) FROM tcell WHERE mhc_restriction_name != ''")

    lines.append(f"Unique MHC alleles:")
    lines.append(f"  receptor (normalized):    {rec_alleles_cnt:>8,}")
    lines.append(f"  tcell:                    {tc_alleles_cnt:>8,}")

    has_mhc_restrict = _table_exists(cur, "mysql_mhc_allele_restriction")
    if has_mhc_restrict:
        restrict_cnt = _scalar(cur, "SELECT COUNT(*) FROM mysql_mhc_allele_restriction")
        lines.append(f"  mhc_allele_restriction:   {restrict_cnt:>8,}")

    if has_mhc_bind := _table_exists(cur, "mysql_mhc_bind"):
        bind_alleles = _scalar(cur,
            "SELECT COUNT(DISTINCT mhc_allele_name) FROM mysql_mhc_bind WHERE mhc_allele_name != ''")
        lines.append(f"  mhc_bind:                 {bind_alleles:>8,}")
    lines.append("")

    # Class distribution from authoritative source
    if has_mhc_restrict:
        class_dist = cur.execute("""
            SELECT class, COUNT(*) AS cnt
            FROM mysql_mhc_allele_restriction
            WHERE class != ''
            GROUP BY class ORDER BY cnt DESC
        """).fetchall()
        lines.append(f"MHC class distribution (from mhc_allele_restriction, authoritative):")
        for cls, cnt in class_dist:
            lines.append(f"  {cls:<20s} {cnt:>8,}")
        lines.append("")

    # Heuristic class for receptor alleles
    all_rec_alleles = cur.execute(
        "SELECT DISTINCT mhc_allele FROM receptor_mhc_alleles"
    ).fetchall()
    heuristic_class = {"Class I": 0, "Class II": 0, "Unknown": 0}
    for (a,) in all_rec_alleles:
        heuristic_class[classify_mhc(a)] += 1
    lines.append(f"Receptor allele class (heuristic): {dict(heuristic_class)}")
    lines.append("")

    # Top 15 receptor alleles
    top_rec_alleles = cur.execute("""
        SELECT mhc_allele, COUNT(DISTINCT receptor_rowid) AS cnt
        FROM receptor_mhc_alleles
        GROUP BY mhc_allele ORDER BY cnt DESC LIMIT 15
    """).fetchall()
    lines.append(f"Top 15 receptor alleles by unique receptor count:")
    for a, cnt in top_rec_alleles:
        lines.append(f"  {a:<30s} {cnt:>8,}")
    lines.append("")

    # ---- 4. BINDING DATA ----
    lines.append("4. BINDING DATA")
    lines.append("-" * 80)
    lines.append("")

    if not _table_exists(cur, "mysql_mhc_bind"):
        lines.append("mysql_mhc_bind NOT LOADED — no binding data available.")
        lines.append("Run without --skip_mysql to parse binding data from the MySQL dump.")
        lines.append("")
    else:
        bind_total = _scalar(cur, "SELECT COUNT(*) FROM mysql_mhc_bind")
        bind_quant = _scalar(cur,
            "SELECT COUNT(*) FROM mysql_mhc_bind WHERE as_num_value != ''")
        bind_qual_only = bind_total - bind_quant

        lines.append(f"Total binding measurements:   {bind_total:>10,}")
        lines.append(f"  With numeric value:         {bind_quant:>10,} ({100*bind_quant/max(bind_total,1):.1f}%)")
        lines.append(f"  Qualitative only:           {bind_qual_only:>10,}")
        lines.append("")

        # By char value
        char_dist = cur.execute("""
            SELECT as_char_value, COUNT(*) AS cnt
            FROM mysql_mhc_bind WHERE as_char_value != ''
            GROUP BY as_char_value ORDER BY cnt DESC
        """).fetchall()
        lines.append(f"Qualitative result distribution:")
        for val, cnt in char_dist:
            lines.append(f"  {val:<30s} {cnt:>10,}")
        lines.append("")

        # By assay type (top 15)
        if _table_exists(cur, "mysql_assay_type"):
            by_type = cur.execute("""
                SELECT at.assay_type, COUNT(*) AS cnt
                FROM mysql_mhc_bind mb
                JOIN mysql_assay_type at ON mb.as_type_id = at.assay_type_id
                GROUP BY at.assay_type ORDER BY cnt DESC LIMIT 15
            """).fetchall()
            lines.append(f"Top 15 assay types:")
            for method, cnt in by_type:
                lines.append(f"  {method:<40s} {cnt:>10,}")
            lines.append("")

        # Top 15 alleles
        top_bind_alleles = cur.execute("""
            SELECT mhc_allele_name, COUNT(*) AS cnt
            FROM mysql_mhc_bind WHERE mhc_allele_name != ''
            GROUP BY mhc_allele_name ORDER BY cnt DESC LIMIT 15
        """).fetchall()
        lines.append(f"Top 15 alleles by binding measurement count:")
        for a, cnt in top_bind_alleles:
            lines.append(f"  {a:<30s} {cnt:>10,}")
        lines.append("")

        # Top 15 pMHC pairs via epitope name lookup
        if _table_exists(cur, "_epitope_name_lookup"):
            top_bind_pmhc = cur.execute("""
                SELECT enl.epitope_name, mb.mhc_allele_name, COUNT(*) AS cnt
                FROM mysql_mhc_bind mb
                JOIN _epitope_name_lookup enl ON mb.curated_epitope_id = enl.curated_epitope_id
                WHERE mb.mhc_allele_name != ''
                GROUP BY enl.epitope_name, mb.mhc_allele_name
                ORDER BY cnt DESC LIMIT 15
            """).fetchall()
            lines.append(f"Top 15 pMHC pairs by binding measurement count:")
            for pep, allele, cnt in top_bind_pmhc:
                lines.append(f"  {pep:<25s} + {allele:<20s} {cnt:>8,}")
            lines.append("")

    # ---- 5. T-CELL EVIDENCE ----
    lines.append("5. T-CELL EVIDENCE")
    lines.append("-" * 80)
    lines.append("")

    tcell_total = _scalar(cur, "SELECT COUNT(*) FROM tcell")
    tcell_pos = _scalar(cur, """
        SELECT COUNT(*) FROM tcell
        WHERE assay_qualitative_measurement IN
            ('Positive', 'Positive-Low', 'Positive-High', 'Positive-Intermediate')
    """)
    tcell_neg = _scalar(cur, """
        SELECT COUNT(*) FROM tcell
        WHERE assay_qualitative_measurement = 'Negative'
    """)

    lines.append(f"Total T-cell assays:   {tcell_total:>10,}")
    lines.append(f"  Positive:            {tcell_pos:>10,} ({100*tcell_pos/max(tcell_total,1):.1f}%)")
    lines.append(f"  Negative:            {tcell_neg:>10,} ({100*tcell_neg/max(tcell_total,1):.1f}%)")
    lines.append("")

    # By method (top 15)
    tc_methods = cur.execute("""
        SELECT assay_method, COUNT(*) AS cnt
        FROM tcell WHERE assay_method != ''
        GROUP BY assay_method ORDER BY cnt DESC LIMIT 15
    """).fetchall()
    lines.append(f"Top 15 assay methods:")
    for method, cnt in tc_methods:
        lines.append(f"  {method:<40s} {cnt:>10,}")
    lines.append("")

    # MHC class from mhc_restriction_class
    tc_mhc_class = cur.execute("""
        SELECT mhc_restriction_class, COUNT(*) AS cnt
        FROM tcell WHERE mhc_restriction_class != ''
        GROUP BY mhc_restriction_class ORDER BY cnt DESC
    """).fetchall()
    lines.append(f"T-cell assays by MHC restriction class:")
    for cls, cnt in tc_mhc_class:
        lines.append(f"  {cls:<20s} {cnt:>10,}")
    lines.append("")

    # Top 15 epitope-MHC pairs
    tc_top_pmhc = cur.execute("""
        SELECT epitope_name, mhc_restriction_name, COUNT(*) AS cnt
        FROM tcell
        WHERE epitope_name != '' AND mhc_restriction_name != ''
            AND assay_qualitative_measurement IN
                ('Positive', 'Positive-Low', 'Positive-High', 'Positive-Intermediate')
        GROUP BY epitope_name, mhc_restriction_name
        ORDER BY cnt DESC LIMIT 15
    """).fetchall()
    lines.append(f"Top 15 epitope-MHC pairs (positive assays):")
    for ep, mhc, cnt in tc_top_pmhc:
        lines.append(f"  {ep:<25s} + {mhc:<20s} {cnt:>8,}")
    lines.append("")

    # ---- 6. RECEPTOR COVERAGE ----
    lines.append("6. RECEPTOR COVERAGE")
    lines.append("-" * 80)
    lines.append("")

    rec_total = _scalar(cur, "SELECT COUNT(*) FROM receptor")
    rec_paired = _scalar(cur, f"""
        SELECT COUNT(*) FROM receptor
        WHERE {CDR3A} IS NOT NULL AND {CDR3B} IS NOT NULL
    """)
    rec_unpaired = rec_total - rec_paired

    lines.append(f"Total receptor records:  {rec_total:>10,}")
    lines.append(f"  Paired (a+b):          {rec_paired:>10,} ({100*rec_paired/max(rec_total,1):.1f}%)")
    lines.append(f"  Unpaired:              {rec_unpaired:>10,}")
    lines.append("")

    # Annotation rates
    cdr3a_cnt = _scalar(cur, f"SELECT COUNT(*) FROM receptor WHERE {CDR3A} IS NOT NULL")
    cdr3b_cnt = _scalar(cur, f"SELECT COUNT(*) FROM receptor WHERE {CDR3B} IS NOT NULL")
    vga_cnt = _scalar(cur, f"SELECT COUNT(*) FROM receptor WHERE {VGA} IS NOT NULL")
    vgb_cnt = _scalar(cur, f"SELECT COUNT(*) FROM receptor WHERE {VGB} IS NOT NULL")
    jga_cnt = _scalar(cur, f"SELECT COUNT(*) FROM receptor WHERE {JGA} IS NOT NULL")
    jgb_cnt = _scalar(cur, f"SELECT COUNT(*) FROM receptor WHERE {JGB} IS NOT NULL")

    lines.append(f"CDR3/V/J annotation rates:")
    lines.append(f"  CDR3 alpha: {cdr3a_cnt:>9,} / {rec_total:,} ({100*cdr3a_cnt/max(rec_total,1):.1f}%)")
    lines.append(f"  CDR3 beta:  {cdr3b_cnt:>9,} / {rec_total:,} ({100*cdr3b_cnt/max(rec_total,1):.1f}%)")
    lines.append(f"  V alpha:    {vga_cnt:>9,} / {rec_total:,} ({100*vga_cnt/max(rec_total,1):.1f}%)")
    lines.append(f"  V beta:     {vgb_cnt:>9,} / {rec_total:,} ({100*vgb_cnt/max(rec_total,1):.1f}%)")
    lines.append(f"  J alpha:    {jga_cnt:>9,} / {rec_total:,} ({100*jga_cnt/max(rec_total,1):.1f}%)")
    lines.append(f"  J beta:     {jgb_cnt:>9,} / {rec_total:,} ({100*jgb_cnt/max(rec_total,1):.1f}%)")
    lines.append("")

    # Top 15 pMHC by unique TCR count
    top_pmhc_tcr = cur.execute(f"""
        SELECT r.epitope_name, rma.mhc_allele,
               COUNT(DISTINCT COALESCE({CDR3A}, '') || '|' || COALESCE({CDR3B}, '')) AS tcr_cnt
        FROM receptor r
        JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
        WHERE r.epitope_name != ''
          AND ({CDR3A} IS NOT NULL OR {CDR3B} IS NOT NULL)
        GROUP BY r.epitope_name, rma.mhc_allele
        ORDER BY tcr_cnt DESC LIMIT 15
    """).fetchall()
    lines.append(f"Top 15 pMHC pairs by unique TCR count:")
    for ep, allele, cnt in top_pmhc_tcr:
        lines.append(f"  {ep:<25s} + {allele:<20s} {cnt:>8,}")
    lines.append("")

    # ---- 7. CROSS-TABLE CONCORDANCE ----
    lines.append("7. CROSS-TABLE CONCORDANCE")
    lines.append("-" * 80)
    lines.append("")

    # Epitopes with multi-source support
    cov_path = os.path.join(output_dir, "epitope_mhc_coverage.tsv")
    if os.path.exists(cov_path):
        multi_source = []
        with open(cov_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sources = row["evidence_sources"].split(";")
                if len(sources) >= 2:
                    multi_source.append((
                        row["epitope_name"],
                        int(row["receptor_count"]),
                        int(row["tcell_total"]),
                        int(row["mhc_bind_measurements"]),
                        row["evidence_sources"],
                    ))
        multi_source.sort(key=lambda x: -(x[1] + x[2] + x[3]))

        lines.append(f"Epitopes with multi-source support: {len(multi_source)}")
        lines.append("")
        lines.append(f"Top 20 multi-source epitopes:")
        lines.append(f"  {'Epitope':<30s} {'Receptor':>10s} {'Tcell':>10s} {'MHC_bind':>10s} {'Sources'}")
        for ep, rec, tc, mb, src in multi_source[:20]:
            lines.append(f"  {ep:<30s} {rec:>10,} {tc:>10,} {mb:>10,} {src}")
        lines.append("")

    # Overlap category distribution from tcell_vs_receptor_overlap.tsv
    overlap_path = os.path.join(output_dir, "tcell_vs_receptor_overlap.tsv")
    if os.path.exists(overlap_path):
        cat_counts = {}
        with open(overlap_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cat = row["overlap_category"]
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        lines.append(f"Tcell vs receptor overlap categories:")
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat:<25s} {cnt:>8,} epitopes")
        lines.append("")

    # Alleles across multiple tables
    rec_allele_set = set(r[0] for r in cur.execute(
        "SELECT DISTINCT mhc_allele FROM receptor_mhc_alleles"
    ).fetchall())
    tc_allele_set = set(r[0] for r in cur.execute(
        "SELECT DISTINCT mhc_restriction_name FROM tcell WHERE mhc_restriction_name != ''"
    ).fetchall())
    bind_allele_set = set()
    if _table_exists(cur, "mysql_mhc_bind"):
        bind_allele_set = set(r[0] for r in cur.execute(
            "SELECT DISTINCT mhc_allele_name FROM mysql_mhc_bind WHERE mhc_allele_name != ''"
        ).fetchall())

    all_allele = rec_allele_set | tc_allele_set | bind_allele_set
    in_all_three = rec_allele_set & tc_allele_set & bind_allele_set
    in_two = ((rec_allele_set & tc_allele_set) | (rec_allele_set & bind_allele_set) |
              (tc_allele_set & bind_allele_set)) - in_all_three

    lines.append(f"MHC allele cross-table coverage:")
    lines.append(f"  Total unique alleles:     {len(all_allele):>6,}")
    lines.append(f"  In all 3 sources:         {len(in_all_three):>6,}")
    lines.append(f"  In exactly 2 sources:     {len(in_two):>6,}")
    lines.append(f"  In 1 source only:         {len(all_allele - in_all_three - in_two):>6,}")
    lines.append("")

    # ---- 8. GAPS AND RECOMMENDATIONS ----
    lines.append("8. GAPS AND RECOMMENDATIONS")
    lines.append("-" * 80)
    lines.append("")

    # Read gaps TSV
    gaps_path = os.path.join(output_dir, "data_source_gaps.tsv")
    if os.path.exists(gaps_path):
        gap_type_counts = {}
        with open(gaps_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                key = f"{row['entity_type']}:{row['gap_type']}"
                gap_type_counts[key] = gap_type_counts.get(key, 0) + 1

        lines.append(f"Data gap summary:")
        for gap, cnt in sorted(gap_type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {gap:<50s} {cnt:>8,}")
        lines.append("")

    # Epitopes with binding data but no TCR data
    if _table_exists(cur, "mysql_mhc_bind") and _table_exists(cur, "_epitope_name_lookup"):
        bind_no_rec = cur.execute("""
            SELECT enl.epitope_name, COUNT(*) AS cnt
            FROM mysql_mhc_bind mb
            JOIN _epitope_name_lookup enl ON mb.curated_epitope_id = enl.curated_epitope_id
            WHERE enl.epitope_name NOT IN (SELECT DISTINCT epitope_name FROM receptor WHERE epitope_name != '')
            GROUP BY enl.epitope_name
            ORDER BY cnt DESC LIMIT 15
        """).fetchall()
        if bind_no_rec:
            lines.append(f"Top 15 epitopes with binding data but no TCR receptor data:")
            for ep, cnt in bind_no_rec:
                lines.append(f"  {ep:<30s} {cnt:>8,} binding measurements")
            lines.append("")

    # Alleles with binding but no receptor data
    if _table_exists(cur, "mysql_mhc_bind"):
        bind_alleles_no_rec = cur.execute("""
            SELECT mhc_allele_name, COUNT(*) AS cnt
            FROM mysql_mhc_bind
            WHERE mhc_allele_name != ''
              AND mhc_allele_name NOT IN (SELECT DISTINCT mhc_allele FROM receptor_mhc_alleles)
            GROUP BY mhc_allele_name
            ORDER BY cnt DESC LIMIT 15
        """).fetchall()
        if bind_alleles_no_rec:
            lines.append(f"Top 15 alleles with binding data but no TCR receptor data:")
            for a, cnt in bind_alleles_no_rec:
                lines.append(f"  {a:<30s} {cnt:>8,} binding measurements")
            lines.append("")

    # Modeling implications
    lines.append(f"Modeling implications:")
    lines.append(f"  a) The receptor table ({rec_epi:,} epitopes) is the primary TCR-pMHC training source,")
    lines.append(f"     but only covers a small fraction of known IEDB epitopes ({epi_epi:,} total).")
    lines.append(f"  b) MHC binding data provides complementary pMHC affinity labels that can")
    lines.append(f"     augment TCR specificity models (semi-supervised learning).")
    lines.append(f"  c) T-cell assays ({tcell_total:,} total) capture functional reactivity and can be")
    lines.append(f"     used as weak labels for epitopes not in the receptor table.")

    if _table_exists(cur, "mysql_mhc_bind"):
        bind_total = _scalar(cur, "SELECT COUNT(*) FROM mysql_mhc_bind")
        lines.append(f"  d) {bind_total:,} binding measurements provide quantitative affinity data")
        lines.append(f"     (IC50/Kd) for training MHC binding predictors.")

    lines.append(f"  e) Cross-table epitope overlap is limited — integrating all sources")
    lines.append(f"     maximizes coverage for multi-task learning approaches.")
    lines.append("")

    # Footer
    lines.append("=" * 80)
    lines.append(f"Output files in {output_dir}/:")
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith(".tsv"):
            fpath = os.path.join(output_dir, fname)
            with open(fpath) as f:
                row_count = sum(1 for _ in f) - 1  # subtract header
            lines.append(f"  {fname:<40s} {row_count:>8,} rows")
    lines.append("=" * 80)

    summary_path = os.path.join(output_dir, "PMHC_SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="IEDB Peptide-MHC Deep Dive Analysis"
    )
    parser.add_argument(
        "--db_path", default=DB_PATH,
        help=f"Path to IEDB SQLite database (default: {DB_PATH})",
    )
    parser.add_argument(
        "--output_dir", default=OUTPUT_DIR,
        help=f"Output directory for TSVs and summary (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--iedb_dir", default="data/databases/IEDB",
        help="Root directory containing IEDB exports (default: data/databases/IEDB)",
    )
    parser.add_argument(
        "--skip_mysql", action="store_true",
        help="Skip MySQL dump parsing even if tables are missing",
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip mhc_ligand re-download attempt",
    )
    args = parser.parse_args()

    db_path = args.db_path
    output_dir = args.output_dir
    iedb_dir = args.iedb_dir
    mysql_dump_path = os.path.join(iedb_dir, "full_database", "iedb_public.sql.gz")

    if not os.path.exists(db_path):
        print(f"ERROR: SQLite DB not found at {db_path}")
        print("Run iedb_sqlite_summary.py first to build the database.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"IEDB pMHC Deep Dive Analysis")
    print(f"  Database: {db_path}")
    print(f"  Output:   {output_dir}")
    print()

    # Step 1: Ensure MySQL tables
    if not args.skip_mysql:
        print("Step 1: Ensuring MySQL tables...")
        try:
            ensure_mysql_tables(db_path, mysql_dump_path)
        except Exception as e:
            print(f"  WARNING: MySQL table loading failed: {e}")
            print(f"  Continuing with available data...")
    else:
        print("Step 1: Skipping MySQL table loading (--skip_mysql)")
    print()

    # Step 2: Ensure mhc_ligand
    if not args.skip_download:
        print("Step 2: Checking MHC ligand data...")
        try:
            ensure_mhc_ligand(iedb_dir, db_path)
        except Exception as e:
            print(f"  WARNING: MHC ligand download failed: {e}")
    else:
        print("Step 2: Skipping MHC ligand download (--skip_download)")
    print()

    # Step 3: Run analyses
    print("Step 3: Running pMHC analyses...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create epitope name lookup table
    has_lookup = _create_epitope_lookup(cur)

    exports = [
        ("binding_affinity_overview", lambda c, o: export_binding_affinity_overview(c, o)),
        ("mhc_allele_metadata", lambda c, o: export_mhc_allele_metadata(c, o)),
        ("epitope_mhc_coverage", lambda c, o: export_epitope_mhc_coverage(c, o, has_lookup)),
        ("tcell_vs_receptor_overlap", lambda c, o: export_tcell_vs_receptor_overlap(c, o)),
        ("data_source_gaps", lambda c, o: export_data_source_gaps(c, o, has_lookup)),
    ]

    for name, func in exports:
        print(f"\nExporting {name}...")
        func(cur, output_dir)

    # Step 4: Write summary
    print(f"\nStep 4: Writing PMHC_SUMMARY.txt...")
    write_pmhc_summary(cur, output_dir)

    conn.close()

    # Step 5: Verification
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check MySQL tables
    for t in ["mysql_mhc_bind", "mysql_mhc_allele_restriction", "mysql_assay_type"]:
        if _table_exists(cur, t):
            cnt = _scalar(cur, f'SELECT COUNT(*) FROM "{t}"')
            print(f"  {t}: {cnt:,} rows [OK]")
        else:
            print(f"  {t}: NOT PRESENT [WARN]")

    # Check mhc_bind has quant + qual
    if _table_exists(cur, "mysql_mhc_bind"):
        quant = _scalar(cur, "SELECT COUNT(*) FROM mysql_mhc_bind WHERE as_num_value != ''")
        qual = _scalar(cur, "SELECT COUNT(*) FROM mysql_mhc_bind WHERE as_char_value != ''")
        print(f"  mysql_mhc_bind: {quant:,} quantitative, {qual:,} qualitative values")

    # Check TSV files
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith(".tsv"):
            fpath = os.path.join(output_dir, fname)
            with open(fpath) as f:
                header = f.readline().strip()
                row_count = sum(1 for _ in f)
            print(f"  {fname}: {row_count} rows, columns: {header[:80]}...")

    # Check epitope_mhc_coverage has receptor epitopes
    cov_path = os.path.join(output_dir, "epitope_mhc_coverage.tsv")
    if os.path.exists(cov_path):
        rec_in_cov = 0
        with open(cov_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if int(row["receptor_count"]) > 0:
                    rec_in_cov += 1
        rec_epi_count = _scalar(cur,
            "SELECT COUNT(DISTINCT epitope_name) FROM receptor WHERE epitope_name != ''")
        print(f"  epitope_mhc_coverage: {rec_in_cov} receptor epitopes "
              f"(expected {rec_epi_count}) {'[OK]' if rec_in_cov == rec_epi_count else '[MISMATCH]'}")

    # Check cross-table join
    if _table_exists(cur, "mysql_mhc_bind") and _table_exists(cur, "mysql_curated_epitope"):
        cross_count = cur.execute("""
            SELECT COUNT(DISTINCT enl.epitope_name)
            FROM mysql_mhc_bind mb
            JOIN (
                SELECT DISTINCT col_0 AS curated_epitope_id, col_2 AS epitope_name
                FROM mysql_curated_epitope WHERE col_2 != ''
            ) enl ON mb.curated_epitope_id = enl.curated_epitope_id
            WHERE enl.epitope_name IN (SELECT DISTINCT epitope_name FROM receptor WHERE epitope_name != '')
        """).fetchone()[0]
        print(f"  Cross-table join: {cross_count} receptor epitopes also in mhc_bind "
              f"{'[OK]' if cross_count > 0 else '[WARN: no overlap]'}")

    # Check PMHC_SUMMARY.txt has all 8 sections
    summary_path = os.path.join(output_dir, "PMHC_SUMMARY.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary_text = f.read()
        for section in range(1, 9):
            marker = f"{section}. "
            if marker in summary_text:
                print(f"  PMHC_SUMMARY.txt section {section}: present")
            else:
                print(f"  PMHC_SUMMARY.txt section {section}: MISSING")

    conn.close()
    print(f"\nDone. Output in {output_dir}/")


if __name__ == "__main__":
    main()
