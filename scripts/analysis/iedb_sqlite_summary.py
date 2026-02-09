#!/usr/bin/env python3
"""
IEDB SQLite Loader and Summary Statistics Generator

Loads IEDB CSV exports (receptor, tcell, epitope, mhc_ligand) and MySQL SQL dump
into a SQLite database and produces comprehensive summary statistics on unique TCRs,
epitopes, MHC alleles, peptide-MHC complexes, TCR-pMHC complexes, binding data,
and quantitative assay measurements.

Usage:
    python scripts/analysis/iedb_sqlite_summary.py
    python scripts/analysis/iedb_sqlite_summary.py --iedb_dir data/databases/IEDB --output_dir data/analysis/iedb
    python scripts/analysis/iedb_sqlite_summary.py --skip_load  # reuse existing DB
    python scripts/analysis/iedb_sqlite_summary.py --skip_mysql  # skip MySQL dump parsing
"""

import argparse
import csv
import gzip
import json
import os
import re
import sqlite3
import sys
import time


# ---------------------------------------------------------------------------
# Column name generation
# ---------------------------------------------------------------------------

def make_column_names(group_row, sub_row):
    """Combine 2-row CSV headers into clean snake_case column names.

    Deduplicates with _2, _3, ... suffixes.
    """
    names = []
    seen = {}
    for g, s in zip(group_row, sub_row):
        g = g.strip()
        s = s.strip()
        raw = f"{g}_{s}" if g else s
        # snake_case: lower, replace non-alnum with _, collapse multiples
        col = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
        if col in seen:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
        else:
            seen[col] = 1
        names.append(col)
    return names


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_receptor(db_path, csv_path):
    """Load receptor CSV (all 70 columns) into SQLite."""
    print(f"\n{'='*60}")
    print(f"Loading receptor table from {csv_path}")
    print(f"{'='*60}")
    t0 = time.time()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        group_row = next(reader)
        sub_row = next(reader)

    col_names = make_column_names(group_row, sub_row)
    ncols = len(col_names)
    print(f"  Columns: {ncols}")

    # Create table
    col_defs = ", ".join(f'"{c}" TEXT' for c in col_names)
    cur.execute(f"DROP TABLE IF EXISTS receptor")
    cur.execute(f"CREATE TABLE receptor (rowid INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs})")

    # Insert rows
    placeholders = ", ".join(["?"] * ncols)
    insert_sql = f"INSERT INTO receptor ({', '.join(f'\"' + c + '\"' for c in col_names)}) VALUES ({placeholders})"

    row_count = 0
    batch = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip group row
        next(reader)  # skip sub row
        for row in reader:
            # Pad or truncate to expected column count
            if len(row) < ncols:
                row = row + [""] * (ncols - len(row))
            elif len(row) > ncols:
                row = row[:ncols]
            batch.append(row)
            if len(batch) >= 50_000:
                cur.executemany(insert_sql, batch)
                row_count += len(batch)
                batch = []
                print(f"  ... {row_count:,} rows loaded", end="\r")

    if batch:
        cur.executemany(insert_sql, batch)
        row_count += len(batch)

    print(f"  Loaded {row_count:,} rows in {time.time()-t0:.1f}s")

    # Create indexes
    print("  Creating indexes...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_receptor_type ON receptor(receptor_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_receptor_epitope ON receptor(epitope_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_receptor_mhc ON receptor(assay_mhc_allele_names)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_receptor_c1_cdr3_cur ON receptor(chain_1_cdr3_curated)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_receptor_c1_cdr3_calc ON receptor(chain_1_cdr3_calculated)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_receptor_c2_cdr3_cur ON receptor(chain_2_cdr3_curated)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_receptor_c2_cdr3_calc ON receptor(chain_2_cdr3_calculated)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_receptor_id ON receptor(receptor_iedb_receptor_id)")

    conn.commit()

    # Normalize MHC alleles into junction table
    print("  Normalizing MHC alleles into junction table...")
    cur.execute("DROP TABLE IF EXISTS receptor_mhc_alleles")
    cur.execute("""
        CREATE TABLE receptor_mhc_alleles (
            receptor_rowid INTEGER,
            mhc_allele TEXT,
            FOREIGN KEY (receptor_rowid) REFERENCES receptor(rowid)
        )
    """)

    cur.execute("SELECT rowid, assay_mhc_allele_names FROM receptor WHERE assay_mhc_allele_names != ''")
    allele_batch = []
    for rid, alleles_str in cur.fetchall():
        for allele in alleles_str.split(","):
            allele = allele.strip()
            if allele:
                allele_batch.append((rid, allele))
        if len(allele_batch) >= 100_000:
            cur.executemany("INSERT INTO receptor_mhc_alleles VALUES (?, ?)", allele_batch)
            allele_batch = []
    if allele_batch:
        cur.executemany("INSERT INTO receptor_mhc_alleles VALUES (?, ?)", allele_batch)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_rma_allele ON receptor_mhc_alleles(mhc_allele)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rma_rowid ON receptor_mhc_alleles(receptor_rowid)")
    conn.commit()

    allele_count = cur.execute("SELECT COUNT(*) FROM receptor_mhc_alleles").fetchone()[0]
    print(f"  Junction table: {allele_count:,} (receptor, allele) pairs")

    conn.close()
    print(f"  Receptor table complete ({time.time()-t0:.1f}s total)")


def load_tcell(db_path, csv_path):
    """Load tcell CSV (20 of 161 columns) into SQLite."""
    print(f"\n{'='*60}")
    print(f"Loading tcell table from {csv_path}")
    print(f"{'='*60}")
    t0 = time.time()

    # Column indices and names for the subset we extract
    col_indices = [0, 1, 3, 9, 10, 11, 23, 43, 118, 119, 120, 122, 123, 124, 125, 126, 127, 134, 141, 145]
    col_names = [
        "assay_iedb_iri",
        "reference_iedb_iri",
        "reference_pmid",
        "epitope_iedb_iri",
        "epitope_object_type",
        "epitope_name",
        "epitope_source_organism",
        "host_name",
        "assay_method",
        "assay_response_measured",
        "assay_units",
        "assay_qualitative_measurement",
        "assay_measurement_inequality",
        "assay_quantitative_measurement",
        "assay_num_subjects_tested",
        "assay_num_subjects_positive",
        "assay_response_frequency",
        "effector_cell_tcr_name",
        "mhc_restriction_name",
        "mhc_restriction_class",
    ]
    ncols = len(col_names)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    col_defs = ", ".join(f'"{c}" TEXT' for c in col_names)
    cur.execute("DROP TABLE IF EXISTS tcell")
    cur.execute(f"CREATE TABLE tcell (rowid INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs})")

    placeholders = ", ".join(["?"] * ncols)
    insert_sql = f"INSERT INTO tcell ({', '.join(f'\"' + c + '\"' for c in col_names)}) VALUES ({placeholders})"

    row_count = 0
    batch = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip group row
        next(reader)  # skip sub row
        for row in reader:
            extracted = []
            for idx in col_indices:
                if idx < len(row):
                    extracted.append(row[idx])
                else:
                    extracted.append("")
            batch.append(extracted)
            if len(batch) >= 50_000:
                cur.executemany(insert_sql, batch)
                row_count += len(batch)
                batch = []
                print(f"  ... {row_count:,} rows loaded", end="\r")

    if batch:
        cur.executemany(insert_sql, batch)
        row_count += len(batch)

    print(f"  Loaded {row_count:,} rows in {time.time()-t0:.1f}s")

    print("  Creating indexes...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tcell_epitope ON tcell(epitope_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tcell_mhc ON tcell(mhc_restriction_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tcell_mhc_class ON tcell(mhc_restriction_class)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tcell_qual ON tcell(assay_qualitative_measurement)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tcell_quant ON tcell(assay_quantitative_measurement)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tcell_tcr ON tcell(effector_cell_tcr_name)")

    conn.commit()
    conn.close()
    print(f"  Tcell table complete ({time.time()-t0:.1f}s total)")


def load_epitope(db_path, csv_path):
    """Load epitope CSV (all 32 columns) into SQLite."""
    print(f"\n{'='*60}")
    print(f"Loading epitope table from {csv_path}")
    print(f"{'='*60}")
    t0 = time.time()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        group_row = next(reader)
        sub_row = next(reader)

    col_names = make_column_names(group_row, sub_row)
    ncols = len(col_names)
    print(f"  Columns: {ncols}")

    col_defs = ", ".join(f'"{c}" TEXT' for c in col_names)
    cur.execute("DROP TABLE IF EXISTS epitope")
    cur.execute(f"CREATE TABLE epitope (rowid INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs})")

    placeholders = ", ".join(["?"] * ncols)
    insert_sql = f"INSERT INTO epitope ({', '.join(f'\"' + c + '\"' for c in col_names)}) VALUES ({placeholders})"

    row_count = 0
    batch = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        for row in reader:
            if len(row) < ncols:
                row = row + [""] * (ncols - len(row))
            elif len(row) > ncols:
                row = row[:ncols]
            batch.append(row)
            if len(batch) >= 100_000:
                cur.executemany(insert_sql, batch)
                row_count += len(batch)
                batch = []
                print(f"  ... {row_count:,} rows loaded", end="\r")

    if batch:
        cur.executemany(insert_sql, batch)
        row_count += len(batch)

    print(f"  Loaded {row_count:,} rows in {time.time()-t0:.1f}s")

    print("  Creating indexes...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_epitope_name ON epitope(epitope_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_epitope_type ON epitope(epitope_object_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_epitope_organism ON epitope(epitope_source_organism)")

    conn.commit()
    conn.close()
    print(f"  Epitope table complete ({time.time()-t0:.1f}s total)")


def load_mhc_ligand(db_path, csv_path):
    """Load MHC ligand CSV (all columns) into SQLite.

    The MHC ligand export uses the same 2-row header as other IEDB v3 exports.
    If the file doesn't exist, prints a warning and returns without error.
    """
    if not os.path.exists(csv_path):
        print(f"\n  WARNING: MHC ligand CSV not found at {csv_path} — skipping")
        return

    print(f"\n{'='*60}")
    print(f"Loading mhc_ligand table from {csv_path}")
    print(f"{'='*60}")
    t0 = time.time()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        group_row = next(reader)
        sub_row = next(reader)

    col_names = make_column_names(group_row, sub_row)
    ncols = len(col_names)
    print(f"  Columns: {ncols}")

    col_defs = ", ".join(f'"{c}" TEXT' for c in col_names)
    cur.execute("DROP TABLE IF EXISTS mhc_ligand")
    cur.execute(f"CREATE TABLE mhc_ligand (rowid INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs})")

    placeholders = ", ".join(["?"] * ncols)
    insert_sql = f"INSERT INTO mhc_ligand ({', '.join(f'\"' + c + '\"' for c in col_names)}) VALUES ({placeholders})"

    row_count = 0
    batch = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip group row
        next(reader)  # skip sub row
        for row in reader:
            if len(row) < ncols:
                row = row + [""] * (ncols - len(row))
            elif len(row) > ncols:
                row = row[:ncols]
            batch.append(row)
            if len(batch) >= 100_000:
                cur.executemany(insert_sql, batch)
                row_count += len(batch)
                batch = []
                print(f"  ... {row_count:,} rows loaded", end="\r")

    if batch:
        cur.executemany(insert_sql, batch)
        row_count += len(batch)

    print(f"  Loaded {row_count:,} rows in {time.time()-t0:.1f}s")

    print("  Creating indexes...")
    # Find likely column names for indexing — these depend on the actual CSV headers
    epi_col = next((c for c in col_names if "epitope" in c and "name" in c), None)
    mhc_col = next((c for c in col_names if "mhc" in c and "allele" in c and "name" in c), None)
    qual_col = next((c for c in col_names if "qualitative" in c and "measure" in c), None)

    if epi_col:
        cur.execute(f'CREATE INDEX IF NOT EXISTS idx_mhclig_epitope ON mhc_ligand("{epi_col}")')
    if mhc_col:
        cur.execute(f'CREATE INDEX IF NOT EXISTS idx_mhclig_mhc ON mhc_ligand("{mhc_col}")')
    if qual_col:
        cur.execute(f'CREATE INDEX IF NOT EXISTS idx_mhclig_qual ON mhc_ligand("{qual_col}")')

    conn.commit()
    conn.close()
    print(f"  MHC ligand table complete ({time.time()-t0:.1f}s total)")


# ---------------------------------------------------------------------------
# MySQL dump parsing
# ---------------------------------------------------------------------------

def parse_mysql_values(values_str):
    """Parse a MySQL VALUES clause into a list of row tuples.

    Handles: quoted strings (with escaped quotes \\'), NULL, numeric literals,
    nested parentheses in text, and empty strings.
    """
    rows = []
    i = 0
    n = len(values_str)

    while i < n:
        # Find start of a tuple
        while i < n and values_str[i] != '(':
            i += 1
        if i >= n:
            break
        i += 1  # skip opening (

        row = []
        while i < n:
            # Skip whitespace
            while i < n and values_str[i] in (' ', '\t'):
                i += 1
            if i >= n:
                break

            if values_str[i] == ')':
                i += 1  # skip closing )
                break
            elif values_str[i] == ',':
                i += 1  # skip comma between values
                continue
            elif values_str[i] == "'":
                # Quoted string
                i += 1  # skip opening quote
                chars = []
                while i < n:
                    if values_str[i] == '\\' and i + 1 < n:
                        # Escaped character
                        chars.append(values_str[i + 1])
                        i += 2
                    elif values_str[i] == "'":
                        i += 1  # skip closing quote
                        break
                    else:
                        chars.append(values_str[i])
                        i += 1
                row.append(''.join(chars))
            elif values_str[i:i+4].upper() == 'NULL':
                row.append(None)
                i += 4
            else:
                # Numeric or other unquoted literal
                start = i
                while i < n and values_str[i] not in (',', ')'):
                    i += 1
                row.append(values_str[start:i].strip())

        rows.append(tuple(row))

    return rows


def load_mysql_dump(db_path, sql_gz_path):
    """Parse MySQL SQL dump and extract binding-related tables into SQLite.

    Extracts INSERT statements for target tables and loads them with mysql_ prefix.
    """
    if not os.path.exists(sql_gz_path):
        print(f"\n  WARNING: MySQL dump not found at {sql_gz_path} — skipping")
        return

    print(f"\n{'='*60}")
    print(f"Parsing MySQL dump from {sql_gz_path}")
    print(f"{'='*60}")
    t0 = time.time()

    # Target tables and their column definitions
    # Column lists derived from MySQL CREATE TABLE schemas
    target_tables = {
        "mhc_bind": [
            "mhc_bind_id", "reference_id", "curated_epitope_id", "as_location",
            "as_type_id", "as_char_value", "as_num_value", "as_inequality",
            "as_comments", "mhc_allele_restriction_id", "mhc_allele_name", "complex_id",
        ],
        "mhc_elution": None,  # will discover columns from INSERT statement
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
        "epitope": None,  # will discover columns
        "curated_epitope": None,  # will discover columns
    }

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Track rows loaded per table
    table_counts = {t: 0 for t in target_tables}

    # Compile regex patterns for each target table
    insert_patterns = {}
    for table_name in target_tables:
        pattern = re.compile(
            r"INSERT INTO `" + re.escape(table_name) + r"`\s+(?:\([^)]*\)\s+)?VALUES\s+(.*)",
            re.IGNORECASE
        )
        insert_patterns[table_name] = pattern

    # Tables that have been created (schema discovered from first INSERT)
    created_tables = set()

    print(f"  Scanning for INSERT statements...")
    line_count = 0

    with gzip.open(sql_gz_path, 'rb') as f:
        for raw_line in f:
            line_count += 1
            if line_count % 1_000_000 == 0:
                print(f"  ... scanned {line_count:,} lines", end="\r")

            # Quick check before decoding
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

                # Create table on first encounter
                if sqlite_table not in created_tables:
                    col_names = target_tables[table_name]
                    if col_names is None:
                        # Discover columns: generate generic names
                        col_names = [f"col_{i}" for i in range(ncols_data)]
                        target_tables[table_name] = col_names
                    # Adjust column count if data has different width
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

                # Normalize rows to expected column count
                normalized = []
                for row in rows:
                    if len(row) < expected_ncols:
                        row = row + (None,) * (expected_ncols - len(row))
                    elif len(row) > expected_ncols:
                        row = row[:expected_ncols]
                    # Convert None to empty string for TEXT columns
                    normalized.append(tuple('' if v is None else v for v in row))

                cur.executemany(insert_sql, normalized)
                table_counts[table_name] += len(normalized)
                break  # only match one table per line

    conn.commit()

    print(f"\n  Scanned {line_count:,} lines in {time.time()-t0:.1f}s")
    print(f"  Tables loaded from MySQL dump:")
    for table_name, count in table_counts.items():
        sqlite_table = f"mysql_{table_name}"
        if count > 0:
            print(f"    {sqlite_table}: {count:,} rows")
        else:
            print(f"    {sqlite_table}: (not found in dump)")

    # Create indexes on key tables
    print("  Creating indexes...")
    if table_counts["mhc_bind"] > 0:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_mhcbind_epitope ON mysql_mhc_bind(curated_epitope_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_mhcbind_allele ON mysql_mhc_bind(mhc_allele_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_mhcbind_type ON mysql_mhc_bind(as_type_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mysql_mhcbind_char ON mysql_mhc_bind(as_char_value)")
    if table_counts["mhc_elution"] > 0:
        # Generic index on first column (likely an ID)
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
    conn.close()
    print(f"  MySQL dump parsing complete ({time.time()-t0:.1f}s total)")


# ---------------------------------------------------------------------------
# Summary queries
# ---------------------------------------------------------------------------

def _table_exists(cur, table_name):
    """Check if a table exists in the SQLite database."""
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def classify_mhc(allele):
    """Heuristic MHC class assignment from allele name."""
    a = allele.upper()
    # Class I: HLA-A, HLA-B, HLA-C, H-2K, H-2D, H-2L (mouse)
    if re.match(r"HLA-[ABC]", a) or re.match(r"H-2[KDL]", a):
        return "Class I"
    # Class II: HLA-D*, H-2I (mouse)
    if re.match(r"HLA-D", a) or re.match(r"H-2I", a):
        return "Class II"
    return "Unknown"


def run_summary(db_path):
    """Run all summary queries and return results dict."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    results = {}

    # ------------------------------------------------------------------
    # 6a. Unique TCRs (from receptor table)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 6a: Unique TCRs (receptor table)")
    print("=" * 60)

    total_rows = cur.execute("SELECT COUNT(*) FROM receptor").fetchone()[0]
    print(f"  Total receptor rows: {total_rows:,}")

    # By receptor type
    type_counts = cur.execute(
        "SELECT receptor_type, COUNT(*) as cnt FROM receptor GROUP BY receptor_type ORDER BY cnt DESC"
    ).fetchall()
    type_dict = {r[0]: r[1] for r in type_counts}
    print(f"  By receptor type: {dict(type_dict)}")

    # CDR3 coalesce pattern
    CDR3A = "COALESCE(NULLIF(chain_1_cdr3_curated, ''), NULLIF(chain_1_cdr3_calculated, ''))"
    CDR3B = "COALESCE(NULLIF(chain_2_cdr3_curated, ''), NULLIF(chain_2_cdr3_calculated, ''))"

    # Paired TCRs: both chain 1 and chain 2 CDR3 present
    paired_count = cur.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT {CDR3A} as cdr3a, {CDR3B} as cdr3b
            FROM receptor
            WHERE {CDR3A} IS NOT NULL AND {CDR3B} IS NOT NULL
        )
    """).fetchone()[0]
    print(f"  Unique paired (CDR3alpha, CDR3beta) pairs: {paired_count:,}")

    # Unpaired alpha-only
    alpha_only = cur.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT {CDR3A} as cdr3a
            FROM receptor
            WHERE {CDR3A} IS NOT NULL AND {CDR3B} IS NULL
        )
    """).fetchone()[0]
    print(f"  Unique unpaired alpha-only CDR3s: {alpha_only:,}")

    # Unpaired beta-only
    beta_only = cur.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT {CDR3B} as cdr3b
            FROM receptor
            WHERE {CDR3B} IS NOT NULL AND {CDR3A} IS NULL
        )
    """).fetchone()[0]
    print(f"  Unique unpaired beta-only CDR3s: {beta_only:,}")

    # Total unique CDR3 alpha and beta (regardless of pairing)
    total_unique_cdr3a = cur.execute(f"""
        SELECT COUNT(DISTINCT {CDR3A}) FROM receptor WHERE {CDR3A} IS NOT NULL
    """).fetchone()[0]
    total_unique_cdr3b = cur.execute(f"""
        SELECT COUNT(DISTINCT {CDR3B}) FROM receptor WHERE {CDR3B} IS NOT NULL
    """).fetchone()[0]
    print(f"  Total unique CDR3 alpha sequences: {total_unique_cdr3a:,}")
    print(f"  Total unique CDR3 beta sequences: {total_unique_cdr3b:,}")

    # V/J gene presence
    VGA = "COALESCE(NULLIF(chain_1_curated_v_gene, ''), NULLIF(chain_1_calculated_v_gene, ''))"
    JGA = "COALESCE(NULLIF(chain_1_curated_j_gene, ''), NULLIF(chain_1_calculated_j_gene, ''))"
    VGB = "COALESCE(NULLIF(chain_2_curated_v_gene, ''), NULLIF(chain_2_calculated_v_gene, ''))"
    JGB = "COALESCE(NULLIF(chain_2_curated_j_gene, ''), NULLIF(chain_2_calculated_j_gene, ''))"

    vga_count = cur.execute(f"SELECT COUNT(*) FROM receptor WHERE {VGA} IS NOT NULL").fetchone()[0]
    jga_count = cur.execute(f"SELECT COUNT(*) FROM receptor WHERE {JGA} IS NOT NULL").fetchone()[0]
    vgb_count = cur.execute(f"SELECT COUNT(*) FROM receptor WHERE {VGB} IS NOT NULL").fetchone()[0]
    jgb_count = cur.execute(f"SELECT COUNT(*) FROM receptor WHERE {JGB} IS NOT NULL").fetchone()[0]
    print(f"  Rows with V gene (chain 1): {vga_count:,} / {total_rows:,}")
    print(f"  Rows with J gene (chain 1): {jga_count:,} / {total_rows:,}")
    print(f"  Rows with V gene (chain 2): {vgb_count:,} / {total_rows:,}")
    print(f"  Rows with J gene (chain 2): {jgb_count:,} / {total_rows:,}")

    results["tcrs"] = {
        "total_rows": total_rows,
        "by_receptor_type": type_dict,
        "unique_paired_cdr3_pairs": paired_count,
        "unique_unpaired_alpha_only": alpha_only,
        "unique_unpaired_beta_only": beta_only,
        "total_unique_cdr3_alpha": total_unique_cdr3a,
        "total_unique_cdr3_beta": total_unique_cdr3b,
        "rows_with_v_gene_chain1": vga_count,
        "rows_with_j_gene_chain1": jga_count,
        "rows_with_v_gene_chain2": vgb_count,
        "rows_with_j_gene_chain2": jgb_count,
    }

    # ------------------------------------------------------------------
    # 6b. Unique Epitopes
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 6b: Unique Epitopes")
    print("=" * 60)

    # From epitope table
    epi_total = cur.execute("SELECT COUNT(*) FROM epitope").fetchone()[0]
    epi_unique_names = cur.execute(
        "SELECT COUNT(DISTINCT epitope_name) FROM epitope WHERE epitope_name != ''"
    ).fetchone()[0]
    print(f"  Epitope table: {epi_total:,} rows, {epi_unique_names:,} unique epitope names")

    epi_by_type = cur.execute("""
        SELECT epitope_object_type, COUNT(*) as cnt
        FROM epitope WHERE epitope_object_type != ''
        GROUP BY epitope_object_type ORDER BY cnt DESC
    """).fetchall()
    epi_by_type_dict = {r[0]: r[1] for r in epi_by_type}
    print(f"  By object type: {dict(epi_by_type_dict)}")

    epi_top_organisms = cur.execute("""
        SELECT epitope_source_organism, COUNT(*) as cnt
        FROM epitope WHERE epitope_source_organism != ''
        GROUP BY epitope_source_organism ORDER BY cnt DESC LIMIT 15
    """).fetchall()
    print(f"  Top 15 source organisms:")
    for org, cnt in epi_top_organisms:
        print(f"    {org}: {cnt:,}")

    # From receptor table: epitopes with TCR specificity data
    rec_unique_epi = cur.execute(
        "SELECT COUNT(DISTINCT epitope_name) FROM receptor WHERE epitope_name != ''"
    ).fetchone()[0]
    print(f"\n  Receptor table: {rec_unique_epi:,} unique epitopes with TCR data")

    rec_top_epi = cur.execute("""
        SELECT epitope_name, COUNT(*) as cnt
        FROM receptor WHERE epitope_name != ''
        GROUP BY epitope_name ORDER BY cnt DESC LIMIT 25
    """).fetchall()
    print(f"  Top 25 epitopes by receptor count:")
    for ep, cnt in rec_top_epi:
        print(f"    {ep}: {cnt:,}")

    # From tcell table: epitopes from positive assays
    tcell_pos_epi = cur.execute("""
        SELECT COUNT(DISTINCT epitope_name)
        FROM tcell
        WHERE epitope_name != ''
          AND assay_qualitative_measurement = 'Positive'
    """).fetchone()[0]
    print(f"\n  Tcell table (positive assays): {tcell_pos_epi:,} unique epitopes")

    tcell_top_epi = cur.execute("""
        SELECT epitope_name, COUNT(*) as cnt
        FROM tcell
        WHERE epitope_name != ''
          AND assay_qualitative_measurement = 'Positive'
        GROUP BY epitope_name ORDER BY cnt DESC LIMIT 25
    """).fetchall()
    print(f"  Top 25 epitopes by positive assay count:")
    for ep, cnt in tcell_top_epi:
        print(f"    {ep}: {cnt:,}")

    results["epitopes"] = {
        "epitope_table_total_rows": epi_total,
        "epitope_table_unique_names": epi_unique_names,
        "epitope_table_by_type": epi_by_type_dict,
        "epitope_table_top_organisms": [(r[0], r[1]) for r in epi_top_organisms],
        "receptor_table_unique_epitopes": rec_unique_epi,
        "receptor_table_top25_epitopes": [(r[0], r[1]) for r in rec_top_epi],
        "tcell_positive_unique_epitopes": tcell_pos_epi,
        "tcell_positive_top25_epitopes": [(r[0], r[1]) for r in tcell_top_epi],
    }

    # ------------------------------------------------------------------
    # 6c. Unique MHC Alleles
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 6c: Unique MHC Alleles")
    print("=" * 60)

    # From receptor table (normalized junction table)
    rec_unique_mhc = cur.execute(
        "SELECT COUNT(DISTINCT mhc_allele) FROM receptor_mhc_alleles"
    ).fetchone()[0]
    print(f"  Receptor table (normalized): {rec_unique_mhc:,} unique MHC alleles")

    rec_top_mhc = cur.execute("""
        SELECT mhc_allele, COUNT(DISTINCT receptor_rowid) as cnt
        FROM receptor_mhc_alleles
        GROUP BY mhc_allele ORDER BY cnt DESC LIMIT 25
    """).fetchall()
    print(f"  Top 25 MHC alleles by unique receptor count:")
    for allele, cnt in rec_top_mhc:
        print(f"    {allele}: {cnt:,}")

    # MHC class breakdown (heuristic)
    all_alleles = cur.execute(
        "SELECT DISTINCT mhc_allele FROM receptor_mhc_alleles"
    ).fetchall()
    class_counts = {"Class I": 0, "Class II": 0, "Unknown": 0}
    for (allele,) in all_alleles:
        cls = classify_mhc(allele)
        class_counts[cls] += 1
    print(f"  MHC class breakdown (heuristic): {class_counts}")

    # From tcell table
    tcell_unique_mhc = cur.execute(
        "SELECT COUNT(DISTINCT mhc_restriction_name) FROM tcell WHERE mhc_restriction_name != ''"
    ).fetchone()[0]
    print(f"\n  Tcell table: {tcell_unique_mhc:,} unique MHC restriction names")

    tcell_mhc_by_class = cur.execute("""
        SELECT mhc_restriction_class, COUNT(DISTINCT mhc_restriction_name) as cnt
        FROM tcell WHERE mhc_restriction_name != '' AND mhc_restriction_class != ''
        GROUP BY mhc_restriction_class ORDER BY cnt DESC
    """).fetchall()
    tcell_mhc_class_dict = {r[0]: r[1] for r in tcell_mhc_by_class}
    print(f"  Tcell MHC by class: {dict(tcell_mhc_class_dict)}")

    tcell_top_mhc = cur.execute("""
        SELECT mhc_restriction_name, COUNT(*) as cnt
        FROM tcell WHERE mhc_restriction_name != ''
        GROUP BY mhc_restriction_name ORDER BY cnt DESC LIMIT 25
    """).fetchall()
    print(f"  Top 25 MHC restrictions by assay count:")
    for allele, cnt in tcell_top_mhc:
        print(f"    {allele}: {cnt:,}")

    results["mhc_alleles"] = {
        "receptor_unique_alleles": rec_unique_mhc,
        "receptor_top25_alleles": [(r[0], r[1]) for r in rec_top_mhc],
        "receptor_class_breakdown": class_counts,
        "tcell_unique_mhc_names": tcell_unique_mhc,
        "tcell_mhc_by_class": tcell_mhc_class_dict,
        "tcell_top25_mhc": [(r[0], r[1]) for r in tcell_top_mhc],
    }

    # ------------------------------------------------------------------
    # 6d. Unique Peptide-MHC Complexes
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 6d: Unique Peptide-MHC Complexes")
    print("=" * 60)

    # From receptor table via junction table
    rec_pmhc = cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT r.epitope_name, rma.mhc_allele
            FROM receptor r
            JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
            WHERE r.epitope_name != ''
        )
    """).fetchone()[0]
    print(f"  Receptor table: {rec_pmhc:,} unique (peptide, MHC allele) pairs")

    # pMHC by class
    rec_pmhc_rows = cur.execute("""
        SELECT DISTINCT r.epitope_name, rma.mhc_allele
        FROM receptor r
        JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
        WHERE r.epitope_name != ''
    """).fetchall()
    pmhc_class = {"Class I": 0, "Class II": 0, "Unknown": 0}
    for _, allele in rec_pmhc_rows:
        cls = classify_mhc(allele)
        pmhc_class[cls] += 1
    print(f"  pMHC by MHC class: {pmhc_class}")

    # Top 25 pMHC pairs by unique TCR count
    rec_top_pmhc = cur.execute(f"""
        SELECT r.epitope_name, rma.mhc_allele,
               COUNT(DISTINCT {CDR3A} || '|' || {CDR3B}) as tcr_cnt
        FROM receptor r
        JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
        WHERE r.epitope_name != ''
          AND ({CDR3A} IS NOT NULL OR {CDR3B} IS NOT NULL)
        GROUP BY r.epitope_name, rma.mhc_allele
        ORDER BY tcr_cnt DESC LIMIT 25
    """).fetchall()
    print(f"  Top 25 pMHC pairs by unique TCR count:")
    for ep, allele, cnt in rec_top_pmhc:
        print(f"    {ep} + {allele}: {cnt:,}")

    # From tcell table (positive assays)
    tcell_pmhc = cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT epitope_name, mhc_restriction_name
            FROM tcell
            WHERE epitope_name != '' AND mhc_restriction_name != ''
              AND assay_qualitative_measurement = 'Positive'
        )
    """).fetchone()[0]
    print(f"\n  Tcell table (positive): {tcell_pmhc:,} unique (epitope, MHC) pairs")

    tcell_top_pmhc = cur.execute("""
        SELECT epitope_name, mhc_restriction_name, COUNT(*) as cnt
        FROM tcell
        WHERE epitope_name != '' AND mhc_restriction_name != ''
          AND assay_qualitative_measurement = 'Positive'
        GROUP BY epitope_name, mhc_restriction_name
        ORDER BY cnt DESC LIMIT 25
    """).fetchall()
    print(f"  Top 25 pMHC pairs by positive assay count:")
    for ep, allele, cnt in tcell_top_pmhc:
        print(f"    {ep} + {allele}: {cnt:,}")

    results["pmhc"] = {
        "receptor_unique_pmhc_pairs": rec_pmhc,
        "receptor_pmhc_by_class": pmhc_class,
        "receptor_top25_pmhc": [(r[0], r[1], r[2]) for r in rec_top_pmhc],
        "tcell_positive_unique_pmhc": tcell_pmhc,
        "tcell_positive_top25_pmhc": [(r[0], r[1], r[2]) for r in tcell_top_pmhc],
    }

    # ------------------------------------------------------------------
    # 6e. Unique TCR-Peptide-MHC Complexes
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 6e: Unique TCR-Peptide-MHC Complexes")
    print("=" * 60)

    # Paired: (CDR3alpha, CDR3beta, peptide, MHC) 4-tuples
    paired_tcrpmhc = cur.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT {CDR3A} as cdr3a, {CDR3B} as cdr3b,
                   r.epitope_name, rma.mhc_allele
            FROM receptor r
            JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
            WHERE {CDR3A} IS NOT NULL AND {CDR3B} IS NOT NULL
              AND r.epitope_name != ''
        )
    """).fetchone()[0]
    print(f"  Paired (CDR3a, CDR3b, epitope, MHC) 4-tuples: {paired_tcrpmhc:,}")

    # Unpaired beta: (CDR3beta, peptide, MHC)
    unpaired_beta_tcrpmhc = cur.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT {CDR3B} as cdr3b, r.epitope_name, rma.mhc_allele
            FROM receptor r
            JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
            WHERE {CDR3B} IS NOT NULL AND {CDR3A} IS NULL
              AND r.epitope_name != ''
        )
    """).fetchone()[0]
    print(f"  Unpaired beta (CDR3b, epitope, MHC) 3-tuples: {unpaired_beta_tcrpmhc:,}")

    # Unpaired alpha: (CDR3alpha, peptide, MHC)
    unpaired_alpha_tcrpmhc = cur.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT {CDR3A} as cdr3a, r.epitope_name, rma.mhc_allele
            FROM receptor r
            JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
            WHERE {CDR3A} IS NOT NULL AND {CDR3B} IS NULL
              AND r.epitope_name != ''
        )
    """).fetchone()[0]
    print(f"  Unpaired alpha (CDR3a, epitope, MHC) 3-tuples: {unpaired_alpha_tcrpmhc:,}")

    # Distribution by receptor type
    tcrpmhc_by_type = cur.execute(f"""
        SELECT r.receptor_type, COUNT(*) as cnt
        FROM receptor r
        JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
        WHERE r.epitope_name != ''
          AND ({CDR3A} IS NOT NULL OR {CDR3B} IS NOT NULL)
        GROUP BY r.receptor_type ORDER BY cnt DESC
    """).fetchall()
    tcrpmhc_type_dict = {r[0]: r[1] for r in tcrpmhc_by_type}
    print(f"  TCR-pMHC rows by receptor type: {dict(tcrpmhc_type_dict)}")

    # Top 25 most-observed complexes (paired)
    top_complexes = cur.execute(f"""
        SELECT {CDR3A} as cdr3a, {CDR3B} as cdr3b,
               r.epitope_name, rma.mhc_allele, COUNT(*) as cnt
        FROM receptor r
        JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
        WHERE {CDR3A} IS NOT NULL AND {CDR3B} IS NOT NULL
          AND r.epitope_name != ''
        GROUP BY cdr3a, cdr3b, r.epitope_name, rma.mhc_allele
        ORDER BY cnt DESC LIMIT 25
    """).fetchall()
    print(f"  Top 25 most-observed paired TCR-pMHC complexes:")
    for cdr3a, cdr3b, ep, allele, cnt in top_complexes:
        print(f"    {cdr3a} / {cdr3b} + {ep} + {allele}: {cnt}")

    results["tcr_pmhc"] = {
        "paired_4tuples": paired_tcrpmhc,
        "unpaired_beta_3tuples": unpaired_beta_tcrpmhc,
        "unpaired_alpha_3tuples": unpaired_alpha_tcrpmhc,
        "by_receptor_type": tcrpmhc_type_dict,
        "top25_paired_complexes": [
            {"cdr3a": r[0], "cdr3b": r[1], "epitope": r[2], "mhc": r[3], "count": r[4]}
            for r in top_complexes
        ],
    }

    # ------------------------------------------------------------------
    # 6f. Tcell Quantitative Stats
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 6f: Tcell Quantitative Stats")
    print("=" * 60)

    tcell_total = cur.execute("SELECT COUNT(*) FROM tcell").fetchone()[0]
    print(f"  Total tcell rows: {tcell_total:,}")

    # Rows with quantitative measurements
    quant_rows = cur.execute(
        "SELECT COUNT(*) FROM tcell WHERE assay_quantitative_measurement != ''"
    ).fetchone()[0]
    print(f"  Rows with quantitative measurement: {quant_rows:,} ({100*quant_rows/max(tcell_total,1):.1f}%)")

    # By assay method (for rows with quantitative data)
    quant_by_method = cur.execute("""
        SELECT assay_method, COUNT(*) as cnt
        FROM tcell WHERE assay_quantitative_measurement != ''
        GROUP BY assay_method ORDER BY cnt DESC LIMIT 25
    """).fetchall()
    print(f"  Top assay methods with quantitative data:")
    for method, cnt in quant_by_method:
        print(f"    {method}: {cnt:,}")

    # Distribution of units
    units_dist = cur.execute("""
        SELECT assay_units, COUNT(*) as cnt
        FROM tcell WHERE assay_units != ''
        GROUP BY assay_units ORDER BY cnt DESC
    """).fetchall()
    units_dict = {r[0]: r[1] for r in units_dist}
    print(f"  Assay units distribution:")
    for unit, cnt in units_dist:
        print(f"    {unit}: {cnt:,}")

    # Population stats
    subjects_tested_rows = cur.execute(
        "SELECT COUNT(*) FROM tcell WHERE assay_num_subjects_tested != ''"
    ).fetchone()[0]
    subjects_positive_rows = cur.execute(
        "SELECT COUNT(*) FROM tcell WHERE assay_num_subjects_positive != ''"
    ).fetchone()[0]
    freq_rows = cur.execute(
        "SELECT COUNT(*) FROM tcell WHERE assay_response_frequency != ''"
    ).fetchone()[0]
    print(f"  Rows with subjects tested: {subjects_tested_rows:,} ({100*subjects_tested_rows/max(tcell_total,1):.1f}%)")
    print(f"  Rows with subjects positive: {subjects_positive_rows:,} ({100*subjects_positive_rows/max(tcell_total,1):.1f}%)")
    print(f"  Rows with response frequency: {freq_rows:,} ({100*freq_rows/max(tcell_total,1):.1f}%)")

    # Cross-tab: qualitative measurement vs presence of quantitative data
    qual_quant_xtab = cur.execute("""
        SELECT assay_qualitative_measurement,
               SUM(CASE WHEN assay_quantitative_measurement != '' THEN 1 ELSE 0 END) as with_quant,
               SUM(CASE WHEN assay_quantitative_measurement = '' THEN 1 ELSE 0 END) as without_quant
        FROM tcell
        WHERE assay_qualitative_measurement != ''
        GROUP BY assay_qualitative_measurement ORDER BY (with_quant + without_quant) DESC
    """).fetchall()
    print(f"  Qualitative vs quantitative cross-tab:")
    for qual, with_q, without_q in qual_quant_xtab:
        print(f"    {qual}: {with_q:,} with quant, {without_q:,} without")

    results["tcell_quantitative"] = {
        "total_tcell_rows": tcell_total,
        "rows_with_quantitative": quant_rows,
        "top_methods_with_quant": [(r[0], r[1]) for r in quant_by_method],
        "units_distribution": units_dict,
        "rows_with_subjects_tested": subjects_tested_rows,
        "rows_with_subjects_positive": subjects_positive_rows,
        "rows_with_response_frequency": freq_rows,
        "qualitative_vs_quantitative": [
            {"qualitative": r[0], "with_quant": r[1], "without_quant": r[2]}
            for r in qual_quant_xtab
        ],
    }

    # ------------------------------------------------------------------
    # 6g. MHC Binding Summary (from mysql_mhc_bind)
    # ------------------------------------------------------------------
    _has_mhc_bind = _table_exists(cur, "mysql_mhc_bind")
    if _has_mhc_bind:
        print("\n" + "=" * 60)
        print("SECTION 6g: MHC Binding Summary (mysql_mhc_bind)")
        print("=" * 60)

        bind_total = cur.execute("SELECT COUNT(*) FROM mysql_mhc_bind").fetchone()[0]
        print(f"  Total binding assay entries: {bind_total:,}")

        # Unique (epitope, MHC allele) pairs
        bind_unique_pairs = cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT curated_epitope_id, mhc_allele_name
                FROM mysql_mhc_bind WHERE mhc_allele_name != ''
            )
        """).fetchone()[0]
        print(f"  Unique (epitope_id, MHC allele) pairs: {bind_unique_pairs:,}")

        # Entries with quantitative vs qualitative-only
        bind_with_num = cur.execute(
            "SELECT COUNT(*) FROM mysql_mhc_bind WHERE as_num_value != ''"
        ).fetchone()[0]
        bind_qual_only = bind_total - bind_with_num
        print(f"  With numeric value: {bind_with_num:,}")
        print(f"  Qualitative only: {bind_qual_only:,}")

        # Breakdown by as_char_value (Positive/Negative/etc.)
        bind_by_char = cur.execute("""
            SELECT as_char_value, COUNT(*) as cnt
            FROM mysql_mhc_bind WHERE as_char_value != ''
            GROUP BY as_char_value ORDER BY cnt DESC
        """).fetchall()
        print(f"  By qualitative value:")
        for val, cnt in bind_by_char:
            print(f"    {val}: {cnt:,}")

        # Breakdown by assay method (join with assay_type if available)
        _has_assay_type = _table_exists(cur, "mysql_assay_type")
        bind_by_method = []
        if _has_assay_type:
            bind_by_method = cur.execute("""
                SELECT at.assay_type, COUNT(*) as cnt
                FROM mysql_mhc_bind mb
                JOIN mysql_assay_type at ON mb.as_type_id = at.assay_type_id
                GROUP BY at.assay_type ORDER BY cnt DESC LIMIT 25
            """).fetchall()
            print(f"  Top 25 assay methods:")
            for method, cnt in bind_by_method:
                print(f"    {method}: {cnt:,}")

        # Top 25 MHC alleles by binding measurements
        bind_top_mhc = cur.execute("""
            SELECT mhc_allele_name, COUNT(*) as cnt
            FROM mysql_mhc_bind WHERE mhc_allele_name != ''
            GROUP BY mhc_allele_name ORDER BY cnt DESC LIMIT 25
        """).fetchall()
        print(f"  Top 25 MHC alleles by binding measurement count:")
        for allele, cnt in bind_top_mhc:
            print(f"    {allele}: {cnt:,}")

        # Top 25 peptide-MHC pairs (join with epitope tables if available)
        _has_curated = _table_exists(cur, "mysql_curated_epitope")
        _has_mysql_epi = _table_exists(cur, "mysql_epitope")
        bind_top_pmhc = []
        if _has_curated and _has_mysql_epi:
            bind_top_pmhc = cur.execute("""
                SELECT e.col_2 as peptide, mb.mhc_allele_name, COUNT(*) as cnt
                FROM mysql_mhc_bind mb
                JOIN mysql_curated_epitope ce ON mb.curated_epitope_id = ce.col_0
                JOIN mysql_epitope e ON ce.col_3 = e.col_0
                WHERE mb.mhc_allele_name != '' AND e.col_2 != ''
                GROUP BY e.col_2, mb.mhc_allele_name
                ORDER BY cnt DESC LIMIT 25
            """).fetchall()
            print(f"  Top 25 peptide-MHC pairs by measurement count:")
            for pep, allele, cnt in bind_top_pmhc:
                print(f"    {pep} + {allele}: {cnt:,}")

        results["mhc_binding"] = {
            "total_entries": bind_total,
            "unique_epitope_mhc_pairs": bind_unique_pairs,
            "with_numeric_value": bind_with_num,
            "qualitative_only": bind_qual_only,
            "by_char_value": {r[0]: r[1] for r in bind_by_char},
            "top25_assay_methods": [(r[0], r[1]) for r in bind_by_method],
            "top25_mhc_alleles": [(r[0], r[1]) for r in bind_top_mhc],
            "top25_peptide_mhc": [(r[0], r[1], r[2]) for r in bind_top_pmhc],
        }

    # ------------------------------------------------------------------
    # 6h. MHC Elution Summary (from mysql_mhc_elution)
    # ------------------------------------------------------------------
    _has_mhc_elution = _table_exists(cur, "mysql_mhc_elution")
    if _has_mhc_elution:
        print("\n" + "=" * 60)
        print("SECTION 6h: MHC Elution Summary (mysql_mhc_elution)")
        print("=" * 60)

        elut_total = cur.execute("SELECT COUNT(*) FROM mysql_mhc_elution").fetchone()[0]
        print(f"  Total elution entries: {elut_total:,}")

        # Column count for orientation
        elut_cols = cur.execute("PRAGMA table_info(mysql_mhc_elution)").fetchall()
        print(f"  Columns: {len(elut_cols)}")

        results["mhc_elution"] = {
            "total_entries": elut_total,
            "num_columns": len(elut_cols),
        }

    # ------------------------------------------------------------------
    # 6i. MHC Ligand CSV Summary (if loaded)
    # ------------------------------------------------------------------
    _has_mhc_ligand = _table_exists(cur, "mhc_ligand")
    if _has_mhc_ligand:
        print("\n" + "=" * 60)
        print("SECTION 6i: MHC Ligand CSV Summary")
        print("=" * 60)

        lig_total = cur.execute("SELECT COUNT(*) FROM mhc_ligand").fetchone()[0]
        print(f"  Total rows: {lig_total:,}")

        # Get column names to find appropriate columns
        lig_cols_info = cur.execute("PRAGMA table_info(mhc_ligand)").fetchall()
        lig_col_names = [c[1] for c in lig_cols_info]

        # Find relevant columns by name pattern
        lig_epi_col = next((c for c in lig_col_names if "epitope" in c and "name" in c), None)
        lig_mhc_col = next((c for c in lig_col_names if "mhc" in c and "allele" in c and "name" in c), None)
        lig_qual_col = next((c for c in lig_col_names if "qualitative" in c and "measure" in c), None)
        lig_method_col = next((c for c in lig_col_names if "assay" in c and "method" in c), None)

        lig_results = {"total_rows": lig_total}

        if lig_epi_col:
            lig_unique_pep = cur.execute(
                f'SELECT COUNT(DISTINCT "{lig_epi_col}") FROM mhc_ligand WHERE "{lig_epi_col}" != \'\''
            ).fetchone()[0]
            print(f"  Unique peptides/epitopes: {lig_unique_pep:,}")
            lig_results["unique_peptides"] = lig_unique_pep

        if lig_mhc_col:
            lig_unique_mhc = cur.execute(
                f'SELECT COUNT(DISTINCT "{lig_mhc_col}") FROM mhc_ligand WHERE "{lig_mhc_col}" != \'\''
            ).fetchone()[0]
            print(f"  Unique MHC alleles: {lig_unique_mhc:,}")
            lig_results["unique_mhc_alleles"] = lig_unique_mhc

        if lig_method_col:
            lig_by_method = cur.execute(f"""
                SELECT "{lig_method_col}", COUNT(*) as cnt
                FROM mhc_ligand WHERE "{lig_method_col}" != ''
                GROUP BY "{lig_method_col}" ORDER BY cnt DESC LIMIT 15
            """).fetchall()
            print(f"  By assay method (top 15):")
            for method, cnt in lig_by_method:
                print(f"    {method}: {cnt:,}")
            lig_results["top15_assay_methods"] = [(r[0], r[1]) for r in lig_by_method]

        if lig_qual_col:
            lig_by_qual = cur.execute(f"""
                SELECT "{lig_qual_col}", COUNT(*) as cnt
                FROM mhc_ligand WHERE "{lig_qual_col}" != ''
                GROUP BY "{lig_qual_col}" ORDER BY cnt DESC
            """).fetchall()
            print(f"  By qualitative measurement:")
            for qual, cnt in lig_by_qual:
                print(f"    {qual}: {cnt:,}")
            lig_results["by_qualitative"] = {r[0]: r[1] for r in lig_by_qual}

        if lig_epi_col and lig_mhc_col:
            lig_top_pmhc = cur.execute(f"""
                SELECT "{lig_epi_col}", "{lig_mhc_col}", COUNT(*) as cnt
                FROM mhc_ligand
                WHERE "{lig_epi_col}" != '' AND "{lig_mhc_col}" != ''
                GROUP BY "{lig_epi_col}", "{lig_mhc_col}"
                ORDER BY cnt DESC LIMIT 25
            """).fetchall()
            print(f"  Top 25 peptide-MHC pairs:")
            for pep, allele, cnt in lig_top_pmhc:
                print(f"    {pep} + {allele}: {cnt:,}")
            lig_results["top25_peptide_mhc"] = [(r[0], r[1], r[2]) for r in lig_top_pmhc]

        results["mhc_ligand"] = lig_results

    conn.close()
    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(results):
    """Format results dict into a human-readable text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("IEDB Summary Statistics")
    lines.append("=" * 70)

    # TCRs
    t = results["tcrs"]
    lines.append("\n" + "-" * 70)
    lines.append("UNIQUE TCRs (from receptor table)")
    lines.append("-" * 70)
    lines.append(f"Total receptor rows: {t['total_rows']:,}")
    lines.append(f"By receptor type:")
    for rtype, cnt in t["by_receptor_type"].items():
        lines.append(f"  {rtype}: {cnt:,}")
    lines.append(f"Unique paired (CDR3alpha, CDR3beta) pairs: {t['unique_paired_cdr3_pairs']:,}")
    lines.append(f"Unique unpaired alpha-only CDR3s: {t['unique_unpaired_alpha_only']:,}")
    lines.append(f"Unique unpaired beta-only CDR3s: {t['unique_unpaired_beta_only']:,}")
    lines.append(f"Total unique CDR3 alpha sequences: {t['total_unique_cdr3_alpha']:,}")
    lines.append(f"Total unique CDR3 beta sequences: {t['total_unique_cdr3_beta']:,}")
    lines.append(f"Rows with V gene (chain 1): {t['rows_with_v_gene_chain1']:,} / {t['total_rows']:,}")
    lines.append(f"Rows with J gene (chain 1): {t['rows_with_j_gene_chain1']:,} / {t['total_rows']:,}")
    lines.append(f"Rows with V gene (chain 2): {t['rows_with_v_gene_chain2']:,} / {t['total_rows']:,}")
    lines.append(f"Rows with J gene (chain 2): {t['rows_with_j_gene_chain2']:,} / {t['total_rows']:,}")

    # Epitopes
    e = results["epitopes"]
    lines.append("\n" + "-" * 70)
    lines.append("UNIQUE EPITOPES")
    lines.append("-" * 70)
    lines.append(f"Epitope table: {e['epitope_table_total_rows']:,} rows, {e['epitope_table_unique_names']:,} unique names")
    lines.append(f"By object type:")
    for otype, cnt in e["epitope_table_by_type"].items():
        lines.append(f"  {otype}: {cnt:,}")
    lines.append(f"Top 15 source organisms:")
    for org, cnt in e["epitope_table_top_organisms"]:
        lines.append(f"  {org}: {cnt:,}")
    lines.append(f"\nReceptor table: {e['receptor_table_unique_epitopes']:,} unique epitopes with TCR data")
    lines.append(f"Top 25 epitopes by receptor count:")
    for ep, cnt in e["receptor_table_top25_epitopes"]:
        lines.append(f"  {ep}: {cnt:,}")
    lines.append(f"\nTcell table (positive assays): {e['tcell_positive_unique_epitopes']:,} unique epitopes")
    lines.append(f"Top 25 epitopes by positive assay count:")
    for ep, cnt in e["tcell_positive_top25_epitopes"]:
        lines.append(f"  {ep}: {cnt:,}")

    # MHC
    m = results["mhc_alleles"]
    lines.append("\n" + "-" * 70)
    lines.append("UNIQUE MHC ALLELES")
    lines.append("-" * 70)
    lines.append(f"Receptor table (normalized): {m['receptor_unique_alleles']:,} unique alleles")
    lines.append(f"MHC class breakdown (heuristic): {m['receptor_class_breakdown']}")
    lines.append(f"Top 25 alleles by unique receptor count:")
    for allele, cnt in m["receptor_top25_alleles"]:
        lines.append(f"  {allele}: {cnt:,}")
    lines.append(f"\nTcell table: {m['tcell_unique_mhc_names']:,} unique MHC restriction names")
    lines.append(f"Tcell MHC by class: {m['tcell_mhc_by_class']}")
    lines.append(f"Top 25 MHC restrictions by assay count:")
    for allele, cnt in m["tcell_top25_mhc"]:
        lines.append(f"  {allele}: {cnt:,}")

    # pMHC
    p = results["pmhc"]
    lines.append("\n" + "-" * 70)
    lines.append("UNIQUE PEPTIDE-MHC COMPLEXES")
    lines.append("-" * 70)
    lines.append(f"Receptor table: {p['receptor_unique_pmhc_pairs']:,} unique (peptide, MHC) pairs")
    lines.append(f"By MHC class: {p['receptor_pmhc_by_class']}")
    lines.append(f"Top 25 pMHC pairs by unique TCR count:")
    for ep, allele, cnt in p["receptor_top25_pmhc"]:
        lines.append(f"  {ep} + {allele}: {cnt:,}")
    lines.append(f"\nTcell table (positive): {p['tcell_positive_unique_pmhc']:,} unique (epitope, MHC) pairs")
    lines.append(f"Top 25 pMHC pairs by positive assay count:")
    for ep, allele, cnt in p["tcell_positive_top25_pmhc"]:
        lines.append(f"  {ep} + {allele}: {cnt:,}")

    # TCR-pMHC
    c = results["tcr_pmhc"]
    lines.append("\n" + "-" * 70)
    lines.append("UNIQUE TCR-PEPTIDE-MHC COMPLEXES")
    lines.append("-" * 70)
    lines.append(f"Paired (CDR3a, CDR3b, epitope, MHC) 4-tuples: {c['paired_4tuples']:,}")
    lines.append(f"Unpaired beta (CDR3b, epitope, MHC) 3-tuples: {c['unpaired_beta_3tuples']:,}")
    lines.append(f"Unpaired alpha (CDR3a, epitope, MHC) 3-tuples: {c['unpaired_alpha_3tuples']:,}")
    lines.append(f"TCR-pMHC rows by receptor type: {c['by_receptor_type']}")
    lines.append(f"Top 25 most-observed paired TCR-pMHC complexes:")
    for cx in c["top25_paired_complexes"]:
        lines.append(f"  {cx['cdr3a']} / {cx['cdr3b']} + {cx['epitope']} + {cx['mhc']}: {cx['count']}")

    # Tcell Quantitative Stats
    if "tcell_quantitative" in results:
        tq = results["tcell_quantitative"]
        lines.append("\n" + "-" * 70)
        lines.append("TCELL QUANTITATIVE STATS")
        lines.append("-" * 70)
        lines.append(f"Total tcell rows: {tq['total_tcell_rows']:,}")
        lines.append(f"Rows with quantitative measurement: {tq['rows_with_quantitative']:,}")
        lines.append(f"Top assay methods with quantitative data:")
        for method, cnt in tq["top_methods_with_quant"]:
            lines.append(f"  {method}: {cnt:,}")
        lines.append(f"Assay units distribution:")
        for unit, cnt in tq["units_distribution"].items():
            lines.append(f"  {unit}: {cnt:,}")
        lines.append(f"Rows with subjects tested: {tq['rows_with_subjects_tested']:,}")
        lines.append(f"Rows with subjects positive: {tq['rows_with_subjects_positive']:,}")
        lines.append(f"Rows with response frequency: {tq['rows_with_response_frequency']:,}")
        lines.append(f"Qualitative vs quantitative cross-tab:")
        for entry in tq["qualitative_vs_quantitative"]:
            lines.append(f"  {entry['qualitative']}: {entry['with_quant']:,} with quant, {entry['without_quant']:,} without")

    # MHC Binding
    if "mhc_binding" in results:
        mb = results["mhc_binding"]
        lines.append("\n" + "-" * 70)
        lines.append("MHC BINDING (from MySQL dump)")
        lines.append("-" * 70)
        lines.append(f"Total binding assay entries: {mb['total_entries']:,}")
        lines.append(f"Unique (epitope_id, MHC allele) pairs: {mb['unique_epitope_mhc_pairs']:,}")
        lines.append(f"With numeric value: {mb['with_numeric_value']:,}")
        lines.append(f"Qualitative only: {mb['qualitative_only']:,}")
        lines.append(f"By qualitative value:")
        for val, cnt in mb["by_char_value"].items():
            lines.append(f"  {val}: {cnt:,}")
        if mb["top25_assay_methods"]:
            lines.append(f"Top 25 assay methods:")
            for method, cnt in mb["top25_assay_methods"]:
                lines.append(f"  {method}: {cnt:,}")
        lines.append(f"Top 25 MHC alleles by binding measurement count:")
        for allele, cnt in mb["top25_mhc_alleles"]:
            lines.append(f"  {allele}: {cnt:,}")
        if mb["top25_peptide_mhc"]:
            lines.append(f"Top 25 peptide-MHC pairs by measurement count:")
            for pep, allele, cnt in mb["top25_peptide_mhc"]:
                lines.append(f"  {pep} + {allele}: {cnt:,}")

    # MHC Elution
    if "mhc_elution" in results:
        me = results["mhc_elution"]
        lines.append("\n" + "-" * 70)
        lines.append("MHC ELUTION (from MySQL dump)")
        lines.append("-" * 70)
        lines.append(f"Total elution entries: {me['total_entries']:,}")
        lines.append(f"Number of columns: {me['num_columns']}")

    # MHC Ligand CSV
    if "mhc_ligand" in results:
        ml = results["mhc_ligand"]
        lines.append("\n" + "-" * 70)
        lines.append("MHC LIGAND (from CSV export)")
        lines.append("-" * 70)
        lines.append(f"Total rows: {ml['total_rows']:,}")
        if "unique_peptides" in ml:
            lines.append(f"Unique peptides/epitopes: {ml['unique_peptides']:,}")
        if "unique_mhc_alleles" in ml:
            lines.append(f"Unique MHC alleles: {ml['unique_mhc_alleles']:,}")
        if "top15_assay_methods" in ml:
            lines.append(f"By assay method (top 15):")
            for method, cnt in ml["top15_assay_methods"]:
                lines.append(f"  {method}: {cnt:,}")
        if "by_qualitative" in ml:
            lines.append(f"By qualitative measurement:")
            for qual, cnt in ml["by_qualitative"].items():
                lines.append(f"  {qual}: {cnt:,}")
        if "top25_peptide_mhc" in ml:
            lines.append(f"Top 25 peptide-MHC pairs:")
            for pep, allele, cnt in ml["top25_peptide_mhc"]:
                lines.append(f"  {pep} + {allele}: {cnt:,}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load IEDB CSVs into SQLite and generate summary statistics"
    )
    parser.add_argument(
        "--iedb_dir", default="data/databases/IEDB",
        help="Root directory containing IEDB CSV exports (default: data/databases/IEDB)"
    )
    parser.add_argument(
        "--output_dir", default="data/analysis/iedb",
        help="Output directory for SQLite DB and reports (default: data/analysis/iedb)"
    )
    parser.add_argument(
        "--skip_load", action="store_true",
        help="Skip loading CSVs, reuse existing SQLite DB and just run queries"
    )
    parser.add_argument(
        "--skip_mysql", action="store_true",
        help="Skip parsing the MySQL SQL dump (can be slow, ~5-10 min)"
    )
    args = parser.parse_args()

    iedb_dir = args.iedb_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(output_dir, "iedb.sqlite")
    receptor_csv = os.path.join(iedb_dir, "receptor", "tcr_full_v3.csv")
    tcell_csv = os.path.join(iedb_dir, "tcell", "tcell_full_v3.csv")
    epitope_csv = os.path.join(iedb_dir, "epitope", "epitope_full_v3.csv")
    mhc_ligand_csv = os.path.join(iedb_dir, "mhc_ligand", "mhc_ligand_full_v3.csv")
    mysql_dump_path = os.path.join(iedb_dir, "full_database", "iedb_public.sql.gz")

    if not args.skip_load:
        # Verify input files exist
        for path, name in [(receptor_csv, "receptor"), (tcell_csv, "tcell"), (epitope_csv, "epitope")]:
            if not os.path.exists(path):
                print(f"ERROR: {name} CSV not found at {path}")
                sys.exit(1)

        # Remove existing DB to start fresh
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Removed existing database at {db_path}")

        t_total = time.time()
        load_receptor(db_path, receptor_csv)
        load_tcell(db_path, tcell_csv)
        load_epitope(db_path, epitope_csv)

        # Load MHC ligand CSV (optional — warns if missing)
        load_mhc_ligand(db_path, mhc_ligand_csv)

        # Parse MySQL dump (optional — skip with --skip_mysql)
        if not args.skip_mysql:
            load_mysql_dump(db_path, mysql_dump_path)
        else:
            print("\n  Skipping MySQL dump parsing (--skip_mysql)")

        print(f"\nAll tables loaded in {time.time()-t_total:.1f}s")
        print(f"Database: {db_path} ({os.path.getsize(db_path) / 1e9:.2f} GB)")
    else:
        if not os.path.exists(db_path):
            print(f"ERROR: SQLite DB not found at {db_path} (cannot --skip_load)")
            sys.exit(1)
        print(f"Reusing existing database at {db_path}")

    # Run summary queries
    print("\n" + "#" * 60)
    print("RUNNING SUMMARY QUERIES")
    print("#" * 60)
    t_query = time.time()
    results = run_summary(db_path)
    print(f"\nQueries completed in {time.time()-t_query:.1f}s")

    # Format and save report
    report = format_report(results)
    print("\n" + report)

    txt_path = os.path.join(output_dir, "iedb_summary.txt")
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"\nText report saved to {txt_path}")

    json_path = os.path.join(output_dir, "iedb_summary.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON report saved to {json_path}")

    print(f"\nDone. SQLite DB: {db_path}")


if __name__ == "__main__":
    main()
