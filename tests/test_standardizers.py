"""Integration tests for per-database standardizers.

Each test creates minimal mock data matching the real schema, runs the
standardizer, and verifies the output conforms to the 25-column schema.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from quest.data.standardization import TARGET_COLUMNS, VALID_AA


def _verify_output(output_dir: Path):
    """Common verification for all standardizer outputs."""
    parquet_files = list(output_dir.glob("*.parquet"))
    assert len(parquet_files) > 0, "No parquet files produced"

    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()

        # Check schema
        assert list(df.columns) == TARGET_COLUMNS, (
            f"Schema mismatch: {list(df.columns)}"
        )

        # Check all values are strings
        for col in TARGET_COLUMNS:
            assert pd.api.types.is_string_dtype(df[col]), f"{col} is not string type"

        # Validate CDR3 sequences
        for col in ("tra", "trb"):
            for val in df[col]:
                if val:
                    assert all(c in VALID_AA for c in val), (
                        f"Invalid AA in {col}: {val}"
                    )
                    assert len(val) >= 4, f"Short CDR3 in {col}: {val}"

        # Validate gene names are TCR (not IG/BCR)
        for col in (
            "trav_gene", "trad_gene", "traj_gene",
            "trbv_gene", "trbd_gene", "trbj_gene",
        ):
            for val in df[col]:
                if val:
                    assert val.startswith("TR"), (
                        f"Non-TCR gene in {col}: {val}"
                    )

        # Validate CDR columns
        for chain, cdr3_col in [("tra", "tra_cdr3"), ("trb", "trb_cdr3")]:
            for i, row in df.iterrows():
                # CDR3 should equal the chain column when both are non-empty
                if row[chain] and row[cdr3_col]:
                    assert row[cdr3_col] == row[chain], (
                        f"Row {i}: {cdr3_col} ({row[cdr3_col]}) != {chain} ({row[chain]})"
                    )

        for col in ("tra_cdr1", "tra_cdr2", "trb_cdr1", "trb_cdr2"):
            for val in df[col]:
                if val:
                    assert all(c in VALID_AA for c in val), (
                        f"Invalid AA in {col}: {val}"
                    )

        # tra_full/trb_full should be empty in non-stitch mode
        for col in ("tra_full", "trb_full"):
            assert all(df[col] == ""), f"{col} should be empty in default mode"

        # Check source is set
        assert all(df["source"] != ""), "source column has empty values"

    # Check dropped.tsv exists
    dropped_path = output_dir / "dropped.tsv"
    assert dropped_path.exists(), "dropped.tsv not created"

    # Check manifest.json exists
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json not created"

    return True


# -----------------------------------------------------------------------
# BATMAN
# -----------------------------------------------------------------------
class TestBatmanStandardizer:
    def test_end_to_end(self, tmp_path):
        from scripts.data_processing.standardize.batman import BatmanStandardizer

        # Create mock Excel file — BATMAN stores gene numbers without prefix
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "tcr": ["TCR1", "TCR2"],
                "va": ["", ""],
                "vb": ["", ""],
                "cdr3a": ["CAVRDSNYQLIW", "CAVKDSNYQLIW"],
                "cdr3b": ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF"],
                "trav": ["1-2", "12-1"],
                "traj": ["33", "49"],
                "trbv": ["6-5", "20-1"],
                "trbd": ["1*00(80)", ""],
                "trbj": ["2-1", "1-2"],
                "assay": ["SPR", "tetramer"],
                "tcr_source_organism": ["human", "human"],
                "index_peptide": ["GILGFVFTL", "NLVPMVATV"],
                "mhc": ["HLA-A*02:01", "HLA-A*02:01"],
                "pmid": ["12345", "67890"],
                "peptide_type": ["viral", "viral"],
                "peptide": ["GILGFVFTL", "NLVPMVATV"],
                "peptide_activity": ["1.0", "0.5"],
            }
        )
        df.to_excel(
            source_dir / "TCR_pMHCI_mutational_scan_database.xlsx",
            index=False,
        )

        output_dir = tmp_path / "output"
        standardizer = BatmanStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 2
        _verify_output(output_dir)


# -----------------------------------------------------------------------
# TaDB
# -----------------------------------------------------------------------
class TestTadbStandardizer:
    def test_end_to_end(self, tmp_path):
        from scripts.data_processing.standardize.tadb import TadbStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "ACCESSION": ["T000001", "T000002"],
                "Epitope sequence": ["ALCRWGLLL", "ALIHHNTHL"],
                "HLA allele": ["A*0201", "A*0201"],
                "Epitope type": ["Overexpressed", "Overexpressed"],
            }
        )
        df.to_csv(source_dir / "tadb_t_cell_epitopes.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = TadbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 2
        _verify_output(output_dir)


# -----------------------------------------------------------------------
# McPAS
# -----------------------------------------------------------------------
class TestMcpasStandardizer:
    def test_end_to_end(self, tmp_path):
        from scripts.data_processing.standardize.mcpas import McpasStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "CDR3.alpha.aa": ["CAVRDSNYQLIW", ""],
                "CDR3.beta.aa": ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF"],
                "TRAV": ["TRAV1-2", ""],
                "TRAJ": ["TRAJ33", ""],
                "TRBV": ["TRBV6-5", "TRBV20-1"],
                "TRBD": ["", ""],
                "TRBJ": ["TRBJ2-1", "TRBJ1-2"],
                "Epitope.peptide": ["GILGFVFTL", "NLVPMVATV"],
                "MHC": ["HLA-A*02:01", "HLA-A*02:01"],
                "Species": ["Human", "Human"],
            }
        )
        df.to_csv(source_dir / "McPAS-TCR.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = McpasStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] >= 2
        _verify_output(output_dir)

    def test_epitope_slash_exploded(self, tmp_path):
        from scripts.data_processing.standardize.mcpas import McpasStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "CDR3.beta.aa": ["CASSLAPGATNEKLFF"],
                "TRBV": ["TRBV6-5"],
                "TRBJ": ["TRBJ2-1"],
                "Epitope.peptide": ["GILGFVFTL/NLVPMVATV"],
                "MHC": ["HLA-A*02:01"],
            }
        )
        df.to_csv(source_dir / "McPAS-TCR.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = McpasStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        # Should produce 2 rows (one per epitope)
        assert summary["rows"] == 2


# -----------------------------------------------------------------------
# VDJdb
# -----------------------------------------------------------------------
class TestVdjdbStandardizer:
    def test_end_to_end(self, tmp_path):
        from scripts.data_processing.standardize.vdjdb import VdjdbStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "cdr3.alpha": ["CAVRDSNYQLIW", ""],
                "v.alpha": ["TRAV1-2", ""],
                "j.alpha": ["TRAJ33", ""],
                "cdr3.beta": ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF"],
                "v.beta": ["TRBV6-5", "TRBV20-1"],
                "d.beta": ["", ""],
                "j.beta": ["TRBJ2-1", "TRBJ1-2"],
                "species": ["HomoSapiens", "HomoSapiens"],
                "mhc.a": ["HLA-A*02:01", "HLA-A*02:01"],
                "mhc.b": ["B2M", "B2M"],
                "mhc.class": ["MHCI", "MHCI"],
                "antigen.epitope": ["GILGFVFTL", "NLVPMVATV"],
                "meta.study.id": ["study1", "study2"],
            }
        )
        df.to_csv(source_dir / "vdjdb_full.txt", sep="\t", index=False)

        output_dir = tmp_path / "output"
        standardizer = VdjdbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 2
        _verify_output(output_dir)


# -----------------------------------------------------------------------
# OTS
# -----------------------------------------------------------------------
class TestOtsStandardizer:
    def test_end_to_end(self, tmp_path):
        from scripts.data_processing.standardize.ots import OtsStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create mock CSV with JSON metadata row
        lines = [
            '{"metadata": "test"}',
            "cdr3_aa_alpha,v_call_alpha,j_call_alpha,cdr3_aa_beta,v_call_beta,d_call_beta,j_call_beta",
            "CAVRDSNYQLIW,TRAV1-2,TRAJ33,CASSLAPGATNEKLFF,TRBV6-5,TRBD1,TRBJ2-1",
        ]
        (source_dir / "SRR12345_1_Paired_All.csv").write_text("\n".join(lines))

        output_dir = tmp_path / "output"
        standardizer = OtsStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] >= 1
        _verify_output(output_dir)


# -----------------------------------------------------------------------
# CEDAR
# -----------------------------------------------------------------------
def _write_multilevel_csv(path, group_row, field_row, data_rows):
    """Write a CSV with two header rows (multi-level) like CEDAR/IEDB exports."""
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(group_row)
        writer.writerow(field_row)
        for row in data_rows:
            writer.writerow(row)


class TestCedarStandardizer:
    def test_end_to_end(self, tmp_path):
        from scripts.data_processing.standardize.cedar import CedarStandardizer

        source_dir = tmp_path / "source"
        receptor_dir = source_dir / "receptor"
        receptor_dir.mkdir(parents=True)

        # Multi-level header: group row + field row
        # Include CDR3 Start/End Curated columns to verify they're excluded
        groups = [
            "Chain 1", "", "", "", "", "", "",
            "Chain 2", "", "", "", "", "", "",
            "Epitope", "",
            "Assay", "",
        ]
        fields = [
            "CDR3 Curated", "CDR3 Start Curated", "CDR3 End Curated",
            "V Gene Curated", "D Gene Curated", "J Gene Curated", "IRI",
            "CDR3 Curated", "CDR3 Start Curated", "CDR3 End Curated",
            "V Gene Curated", "D Gene Curated", "J Gene Curated", "IRI",
            "Name", "IRI",
            "MHC Allele Names", "Organism Source",
        ]
        data = [
            [
                "CAVRDSNYQLIW", "10", "22",
                "TRAV1-2", "", "TRAJ33", "http://cedar.iedb.org/receptor/100",
                "CASSLAPGATNEKLFF", "5", "21",
                "TRBV6-5", "", "TRBJ2-1", "",
                "GILGFVFTL", "",
                "HLA-A*02:01", "Homo sapiens",
            ],
            [
                "CAVKDSNYQLIW", "10", "22",
                "TRAV12-1", "", "TRAJ49", "",
                "CASSLGQAYEQYF", "5", "18",
                "TRBV20-1", "", "TRBJ1-2", "",
                "NLVPMVATV", "",
                "HLA-A*02:01", "Homo sapiens",
            ],
        ]
        _write_multilevel_csv(receptor_dir / "tcr_full_v3.csv", groups, fields, data)

        output_dir = tmp_path / "output"
        standardizer = CedarStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 2
        _verify_output(output_dir)

        # Verify peptide and MHC are populated
        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert list(result["peptide"]) == ["GILGFVFTL", "NLVPMVATV"]
        assert all(result["mhc_one"] != "")

    def test_no_tcell_file_still_works(self, tmp_path):
        """CEDAR should produce rows even without tcell file."""
        from scripts.data_processing.standardize.cedar import CedarStandardizer

        source_dir = tmp_path / "source"
        receptor_dir = source_dir / "receptor"
        receptor_dir.mkdir(parents=True)

        groups = ["Chain 2", "", "", "Epitope", "Assay"]
        fields = ["CDR3 Curated", "V Gene Curated", "J Gene Curated", "Name", "MHC Allele Names"]
        data = [
            ["CASSLAPGATNEKLFF", "TRBV6-5", "TRBJ2-1", "GILGFVFTL", "HLA-A*02:01"],
        ]
        _write_multilevel_csv(receptor_dir / "tcr_full_v3.csv", groups, fields, data)

        output_dir = tmp_path / "output"
        standardizer = CedarStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 1


# -----------------------------------------------------------------------
# IEDB
# -----------------------------------------------------------------------
class TestIedbStandardizer:
    def test_end_to_end(self, tmp_path):
        from scripts.data_processing.standardize.iedb import IedbStandardizer

        source_dir = tmp_path / "source"
        receptor_dir = source_dir / "receptor"
        tcell_dir = source_dir / "tcell"
        receptor_dir.mkdir(parents=True)
        tcell_dir.mkdir(parents=True)

        # Receptor CSV with assay IDs — uses realistic "alphabeta" type (not "TCR")
        # Includes CDR3 Start/End Curated columns to verify they're excluded
        r_groups = [
            "Receptor", "",
            "Chain 1", "", "", "", "", "",
            "Chain 2", "", "", "", "", "",
            "Assay", "",
        ]
        r_fields = [
            "Type", "IRI",
            "CDR3 Curated", "CDR3 Start Curated", "CDR3 End Curated",
            "V Gene Curated", "D Gene Curated", "J Gene Curated",
            "CDR3 Curated", "CDR3 Start Curated", "CDR3 End Curated",
            "V Gene Curated", "D Gene Curated", "J Gene Curated",
            "IEDB IDs", "Other",
        ]
        r_data = [
            [
                "alphabeta", "http://www.iedb.org/receptor/100",
                "CAVRDSNYQLIW", "10", "22",
                "TRAV1-2", "", "TRAJ33",
                "CASSLAPGATNEKLFF", "5", "21",
                "TRBV6-5", "", "TRBJ2-1",
                "1001", "",
            ],
            [
                "alphabeta", "http://www.iedb.org/receptor/200",
                "CAVKDSNYQLIW", "10", "22",
                "TRAV12-1", "", "TRAJ49",
                "CASSLGQAYEQYF", "5", "18",
                "TRBV20-1", "", "TRBJ1-2",
                "1002, 1003", "",
            ],
        ]
        _write_multilevel_csv(receptor_dir / "tcr_full_v3.csv", r_groups, r_fields, r_data)

        # Tcell CSV with assay IRIs
        t_groups = [
            "Assay ID", "",
            "Epitope", "",
            "MHC", "",
            "Outcome", "",
            "Host", "",
        ]
        t_fields = [
            "IEDB IRI", "Other",
            "Name", "IRI",
            "Allele Names", "Other",
            "Qualitative Measurement", "Other",
            "Organism Source", "Other",
        ]
        t_data = [
            [
                "http://www.iedb.org/assay/1001", "",
                "GILGFVFTL", "",
                "HLA-A*02:01", "",
                "Positive", "",
                "Homo sapiens", "",
            ],
            [
                "http://www.iedb.org/assay/1002", "",
                "NLVPMVATV", "",
                "HLA-A*02:01", "",
                "Positive", "",
                "Homo sapiens", "",
            ],
            [
                "http://www.iedb.org/assay/9999", "",
                "SIINFEKL", "",
                "H-2Kb", "",
                "Negative", "",
                "Mus musculus", "",
            ],
        ]
        _write_multilevel_csv(tcell_dir / "tcell_full_v3.csv", t_groups, t_fields, t_data)

        output_dir = tmp_path / "output"
        standardizer = IedbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 2
        _verify_output(output_dir)

        # Verify peptide, MHC, and binding are populated
        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert list(result["peptide"]) == ["GILGFVFTL", "NLVPMVATV"]
        assert all(result["mhc_one"] != "")
        assert all(result["binding"] == "Positive")

    def test_negative_outcomes_excluded(self, tmp_path):
        """IEDB should filter out negative assay outcomes."""
        from scripts.data_processing.standardize.iedb import IedbStandardizer

        source_dir = tmp_path / "source"
        receptor_dir = source_dir / "receptor"
        tcell_dir = source_dir / "tcell"
        receptor_dir.mkdir(parents=True)
        tcell_dir.mkdir(parents=True)

        r_groups = ["Receptor", "Chain 2", "", "", "Assay"]
        r_fields = ["Type", "CDR3 Curated", "V Gene Curated", "J Gene Curated", "IEDB IDs"]
        r_data = [
            ["alphabeta", "CASSLAPGATNEKLFF", "TRBV6-5", "TRBJ2-1", "2001"],
            ["alphabeta", "CASSLGQAYEQYF", "TRBV20-1", "TRBJ1-2", "2002"],
        ]
        _write_multilevel_csv(receptor_dir / "tcr_full_v3.csv", r_groups, r_fields, r_data)

        t_groups = ["Assay ID", "Epitope", "MHC", "Outcome"]
        t_fields = ["IEDB IRI", "Name", "Allele Names", "Qualitative Measurement"]
        t_data = [
            ["http://www.iedb.org/assay/2001", "GILGFVFTL", "HLA-A*02:01", "Positive"],
            ["http://www.iedb.org/assay/2002", "NLVPMVATV", "HLA-A*02:01", "Negative"],
        ]
        _write_multilevel_csv(tcell_dir / "tcell_full_v3.csv", t_groups, t_fields, t_data)

        output_dir = tmp_path / "output"
        standardizer = IedbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        # Row with negative outcome: no epitope/MHC match → only CDR3 data
        # Row with positive outcome: full data
        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        positive_rows = result[result["binding"] == "Positive"]
        assert len(positive_rows) == 1
        assert positive_rows.iloc[0]["peptide"] == "GILGFVFTL"


# -----------------------------------------------------------------------
# Incremental updates
# -----------------------------------------------------------------------
class TestIncrementalUpdates:
    def test_is_stale_no_manifest(self, tmp_path):
        from scripts.data_processing.standardize.batman import BatmanStandardizer

        standardizer = BatmanStandardizer(
            source_dir=tmp_path / "source",
            output_dir=tmp_path / "output",
        )
        assert standardizer.is_stale() is True

    def test_skips_when_up_to_date(self, tmp_path):
        from scripts.data_processing.standardize.tadb import TadbStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "ACCESSION": ["T000001"],
                "Epitope sequence": ["ALCRWGLLL"],
                "HLA allele": ["A*0201"],
                "Epitope type": ["Overexpressed"],
            }
        )
        df.to_csv(source_dir / "tadb.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = TadbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )

        # First run
        summary1 = standardizer.run(force=True)
        assert summary1["status"] == "completed"

        # Second run (should skip)
        summary2 = standardizer.run(force=False)
        assert summary2["status"] == "skipped"


# -----------------------------------------------------------------------
# Verify method
# -----------------------------------------------------------------------
class TestVerify:
    def test_verify_after_run(self, tmp_path):
        from scripts.data_processing.standardize.tadb import TadbStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "ACCESSION": ["T000001", "T000002"],
                "Epitope sequence": ["ALCRWGLLL", "ALIHHNTHL"],
                "HLA allele": ["A*0201", "A*0201"],
                "Epitope type": ["Overexpressed", "Overexpressed"],
            }
        )
        df.to_csv(source_dir / "tadb.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = TadbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        standardizer.run(force=True)

        result = standardizer.verify()
        assert result["total_rows"] == 2
        assert result["parquet_files"] == 1
        assert "null_ratios" in result


# -----------------------------------------------------------------------
# Species filtering
# -----------------------------------------------------------------------
class TestSpeciesFiltering:
    def test_batman_filters_mouse(self, tmp_path):
        from scripts.data_processing.standardize.batman import BatmanStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "tcr": ["TCR1", "TCR2", "TCR3"],
                "va": ["", "", ""],
                "vb": ["", "", ""],
                "cdr3a": ["CAVRDSNYQLIW", "CAVKDSNYQLIW", "CAVMDSNYQLIW"],
                "cdr3b": ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF", "CASSLAPNEKLFF"],
                "trav": ["1-2", "12-1", "1-2"],
                "traj": ["33", "49", "33"],
                "trbv": ["6-5", "20-1", "6-5"],
                "trbd": ["", "", ""],
                "trbj": ["2-1", "1-2", "2-1"],
                "assay": ["SPR", "tetramer", "SPR"],
                "tcr_source_organism": ["human", "Mouse", "Human"],
                "index_peptide": ["GILGFVFTL", "SIINFEKL", "NLVPMVATV"],
                "mhc": ["HLA-A*02:01", "H-2Kb", "HLA-A*02:01"],
                "pmid": ["12345", "67890", "11111"],
                "peptide_type": ["viral", "viral", "viral"],
                "peptide": ["GILGFVFTL", "SIINFEKL", "NLVPMVATV"],
                "peptide_activity": ["1.0", "0.5", "0.8"],
            }
        )
        df.to_excel(
            source_dir / "TCR_pMHCI_mutational_scan_database.xlsx",
            index=False,
        )

        output_dir = tmp_path / "output"
        standardizer = BatmanStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        # Only 2 human rows should survive (mouse row filtered out)
        assert summary["rows"] == 2

    def test_vdjdb_filters_mouse(self, tmp_path):
        from scripts.data_processing.standardize.vdjdb import VdjdbStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "cdr3.alpha": ["CAVRDSNYQLIW", ""],
                "v.alpha": ["TRAV1-2", ""],
                "j.alpha": ["TRAJ33", ""],
                "cdr3.beta": ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF"],
                "v.beta": ["TRBV6-5", "TRBV20-1"],
                "d.beta": ["", ""],
                "j.beta": ["TRBJ2-1", "TRBJ1-2"],
                "species": ["HomoSapiens", "MusMusculus"],
                "mhc.a": ["HLA-A*02:01", "H-2Kb"],
                "mhc.b": ["B2M", "B2M"],
                "mhc.class": ["MHCI", "MHCI"],
                "antigen.epitope": ["GILGFVFTL", "SIINFEKL"],
                "meta.study.id": ["study1", "study2"],
            }
        )
        df.to_csv(source_dir / "vdjdb_full.txt", sep="\t", index=False)

        output_dir = tmp_path / "output"
        standardizer = VdjdbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        # Only HomoSapiens row should survive
        assert summary["rows"] == 1

    def test_mcpas_filters_mouse(self, tmp_path):
        from scripts.data_processing.standardize.mcpas import McpasStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "CDR3.beta.aa": ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF"],
                "TRBV": ["TRBV6-5", "TRBV20-1"],
                "TRBJ": ["TRBJ2-1", "TRBJ1-2"],
                "Epitope.peptide": ["GILGFVFTL", "SIINFEKL"],
                "MHC": ["HLA-A*02:01", "H-2Kb"],
                "Species": ["Human", "Mouse"],
            }
        )
        df.to_csv(source_dir / "McPAS-TCR.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = McpasStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        # Only Human row should survive
        assert summary["rows"] == 1


# -----------------------------------------------------------------------
# TCR-only gene filtering
# -----------------------------------------------------------------------
class TestTCROnlyGeneFiltering:
    def test_ig_genes_rejected(self):
        from quest.data.standardization import normalize_gene

        # TCR genes should pass
        assert normalize_gene("TRBV6-5") != ""
        assert normalize_gene("TRAV1-2") != ""
        assert normalize_gene("TRBJ2-1") != ""

        # IG (BCR) genes should be rejected
        assert normalize_gene("IGHV1-2") == ""
        assert normalize_gene("IGKV1-5") == ""
        assert normalize_gene("IGLV1-40") == ""
        assert normalize_gene("IGHJ4") == ""


# -----------------------------------------------------------------------
# Binding & Score columns
# -----------------------------------------------------------------------
class TestBindingScoreColumns:
    def test_batman_score_and_binding_populated(self, tmp_path):
        from scripts.data_processing.standardize.batman import BatmanStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "tcr": ["TCR1", "TCR2"],
                "va": ["", ""],
                "vb": ["", ""],
                "cdr3a": ["CAVRDSNYQLIW", "CAVKDSNYQLIW"],
                "cdr3b": ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF"],
                "trav": ["1-2", "12-1"],
                "traj": ["33", "49"],
                "trbv": ["6-5", "20-1"],
                "trbd": ["", ""],
                "trbj": ["2-1", "1-2"],
                "assay": ["SPR", "tetramer"],
                "tcr_source_organism": ["human", "human"],
                "index_peptide": ["GILGFVFTL", "NLVPMVATV"],
                "mhc": ["HLA-A*02:01", "HLA-A*02:01"],
                "pmid": ["12345", "67890"],
                "peptide_type": ["viral", "viral"],
                "peptide": ["GILGFVFTL", "NLVPMVATV"],
                "peptide_activity": ["37", "126"],
            }
        )
        df.to_excel(
            source_dir / "TCR_pMHCI_mutational_scan_database.xlsx",
            index=False,
        )

        output_dir = tmp_path / "output"
        standardizer = BatmanStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        standardizer.run(force=True)

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert "binding" in result.columns
        assert "score" in result.columns
        # score now contains the numeric peptide_activity values
        assert list(result["score"]) == ["37", "126"]
        # binding derived from threshold: both >= 0.1 → "pos"
        assert list(result["binding"]) == ["pos", "pos"]

    def test_vdjdb_score_populated(self, tmp_path):
        from scripts.data_processing.standardize.vdjdb import VdjdbStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "cdr3.alpha": ["CAVRDSNYQLIW", "", "CAVKDSNYQLIW"],
                "v.alpha": ["TRAV1-2", "", "TRAV12-1"],
                "j.alpha": ["TRAJ33", "", "TRAJ49"],
                "cdr3.beta": ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF", "CASSDRGYTFGSGANVLTF"],
                "v.beta": ["TRBV6-5", "TRBV20-1", "TRBV6-5"],
                "d.beta": ["", "", ""],
                "j.beta": ["TRBJ2-1", "TRBJ1-2", "TRBJ2-1"],
                "species": ["HomoSapiens", "HomoSapiens", "HomoSapiens"],
                "mhc.a": ["HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01"],
                "mhc.b": ["B2M", "B2M", "B2M"],
                "mhc.class": ["MHCI", "MHCI", "MHCI"],
                "antigen.epitope": ["GILGFVFTL", "NLVPMVATV", "KLVALGINAV"],
                "vdjdb.score": ["3", "1", "0"],
                "meta.study.id": ["study1", "study2", "study3"],
            }
        )
        df.to_csv(source_dir / "vdjdb_full.txt", sep="\t", index=False)

        output_dir = tmp_path / "output"
        standardizer = VdjdbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        standardizer.run(force=True)

        # Read all parquet files (multiple yields produce multiple parts)
        all_parts = sorted(output_dir.glob("*.parquet"))
        result = pd.concat(
            [pq.read_table(p).to_pandas() for p in all_parts], ignore_index=True,
        )
        assert "score" in result.columns
        assert "binding" in result.columns

        # High-score rows should retain all fields
        high_score = result[result["score"].isin(["3", "1"])]
        assert len(high_score) == 2
        assert all(high_score["peptide"] != "")

        # Score-0 rows should be split into TCR-only and pMHC-only
        score_zero = result[result["score"] == "0"]
        # TCR-only row has no peptide/MHC, pMHC-only row has no TCR
        tcr_only = score_zero[(score_zero["tra"] != "") | (score_zero["trb"] != "")]
        pmhc_only = score_zero[(score_zero["peptide"] != "") | (score_zero["mhc_one"] != "")]
        assert len(tcr_only) >= 1
        assert len(pmhc_only) >= 1
        # TCR-only should have empty peptide
        assert all(tcr_only["peptide"] == "")
        # pMHC-only should have empty TCR
        assert all(pmhc_only["tra"] == "")
        assert all(pmhc_only["trb"] == "")

    def test_trait_binding_from_filename(self, tmp_path):
        """TRAIT extracts pos/neg binder status from filename."""
        from scripts.data_processing.standardize.trait import _parse_trait_filename

        meta_pos = _parse_trait_filename("A0201_GILGFVFTL_BMLF1_EBV_binder_pos.zip")
        assert meta_pos["binding"] == "pos"

        meta_neg = _parse_trait_filename("A0201_GILGFVFTL_BMLF1_EBV_binder_neg.zip")
        assert meta_neg["binding"] == "neg"

    def test_schema_has_25_columns(self):
        assert len(TARGET_COLUMNS) == 25
        assert "binding" in TARGET_COLUMNS
        assert "score" in TARGET_COLUMNS
        # binding and score should come after mhc_two, before source
        mhc_two_idx = TARGET_COLUMNS.index("mhc_two")
        source_idx = TARGET_COLUMNS.index("source")
        binding_idx = TARGET_COLUMNS.index("binding")
        score_idx = TARGET_COLUMNS.index("score")
        assert mhc_two_idx < binding_idx < score_idx < source_idx
        # CDR columns should be present
        for col in ("tra_cdr1", "tra_cdr2", "tra_cdr3", "tra_full",
                     "trb_cdr1", "trb_cdr2", "trb_cdr3", "trb_full"):
            assert col in TARGET_COLUMNS


# -----------------------------------------------------------------------
# ImmuneCODE edge cases
# -----------------------------------------------------------------------
class TestImmunecodeStandardizer:
    def test_only_ci_csv_used(self, tmp_path):
        """MIRA should only process peptide-detail-ci.csv (Class I)."""
        from scripts.data_processing.standardize.immunecode import (
            ImmunecodeStandardizer,
        )

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create Class I file with valid data
        ci_df = pd.DataFrame(
            {
                "TCR BioIdentity": ["CASSLAPGATNEKLFF+TRBV6-5+TRBJ2-1"],
                "Amino Acids": ["GILGFVFTL"],
                "Experiment": ["EXP001"],
            }
        )
        ci_df.to_csv(source_dir / "peptide-detail-ci.csv", index=False)

        # Create Class II file (should be ignored)
        cii_df = pd.DataFrame(
            {
                "TCR BioIdentity": ["CASSLGQAYEQYF+TRBV20-1+TRBJ1-2"],
                "Amino Acids": ["NLVPMVATV"],
                "Experiment": ["EXP002"],
            }
        )
        cii_df.to_csv(source_dir / "peptide-detail-cii.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = ImmunecodeStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        # Only 1 row from ci.csv, not 2
        assert summary["rows"] == 1

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert list(result["peptide"]) == ["GILGFVFTL"]

    def test_comma_separated_peptides_split(self, tmp_path):
        """Comma-separated peptides in Amino Acids column should be split."""
        from scripts.data_processing.standardize.immunecode import (
            ImmunecodeStandardizer,
        )

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        df = pd.DataFrame(
            {
                "TCR BioIdentity": ["CASSLAPGATNEKLFF+TRBV6-5+TRBJ2-1"],
                "Amino Acids": ["GILGFVFTL,NLVPMVATV"],
                "Experiment": ["EXP001"],
            }
        )
        df.to_csv(source_dir / "peptide-detail-ci.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = ImmunecodeStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["rows"] == 2

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert set(result["peptide"]) == {"GILGFVFTL", "NLVPMVATV"}

    def test_long_peptides_filtered(self, tmp_path):
        """Peptides > 25 AA (minigene constructs) should be filtered."""
        from scripts.data_processing.standardize.immunecode import (
            ImmunecodeStandardizer,
        )

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        long_peptide = "A" * 30  # 30 AA — too long
        df = pd.DataFrame(
            {
                "TCR BioIdentity": [
                    "CASSLAPGATNEKLFF+TRBV6-5+TRBJ2-1",
                    "CASSLGQAYEQYF+TRBV20-1+TRBJ1-2",
                ],
                "Amino Acids": ["GILGFVFTL", long_peptide],
                "Experiment": ["EXP001", "EXP002"],
            }
        )
        df.to_csv(source_dir / "peptide-detail-ci.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = ImmunecodeStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        # Only the short peptide row should survive
        assert summary["rows"] == 1

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert list(result["peptide"]) == ["GILGFVFTL"]


# -----------------------------------------------------------------------
# VDJdb study_id edge cases
# -----------------------------------------------------------------------
class TestVdjdbStudyId:
    def test_per_row_study_id_preserved_after_drops(self, tmp_path):
        """Per-row study_id from meta.study.id should survive row drops."""
        from scripts.data_processing.standardize.vdjdb import VdjdbStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Row 2 has invalid CDR3 (too short) and no other valid field
        # so it will be dropped, but row 1 and 3 should retain their study_ids
        df = pd.DataFrame(
            {
                "cdr3.beta": ["CASSLAPGATNEKLFF", "CA", "CASSLGQAYEQYF"],
                "v.beta": ["TRBV6-5", "", "TRBV20-1"],
                "d.beta": ["", "", ""],
                "j.beta": ["TRBJ2-1", "", "TRBJ1-2"],
                "species": ["HomoSapiens", "HomoSapiens", "HomoSapiens"],
                "mhc.a": ["HLA-A*02:01", "", "HLA-A*02:01"],
                "mhc.b": ["B2M", "", "B2M"],
                "antigen.epitope": ["GILGFVFTL", "", "NLVPMVATV"],
                "meta.study.id": ["PMID:111", "PMID:222", "PMID:333"],
            }
        )
        df.to_csv(source_dir / "vdjdb_full.txt", sep="\t", index=False)

        output_dir = tmp_path / "output"
        standardizer = VdjdbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["rows"] == 2

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert list(result["study_id"]) == ["PMID:111", "PMID:333"]


# -----------------------------------------------------------------------
# VDJdb score-based row splitting
# -----------------------------------------------------------------------
class TestVdjdbScoreSplitting:
    def test_score_zero_split_into_tcr_and_pmhc(self, tmp_path):
        """Score-0 row with TCR+peptide+MHC produces TCR-only + pMHC-only rows."""
        from scripts.data_processing.standardize.vdjdb import VdjdbStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "cdr3.alpha": ["CAVRDSNYQLIW"],
                "v.alpha": ["TRAV1-2"],
                "j.alpha": ["TRAJ33"],
                "cdr3.beta": ["CASSLAPGATNEKLFF"],
                "v.beta": ["TRBV6-5"],
                "d.beta": [""],
                "j.beta": ["TRBJ2-1"],
                "species": ["HomoSapiens"],
                "mhc.a": ["HLA-A*02:01"],
                "mhc.b": ["B2M"],
                "mhc.class": ["MHCI"],
                "antigen.epitope": ["GILGFVFTL"],
                "vdjdb.score": ["0"],
                "meta.study.id": ["study1"],
            }
        )
        df.to_csv(source_dir / "vdjdb_full.txt", sep="\t", index=False)

        output_dir = tmp_path / "output"
        standardizer = VdjdbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"

        all_parts = sorted(output_dir.glob("*.parquet"))
        result = pd.concat(
            [pq.read_table(p).to_pandas() for p in all_parts], ignore_index=True,
        )
        # Should produce 2 rows: TCR-only + pMHC-only
        assert len(result) == 2

        # TCR-only row: has CDR3 but no peptide/MHC
        tcr_row = result[(result["tra"] != "") | (result["trb"] != "")]
        assert len(tcr_row) == 1
        assert tcr_row.iloc[0]["tra"] == "CAVRDSNYQLIW"
        assert tcr_row.iloc[0]["trb"] == "CASSLAPGATNEKLFF"
        assert tcr_row.iloc[0]["peptide"] == ""
        assert tcr_row.iloc[0]["mhc_one"] == ""

        # pMHC-only row: has peptide/MHC but no CDR3
        pmhc_row = result[(result["peptide"] != "") | (result["mhc_one"] != "")]
        assert len(pmhc_row) == 1
        assert pmhc_row.iloc[0]["peptide"] == "GILGFVFTL"
        assert pmhc_row.iloc[0]["mhc_one"] == "HLA-A*02:01"
        assert pmhc_row.iloc[0]["tra"] == ""
        assert pmhc_row.iloc[0]["trb"] == ""

    def test_score_one_keeps_all_fields(self, tmp_path):
        """Score-1 row keeps all fields intact (full interaction training)."""
        from scripts.data_processing.standardize.vdjdb import VdjdbStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "cdr3.alpha": ["CAVRDSNYQLIW"],
                "v.alpha": ["TRAV1-2"],
                "j.alpha": ["TRAJ33"],
                "cdr3.beta": ["CASSLAPGATNEKLFF"],
                "v.beta": ["TRBV6-5"],
                "d.beta": [""],
                "j.beta": ["TRBJ2-1"],
                "species": ["HomoSapiens"],
                "mhc.a": ["HLA-A*02:01"],
                "mhc.b": ["B2M"],
                "mhc.class": ["MHCI"],
                "antigen.epitope": ["GILGFVFTL"],
                "vdjdb.score": ["1"],
                "meta.study.id": ["study1"],
            }
        )
        df.to_csv(source_dir / "vdjdb_full.txt", sep="\t", index=False)

        output_dir = tmp_path / "output"
        standardizer = VdjdbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 1

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert len(result) == 1
        row = result.iloc[0]
        assert row["tra"] == "CAVRDSNYQLIW"
        assert row["trb"] == "CASSLAPGATNEKLFF"
        assert row["peptide"] == "GILGFVFTL"
        assert row["mhc_one"] == "HLA-A*02:01"
        assert row["score"] == "1"

    def test_missing_score_treated_as_trusted(self, tmp_path):
        """Row without vdjdb.score column is treated as score>=1 (full row)."""
        from scripts.data_processing.standardize.vdjdb import VdjdbStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "cdr3.alpha": ["CAVRDSNYQLIW"],
                "v.alpha": ["TRAV1-2"],
                "j.alpha": ["TRAJ33"],
                "cdr3.beta": ["CASSLAPGATNEKLFF"],
                "v.beta": ["TRBV6-5"],
                "d.beta": [""],
                "j.beta": ["TRBJ2-1"],
                "species": ["HomoSapiens"],
                "mhc.a": ["HLA-A*02:01"],
                "mhc.b": ["B2M"],
                "mhc.class": ["MHCI"],
                "antigen.epitope": ["GILGFVFTL"],
                "meta.study.id": ["study1"],
            }
        )
        df.to_csv(source_dir / "vdjdb_full.txt", sep="\t", index=False)

        output_dir = tmp_path / "output"
        standardizer = VdjdbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 1

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert len(result) == 1
        row = result.iloc[0]
        # All fields should be present (treated as full row)
        assert row["tra"] == "CAVRDSNYQLIW"
        assert row["trb"] == "CASSLAPGATNEKLFF"
        assert row["peptide"] == "GILGFVFTL"
        assert row["mhc_one"] == "HLA-A*02:01"

    def test_score_zero_tcr_only_when_no_peptide(self, tmp_path):
        """Score-0 row with TCR but no peptide/MHC → only TCR-only row emitted."""
        from scripts.data_processing.standardize.vdjdb import VdjdbStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "cdr3.alpha": [""],
                "v.alpha": [""],
                "j.alpha": [""],
                "cdr3.beta": ["CASSLAPGATNEKLFF"],
                "v.beta": ["TRBV6-5"],
                "d.beta": [""],
                "j.beta": ["TRBJ2-1"],
                "species": ["HomoSapiens"],
                "mhc.a": [""],
                "mhc.b": [""],
                "mhc.class": [""],
                "antigen.epitope": [""],
                "vdjdb.score": ["0"],
                "meta.study.id": ["study1"],
            }
        )
        df.to_csv(source_dir / "vdjdb_full.txt", sep="\t", index=False)

        output_dir = tmp_path / "output"
        standardizer = VdjdbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        # Only TCR-only row, no pMHC row (no peptide/MHC data)
        assert summary["rows"] == 1

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert len(result) == 1
        row = result.iloc[0]
        assert row["trb"] == "CASSLAPGATNEKLFF"
        assert row["peptide"] == ""
        assert row["mhc_one"] == ""


# -----------------------------------------------------------------------
# ADC gamma-delta filtering
# -----------------------------------------------------------------------
class TestAdcStandardizer:
    def test_gamma_delta_loci_skipped(self, tmp_path):
        """TRD and TRG loci should be skipped (not mapped to alpha)."""
        from scripts.data_processing.standardize.adc import AdcStandardizer

        source_dir = tmp_path / "source"
        repo_dir = source_dir / "repo1" / "rearrangements"
        repo_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "junction_aa": [
                    "CASSLAPGATNEKLFF",
                    "CAVRDSNYQLIW",
                    "CASSDRGYTF",
                    "CALGELNYQLIW",
                ],
                "v_call": ["TRBV6-5", "TRAV1-2", "TRDV1", "TRGV9"],
                "d_call": ["", "", "", ""],
                "j_call": ["TRBJ2-1", "TRAJ33", "TRDJ1", "TRGJ1"],
                "locus": ["TRB", "TRA", "TRD", "TRG"],
                "productive": ["true", "true", "true", "true"],
            }
        )
        df.to_csv(repo_dir / "rep1.tsv", sep="\t", index=False)

        output_dir = tmp_path / "output"
        standardizer = AdcStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        # Only TRA and TRB rows should be kept (2 rows), TRD and TRG skipped
        assert summary["rows"] == 2

        # Read all parquet files (data split by locus into separate files)
        all_parts = sorted(output_dir.glob("*.parquet"))
        result = pd.concat(
            [pq.read_table(p).to_pandas() for p in all_parts], ignore_index=True,
        )
        # Check that TRB row went to trb column
        trb_rows = result[result["trb"] != ""]
        assert len(trb_rows) == 1
        assert trb_rows.iloc[0]["trb"] == "CASSLAPGATNEKLFF"

    def test_get_column_map_returns_none_for_gamma_delta(self):
        """_get_column_map_for_locus should return None for TRD/TRG."""
        from scripts.data_processing.standardize.adc import AdcStandardizer

        standardizer = AdcStandardizer.__new__(AdcStandardizer)
        assert standardizer._get_column_map_for_locus("TRA") is not None
        assert standardizer._get_column_map_for_locus("TRB") is not None
        assert standardizer._get_column_map_for_locus("TRD") is None
        assert standardizer._get_column_map_for_locus("TRG") is None


# -----------------------------------------------------------------------
# BATMAN score/binding audit changes
# -----------------------------------------------------------------------
class TestBatmanScoreBinding:
    def test_score_is_numeric_binding_is_categorical(self, tmp_path):
        """BATMAN should map peptide_activity to score and derive binding."""
        from scripts.data_processing.standardize.batman import BatmanStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "tcr": ["TCR1", "TCR2", "TCR3"],
                "va": ["", "", ""],
                "vb": ["", "", ""],
                "cdr3a": ["CAVRDSNYQLIW", "CAVKDSNYQLIW", "CAVMDSNYQLIW"],
                "cdr3b": ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF", "CASSLAPNEKLFF"],
                "trav": ["1-2", "12-1", "1-2"],
                "traj": ["33", "49", "33"],
                "trbv": ["6-5", "20-1", "6-5"],
                "trbd": ["", "", ""],
                "trbj": ["2-1", "1-2", "2-1"],
                "assay": ["SPR", "tetramer", "SPR"],
                "tcr_source_organism": ["human", "human", "human"],
                "index_peptide": ["GILGFVFTL", "NLVPMVATV", "GILGFVFTL"],
                "mhc": ["HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01"],
                "pmid": ["12345", "67890", "11111"],
                "peptide_type": ["viral", "viral", "viral"],
                "peptide": ["GILGFVFTL", "NLVPMVATV", "GILGFVFTL"],
                # Strong activation, weak activation, no activation
                "peptide_activity": ["0.8", "0.3", "0.05"],
            }
        )
        df.to_excel(
            source_dir / "TCR_pMHCI_mutational_scan_database.xlsx",
            index=False,
        )

        output_dir = tmp_path / "output"
        standardizer = BatmanStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        standardizer.run(force=True)

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        # score should contain the numeric peptide_activity values
        assert list(result["score"]) == ["0.8", "0.3", "0.05"]
        # binding should be derived: pos (>=0.1) or neg (<0.1)
        assert list(result["binding"]) == ["pos", "pos", "neg"]


# -----------------------------------------------------------------------
# McPAS identification method parsing
# -----------------------------------------------------------------------
class TestMcpasIdentificationMethod:
    def test_tetramer_gets_pos_binding(self, tmp_path):
        """McPAS records with tetramer method should get binding='pos'."""
        from scripts.data_processing.standardize.mcpas import McpasStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "CDR3.beta.aa": [
                    "CASSLAPGATNEKLFF",
                    "CASSLGQAYEQYF",
                    "CASSDRGYTFGSGANVLTF",
                ],
                "TRBV": ["TRBV6-5", "TRBV20-1", "TRBV6-5"],
                "TRBJ": ["TRBJ2-1", "TRBJ1-2", "TRBJ2-1"],
                "Epitope.peptide": ["GILGFVFTL", "NLVPMVATV", "KLVALGINAV"],
                "MHC": ["HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01"],
                "Species": ["Human", "Human", "Human"],
                "Antigen.identification.method": [
                    "tetramer",
                    "stimulation",
                    "",
                ],
            }
        )
        df.to_csv(source_dir / "McPAS-TCR.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = McpasStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        standardizer.run(force=True)

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert len(result) == 3
        # Tetramer and stimulation should get "pos"
        assert result.iloc[0]["binding"] == "pos"
        assert result.iloc[1]["binding"] == "pos"
        # Empty method should leave binding empty
        assert result.iloc[2]["binding"] == ""

    def test_no_method_column_still_works(self, tmp_path):
        """McPAS should work even without Antigen.identification.method column."""
        from scripts.data_processing.standardize.mcpas import McpasStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        df = pd.DataFrame(
            {
                "CDR3.beta.aa": ["CASSLAPGATNEKLFF"],
                "TRBV": ["TRBV6-5"],
                "TRBJ": ["TRBJ2-1"],
                "Epitope.peptide": ["GILGFVFTL"],
                "MHC": ["HLA-A*02:01"],
                "Species": ["Human"],
            }
        )
        df.to_csv(source_dir / "McPAS-TCR.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = McpasStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 1


# -----------------------------------------------------------------------
# IEDB pMHC standardizer
# -----------------------------------------------------------------------
class TestIedbPmhcStandardizer:
    def test_mhc_ligand_csv(self, tmp_path):
        """IEDB pMHC should load mhc_ligand CSV with multi-level headers."""
        from scripts.data_processing.standardize.iedb_pmhc import (
            IedbPmhcStandardizer,
        )

        source_dir = tmp_path / "source"
        ligand_dir = source_dir / "mhc_ligand"
        ligand_dir.mkdir(parents=True)
        # Create empty full_database dir so source_checksums works
        (source_dir / "full_database").mkdir(parents=True)

        # Multi-level header CSV like CEDAR/IEDB exports
        groups = ["Epitope", "", "MHC", "", "Host"]
        fields = ["Name", "IRI", "Allele Names", "Other", "Organism Source"]
        data = [
            ["GILGFVFTL", "", "HLA-A*02:01", "", "Homo sapiens"],
            ["NLVPMVATV", "", "HLA-A*02:01", "", "Homo sapiens"],
            ["SIINFEKL", "", "H-2Kb", "", "Mus musculus"],
        ]
        _write_multilevel_csv(
            ligand_dir / "mhc_ligand_full_v3.csv", groups, fields, data,
        )

        output_dir = tmp_path / "output"
        standardizer = IedbPmhcStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        # Mouse record should be filtered out by MHC normalization
        # (H-2Kb is non-human)
        assert summary["rows"] == 2
        _verify_output(output_dir)

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        # All records should have source=iedb_pmhc
        assert all(result["source"] == "iedb_pmhc")
        # All records should be positive (eluted ligands)
        assert all(result["binding"] == "pos")
        # No TCR data
        assert all(result["tra"] == "")
        assert all(result["trb"] == "")
        # Peptides should be present
        assert set(result["peptide"]) == {"GILGFVFTL", "NLVPMVATV"}

    def test_empty_sources_graceful(self, tmp_path):
        """IEDB pMHC should handle missing/empty data gracefully."""
        from scripts.data_processing.standardize.iedb_pmhc import (
            IedbPmhcStandardizer,
        )

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        output_dir = tmp_path / "output"
        standardizer = IedbPmhcStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        # Should complete with 0 rows (no data found)
        assert summary["rows"] == 0

    def test_mysql_dump_parsing(self, tmp_path):
        """IEDB pMHC should parse mhc_bind from MySQL dump."""
        from scripts.data_processing.standardize.iedb_pmhc import (
            _load_mhc_bind_from_mysql,
        )
        import gzip

        source_dir = tmp_path / "source"
        db_dir = source_dir / "full_database"
        db_dir.mkdir(parents=True)

        # Create a minimal MySQL dump with mhc_bind, curated_epitope, epitope
        lines = [
            # epitope table: col_0=epitope_id, col_1=description
            "INSERT INTO `epitope` VALUES ('100','GILGFVFTL','','','','','','','','');",
            "INSERT INTO `epitope` VALUES ('200','NLVPMVATV','','','','','','','','');",
            # curated_epitope: col_0=curated_epitope_id, ..., col_6=epitope_id
            "INSERT INTO `curated_epitope` VALUES ('1','','','','','','100','','');",
            "INSERT INTO `curated_epitope` VALUES ('2','','','','','','200','','');",
            # mhc_bind: id, ref_id, curated_epi_id, location, type_id,
            #           char_value, num_value, inequality, comments,
            #           restriction_id, allele_name, complex_id
            "INSERT INTO `mhc_bind` VALUES ('1','1','1','','','Positive','50','','','','HLA-A*02:01','');",
            "INSERT INTO `mhc_bind` VALUES ('2','1','2','','','Negative','5000','','','','HLA-A*02:01','');",
        ]

        sql_gz_path = db_dir / "iedb_public.sql.gz"
        with gzip.open(sql_gz_path, "wb") as f:
            for line in lines:
                f.write((line + "\n").encode("utf-8"))

        result = _load_mhc_bind_from_mysql(sql_gz_path)
        assert len(result) == 2

        pos_row = result[result["binding"] == "pos"]
        assert len(pos_row) == 1
        assert pos_row.iloc[0]["peptide"] == "GILGFVFTL"
        assert pos_row.iloc[0]["score"] == "50"

        neg_row = result[result["binding"] == "neg"]
        assert len(neg_row) == 1
        assert neg_row.iloc[0]["peptide"] == "NLVPMVATV"


# -----------------------------------------------------------------------
# CEDAR pMHC standardizer
# -----------------------------------------------------------------------
class TestCedarPmhcStandardizer:
    def test_mhc_ligand_csv(self, tmp_path):
        """CEDAR pMHC should load mhc_ligand CSV."""
        from scripts.data_processing.standardize.cedar_pmhc import (
            CedarPmhcStandardizer,
        )

        source_dir = tmp_path / "source"
        ligand_dir = source_dir / "mhc_ligand"
        ligand_dir.mkdir(parents=True)

        groups = ["Epitope", "", "MHC", "Host"]
        fields = ["Name", "IRI", "Allele Names", "Organism Source"]
        data = [
            ["GILGFVFTL", "", "HLA-A*02:01", "Homo sapiens"],
            ["NLVPMVATV", "", "HLA-B*07:02", "Homo sapiens"],
        ]
        _write_multilevel_csv(
            ligand_dir / "mhc_ligand_full_v3.csv", groups, fields, data,
        )

        output_dir = tmp_path / "output"
        standardizer = CedarPmhcStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 2
        _verify_output(output_dir)

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert all(result["source"] == "cedar_pmhc")
        assert all(result["binding"] == "pos")
        assert all(result["tra"] == "")
        assert all(result["trb"] == "")

    def test_empty_csv_graceful(self, tmp_path):
        """CEDAR pMHC should handle empty CSV gracefully."""
        from scripts.data_processing.standardize.cedar_pmhc import (
            CedarPmhcStandardizer,
        )

        source_dir = tmp_path / "source"
        ligand_dir = source_dir / "mhc_ligand"
        ligand_dir.mkdir(parents=True)
        # Create 0-byte file
        (ligand_dir / "mhc_ligand_full_v3.csv").touch()

        output_dir = tmp_path / "output"
        standardizer = CedarPmhcStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["rows"] == 0

    def test_missing_file_graceful(self, tmp_path):
        """CEDAR pMHC should handle missing file gracefully."""
        from scripts.data_processing.standardize.cedar_pmhc import (
            CedarPmhcStandardizer,
        )

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        output_dir = tmp_path / "output"
        standardizer = CedarPmhcStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["rows"] == 0

    def test_api_format_csv(self, tmp_path):
        """CEDAR pMHC should handle flat API CSV format (__ delimited headers)."""
        from scripts.data_processing.standardize.cedar_pmhc import (
            CedarPmhcStandardizer,
        )

        source_dir = tmp_path / "source"
        ligand_dir = source_dir / "mhc_ligand"
        ligand_dir.mkdir(parents=True)

        # API format: single-row flat headers with __ delimiters
        df = pd.DataFrame(
            {
                "epitope__name": ["GILGFVFTL", "NLVPMVATV", "SIINFEKL", "KLEDLERDL"],
                "epitope__object_type": [
                    "Linear peptide",
                    "Linear peptide",
                    "Linear peptide",
                    "Linear peptide",
                ],
                "mhc_restriction__name": [
                    "HLA-A*02:01",
                    "HLA-B*07:02",
                    "H-2Kb",
                    "HLA-A*02:01",
                ],
                "host__name": [
                    "Homo sapiens (human)",
                    "Homo sapiens (human)",
                    "Mus musculus (house mouse)",
                    "Homo sapiens (human)",
                ],
                "assay__qualitative_measurement": [
                    "Positive-High",
                    "Positive-Low",
                    "Positive",
                    "Negative",
                ],
            }
        )
        df.to_csv(ligand_dir / "mhc_ligand_full_v3.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = CedarPmhcStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        # Mouse record filtered out, 3 human records remain
        # (2 positive + 1 negative — all pass standardization)
        assert summary["rows"] >= 2
        _verify_output(output_dir)

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert all(result["source"] == "cedar_pmhc")
        # Check binding labels: positive -> pos, negative -> neg
        pos_rows = result[result["binding"] == "pos"]
        neg_rows = result[result["binding"] == "neg"]
        assert len(pos_rows) >= 2
        assert len(neg_rows) >= 1
        # No TCR data
        assert all(result["tra"] == "")
        assert all(result["trb"] == "")


# -----------------------------------------------------------------------
# Studies 10x chain detection
# -----------------------------------------------------------------------
class TestStudiesStandardizer:
    def test_10x_chain_column_splits_correctly(self, tmp_path):
        """10x format with chain column should map TRA to alpha, TRB to beta."""
        from scripts.data_processing.standardize.studies import StudiesStandardizer

        source_dir = tmp_path / "studies"
        study_dir = source_dir / "GSE_test"
        study_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "barcode": ["AAAA-1", "BBBB-1"],
                "chain": ["TRA", "TRB"],
                "cdr3": ["CAVRDSNYQLIW", "CASSLAPGATNEKLFF"],
                "v_gene": ["TRAV1-2", "TRBV6-5"],
                "d_gene": ["", ""],
                "j_gene": ["TRAJ33", "TRBJ2-1"],
                "productive": ["true", "true"],
            }
        )
        df.to_csv(study_dir / "data.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = StudiesStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 2

        # Read all parquet files (data split by chain into separate files)
        all_parts = sorted(output_dir.glob("*.parquet"))
        result = pd.concat(
            [pq.read_table(p).to_pandas() for p in all_parts], ignore_index=True,
        )
        # TRA row should have CDR3 in 'tra' column
        tra_rows = result[result["tra"] != ""]
        assert len(tra_rows) == 1
        assert tra_rows.iloc[0]["tra"] == "CAVRDSNYQLIW"
        assert tra_rows.iloc[0]["trb"] == ""

        # TRB row should have CDR3 in 'trb' column
        trb_rows = result[result["trb"] != ""]
        assert len(trb_rows) == 1
        assert trb_rows.iloc[0]["trb"] == "CASSLAPGATNEKLFF"
        assert trb_rows.iloc[0]["tra"] == ""

    def test_productive_filter_excludes_empty_string(self, tmp_path):
        """Empty string should NOT be treated as productive."""
        from scripts.data_processing.standardize.studies import StudiesStandardizer

        source_dir = tmp_path / "studies"
        study_dir = source_dir / "GSE_prod_test"
        study_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "amino_acid": ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF", "CASSDRGYTF"],
                "rearrangement": ["ACG", "TGA", "CCC"],
                "v_gene": ["TRBV6-5", "TRBV20-1", "TRBV6-5"],
                "j_gene": ["TRBJ2-1", "TRBJ1-2", "TRBJ2-1"],
                "productive": ["true", "", "false"],
            }
        )
        df.to_csv(study_dir / "data.tsv", sep="\t", index=False)

        output_dir = tmp_path / "output"
        standardizer = StudiesStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        # Only the first row (productive=true) should survive
        assert summary["rows"] == 1


# -----------------------------------------------------------------------
# IEDB without tcell file
# -----------------------------------------------------------------------
class TestIedbWithoutTcell:
    def test_peptide_mhc_from_receptor_only(self, tmp_path):
        """IEDB should populate peptide/MHC from receptor columns even without tcell."""
        from scripts.data_processing.standardize.iedb import IedbStandardizer

        source_dir = tmp_path / "source"
        receptor_dir = source_dir / "receptor"
        receptor_dir.mkdir(parents=True)
        # No tcell directory

        r_groups = [
            "Receptor", "Chain 2", "", "", "Epitope", "Assay",
        ]
        r_fields = [
            "Type", "CDR3 Curated", "V Gene Curated", "J Gene Curated",
            "Name", "MHC Allele Names",
        ]
        r_data = [
            [
                "alphabeta", "CASSLAPGATNEKLFF", "TRBV6-5", "TRBJ2-1",
                "GILGFVFTL", "HLA-A*02:01",
            ],
        ]
        _write_multilevel_csv(
            receptor_dir / "tcr_full_v3.csv", r_groups, r_fields, r_data,
        )

        output_dir = tmp_path / "output"
        standardizer = IedbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 1

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert result.iloc[0]["peptide"] == "GILGFVFTL"
        assert result.iloc[0]["mhc_one"] == "HLA-A*02:01"


# -----------------------------------------------------------------------
# TCRdb Chain column
# -----------------------------------------------------------------------
class TestTcrdbStandardizer:
    def test_chain_column_used(self, tmp_path):
        """When Chain column exists, it should determine chain type."""
        from scripts.data_processing.standardize.tcrdb import TcrdbStandardizer

        source_dir = tmp_path / "source"
        cat_dir = source_dir / "viral"
        cat_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "AASeq": ["CAVRDSNYQLIW", "CASSLAPGATNEKLFF"],
                "Vregion": ["", "TRBV6-5"],
                "Dregion": ["", ""],
                "Jregion": ["TRAJ33", "TRBJ2-1"],
                "Chain": ["alpha", "beta"],
            }
        )
        df.to_csv(cat_dir / "test_data.csv", index=False)

        output_dir = tmp_path / "output"
        standardizer = TcrdbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 2

        # Read all parquet files (data split by chain into separate files)
        all_parts = sorted(output_dir.glob("*.parquet"))
        result = pd.concat(
            [pq.read_table(p).to_pandas() for p in all_parts], ignore_index=True,
        )
        tra_rows = result[result["tra"] != ""]
        assert len(tra_rows) == 1
        assert tra_rows.iloc[0]["tra"] == "CAVRDSNYQLIW"

        trb_rows = result[result["trb"] != ""]
        assert len(trb_rows) == 1
        assert trb_rows.iloc[0]["trb"] == "CASSLAPGATNEKLFF"


# -----------------------------------------------------------------------
# IEDB pMHC mhc_elution parsing
# -----------------------------------------------------------------------
class TestIedbPmhcElution:
    def test_mhc_elution_parsed(self, tmp_path):
        """mhc_elution records should be parsed from SQL dump with correct allele index."""
        from scripts.data_processing.standardize.iedb_pmhc import IedbPmhcStandardizer

        source_dir = tmp_path / "source"
        full_db_dir = source_dir / "full_database"
        full_db_dir.mkdir(parents=True)

        import gzip

        # Build a minimal SQL dump with epitope, curated_epitope, mhc_bind,
        # and mhc_elution INSERT statements.
        lines = []
        # epitope table: id=1 -> GILGFVFTL, id=2 -> NLVPMVATV
        lines.append(
            "INSERT INTO `epitope` VALUES "
            "(1,'GILGFVFTL','','','','','','',''),"
            "(2,'NLVPMVATV','','','','','','','');"
        )
        # curated_epitope: curated_id=10 -> epitope_id=1, curated_id=20 -> epitope_id=2
        lines.append(
            "INSERT INTO `curated_epitope` VALUES "
            "(10,'','','','','','1'),"
            "(20,'','','','','','2');"
        )
        # mhc_bind: 12 columns, allele at index 10
        lines.append(
            "INSERT INTO `mhc_bind` VALUES "
            "(1,1,10,'','','Positive','500','','',1,'HLA-A*02:01',1);"
        )
        # mhc_elution: 37 columns, allele at index 36
        elution_fields = ["''"] * 37
        elution_fields[0] = "1"
        elution_fields[1] = "1"
        elution_fields[2] = "20"
        elution_fields[5] = "'Positive'"
        elution_fields[36] = "'HLA-B*07:02'"
        lines.append(
            "INSERT INTO `mhc_elution` VALUES "
            f"({','.join(elution_fields)});"
        )

        sql_content = "\n".join(lines)
        with gzip.open(full_db_dir / "iedb_public.sql.gz", "wb") as f:
            f.write(sql_content.encode("utf-8"))

        # Also create empty mhc_ligand dir
        (source_dir / "mhc_ligand").mkdir(parents=True)

        output_dir = tmp_path / "output"
        standardizer = IedbPmhcStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        assert summary["rows"] == 2  # 1 bind + 1 elution

        result = pd.concat(
            [pq.read_table(p).to_pandas() for p in sorted(output_dir.glob("*.parquet"))],
            ignore_index=True,
        )
        # Check the bind record
        bind_rows = result[result["mhc_one"].str.contains("A", na=False)]
        assert len(bind_rows) >= 1
        assert bind_rows.iloc[0]["peptide"] == "GILGFVFTL"
        assert bind_rows.iloc[0]["binding"] == "pos"

        # Check the elution record
        elution_rows = result[result["mhc_one"].str.contains("B", na=False)]
        assert len(elution_rows) >= 1
        assert elution_rows.iloc[0]["peptide"] == "NLVPMVATV"
        assert elution_rows.iloc[0]["binding"] == "pos"

    def test_extract_records_helper(self):
        """Test _extract_records with known column indices."""
        from scripts.data_processing.standardize.iedb_pmhc import _extract_records

        curated_to_name = {"10": "GILGFVFTL"}

        # mhc_bind: allele at index 10
        bind_row = ("1", "1", "10", "", "", "Positive", "500", "", "", "1", "HLA-A*02:01", "1")
        records = _extract_records([bind_row], curated_to_name, allele_idx=10)
        assert len(records) == 1
        assert records[0]["binding"] == "pos"
        assert records[0]["score"] == "500"
        assert records[0]["mhc_allele"] == "HLA-A*02:01"

        # mhc_elution: allele at index 36, is_elution=True
        elution_row = list([""] * 37)
        elution_row[2] = "10"
        elution_row[36] = "HLA-B*07:02"
        elution_row = tuple(elution_row)
        records = _extract_records([elution_row], curated_to_name, allele_idx=36, is_elution=True)
        assert len(records) == 1
        assert records[0]["binding"] == "pos"
        assert records[0]["score"] == ""
        assert records[0]["mhc_allele"] == "HLA-B*07:02"


# -----------------------------------------------------------------------
# IMGTHLA Standardizer
# -----------------------------------------------------------------------
class TestImgthlaStandardizer:
    def test_end_to_end(self, tmp_path):
        """Test IMGTHLA standardizer with mock FASTA data."""
        from scripts.data_processing.standardize.imgthla import ImgthlaStandardizer

        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True)

        fasta_content = (
            ">HLA:HLA00001 A*01:01:01:01 365 bp\n"
            "MAVMAPRTLLLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGY\n"
            "VDDTQFVRFDSDAASQKMEPRAPWIEQEGPEYWDQETRNMKAHSQTDRAN\n"
            ">HLA:HLA00002 A*01:01:01:02N 200 bp\n"
            "MAVMAPRTLLLLLSGALALTQTWAGSHSMRYFFTSVSRPGR\n"
            ">HLA:HLA00100 B*07:02:01 365 bp\n"
            "MRVTAPRTVLLLLWGAVALTETWAGSHSMRYFYTSVSRPGRGEPRFITVGY\n"
            ">HLA:HLA10000 DRB1*04:01:01 365 bp\n"
            "MVCLKLPGGSCMTALTVTLMVLSSPLALAGDTRPRFLWQLKFECHFFNGTERV\n"
            ">HLA:HLA20000 DQA1*01:01:01 365 bp\n"
            "MILNKALMLGALALTTVMSPCGGEDIVADHVASCGVNLYQFYGPSGQYTHE\n"
        )
        (source_dir / "hla_prot.fasta").write_text(fasta_content)

        output_dir = tmp_path / "output"
        standardizer = ImgthlaStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        summary = standardizer.run(force=True)
        assert summary["status"] == "completed"
        # A*01:01 deduplicated, so: A*01:01, B*07:02, DRB1*04:01, DQA1*01:01 = 4
        assert summary["rows"] == 4

        result = pd.concat(
            [pq.read_table(p).to_pandas() for p in sorted(output_dir.glob("*.parquet"))],
            ignore_index=True,
        )
        assert list(result.columns) == TARGET_COLUMNS
        assert all(result["source"] == "imgthla")
        assert all(result["peptide"] == "")

        # Class I: mhc_one has sequence
        a_rows = result[result["study_id"] == "HLA-A*01:01"]
        assert len(a_rows) == 1
        assert a_rows.iloc[0]["mhc_one"] != ""
        assert a_rows.iloc[0]["mhc_two"] == ""

        # Class II beta: mhc_two has sequence
        drb_rows = result[result["study_id"] == "HLA-DRB1*04:01"]
        assert len(drb_rows) == 1
        assert drb_rows.iloc[0]["mhc_one"] == ""
        assert drb_rows.iloc[0]["mhc_two"] != ""

        # Class II alpha: mhc_one has sequence
        dqa_rows = result[result["study_id"] == "HLA-DQA1*01:01"]
        assert len(dqa_rows) == 1
        assert dqa_rows.iloc[0]["mhc_one"] != ""
        assert dqa_rows.iloc[0]["mhc_two"] == ""

    def test_allele_deduplication(self, tmp_path):
        """Multiple full-length alleles for same 4-digit should be deduplicated."""
        from scripts.data_processing.standardize.imgthla import _parse_hla_fasta

        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True)

        fasta_content = (
            ">HLA:HLA00001 A*01:01:01:01 365 bp\n"
            "SHORTSEQ\n"
            ">HLA:HLA00002 A*01:01:01:02 365 bp\n"
            "LONGERLONGERSEQ\n"
        )
        (source_dir / "hla_prot.fasta").write_text(fasta_content)

        records = _parse_hla_fasta(source_dir / "hla_prot.fasta")
        assert len(records) == 1
        assert records[0]["allele_name"] == "HLA-A*01:01"
        assert records[0]["sequence"] == "LONGERLONGERSEQ"
