"""Integration tests for per-database standardizers.

Each test creates minimal mock data matching the real schema, runs the
standardizer, and verifies the output conforms to the 15-column schema.
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
    def test_batman_binding_populated(self, tmp_path):
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
        assert list(result["binding"]) == ["37", "126"]
        assert all(result["score"] == "")

    def test_vdjdb_score_populated(self, tmp_path):
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
                "vdjdb.score": ["3", "1"],
                "meta.study.id": ["study1", "study2"],
            }
        )
        df.to_csv(source_dir / "vdjdb_full.txt", sep="\t", index=False)

        output_dir = tmp_path / "output"
        standardizer = VdjdbStandardizer(
            source_dir=source_dir, output_dir=output_dir,
        )
        standardizer.run(force=True)

        result = pq.read_table(output_dir / "part_0000.parquet").to_pandas()
        assert "score" in result.columns
        assert "binding" in result.columns
        assert list(result["score"]) == ["3", "1"]
        assert all(result["binding"] == "")

    def test_trait_binding_from_filename(self, tmp_path):
        """TRAIT extracts pos/neg binder status from filename."""
        from scripts.data_processing.standardize.trait import _parse_trait_filename

        meta_pos = _parse_trait_filename("A0201_GILGFVFTL_BMLF1_EBV_binder_pos.zip")
        assert meta_pos["binding"] == "pos"

        meta_neg = _parse_trait_filename("A0201_GILGFVFTL_BMLF1_EBV_binder_neg.zip")
        assert meta_neg["binding"] == "neg"

    def test_schema_has_15_columns(self):
        assert len(TARGET_COLUMNS) == 15
        assert "binding" in TARGET_COLUMNS
        assert "score" in TARGET_COLUMNS
        # binding and score should come after mhc_two, before source
        mhc_two_idx = TARGET_COLUMNS.index("mhc_two")
        source_idx = TARGET_COLUMNS.index("source")
        binding_idx = TARGET_COLUMNS.index("binding")
        score_idx = TARGET_COLUMNS.index("score")
        assert mhc_two_idx < binding_idx < score_idx < source_idx


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
