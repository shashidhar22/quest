"""Unit tests for quest.data.standardization normalization functions."""

import pandas as pd
import pytest

from quest.data.standardization import (
    TARGET_COLUMNS,
    _CDR_LOOKUP_CACHE,
    _NORM_CACHE,
    classify_mhc_class,
    clear_norm_cache,
    enrich_cdr_columns,
    load_hla_sequences,
    normalize_cdr3,
    normalize_gene,
    normalize_mhc_allele,
    normalize_peptide,
    resolve_allele_to_sequence,
    split_mhc_to_alpha_beta,
    standardize_dataframe,
)


# -----------------------------------------------------------------------
# normalize_cdr3
# -----------------------------------------------------------------------
class TestNormalizeCdr3:
    def test_valid_sequence(self):
        assert normalize_cdr3("CASSLAPGATNEKLFF") == "CASSLAPGATNEKLFF"

    def test_lowercase(self):
        assert normalize_cdr3("casslapgatneklff") == "CASSLAPGATNEKLFF"

    def test_whitespace(self):
        assert normalize_cdr3("  CASS  ") == "CASS"  # stripped to 4 valid chars

    def test_short_sequence(self):
        assert normalize_cdr3("CA") == ""
        assert normalize_cdr3("CAS") == ""

    def test_exactly_4_chars(self):
        assert normalize_cdr3("CASS") == "CASS"

    def test_non_aa_chars(self):
        assert normalize_cdr3("CASS#@!") == ""
        assert normalize_cdr3("CASS123") == ""

    def test_empty(self):
        assert normalize_cdr3("") == ""
        assert normalize_cdr3(None) == ""

    def test_na_strings(self):
        assert normalize_cdr3("NA") == ""
        assert normalize_cdr3("nan") == ""
        assert normalize_cdr3("None") == ""
        assert normalize_cdr3("NULL") == ""

    def test_float_nan(self):
        import math

        assert normalize_cdr3(float("nan")) == ""

    def test_valid_long(self):
        seq = "CASSLGQAYEQYF"
        assert normalize_cdr3(seq) == seq


# -----------------------------------------------------------------------
# normalize_gene
# -----------------------------------------------------------------------
class TestNormalizeGene:
    def test_standard_imgt(self):
        result = normalize_gene("TRBV20-1*01")
        assert result != ""
        assert "TRBV20" in result

    def test_tcr_prefix(self):
        result = normalize_gene("TCRBV13-01")
        assert result != ""
        assert "TRBV" in result

    def test_leading_zeros(self):
        result = normalize_gene("TRBV06-05")
        assert result != ""

    def test_multiple_alleles(self):
        result = normalize_gene("TRAV1-2,TRAV1-3")
        assert result != ""
        assert "TRAV1" in result

    def test_semicolon_separated(self):
        result = normalize_gene("TRBV20-1;TRBV20-2")
        assert result != ""

    def test_empty(self):
        assert normalize_gene("") == ""
        assert normalize_gene(None) == ""
        assert normalize_gene("NA") == ""

    def test_invalid(self):
        assert normalize_gene("unknown_gene") == ""
        assert normalize_gene("12345") == ""

    def test_alpha_gene(self):
        result = normalize_gene("TRAV1-2")
        assert result != ""
        assert "TRAV" in result

    def test_d_gene(self):
        result = normalize_gene("TRBD1")
        assert result != ""

    def test_j_gene(self):
        result = normalize_gene("TRBJ2-1")
        assert result != ""

    def test_ig_genes_rejected(self):
        """IG (BCR) genes must be rejected — only TCR genes allowed."""
        assert normalize_gene("IGHV1-2") == ""
        assert normalize_gene("IGKV1-5") == ""
        assert normalize_gene("IGLV1-40") == ""
        assert normalize_gene("IGHJ4") == ""
        assert normalize_gene("IGKJ1") == ""
        assert normalize_gene("IGHD3-10") == ""

    def test_tcr_genes_accepted(self):
        """TCR genes from all chains should be accepted."""
        assert normalize_gene("TRAV1-2") != ""
        assert normalize_gene("TRBV6-5") != ""
        assert normalize_gene("TRBJ2-1") != ""
        assert normalize_gene("TRBD1") != ""
        assert normalize_gene("TRAJ33") != ""
        assert normalize_gene("TRDV1") != ""
        assert normalize_gene("TRGV9") != ""


# -----------------------------------------------------------------------
# normalize_mhc_allele
# -----------------------------------------------------------------------
class TestNormalizeMhcAllele:
    def test_standard_hla(self):
        assert normalize_mhc_allele("HLA-A*02:01") == "HLA-A*02:01"

    def test_compact_format(self):
        result = normalize_mhc_allele("A0201")
        assert result != ""
        assert "02" in result and "01" in result

    def test_missing_prefix(self):
        result = normalize_mhc_allele("A*02:01")
        assert result != ""
        assert "HLA" in result or "A*02:01" in result

    def test_class_ii(self):
        result = normalize_mhc_allele("HLA-DRB1*04:01")
        assert result != ""
        assert "DRB1" in result

    def test_empty(self):
        assert normalize_mhc_allele("") == ""
        assert normalize_mhc_allele(None) == ""
        assert normalize_mhc_allele("NA") == ""

    def test_non_allele_text(self):
        assert normalize_mhc_allele("class I") == ""
        assert normalize_mhc_allele("See reference for details") == ""

    def test_mouse_mhc(self):
        result = normalize_mhc_allele("H-2Kb")
        # Should normalize or return empty
        assert isinstance(result, str)


# -----------------------------------------------------------------------
# classify_mhc_class
# -----------------------------------------------------------------------
class TestClassifyMhcClass:
    def test_class_i_hla_a(self):
        assert classify_mhc_class("HLA-A*02:01") == "I"

    def test_class_i_hla_b(self):
        assert classify_mhc_class("HLA-B*07:02") == "I"

    def test_class_i_hla_c(self):
        assert classify_mhc_class("HLA-C*07:01") == "I"

    def test_class_ii_drb(self):
        assert classify_mhc_class("HLA-DRB1*04:01") == "II"

    def test_class_ii_dqb(self):
        assert classify_mhc_class("HLA-DQB1*06:02") == "II"

    def test_class_ii_dpb(self):
        assert classify_mhc_class("HLA-DPB1*04:01") == "II"

    def test_empty(self):
        assert classify_mhc_class("") == ""

    def test_mouse_class_i(self):
        assert classify_mhc_class("H-2Kb") == "I"

    def test_mouse_class_ii(self):
        assert classify_mhc_class("H-2IAb") == "II"


# -----------------------------------------------------------------------
# split_mhc_to_alpha_beta
# -----------------------------------------------------------------------
class TestSplitMhc:
    def test_class_i(self):
        m1, m2 = split_mhc_to_alpha_beta("HLA-A*02:01")
        assert m1 != ""
        assert m2 == ""

    def test_class_ii_beta(self):
        m1, m2 = split_mhc_to_alpha_beta("HLA-DRB1*04:01")
        assert m2 != ""  # Beta chain goes to mhc_two

    def test_slash_separated(self):
        m1, m2 = split_mhc_to_alpha_beta("HLA-DQA1*01:02/HLA-DQB1*06:02")
        assert m1 != "" or m2 != ""

    def test_empty(self):
        m1, m2 = split_mhc_to_alpha_beta("")
        assert m1 == ""
        assert m2 == ""

    def test_none(self):
        m1, m2 = split_mhc_to_alpha_beta(None)
        assert m1 == ""
        assert m2 == ""


# -----------------------------------------------------------------------
# normalize_peptide
# -----------------------------------------------------------------------
class TestNormalizePeptide:
    def test_valid_peptide(self):
        assert normalize_peptide("GILGFVFTL") == "GILGFVFTL"

    def test_lowercase(self):
        assert normalize_peptide("gilgfvftl") == "GILGFVFTL"

    def test_whitespace(self):
        assert normalize_peptide("  GILGFVFTL  ") == "GILGFVFTL"

    def test_empty(self):
        assert normalize_peptide("") == ""
        assert normalize_peptide(None) == ""

    def test_na_strings(self):
        assert normalize_peptide("NA") == ""
        assert normalize_peptide("nan") == ""

    def test_non_sequence_text(self):
        assert normalize_peptide("See reference for details") == ""

    def test_long_text(self):
        assert normalize_peptide("A" * 101) == ""

    def test_single_aa(self):
        assert normalize_peptide("A") == "A"


# -----------------------------------------------------------------------
# standardize_dataframe
# -----------------------------------------------------------------------
class TestStandardizeDataframe:
    def test_basic_mapping(self):
        df = pd.DataFrame(
            {
                "cdr3a": ["CASSTLGQAYEQYF"],
                "cdr3b": ["CASSLAPGATNEKLFF"],
                "epitope": ["GILGFVFTL"],
                "mhc": ["HLA-A*02:01"],
            }
        )
        column_map = {
            "cdr3a": "tra",
            "cdr3b": "trb",
            "epitope": "peptide",
            "mhc": "mhc_one",
        }
        result, dropped = standardize_dataframe(df, column_map, source="test")
        assert list(result.columns) == TARGET_COLUMNS
        assert len(result) == 1
        assert result.iloc[0]["source"] == "test"

    def test_missing_columns_filled(self):
        df = pd.DataFrame({"cdr3b": ["CASSLAPGATNEKLFF"]})
        column_map = {"cdr3b": "trb"}
        result, dropped = standardize_dataframe(df, column_map, source="test")
        assert list(result.columns) == TARGET_COLUMNS
        assert result.iloc[0]["tra"] == ""
        assert result.iloc[0]["trav_gene"] == ""

    def test_no_valid_field_dropped(self):
        df = pd.DataFrame({"other": ["some_value"]})
        column_map = {}
        result, dropped = standardize_dataframe(df, column_map, source="test")
        assert len(result) == 0
        assert len(dropped) == 1
        assert dropped.iloc[0]["reason"] == "no_valid_field"

    def test_invalid_cdr3_logged(self):
        df = pd.DataFrame(
            {
                "cdr3b": ["CASS#@!"],
                "epitope": ["GILGFVFTL"],
            }
        )
        column_map = {"cdr3b": "trb", "epitope": "peptide"}
        result, dropped = standardize_dataframe(df, column_map, source="test")
        # Row kept because peptide is valid
        assert len(result) == 1
        assert result.iloc[0]["trb"] == ""
        # But CDR3 issue is logged
        cdr3_drops = dropped[dropped["field"] == "trb"]
        assert len(cdr3_drops) >= 1

    def test_study_id(self):
        df = pd.DataFrame({"cdr3b": ["CASSLAPGATNEKLFF"]})
        column_map = {"cdr3b": "trb"}
        result, dropped = standardize_dataframe(
            df, column_map, source="test", study_id="study1"
        )
        assert result.iloc[0]["study_id"] == "study1"

    def test_all_types_string(self):
        df = pd.DataFrame(
            {
                "cdr3b": ["CASSLAPGATNEKLFF"],
                "peptide": ["GILGFVFTL"],
            }
        )
        column_map = {"cdr3b": "trb", "peptide": "peptide"}
        result, dropped = standardize_dataframe(df, column_map, source="test")
        for col in TARGET_COLUMNS:
            # pandas 3.x uses StringDtype by default; accept any string-like dtype
            assert pd.api.types.is_string_dtype(result[col])

    def test_unique_then_map_high_duplication(self):
        """Verify unique-then-map produces identical results with high duplication."""
        # 1000 rows with only 3 unique genes — exercises the lookup dict path
        genes = ["TRBV6-5", "TRBV20-1", "TRBJ2-1"] * 333 + ["TRBV6-5"]
        cdr3s = ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF", "CASSDRGQAYEQYF"] * 333 + [
            "CASSLAPGATNEKLFF"
        ]
        df = pd.DataFrame(
            {
                "cdr3b": cdr3s,
                "vb": genes,
                "peptide": ["GILGFVFTL"] * 1000,
            }
        )
        column_map = {"cdr3b": "trb", "vb": "trbv_gene", "peptide": "peptide"}
        result, dropped = standardize_dataframe(df, column_map, source="test")
        assert len(result) == 1000
        # All 3 unique genes should normalize to valid TRBV/TRBJ names
        unique_genes = result["trbv_gene"].unique()
        assert len(unique_genes) == 3
        for g in unique_genes:
            assert g.startswith("TR"), f"Gene not normalized: {g}"

    def test_global_cache_reuse_across_calls(self):
        """Global normalization cache should be populated on first call and
        reused on subsequent calls, producing identical results."""
        clear_norm_cache()

        df = pd.DataFrame(
            {
                "cdr3b": ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF"],
                "vb": ["TRBV6-5", "TRBV20-1"],
                "peptide": ["GILGFVFTL", "NLVPMVATV"],
                "mhc": ["HLA-A*02:01", "HLA-B*07:02"],
            }
        )
        column_map = {
            "cdr3b": "trb",
            "vb": "trbv_gene",
            "peptide": "peptide",
            "mhc": "mhc_one",
        }

        # First call — populates the cache
        result1, _ = standardize_dataframe(df, column_map, source="test")
        assert len(_NORM_CACHE) > 0, "Cache should be populated after first call"
        cache_snapshot = {k: dict(v) for k, v in _NORM_CACHE.items()}

        # Second call — should reuse cached values and produce identical output
        result2, _ = standardize_dataframe(df, column_map, source="test")
        pd.testing.assert_frame_equal(result1, result2)

        # Cache should not have grown (same unique values)
        for key in cache_snapshot:
            assert cache_snapshot[key] == dict(_NORM_CACHE[key])

        clear_norm_cache()
        assert len(_NORM_CACHE) == 0, "Cache should be empty after clear"


# -----------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------
class TestSchema:
    def test_schema_has_25_columns(self):
        assert len(TARGET_COLUMNS) == 25

    def test_cdr_columns_present(self):
        for col in (
            "tra_cdr1", "tra_cdr2", "tra_cdr3", "tra_full",
            "trb_cdr1", "trb_cdr2", "trb_cdr3", "trb_full",
        ):
            assert col in TARGET_COLUMNS

    def test_mhc_allele_columns_present(self):
        assert "mhc_one_allele" in TARGET_COLUMNS
        assert "mhc_two_allele" in TARGET_COLUMNS

    def test_column_ordering(self):
        # Alpha chain columns grouped together
        tra_idx = TARGET_COLUMNS.index("tra")
        tra_full_idx = TARGET_COLUMNS.index("tra_full")
        trb_idx = TARGET_COLUMNS.index("trb")
        assert tra_idx < tra_full_idx < trb_idx

        # Beta chain columns grouped together
        trb_full_idx = TARGET_COLUMNS.index("trb_full")
        peptide_idx = TARGET_COLUMNS.index("peptide")
        assert trb_idx < trb_full_idx < peptide_idx


# -----------------------------------------------------------------------
# enrich_cdr_columns
# -----------------------------------------------------------------------
class TestEnrichCdrColumns:
    def test_cdr3_copied_from_tra_trb(self):
        clear_norm_cache()
        df = pd.DataFrame({col: "" for col in TARGET_COLUMNS}, index=[0, 1])
        df["tra"] = ["CAVRDSNYQLIW", ""]
        df["trb"] = ["CASSLAPGATNEKLFF", "CASSLGQAYEQYF"]
        df = enrich_cdr_columns(df)
        assert df.iloc[0]["tra_cdr3"] == "CAVRDSNYQLIW"
        assert df.iloc[0]["trb_cdr3"] == "CASSLAPGATNEKLFF"
        assert df.iloc[1]["tra_cdr3"] == ""
        assert df.iloc[1]["trb_cdr3"] == "CASSLGQAYEQYF"

    def test_cdr1_cdr2_from_vgene(self):
        clear_norm_cache()
        try:
            import tidytcells.tr as tr
        except ImportError:
            pytest.skip("tidytcells not available")

        df = pd.DataFrame({col: "" for col in TARGET_COLUMNS}, index=[0])
        df["tra"] = "CAVRDSNYQLIW"
        df["trav_gene"] = "TRAV1-2*01"
        df["trb"] = "CASSLAPGATNEKLFF"
        df["trbv_gene"] = "TRBV6-5*01"
        df = enrich_cdr_columns(df)

        # CDR1/CDR2 should be populated
        assert df.iloc[0]["tra_cdr1"] != ""
        assert df.iloc[0]["tra_cdr2"] != ""
        assert df.iloc[0]["trb_cdr1"] != ""
        assert df.iloc[0]["trb_cdr2"] != ""
        # All CDR values should be valid AA
        for col in ("tra_cdr1", "tra_cdr2", "trb_cdr1", "trb_cdr2"):
            val = df.iloc[0][col]
            assert all(c in "ACDEFGHIKLMNPQRSTVWYX" for c in val), f"Invalid AA in {col}: {val}"

    def test_vgene_without_allele_tries_star01(self):
        clear_norm_cache()
        try:
            import tidytcells.tr as tr
        except ImportError:
            pytest.skip("tidytcells not available")

        df = pd.DataFrame({col: "" for col in TARGET_COLUMNS}, index=[0])
        df["trb"] = "CASSLAPGATNEKLFF"
        df["trbv_gene"] = "TRBV6-5"
        df = enrich_cdr_columns(df)
        # Should still populate via *01 fallback
        assert df.iloc[0]["trb_cdr1"] != ""
        assert df.iloc[0]["trb_cdr2"] != ""

    def test_empty_vgene_gives_empty_cdr12(self):
        clear_norm_cache()
        df = pd.DataFrame({col: "" for col in TARGET_COLUMNS}, index=[0])
        df["tra"] = "CAVRDSNYQLIW"
        df["trav_gene"] = ""
        df = enrich_cdr_columns(df)
        assert df.iloc[0]["tra_cdr1"] == ""
        assert df.iloc[0]["tra_cdr2"] == ""
        # CDR3 still populated
        assert df.iloc[0]["tra_cdr3"] == "CAVRDSNYQLIW"

    def test_full_columns_empty_without_stitch(self):
        clear_norm_cache()
        df = pd.DataFrame({col: "" for col in TARGET_COLUMNS}, index=[0])
        df["tra"] = "CAVRDSNYQLIW"
        df["trav_gene"] = "TRAV1-2*01"
        df["traj_gene"] = "TRAJ33*01"
        df = enrich_cdr_columns(df, stitch=False)
        assert df.iloc[0]["tra_full"] == ""
        assert df.iloc[0]["trb_full"] == ""

    def test_empty_dataframe(self):
        clear_norm_cache()
        df = pd.DataFrame(columns=TARGET_COLUMNS)
        result = enrich_cdr_columns(df)
        assert list(result.columns) == TARGET_COLUMNS

    def test_standardize_dataframe_includes_cdr_columns(self):
        clear_norm_cache()
        df = pd.DataFrame({
            "cdr3b": ["CASSLAPGATNEKLFF"],
            "vb": ["TRBV6-5*01"],
            "peptide": ["GILGFVFTL"],
        })
        column_map = {"cdr3b": "trb", "vb": "trbv_gene", "peptide": "peptide"}
        result, dropped = standardize_dataframe(df, column_map, source="test")
        assert list(result.columns) == TARGET_COLUMNS
        assert len(result) == 1
        assert result.iloc[0]["trb_cdr3"] == "CASSLAPGATNEKLFF"

    def test_clear_norm_cache_clears_cdr_cache(self):
        _CDR_LOOKUP_CACHE["test_gene"] = {"cdr1": "AAA", "cdr2": "BBB"}
        clear_norm_cache()
        assert len(_CDR_LOOKUP_CACHE) == 0


# -----------------------------------------------------------------------
# normalize_mhc_allele — NetMHCpan format fix
# -----------------------------------------------------------------------
class TestNetMHCpanAlleleFix:
    def test_no_asterisk_class_i(self):
        """HLA-A02:01 (NetMHCpan format) should normalize to HLA-A*02:01."""
        assert normalize_mhc_allele("HLA-A02:01") == "HLA-A*02:01"

    def test_no_asterisk_no_prefix(self):
        """A02:01 (no HLA- prefix, no asterisk) should normalize."""
        assert normalize_mhc_allele("A02:01") == "HLA-A*02:01"

    def test_no_asterisk_class_i_b(self):
        assert normalize_mhc_allele("HLA-B07:02") == "HLA-B*07:02"

    def test_no_asterisk_class_ii(self):
        assert normalize_mhc_allele("HLA-DRB104:01") == "HLA-DRB1*04:01"

    def test_standard_format_unchanged(self):
        """Standard HLA-A*02:01 should still work."""
        assert normalize_mhc_allele("HLA-A*02:01") == "HLA-A*02:01"
        assert normalize_mhc_allele("HLA-DRB1*04:01") == "HLA-DRB1*04:01"


# -----------------------------------------------------------------------
# HLA sequence resolution
# -----------------------------------------------------------------------
class TestHlaResolution:
    def setup_method(self):
        clear_norm_cache()

    def teardown_method(self):
        clear_norm_cache()

    def test_resolve_allele_without_loading(self):
        """resolve_allele_to_sequence returns '' if HLA dict not loaded."""
        assert resolve_allele_to_sequence("HLA-A*02:01") == ""

    def test_resolve_allele_with_mock_dict(self):
        """resolve_allele_to_sequence with manually set dict."""
        import quest.data.standardization as mod

        mod._HLA_SEQ_DICT = {"A*02:01": "MAVMAPRTLLL", "B*07:02": "MRVTAPRTVLL"}
        mod._HLA_PREFIX_CACHE = {"A*02:01": "A*02:01", "A*02": "A*02:01",
                                  "B*07:02": "B*07:02", "B*07": "B*07:02"}
        try:
            assert resolve_allele_to_sequence("HLA-A*02:01") == "MAVMAPRTLLL"
            assert resolve_allele_to_sequence("HLA-B*07:02") == "MRVTAPRTVLL"
            assert resolve_allele_to_sequence("HLA-C*01:01") == ""
            assert resolve_allele_to_sequence("") == ""
        finally:
            mod._HLA_SEQ_DICT = None
            mod._HLA_PREFIX_CACHE = None

    def test_resolve_strips_hla_prefix(self):
        """Lookup key strips HLA- prefix."""
        import quest.data.standardization as mod

        mod._HLA_SEQ_DICT = {"A*02:01": "MAVMAPRTLLL"}
        mod._HLA_PREFIX_CACHE = {"A*02:01": "A*02:01"}
        try:
            assert resolve_allele_to_sequence("HLA-A*02:01") == "MAVMAPRTLLL"
            assert resolve_allele_to_sequence("A*02:01") == "MAVMAPRTLLL"
        finally:
            mod._HLA_SEQ_DICT = None
            mod._HLA_PREFIX_CACHE = None

    def test_resolve_truncates_to_4digit(self):
        """Higher-resolution alleles truncated to 4-digit for lookup."""
        import quest.data.standardization as mod

        mod._HLA_SEQ_DICT = {"A*02:01": "MAVMAPRTLLL"}
        mod._HLA_PREFIX_CACHE = {"A*02:01": "A*02:01", "A*02": "A*02:01"}
        try:
            assert resolve_allele_to_sequence("HLA-A*02:01:01:01") == "MAVMAPRTLLL"
        finally:
            mod._HLA_SEQ_DICT = None
            mod._HLA_PREFIX_CACHE = None

    def test_standardize_with_hla_dir(self, tmp_path):
        """standardize_dataframe with hla_dir resolves alleles to sequences."""
        # Create a mock FASTA file
        fasta_content = (
            ">HLA:HLA00001 A*02:01 365 bp\n"
            "MAVMAPRTLLL\n"
            ">HLA:HLA00002 B*07:02 365 bp\n"
            "MRVTAPRTVLL\n"
        )
        fasta_file = tmp_path / "A_prot.fasta"
        fasta_file.write_text(fasta_content)

        df = pd.DataFrame({
            "cdr3b": ["CASSLAPGATNEKLFF"],
            "peptide": ["GILGFVFTL"],
            "mhc": ["HLA-A*02:01"],
        })
        column_map = {"cdr3b": "trb", "peptide": "peptide", "mhc": "mhc_one"}

        result, dropped = standardize_dataframe(
            df, column_map, source="test", hla_dir=str(tmp_path)
        )

        assert list(result.columns) == TARGET_COLUMNS
        assert len(result) == 1
        # mhc_one should now be the AA sequence
        assert result.iloc[0]["mhc_one"] == "MAVMAPRTLLL"
        # mhc_one_allele should preserve the allele name
        assert result.iloc[0]["mhc_one_allele"] == "HLA-A*02:01"

    def test_unresolved_allele_tracked(self, tmp_path):
        """Unresolvable alleles are blanked and tracked in dropped_df."""
        # Create a mock FASTA file with only A*02:01
        fasta_content = (
            ">HLA:HLA00001 A*02:01 365 bp\n"
            "MAVMAPRTLLL\n"
        )
        fasta_file = tmp_path / "A_prot.fasta"
        fasta_file.write_text(fasta_content)

        df = pd.DataFrame({
            "cdr3b": ["CASSLAPGATNEKLFF"],
            "peptide": ["GILGFVFTL"],
            "mhc": ["HLA-C*99:99"],  # Not in FASTA
        })
        column_map = {"cdr3b": "trb", "peptide": "peptide", "mhc": "mhc_one"}

        result, dropped = standardize_dataframe(
            df, column_map, source="test", hla_dir=str(tmp_path)
        )

        assert len(result) == 1
        # mhc_one should be blanked
        assert result.iloc[0]["mhc_one"] == ""
        # mhc_one_allele preserves the allele name
        assert result.iloc[0]["mhc_one_allele"] == "HLA-C*99:99"
        # Dropped records should track the unresolved allele
        unresolved = dropped[dropped["reason"] == "mhc_unresolved"]
        assert len(unresolved) >= 1
        assert unresolved.iloc[0]["field"] == "mhc_one"

    def test_no_hla_dir_preserves_allele_names(self):
        """Without hla_dir, mhc_one contains allele names (backward compat)."""
        clear_norm_cache()
        df = pd.DataFrame({
            "cdr3b": ["CASSLAPGATNEKLFF"],
            "peptide": ["GILGFVFTL"],
            "mhc": ["HLA-A*02:01"],
        })
        column_map = {"cdr3b": "trb", "peptide": "peptide", "mhc": "mhc_one"}

        result, dropped = standardize_dataframe(
            df, column_map, source="test"
        )

        assert result.iloc[0]["mhc_one"] == "HLA-A*02:01"
        assert result.iloc[0]["mhc_one_allele"] == "HLA-A*02:01"

    def test_output_has_25_columns(self):
        """Output DataFrame always has exactly 25 TARGET_COLUMNS."""
        clear_norm_cache()
        df = pd.DataFrame({"cdr3b": ["CASSLAPGATNEKLFF"]})
        column_map = {"cdr3b": "trb"}
        result, _ = standardize_dataframe(df, column_map, source="test")
        assert list(result.columns) == TARGET_COLUMNS
        assert len(TARGET_COLUMNS) == 25
