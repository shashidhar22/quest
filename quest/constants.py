PAD_TOKEN_ID = 0
BPE_TOKENS = {
        "cls_token": [None],
        "sep_token": [None],
        "pad_token": ["[PAD]"], 
        "unk_token": ["[UNK]"],
        "end_token": ["[END]"],
        "tra_tokens": ["[TRA]", "[ETRA]"],
        "trb_tokens": ["[TRB]", "[ETRB]"],
        "pep_tokens": ["[PEP]", "[EPEP]"],
        "mho_tokens": ["MHO", "[EMHO]"],
        "mht_tokens": ["MHT", "[EMHT]"], 
}
PROT_BERT_TOKENS = {
    "cls_token": ["[CLS]"],
    "sep_token": ["[SEP]"],
    "end_token": [None],
    "unk_token": [None],
    "pad_token": [None],
    "tra_tokens": [None, None],
    "trb_tokens": [None, None],
    "pep_tokens": [None, None],
    "mho_tokens": [None, None],
    "mht_tokens": [None, None], 
}

BERT_TOKENS = {
    "cls_token": ["[CLS]"],
    "sep_token": ["[SEP]"],
    "end_token": [None],
    "unk_token": [None],
    "pad_token": [None],
    "tra_tokens": [None, None],
    "trb_tokens": [None, None],
    "pep_tokens": [None, None],
    "mho_tokens": [None, None],
    "mht_tokens": [None, None], 
}

TRA_TOKENS = ["[TRA]", "[ETRA]"]
TRB_TOKENS = ["[TRB]", "[ETRB]"]
PEP_TOKENS = ["[PEP]", "[EPEP]"]
MHC1_TOKENS = ["[MHO]", "[EMHO]"]
MHC2_TOKENS = ["[MHT]", "[EMHT]"]
PAD_TOKEN_STR = "[PAD]"
UNK_TOKEN_STR = "[UNK]"
TRA_START = "[TRA]"
TRB_START = "[TRB]"
PEP_START = "[PEP]"
MHC1_START= "[MHO]"
MHC2_START= "[MHT]"
END_TOKEN = "[END]"
TRA_LENGTH = 8
TRB_LENGTH = 8
PEPTIDE_LENGTH = 4