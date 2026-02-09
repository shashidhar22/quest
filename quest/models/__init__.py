"""
QUEST Models Package

Model definitions for TCR and MHC interaction modeling, including:
- Seq2seq models for conditional sequence generation (TCR and peptide-MHC)
- Cross-encoder for TCR alpha-beta pairing classification
- Dual encoder with contrastive learning for TCR pairing
- Reusable components: positional encoding, attention pooling, LoRA utilities
"""

# Positional encoding
from .positional_encoding import PositionalEncoding

# Attention pooling and projection heads
from .attention_pooling import AttentionPooling, DeepProjectionHead

# LoRA utilities
from .lora_utils import apply_lora_to_encoder, load_encoder_checkpoint

# Custom decoder with self-attention dropout
from .seq2seq_decoder import SelfAttnDropoutDecoder, SelfAttnDropoutDecoderLayer

# TCR Seq2Seq model and BLOSUM62 loss
from .seq2seq_model import TCRSeq2SeqModel, BLOSUM62SoftLoss

# Peptide-MHC Seq2Seq model
from .pmhc_seq2seq_model import PeptideMHCSeq2SeqModel

# Cross-encoder for TCR pairing
from .cross_encoder import TCRCrossEncoder, CrossEncoderBCELoss

# Dual encoder and contrastive learning components
from .dual_encoder import TCRDualEncoder, MomentumEncoder, DistractorManager

# Contrastive loss
from .contrastive_loss import DistributedRobustInfoNCE, gather_embeddings

__all__ = [
    # Positional encoding
    "PositionalEncoding",
    # Pooling and projection
    "AttentionPooling",
    "DeepProjectionHead",
    # LoRA utilities
    "apply_lora_to_encoder",
    "load_encoder_checkpoint",
    # Seq2seq decoder
    "SelfAttnDropoutDecoder",
    "SelfAttnDropoutDecoderLayer",
    # TCR seq2seq
    "TCRSeq2SeqModel",
    "BLOSUM62SoftLoss",
    # Peptide-MHC seq2seq
    "PeptideMHCSeq2SeqModel",
    # Cross-encoder
    "TCRCrossEncoder",
    "CrossEncoderBCELoss",
    # Dual encoder
    "TCRDualEncoder",
    "MomentumEncoder",
    "DistractorManager",
    # Contrastive loss
    "DistributedRobustInfoNCE",
    "gather_embeddings",
]
