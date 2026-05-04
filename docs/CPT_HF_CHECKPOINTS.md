# CPT checkpoints on HuggingFace Hub

LoRA adapter checkpoints from the NeurIPS Track A continuous pre-training (CPT)
sweep on the `foundation_100M` corpus. All three are private repos under the
`sravisha` namespace and have been verified end-to-end (download, cache, param
counts, inference).

## Repos

| Repo | Base model | Adapter size | Total params | Adapter params |
|---|---|---|---|---|
| [`sravisha/esm2-8m-cpt-fnd100m`](https://huggingface.co/sravisha/esm2-8m-cpt-fnd100m) | `facebook/esm2_t6_8M_UR50D` | 4.5 MB | 8.64 M | 1,126,400 |
| [`sravisha/esm2-35m-cpt-fnd100m`](https://huggingface.co/sravisha/esm2-35m-cpt-fnd100m) | `facebook/esm2_t12_35M_UR50D` | 13.4 MB | 36.85 M | 3,348,480 |
| [`sravisha/esmc300m-foundation100m-c3-lora-r32`](https://huggingface.co/sravisha/esmc300m-foundation100m-c3-lora-r32) | `EvolutionaryScale/esmc-300m-2024-12` | ~58 MB | 347.74 M | 14,745,600 |

All three were trained with PEFT LoRA: `r=32`, `α=64`, `dropout=0.05`, MLM
objective (15% masking), bf16, AdamW (lr=2e-4, cosine, warmup 0.1, wd=0.01),
1 epoch.

## Final eval metrics (epoch end)

| Model | Val loss | Val accuracy | Best val loss |
|---|---|---|---|
| ESM2-8M  | 0.5079 | 0.8457 | 0.5055 |
| ESM2-35M | 0.5034 | 0.8472 | 0.4986 |
| ESMC-300M | (see WandB) | | |

## Loading

### ESM-2 adapters (8M, 35M)

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
tok  = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = PeftModel.from_pretrained(base, "sravisha/esm2-8m-cpt-fnd100m")
```

Repos contain PEFT adapters, **not full models** — `AutoModelForMaskedLM.from_pretrained(repo)`
on the adapter repo will fail. Always load the base first, then attach the adapter.

### ESM-C 300M adapter

ESMC isn't a HuggingFace transformers model — it's loaded via the EvolutionaryScale
`esm` package, then the PEFT adapter is attached on top.

```python
import torch
from esm.models.esmc import ESMC
from peft import PeftModel

base = ESMC.from_pretrained("esmc_300m", device=torch.device("cuda"))

# ESMC has no .config attribute; PEFT requires one. Shim it:
class _ConfigShim(dict):
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
base.config = _ConfigShim(use_return_dict=True, tie_word_embeddings=False)

model = PeftModel.from_pretrained(base, "sravisha/esmc300m-foundation100m-c3-lora-r32")

# Forward call must bypass PEFT's HF-style wrapper (ESMC doesn't accept input_ids):
out = model.base_model.model(sequence_tokens=input_ids, sequence_id=attention_mask.bool())
logits = out.sequence_logits
```

## Verification (2026-05-04)

Cold-cache load (download from Hub) vs warm-cache load (HF cache hit), in throwaway
`HF_HOME` directories:

| Repo | Cold load | Cache hit | Speedup | Inference |
|---|---|---|---|---|
| esm2-8m-cpt-fnd100m | 3.2 s | 1.2 s | 2.7× | logits (1, 19, 33), finite |
| esm2-35m-cpt-fnd100m | 3.4 s | 1.3 s | 2.6× | logits (1, 19, 33), finite |
| esmc300m-foundation100m-c3-lora-r32 | 21.4 s | 1.0 s | 21× | logits (1, 19, 64), finite |

For each repo: cold-load + warm-load yield identical param counts (no drift), and
adapter param counts match the values in the README. ESMC's slow cold load is the
660 MB base-model snapshot download; the adapter itself is ~58 MB.

## Environment notes

- ESM-2 adapters work with `transformers ≥ 4.57.1` (current quest pin).
- The ESM-C path requires `pip install esm` (EvolutionaryScale 3.2.x), which
  pins `transformers < 4.48.2` and so conflicts with quest's
  `transformers ≥ 4.57.1`. **Use a separate venv** for ESMC inference work to
  avoid breaking the quest install.
- Setting `INFRA_PROVIDER` to anything causes ESMC's `data_root()` to return
  `Path("")` and try to load weights from cwd — leave it unset for HF download
  to work.
- ESMC defaults to flash-attention; CPU-only loading will fail. Use a GPU device.

## Tracking

WandB project [`neurips`](https://wandb.ai/shashidhar-r-shankar-fred-hutchinson-cancer-center/neurips),
group `cpt-foundation-100m`:

- ESM2-8M: <https://wandb.ai/shashidhar-r-shankar-fred-hutchinson-cancer-center/neurips/runs/pjg24itx>
- ESM2-35M: <https://wandb.ai/shashidhar-r-shankar-fred-hutchinson-cancer-center/neurips/runs/jskm0fnz>
- ESM-C 300M: (run on the g5.48xlarge box, separate session)

## Provenance

- Trainer: `scripts/training/esm_native_trainer.py` (PEFT LoRA, dynamic-padding
  MLM collator, memmap-backed parquet dataloader).
- Data: `benchmark_v2/tokenized/foundation/foundation_100M_C3.parquet` (train),
  `foundation_val_C3.parquet` (val, sampled to 50K rows for periodic eval).
- Hardware: ESM-2 runs on g6.12xlarge (4× L4); ESM-C run on g5.48xlarge.
