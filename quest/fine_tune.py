import os, math, numpy as np, wandb, torch, time
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
import argparse, json, pathlib
from transformers import TrainerCallback


# ---------------------------------------------------------------------
# 1.  Mapping of easy-to-type keys → HF hub IDs and objective types
# ---------------------------------------------------------------------
MODEL_ZOO = {
    "bert":   {"hf_id": "bert-base-cased",
               "objective": "mlm"},          # masked-language-model
    "gpt2":   {"hf_id": "gpt2",
               "objective": "clm"},          # causal-language-model
    "llama3": {"hf_id": "meta-llama/Meta-Llama-3-7B",   # adjust if you use the Instruct variant
               "objective": "clm"},
    "esm3":   {"hf_id": "facebook/esm3_t48_15B_UR50D",  # update to the exact ESM-3 ID you need
               "objective": "mlm"},
    "protbert": {"hf_id": "Rostlab/prot_bert_bfd",
                 "objective": "mlm"},          # masked-language-model
}

# ---------------------------------------------------------------------
# 2.  Tiny callback to free GPU memory between epochs
# ---------------------------------------------------------------------
class ClearCudaCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        return control

def is_main():
    return int(os.environ.get("RANK", 0)) == 0

# ---------------------------------------------------------------------
# 3.  Metric helper (accuracy for MLM, perplexity for both)
# ---------------------------------------------------------------------
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    perplexity = math.exp(eval_pred.metrics["eval_loss"]) \
                 if "eval_loss" in eval_pred.metrics and eval_pred.metrics["eval_loss"] < 20 else float("inf")

    # Accuracy only makes sense for masked LM (ignored labels == -100)
    mask = labels != -100
    acc = (np.argmax(logits, axis=-1)[mask] == labels[mask]).mean() if mask.sum() else 0.0
    return {"accuracy": acc, "perplexity": perplexity}

# ---------------------------------------------------------------------
# 4.  Master training function
# ---------------------------------------------------------------------
def train_lm(config: dict):
    """Train either an MLM or CLM picked via `config["model_key"]`."""

    spec = MODEL_ZOO[config["model_key"].lower()]
    hf_id, objective = spec["hf_id"], spec["objective"]

    # ----------  W&B  ----------
    if is_main() and wandb.run is None:
        wandb.init(project="quest", config=config)

    # ----------  Data  ----------
    ds = load_from_disk(config["dataset"])       # expect "train" / "validation" splits
    tokenizer = AutoTokenizer.from_pretrained(hf_id,
                                              trust_remote_code=True)  # handles Llama & ESM

    # Some models (GPT-2, Llama) lack a pad token → add one on the fly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok_fn(batch):
        return tokenizer(batch["sequence"],
                         truncation=True,
                         padding="max_length",
                         max_length=config.get("max_length", 1024))

    ds = ds.map(tok_fn, batched=True)
    cols = ["input_ids", "attention_mask"]
    if objective == "clm":
        # For causal LM we set labels = input_ids (shift is handled inside HF)
        ds = ds.map(lambda x: {"labels": x["input_ids"]})
        cols += ["labels"]
    ds.set_format("torch", columns=cols)

    # ----------  Model  ----------
    if objective == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(hf_id,
                                                     trust_remote_code=True)
        collator = DataCollatorForLanguageModeling(tokenizer,
                                                   mlm=True, mlm_probability=0.15)
    else:  # causal LM
        model = AutoModelForCausalLM.from_pretrained(hf_id,
                                                     trust_remote_code=True)
        collator = DataCollatorForLanguageModeling(tokenizer,
                                                   mlm=False)  # simply shifts labels
    # Memory tweak for big models on smaller GPUs
    model.gradient_checkpointing_enable()

    # ----------  Trainer args  ----------
    targs = TrainingArguments(
        output_dir=config["checkpoint_path"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        report_to="wandb",
        fp16=True,
        gradient_accumulation_steps=config.get("grad_accum", 1),
        run_name=wandb.run.name if wandb.run else None,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[ClearCudaCacheCallback()],
    )

    # ----------  Train / evaluate  ----------
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    wandb.log(metrics)
    wandb.finish()


# ---------------------------------------------------------------------
# 5.  Example CLI wrapper  (optional)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON/YAML with training config")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    if cfg_path.suffix == ".json":
        cfg = json.loads(cfg_path.read_text())
    else:
        import yaml; cfg = yaml.safe_load(cfg_path.read_text())

    train_lm(cfg)
