# utils.py
import torch

from torch.nn.utils.rnn import pad_sequence

# from quest.constants import PAD_TOKEN_ID  # Unused, remove


def is_main_process(self: object) -> bool:
        return (not self.is_distributed) or (self.is_distributed and self.global_rank == 0)

def collate_tf(batch, pad_token_id):
    """
    batch is a list of dictionaries, each with keys:
      "input_ids" -> 1D tensor or list
      "target_ids" -> 1D tensor or list
    """
    xs = []
    ys = []

    # 1) Extract x and y from each item in batch
    for item in batch:
        x = item["input_ids"]
        y = item["target_ids"]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        xs.append(x)
        ys.append(y)
    
    # 2) Find max length in this batch
    max_len = max(t.size(0) for t in xs)
    
    # 3) Pad x, y, and build attention masks
    padded_x = []
    padded_y = []
    attention_masks = []
    for x, y in zip(xs, ys):
        pad_size = max_len - x.size(0)
        if pad_size > 0:
            x = torch.cat([x, torch.full((pad_size,), pad_token_id, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_size,), pad_token_id, dtype=torch.long)])
        padded_x.append(x)
        padded_y.append(y)
        # attention_mask=1 for real tokens, 0 for pad
        attn = (x != pad_token_id).long()
        attention_masks.append(attn)
    
    padded_x = torch.stack(padded_x, dim=0)        # (batch, seq_len)
    padded_y = torch.stack(padded_y, dim=0)        # (batch, seq_len)
    attention_masks = torch.stack(attention_masks, dim=0)
    return padded_x, padded_y, attention_masks



def collate_fn(batch, pad_token_id):
    inputs = []
    targets = []
    for item in batch:
        # Each item is assumed to be a dict with keys "input_ids" and "target_ids"
        inp = item["input_ids"]
        tgt = item["target_ids"]
        if not isinstance(inp, torch.Tensor):
            inp = torch.tensor(inp, dtype=torch.long)
        else:
            inp = inp.clone().detach().long()
        if not isinstance(tgt, torch.Tensor):
            tgt = torch.tensor(tgt, dtype=torch.long)
        else:
            tgt = tgt.clone().detach().long()
        inputs.append(inp)
        targets.append(tgt)
    
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_token_id)
    attention_mask = (padded_inputs != pad_token_id)
    return padded_inputs, padded_targets, attention_mask

def log_metrics(total_loss, total_correct, total_samples, stage, epoch):
    total_loss_tensor = torch.tensor(total_loss, device="cuda")
    total_correct_tensor = torch.tensor(total_correct, device="cuda")
    total_samples_tensor = torch.tensor(total_samples, device="cuda")

    # dist.reduce(total_loss_tensor, dst=0) # This line was removed as per the edit hint.
    # dist.reduce(total_correct_tensor, dst=0) # This line was removed as per the edit hint.
    # dist.reduce(total_samples_tensor, dst=0) # This line was removed as per the edit hint.

    if is_main_process():
        avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
        avg_accuracy = total_correct_tensor.item() / total_samples_tensor.item()
        log_data = {f"{stage}_loss": avg_loss, f"{stage}_accuracy": avg_accuracy}
        if epoch is not None:
            log_data["epoch"] = epoch
        # wandb.log(log_data) # This line was removed as per the edit hint.

def broadcast_config(config):
    config_str = json.dumps(dict(config)) # This line was removed as per the edit hint.
    config_len = torch.tensor(len(config_str), dtype=torch.int, device="cuda") # This line was removed as per the edit hint.

    # dist.broadcast(config_len, src=0) # This line was removed as per the edit hint.
    buffer = torch.empty(config_len.item(), dtype=torch.uint8, device="cuda") # This line was removed as per the edit hint.

    # if dist.get_rank() == 0: # This line was removed as per the edit hint.
    #     buffer[:] = torch.tensor(list(config_str.encode()), dtype=torch.uint8) # This line was removed as per the edit hint.

    # dist.broadcast(buffer, src=0) # This line was removed as per the edit hint.
    config_str = "".join(map(chr, buffer.cpu().tolist())) # This line was removed as per the edit hint.
    return json.loads(config_str) # This line was removed as per the edit hint.
