# utils.py
import torch

from torch.nn.utils.rnn import pad_sequence


def is_main_process(is_distributed: bool, global_rank: int) -> bool:
    """Check if current process is the main process."""
    return (not is_distributed) or (is_distributed and global_rank == 0)

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

