# utils.py

import torch
import torch.distributed as dist
import json
import wandb

def is_main_process(self):
        return (not self.is_distributed) or (self.is_distributed and self.global_rank == 0)

def log_metrics(total_loss, total_correct, total_samples, stage, epoch):
    total_loss_tensor = torch.tensor(total_loss, device="cuda")
    total_correct_tensor = torch.tensor(total_correct, device="cuda")
    total_samples_tensor = torch.tensor(total_samples, device="cuda")

    dist.reduce(total_loss_tensor, dst=0)
    dist.reduce(total_correct_tensor, dst=0)
    dist.reduce(total_samples_tensor, dst=0)

    if is_main_process():
        avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
        avg_accuracy = total_correct_tensor.item() / total_samples_tensor.item()
        log_data = {f"{stage}_loss": avg_loss, f"{stage}_accuracy": avg_accuracy}
        if epoch is not None:
            log_data["epoch"] = epoch
        wandb.log(log_data)

def broadcast_config(config):
    config_str = json.dumps(dict(config))
    config_len = torch.tensor(len(config_str), dtype=torch.int, device="cuda")

    dist.broadcast(config_len, src=0)
    buffer = torch.empty(config_len.item(), dtype=torch.uint8, device="cuda")

    if dist.get_rank() == 0:
        buffer[:] = torch.tensor(list(config_str.encode()), dtype=torch.uint8)

    dist.broadcast(buffer, src=0)
    config_str = "".join(map(chr, buffer.cpu().tolist()))
    return json.loads(config_str)
