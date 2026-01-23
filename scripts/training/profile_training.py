#!/usr/bin/env python3
"""
Training Profiler for TCR Seq2Seq Trainer

Identifies bottlenecks and optimal batch size for your hardware.
Run on p4d.24xlarge to find optimal configuration.

Usage:
    python scripts/training/profile_training.py \
        --data_path data/deduplicated/full/foundation_permutations/ \
        --task PEPTIDE
"""

import argparse
import gc
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.training.tcr_seq2seq_trainer import (
    TCRSeq2SeqModel,
    TCRSeq2SeqDataset,
    Seq2SeqCollator,
    GenerationTask,
    apply_lora_to_encoder,
    PEFT_AVAILABLE,
)
from transformers import AutoTokenizer


@dataclass
class ProfileResult:
    """Container for profiling results."""
    batch_size: int
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float
    gpu_memory_max_gb: float
    data_load_time_ms: float
    forward_time_ms: float
    backward_time_ms: float
    optimizer_step_time_ms: float
    total_step_time_ms: float
    throughput_samples_per_sec: float
    gpu_utilization_pct: float
    success: bool
    error: Optional[str] = None


def get_gpu_memory_gb() -> Tuple[float, float, float]:
    """Get GPU memory stats in GB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_memory = torch.cuda.max_memory_allocated() / 1e9
    return allocated, reserved, max_memory


def get_gpu_utilization() -> float:
    """Get GPU utilization percentage using nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Get first GPU's utilization
            utilizations = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            return utilizations[0] if utilizations else 0.0
    except Exception:
        pass
    return 0.0


@contextmanager
def cuda_timer():
    """Context manager for accurate CUDA timing."""
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        yield lambda: (end_event.record(), torch.cuda.synchronize(), start_event.elapsed_time(end_event))[-1]
    else:
        start = time.perf_counter()
        yield lambda: (time.perf_counter() - start) * 1000


def profile_batch_size(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    num_warmup: int = 3,
    num_profile: int = 10,
) -> ProfileResult:
    """
    Profile training performance for a given batch size.

    Args:
        model: The model to profile
        dataloader: DataLoader with the specified batch size
        optimizer: Optimizer
        device: CUDA device
        batch_size: Batch size being profiled
        num_warmup: Number of warmup iterations
        num_profile: Number of iterations to profile

    Returns:
        ProfileResult with timing and memory stats
    """
    model.train()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    data_load_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    total_times = []
    gpu_utils = []

    try:
        data_iter = iter(dataloader)

        # Warmup iterations (compile kernels, fill caches)
        print(f"  Warming up ({num_warmup} iterations)...", end=" ", flush=True)
        for _ in range(num_warmup):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            encoder_ids = batch["encoder_input_ids"].to(device)
            encoder_mask = batch["encoder_attention_mask"].to(device)
            decoder_ids = batch["decoder_input_ids"].to(device)
            decoder_mask = batch["decoder_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    encoder_input_ids=encoder_ids,
                    encoder_attention_mask=encoder_mask,
                    decoder_input_ids=decoder_ids,
                    decoder_attention_mask=decoder_mask,
                    labels=labels,
                )
            outputs["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        print("done")

        # Profile iterations
        print(f"  Profiling ({num_profile} iterations)...", end=" ", flush=True)
        for i in range(num_profile):
            # Data loading time
            t0 = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            encoder_ids = batch["encoder_input_ids"].to(device, non_blocking=True)
            encoder_mask = batch["encoder_attention_mask"].to(device, non_blocking=True)
            decoder_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
            decoder_mask = batch["decoder_attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            torch.cuda.synchronize()
            data_load_times.append((time.perf_counter() - t0) * 1000)

            # Forward pass
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    encoder_input_ids=encoder_ids,
                    encoder_attention_mask=encoder_mask,
                    decoder_input_ids=decoder_ids,
                    decoder_attention_mask=decoder_mask,
                    labels=labels,
                )
            end_event.record()
            torch.cuda.synchronize()
            forward_times.append(start_event.elapsed_time(end_event))

            # Backward pass
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            outputs["loss"].backward()
            end_event.record()
            torch.cuda.synchronize()
            backward_times.append(start_event.elapsed_time(end_event))

            # Optimizer step
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            optimizer.step()
            optimizer.zero_grad()
            end_event.record()
            torch.cuda.synchronize()
            optimizer_times.append(start_event.elapsed_time(end_event))

            # GPU utilization sample
            gpu_utils.append(get_gpu_utilization())

        print("done")

        # Calculate statistics
        allocated, reserved, max_mem = get_gpu_memory_gb()

        avg_data = sum(data_load_times) / len(data_load_times)
        avg_forward = sum(forward_times) / len(forward_times)
        avg_backward = sum(backward_times) / len(backward_times)
        avg_optimizer = sum(optimizer_times) / len(optimizer_times)
        avg_total = avg_data + avg_forward + avg_backward + avg_optimizer

        throughput = (batch_size * 1000) / avg_total  # samples per second
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0

        return ProfileResult(
            batch_size=batch_size,
            gpu_memory_allocated_gb=allocated,
            gpu_memory_reserved_gb=reserved,
            gpu_memory_max_gb=max_mem,
            data_load_time_ms=avg_data,
            forward_time_ms=avg_forward,
            backward_time_ms=avg_backward,
            optimizer_step_time_ms=avg_optimizer,
            total_step_time_ms=avg_total,
            throughput_samples_per_sec=throughput,
            gpu_utilization_pct=avg_gpu_util,
            success=True,
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return ProfileResult(
                batch_size=batch_size,
                gpu_memory_allocated_gb=0,
                gpu_memory_reserved_gb=0,
                gpu_memory_max_gb=0,
                data_load_time_ms=0,
                forward_time_ms=0,
                backward_time_ms=0,
                optimizer_step_time_ms=0,
                total_step_time_ms=0,
                throughput_samples_per_sec=0,
                gpu_utilization_pct=0,
                success=False,
                error="OOM",
            )
        raise


def find_optimal_batch_size(
    model: torch.nn.Module,
    tokenizer,
    dataset,
    device: torch.device,
    start_batch: int = 8,
    max_batch: int = 64,
    step: int = 4,
    num_workers: int = 4,
) -> Tuple[int, List[ProfileResult]]:
    """
    Binary search for optimal batch size.

    Returns:
        Tuple of (optimal_batch_size, list of profile results)
    """
    results = []
    optimal_batch = start_batch
    best_throughput = 0.0

    collator = Seq2SeqCollator(
        tokenizer=tokenizer,
        max_encoder_length=512,
        max_decoder_length=256,
    )

    batch_sizes = list(range(start_batch, max_batch + 1, step))

    print(f"\nTesting batch sizes: {batch_sizes}")
    print("=" * 70)

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create new dataloader with this batch size
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Create fresh optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
        )

        result = profile_batch_size(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
        )

        results.append(result)

        if result.success:
            print(f"  Memory: {result.gpu_memory_max_gb:.2f} GB peak")
            print(f"  Timing breakdown:")
            print(f"    Data load:  {result.data_load_time_ms:7.2f} ms ({100*result.data_load_time_ms/result.total_step_time_ms:5.1f}%)")
            print(f"    Forward:    {result.forward_time_ms:7.2f} ms ({100*result.forward_time_ms/result.total_step_time_ms:5.1f}%)")
            print(f"    Backward:   {result.backward_time_ms:7.2f} ms ({100*result.backward_time_ms/result.total_step_time_ms:5.1f}%)")
            print(f"    Optimizer:  {result.optimizer_step_time_ms:7.2f} ms ({100*result.optimizer_step_time_ms/result.total_step_time_ms:5.1f}%)")
            print(f"    Total:      {result.total_step_time_ms:7.2f} ms")
            print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")

            if result.throughput_samples_per_sec > best_throughput:
                best_throughput = result.throughput_samples_per_sec
                optimal_batch = batch_size
        else:
            print(f"  FAILED: {result.error}")
            break  # Stop on OOM

        # Cleanup
        del dataloader, optimizer

    return optimal_batch, results


def profile_num_workers(
    model: torch.nn.Module,
    tokenizer,
    dataset,
    device: torch.device,
    batch_size: int,
    worker_counts: List[int] = [0, 2, 4, 8, 12, 16],
) -> Dict[int, float]:
    """Test different num_workers to find data loading bottleneck."""

    collator = Seq2SeqCollator(
        tokenizer=tokenizer,
        max_encoder_length=512,
        max_decoder_length=256,
    )

    results = {}

    print(f"\nTesting num_workers with batch_size={batch_size}")
    print("=" * 50)

    for num_workers in worker_counts:
        print(f"\nWorkers: {num_workers}")

        gc.collect()
        torch.cuda.empty_cache()

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        result = profile_batch_size(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            num_warmup=2,
            num_profile=5,
        )

        if result.success:
            results[num_workers] = result.data_load_time_ms
            data_pct = 100 * result.data_load_time_ms / result.total_step_time_ms
            print(f"  Data load: {result.data_load_time_ms:.2f} ms ({data_pct:.1f}% of step)")
            print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
        else:
            print(f"  FAILED: {result.error}")

        del dataloader, optimizer

    return results


def print_recommendations(results: List[ProfileResult], optimal_batch: int, worker_results: Dict[int, float]):
    """Print optimization recommendations based on profiling results."""

    print("\n" + "=" * 70)
    print("PROFILING SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    # Find best result
    best_result = None
    for r in results:
        if r.success and r.batch_size == optimal_batch:
            best_result = r
            break

    if best_result is None:
        print("No successful profile results found!")
        return

    print(f"\n📊 OPTIMAL CONFIGURATION FOUND:")
    print(f"   Batch size: {optimal_batch}")
    print(f"   Peak memory: {best_result.gpu_memory_max_gb:.2f} GB")
    print(f"   Throughput: {best_result.throughput_samples_per_sec:.1f} samples/sec")

    # Identify bottleneck
    total = best_result.total_step_time_ms
    data_pct = 100 * best_result.data_load_time_ms / total
    forward_pct = 100 * best_result.forward_time_ms / total
    backward_pct = 100 * best_result.backward_time_ms / total

    print(f"\n⏱️  TIME BREAKDOWN:")
    print(f"   Data loading: {data_pct:.1f}%")
    print(f"   Forward pass: {forward_pct:.1f}%")
    print(f"   Backward pass: {backward_pct:.1f}%")

    bottleneck = "compute"
    if data_pct > 30:
        bottleneck = "data_loading"

    print(f"\n🔍 BOTTLENECK: {bottleneck.upper()}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    if bottleneck == "data_loading":
        print("   • Increase num_workers (try 8-16 on p4d.24xlarge)")
        print("   • Enable persistent_workers=True")
        print("   • Increase prefetch_factor to 4")
        if worker_results:
            best_workers = min(worker_results, key=worker_results.get)
            print(f"   • Best num_workers tested: {best_workers}")
    else:
        print("   • GPU is compute-bound (good!)")
        print("   • Current num_workers is sufficient")

    # Memory headroom
    total_gpu_memory = 40.0  # A100 40GB
    memory_used_pct = 100 * best_result.gpu_memory_max_gb / total_gpu_memory
    memory_headroom = total_gpu_memory - best_result.gpu_memory_max_gb

    print(f"\n🧠 MEMORY ANALYSIS:")
    print(f"   Peak usage: {best_result.gpu_memory_max_gb:.2f} GB ({memory_used_pct:.1f}%)")
    print(f"   Headroom: {memory_headroom:.2f} GB")

    if memory_headroom > 10:
        suggested_increase = int(optimal_batch * 1.3)
        print(f"   • Consider trying batch_size={suggested_increase} for more throughput")
    elif memory_headroom < 5:
        print(f"   • Memory is tight, current batch_size is near optimal")

    # Final command
    best_workers = min(worker_results, key=worker_results.get) if worker_results else 8

    print(f"\n🚀 RECOMMENDED COMMAND:")
    print(f"""
python scripts/training/tcr_seq2seq_trainer.py \\
    --data_path <your_data_path> \\
    --output_dir ./output/optimized \\
    --task PEPTIDE \\
    --batch_size {optimal_batch} \\
    --gradient_accumulation_steps 2 \\
    --num_workers {best_workers} \\
    --learning_rate 2e-4 \\
    --use_lora
""")

    # Effective batch size calculation
    eff_batch = optimal_batch * 2 * 8  # grad_accum=2, 8 GPUs
    print(f"   Effective batch size: {optimal_batch} × 2 × 8 = {eff_batch}")


def main():
    parser = argparse.ArgumentParser(description="Profile TCR Seq2Seq Training")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to parquet data directory")
    parser.add_argument("--task", type=str, default="PEPTIDE",
                        choices=["ALPHA", "BETA", "PEPTIDE"])
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="ESM2 model name")
    parser.add_argument("--start_batch", type=int, default=8,
                        help="Starting batch size to test")
    parser.add_argument("--max_batch", type=int, default=48,
                        help="Maximum batch size to test")
    parser.add_argument("--step", type=int, default=4,
                        help="Batch size step increment")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA (matches your current config)")
    parser.add_argument("--decoder_layers", type=int, default=6)
    parser.add_argument("--decoder_heads", type=int, default=20)
    parser.add_argument("--decoder_dim", type=int, default=1280)
    parser.add_argument("--decoder_ffn_dim", type=int, default=5120)

    args = parser.parse_args()

    print("=" * 70)
    print("TCR SEQ2SEQ TRAINING PROFILER")
    print("=" * 70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    device = torch.device("cuda:0")
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model
    print(f"Creating model...")
    model = TCRSeq2SeqModel(
        encoder_model_name=args.model_name,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_dim=args.decoder_dim,
        decoder_ffn_dim=args.decoder_ffn_dim,
        dropout=0.1,
        torch_dtype=torch.bfloat16,
    )

    # Apply LoRA
    if args.use_lora and PEFT_AVAILABLE:
        print("Applying LoRA to encoder...")
        model = apply_lora_to_encoder(model, {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        })

    # Enable gradient checkpointing
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Load dataset (small subset for profiling)
    print(f"\nLoading dataset from {args.data_path}...")
    task = GenerationTask[args.task.upper()]
    dataset = TCRSeq2SeqDataset(
        data_path=args.data_path,
        permutation_keys=None,  # Auto-discover
        task=task,
        split="train",
        local_rank=0,
    )
    print(f"Dataset size: {len(dataset):,} samples")

    # Profile batch sizes
    optimal_batch, batch_results = find_optimal_batch_size(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        start_batch=args.start_batch,
        max_batch=args.max_batch,
        step=args.step,
        num_workers=4,  # Start with reasonable default
    )

    # Profile num_workers with optimal batch size
    print("\n" + "=" * 70)
    print("PROFILING DATA LOADING (num_workers)")
    print("=" * 70)

    worker_results = profile_num_workers(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        batch_size=optimal_batch,
        worker_counts=[0, 2, 4, 8, 12],
    )

    # Print recommendations
    print_recommendations(batch_results, optimal_batch, worker_results)


if __name__ == "__main__":
    main()
