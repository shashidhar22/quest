#!/usr/bin/env python3
"""
Launch SageMaker training job for fine_tune.py

This script provides a convenient command-line interface to launch AWS SageMaker
training jobs for fine-tuning protein language models on TCR/MHC data.

Features:
- Managed infrastructure with automatic scaling
- Spot Training support for up to 90% cost savings
- Automatic checkpoint management for interruption recovery
- Multi-GPU distributed training
- S3-based dataset and model storage
- W&B integration for experiment tracking

Example Usage (Pre-tokenized):
    python scripts/training/launch_sagemaker_training.py \
        --s3-dataset s3://my-bucket/datasets/tcr-dataset \
        --s3-output s3://my-bucket/models/esm2-finetuned \
        --instance-type ml.p3.2xlarge \
        --spot-instances \
        --mode full_tra \
        --model-path facebook/esm2_t33_650M_UR50D \
        --num-epochs 3 \
        --batch-size 32 \
        --use-lora \
        --fp16 \
        --wandb-project tcr-quest

Example Usage (On-the-fly Tokenization - 48% disk savings):
    python scripts/training/launch_sagemaker_training.py \
        --s3-dataset s3://my-bucket/raw-parquet/tcr-data \
        --s3-output s3://my-bucket/models/esm2-finetuned \
        --instance-type ml.p3.2xlarge \
        --spot-instances \
        --mode mlm \
        --model-path facebook/esm2_t33_650M_UR50D \
        --tokenize-on-fly \
        --tokenizer-type esm2 \
        --num-epochs 3 \
        --batch-size 32 \
        --use-lora \
        --fp16 \
        --gradient-checkpointing \
        --wandb-project tcr-quest

Instance Types:
    - ml.p3.2xlarge:   1x V100 GPU, 16GB VRAM  (single GPU training)
    - ml.p3.8xlarge:   4x V100 GPU, 64GB VRAM  (multi-GPU training)
    - ml.p3.16xlarge:  8x V100 GPU, 128GB VRAM (large-scale training)
    - ml.g5.xlarge:    1x A10G GPU, 24GB VRAM  (cost-effective option)
    - ml.g5.12xlarge:  4x A10G GPU, 96GB VRAM  (multi-GPU cost-effective)
"""

import argparse
import os
import sys

try:
    import sagemaker
    from sagemaker.pytorch import PyTorch
except ImportError:
    print("ERROR: SageMaker SDK not installed.")
    print("Install with: pip install 'sagemaker<3.0'")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Launch SageMaker training job for fine-tuning protein language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # =========================================================================
    # Infrastructure Arguments
    # =========================================================================
    infra_group = parser.add_argument_group('Infrastructure Configuration')
    infra_group.add_argument(
        '--s3-dataset', required=True,
        help='S3 path to dataset directory. For pre-tokenized: HuggingFace dataset. '
             'For on-the-fly: raw parquet files (e.g., s3://bucket/datasets/tcr-data)'
    )
    infra_group.add_argument(
        '--s3-output', required=True,
        help='S3 path for model outputs and checkpoints (e.g., s3://bucket/models/run-1)'
    )
    infra_group.add_argument(
        '--instance-type', default='ml.p3.2xlarge',
        help='SageMaker instance type (default: ml.p3.2xlarge). Common options: '
             'ml.p3.2xlarge (1 GPU), ml.p3.8xlarge (4 GPU), ml.g5.xlarge (A10G)'
    )
    infra_group.add_argument(
        '--instance-count', type=int, default=1,
        help='Number of instances for distributed training (default: 1)'
    )
    infra_group.add_argument(
        '--volume-size', type=int, default=100,
        help='EBS volume size in GB (default: 100)'
    )
    infra_group.add_argument(
        '--role', default=None,
        help='IAM role ARN for SageMaker (default: auto-detect from SageMaker session)'
    )

    # =========================================================================
    # Spot Training Arguments
    # =========================================================================
    spot_group = parser.add_argument_group('Spot Training (Cost Savings)')
    spot_group.add_argument(
        '--spot-instances', action='store_true',
        help='Use managed spot instances for up to 90%% cost savings (highly recommended)'
    )
    spot_group.add_argument(
        '--max-run-hours', type=int, default=24,
        help='Maximum training time in hours (default: 24)'
    )
    spot_group.add_argument(
        '--max-wait-hours', type=int, default=None,
        help='Maximum wait time for spot instances in hours (default: max-run-hours + 1). '
             'Only used with --spot-instances'
    )

    # =========================================================================
    # Model & Dataset Arguments (passed to fine_tune.py)
    # =========================================================================
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--mode', required=True,
        choices=['mlm', 'tra', 'trb', 'full_tra', 'full_trb', 'tra_trb_pairing',
                 'tcr_mhc', 'peptide_mhc', 'specificity'],
        help='Fine-tuning mode (determines masking strategy)'
    )
    model_group.add_argument(
        '--model-path', required=True,
        help='Pre-trained model path or HuggingFace model name '
             '(e.g., facebook/esm2_t33_650M_UR50D, Rostlab/prot_bert)'
    )

    # =========================================================================
    # Training Hyperparameters (passed to fine_tune.py)
    # =========================================================================
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--num-epochs', type=int, default=3,
                            help='Number of training epochs (default: 3)')
    train_group.add_argument('--batch-size', type=int, default=32,
                            help='Per-device training batch size (default: 32)')
    train_group.add_argument('--eval-batch-size', type=int, default=None,
                            help='Per-device eval batch size (default: same as batch-size)')
    train_group.add_argument('--learning-rate', type=float, default=5e-5,
                            help='Learning rate (default: 5e-5)')
    train_group.add_argument('--warmup-steps', type=int, default=500,
                            help='Number of warmup steps (default: 500)')
    train_group.add_argument('--weight-decay', type=float, default=0.01,
                            help='Weight decay (default: 0.01)')
    train_group.add_argument('--gradient-accumulation-steps', type=int, default=1,
                            help='Gradient accumulation steps (default: 1)')

    # =========================================================================
    # Optimization Arguments
    # =========================================================================
    opt_group = parser.add_argument_group('Optimization')
    opt_group.add_argument('--use-lora', action='store_true',
                          help='Use LoRA for parameter-efficient fine-tuning')
    opt_group.add_argument('--lora-r', type=int, default=16,
                          help='LoRA rank (default: 16)')
    opt_group.add_argument('--lora-alpha', type=int, default=32,
                          help='LoRA alpha (default: 32)')
    opt_group.add_argument('--fp16', action='store_true',
                          help='Use FP16 mixed precision training')
    opt_group.add_argument('--bf16', action='store_true',
                          help='Use BF16 mixed precision training')
    opt_group.add_argument('--gradient-checkpointing', action='store_true',
                          help='Enable gradient checkpointing (saves memory)')
    opt_group.add_argument('--max-seq-length', type=int, default=None,
                          help='Maximum sequence length (truncate longer sequences)')

    # =========================================================================
    # Logging & Monitoring
    # =========================================================================
    log_group = parser.add_argument_group('Logging & Monitoring')
    log_group.add_argument('--wandb-project', default=None,
                          help='Weights & Biases project name for experiment tracking')
    log_group.add_argument('--wandb-run-name', default=None,
                          help='W&B run name (default: auto-generated from job name)')
    log_group.add_argument('--job-name', default=None,
                          help='SageMaker training job name (default: auto-generated)')

    # =========================================================================
    # Tokenization (On-the-fly)
    # =========================================================================
    tokenize_group = parser.add_argument_group('On-the-fly Tokenization (Optional)')
    tokenize_group.add_argument(
        '--tokenize-on-fly', action='store_true',
        help='Enable on-the-fly tokenization from raw parquet files (saves 48%% disk space). '
             'Requires --tokenizer-type. First epoch slower, subsequent epochs fast (uses cache).'
    )
    tokenize_group.add_argument(
        '--tokenizer-type',
        choices=['protbert', 'bert', 'esm2', 'esm3'],
        help='Tokenizer type for on-the-fly tokenization (required with --tokenize-on-fly). '
             'Must match model architecture: esm2 for ESM models, protbert for ProtBERT, etc.'
    )
    tokenize_group.add_argument(
        '--tokenization-max-length', type=int, default=512,
        help='Maximum sequence length for tokenization (default: 512)'
    )
    tokenize_group.add_argument(
        '--tokenization-num-workers', type=int, default=None,
        help='Parallel workers for tokenization (default: auto-detect, typically CPU_COUNT - 4)'
    )
    tokenize_group.add_argument(
        '--tokenization-batch-size', type=int, default=5000,
        help='Batch size for tokenization (default: 5000, higher = faster but more RAM)'
    )
    tokenize_group.add_argument(
        '--use-streaming', action='store_true',
        help='Use streaming dataset for truly lazy evaluation. Best for 100M+ examples. '
             'Tokenizes data on-the-fly during training (no pre-processing wait). '
             'Trade-off: Cannot shuffle across full dataset, only within buffer.'
    )

    # =========================================================================
    # Advanced
    # =========================================================================
    adv_group = parser.add_argument_group('Advanced')
    adv_group.add_argument('--framework-version', default='2.8',
                          help='PyTorch framework version (default: 2.8)')
    adv_group.add_argument('--py-version', default='py312',
                          help='Python version (default: py312)')
    adv_group.add_argument('--test', action='store_true',
                          help='Run in test mode with reduced dataset')
    adv_group.add_argument('--skip-mode-filter', action='store_true', default=True,
                          help='Skip mode filtering (dataset already filtered during tokenization)')
    adv_group.add_argument('--no-wait', action='store_true',
                          help='Launch job and exit (don\'t wait for completion). Useful for long jobs with SSO.')

    args = parser.parse_args()

    # =========================================================================
    # Validate Arguments
    # =========================================================================
    if args.spot_instances and args.max_wait_hours is None:
        args.max_wait_hours = args.max_run_hours + 1

    # Validate on-the-fly tokenization arguments
    if args.tokenize_on_fly and not args.tokenizer_type:
        print("\nERROR: --tokenizer-type is required when using --tokenize-on-fly")
        print("Choose from: protbert, bert, esm2, esm3")
        sys.exit(1)

    if args.tokenizer_type and not args.tokenize_on_fly:
        print("\nWARNING: --tokenizer-type specified without --tokenize-on-fly")
        print("Enabling --tokenize-on-fly automatically")
        args.tokenize_on_fly = True

    # =========================================================================
    # Get SageMaker Role and Session
    # =========================================================================
    # Extract bucket name from s3_output to use as default bucket
    default_bucket = args.s3_output.split('/')[2] if args.s3_output.startswith('s3://') else None

    session = sagemaker.Session(default_bucket=default_bucket)

    try:
        if args.role:
            role = args.role
        else:
            role = sagemaker.get_execution_role()
            print(f"Using SageMaker execution role: {role}")
    except Exception as e:
        print(f"\nERROR: Could not determine SageMaker execution role.")
        print(f"Details: {e}")
        print(f"\nPlease specify --role explicitly with your IAM role ARN.")
        sys.exit(1)

    # =========================================================================
    # Build Hyperparameters (passed as CLI args to fine_tune.py)
    # =========================================================================
    hyperparameters = {
        'mode': args.mode,
        'model_path': args.model_path,
        'output_dir': '/opt/ml/checkpoints',  # SageMaker checkpoint directory
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'warmup_steps': args.warmup_steps,
        'weight_decay': args.weight_decay,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
    }

    # Add dataset path OR raw data dir based on tokenization mode
    if args.tokenize_on_fly:
        # On-the-fly tokenization: use raw_data_dir
        hyperparameters['raw_data_dir'] = '/opt/ml/input/data/training'
        hyperparameters['tokenizer_type'] = args.tokenizer_type
        hyperparameters['tokenization_max_length'] = args.tokenization_max_length
        # Only pass num_workers if explicitly set (otherwise auto-detect in fine_tune.py)
        if args.tokenization_num_workers is not None:
            hyperparameters['tokenization_num_workers'] = args.tokenization_num_workers
        hyperparameters['tokenization_batch_size'] = args.tokenization_batch_size

        # Add streaming option
        if args.use_streaming:
            hyperparameters['use_streaming'] = ''
            print(f"\n🌊 STREAMING MODE enabled!")
            print(f"  Tokenization: On-the-fly during training (NO pre-processing wait)")
            print(f"  Best for: 100M+ examples")
            print(f"  Trade-off: Limited shuffling (buffer-based)")
        else:
            print(f"\n✓ On-the-fly tokenization enabled (tokenizer: {args.tokenizer_type})")
            print(f"  Tokenization workers: {'auto-detect' if args.tokenization_num_workers is None else args.tokenization_num_workers}")
            print(f"  Tokenization batch size: {args.tokenization_batch_size}")
            print(f"  First epoch: Will tokenize in parallel, then cache")
            print(f"  Subsequent epochs: Will read from cache (fast)")
            print(f"  Expected disk savings: ~48%")
    else:
        # Pre-tokenized: use dataset_path
        hyperparameters['dataset_path'] = '/opt/ml/input/data/training'  # SageMaker mounts S3 data here

    # Add eval batch size if specified
    if args.eval_batch_size:
        hyperparameters['eval_batch_size'] = args.eval_batch_size

    # Add LoRA parameters
    if args.use_lora:
        hyperparameters['use_lora'] = ''  # Store-true flags
        hyperparameters['lora_r'] = args.lora_r
        hyperparameters['lora_alpha'] = args.lora_alpha

    # Add precision
    if args.fp16:
        hyperparameters['fp16'] = ''
    if args.bf16:
        hyperparameters['bf16'] = ''

    # Add gradient checkpointing
    if args.gradient_checkpointing:
        hyperparameters['gradient_checkpointing'] = ''

    # Add max sequence length
    if args.max_seq_length:
        hyperparameters['max_seq_length'] = args.max_seq_length

    # Add W&B
    if args.wandb_project:
        hyperparameters['wandb_project'] = args.wandb_project
    if args.wandb_run_name:
        hyperparameters['wandb_run_name'] = args.wandb_run_name

    # Add test mode
    if args.test:
        hyperparameters['test'] = ''

    # Add skip mode filter
    if args.skip_mode_filter:
        hyperparameters['skip-mode-filter'] = ''

    # =========================================================================
    # Configure PyTorch Estimator
    # =========================================================================
    estimator_args = {
        'entry_point': 'fine_tune.py',
        'source_dir': 'scripts/training',
        'role': role,
        'instance_type': args.instance_type,
        'instance_count': args.instance_count,
        'framework_version': args.framework_version,
        'py_version': args.py_version,
        'output_path': args.s3_output,
        'hyperparameters': hyperparameters,
        'volume_size': args.volume_size,
        'max_run': args.max_run_hours * 3600,  # Convert to seconds
        'keep_alive_period_in_seconds': 0,  # Don't keep instance alive after job
        'sagemaker_session': session,
        # Enable DDP for multi-GPU training (p4d.24xlarge has 8 GPUs)
        # This uses torchrun instead of DataParallel for efficient multi-GPU
        'distribution': {
            'torch_distributed': {
                'enabled': True
            }
        } if args.instance_type in ['ml.p4d.24xlarge', 'ml.p3.8xlarge', 'ml.p3.16xlarge',
                                     'ml.g5.12xlarge', 'ml.g5.48xlarge'] else None,
    }

    # Add spot training configuration
    if args.spot_instances:
        estimator_args.update({
            'use_spot_instances': True,
            'max_wait': args.max_wait_hours * 3600,  # Convert to seconds
            'checkpoint_s3_uri': f"{args.s3_output}/checkpoints",
            'checkpoint_local_path': '/opt/ml/checkpoints',
        })

    # Add job name if specified (sanitize for SageMaker naming requirements)
    if args.job_name:
        # SageMaker job names must match: [a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}
        # Replace underscores with hyphens, remove other invalid characters
        sanitized_name = args.job_name.replace('_', '-')
        # Remove any characters that aren't alphanumeric or hyphen
        sanitized_name = ''.join(c for c in sanitized_name if c.isalnum() or c == '-')
        # Ensure it doesn't start or end with a hyphen
        sanitized_name = sanitized_name.strip('-')
        # Truncate to 62 characters (SageMaker adds timestamp suffix)
        sanitized_name = sanitized_name[:62]

        if sanitized_name != args.job_name:
            print(f"\nℹ️  Sanitized job name: '{args.job_name}' -> '{sanitized_name}'")

        estimator_args['base_job_name'] = sanitized_name

    # Add W&B API key from environment
    if args.wandb_project:
        wandb_api_key = os.environ.get('WANDB_API_KEY', '')
        if not wandb_api_key:
            print("\nWARNING: WANDB_API_KEY not found in environment.")
            print("W&B logging will be disabled unless you set it in SageMaker environment.")
        estimator_args['environment'] = {'WANDB_API_KEY': wandb_api_key}

    # Create estimator
    estimator = PyTorch(**estimator_args)

    # =========================================================================
    # Print Configuration
    # =========================================================================
    print("\n" + "="*80)
    print("🚀 Launching SageMaker Training Job")
    print("="*80)
    print(f"\n📊 Infrastructure:")
    print(f"   Instance Type:    {args.instance_type}")
    print(f"   Instance Count:   {args.instance_count}")
    print(f"   Volume Size:      {args.volume_size} GB")
    print(f"   Spot Training:    {args.spot_instances}")
    if args.spot_instances:
        print(f"   Max Run Time:     {args.max_run_hours} hours")
        print(f"   Max Wait Time:    {args.max_wait_hours} hours")
        print(f"   Checkpoint S3:    {args.s3_output}/checkpoints")

    print(f"\n📁 Data & Model:")
    print(f"   Dataset (S3):     {args.s3_dataset}")
    print(f"   Output (S3):      {args.s3_output}")
    print(f"   Model:            {args.model_path}")
    print(f"   Mode:             {args.mode}")

    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs:           {args.num_epochs}")
    print(f"   Batch Size:       {args.batch_size}")
    print(f"   Learning Rate:    {args.learning_rate}")
    print(f"   LoRA:             {args.use_lora}")
    if args.use_lora:
        print(f"   LoRA Rank:        {args.lora_r}")
    print(f"   FP16:             {args.fp16}")
    print(f"   BF16:             {args.bf16}")

    if args.wandb_project:
        print(f"\n📈 Logging:")
        print(f"   W&B Project:      {args.wandb_project}")

    print("\n" + "="*80 + "\n")

    # =========================================================================
    # Launch Training Job
    # =========================================================================
    try:
        # Launch job (wait=False allows SSO to expire without affecting training)
        estimator.fit({'training': args.s3_dataset}, wait=not args.no_wait)

        if args.no_wait:
            # Job launched successfully, print info and exit
            print("\n" + "="*80)
            print("🚀 Training job launched successfully!")
            print("="*80)
            print(f"\n📋 Job Info:")
            print(f"   Job Name:         {estimator.latest_training_job.name}")
            print(f"   Output Path:      {args.s3_output}")
            if args.spot_instances:
                print(f"   Checkpoints:      {args.s3_output}/checkpoints")

            print(f"\n📊 Monitor job:")
            print(f"   SageMaker Console: https://console.aws.amazon.com/sagemaker/home#/jobs/{estimator.latest_training_job.name}")
            print(f"   CloudWatch Logs:   https://console.aws.amazon.com/cloudwatch/home")

            print(f"\n💡 Check status later (after re-authenticating SSO):")
            print(f"   aws sagemaker describe-training-job --training-job-name {estimator.latest_training_job.name}")
            print()
        else:
            # Waited for completion
            print("\n" + "="*80)
            print("✅ Training job completed successfully!")
            print("="*80)
            print(f"\n📦 Outputs:")
            print(f"   Job Name:         {estimator.latest_training_job.name}")
            print(f"   Model Artifact:   {args.s3_output}/{estimator.latest_training_job.name}/output/model.tar.gz")
            if args.spot_instances:
                print(f"   Checkpoints:      {args.s3_output}/checkpoints")
            print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Training job interrupted by user")
        print(f"Job Name: {estimator.latest_training_job.name}")
        print(f"You can monitor the job in the SageMaker console.")
        sys.exit(1)

    except Exception as e:
        print("\n" + "="*80)
        print("❌ Training job failed")
        print("="*80)
        print(f"\nError: {e}")
        print(f"\nCheck CloudWatch logs for details:")
        print(f"https://console.aws.amazon.com/cloudwatch/home")
        sys.exit(1)


if __name__ == '__main__':
    main()
