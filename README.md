# Quantitative Understanding of Epitope Specificity in T-cells (QUEST)

## Requirements

To use the distributed training and evaluation scripts, ensure the following Python packages are installed:

- torch (>=2.7.0)
- torchvision
- torchaudio
- cudatoolkit (if using GPU)
- datasets
- transformers
- tokenizers
- accelerate
- wandb
- evaluate
- ray[data,train,tune,serve]
- peft
- tqdm
- scikit-learn
- s3fs

You can install the core requirements with:

```bash
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers datasets accelerate wandb evaluate tqdm scikit-learn "ray[default]" peft boto3 s3fs
```

---

## Running Scripts with Ray

Ray enables distributed training and evaluation across multiple nodes or GPUs. You can use Ray both locally (on a single machine) or on a cluster (e.g., AWS, using a YAML config).

### 1. Launching a Ray Cluster

**From a YAML config (cloud/cluster):**

```bash
ray up <path-to-ray-cluster-yaml>
```

This will start the Ray head and worker nodes as defined in your YAML file.

**Locally (single machine):**

```bash
ray start --head --port=6379
```

### 2. Submitting a Script to the Ray Cluster

**Using `ray submit` (for remote clusters):**

```bash
ray submit <path-to-ray-cluster-yaml> scripts/<your_script>.py -- <script-args>
```

- The `--` separates Ray arguments from your script's arguments.
- Example:

```bash
ray submit <path-to-ray-cluster-yaml> scripts/ray_train.py -- --dataset <DATASET_PATH> --model lstm --use-ray --num_workers 4
```

**Using `ray job submit` (Ray 2.x+):**

If your cluster is already running, you can submit a job from your local machine:

```bash
ray job submit --address='ray://<head-node-ip>:10001' -- python scripts/<your_script>.py <script-args>
```

### 3. Monitoring Ray Jobs

- **Ray Dashboard:**
  - Access the Ray dashboard at `http://<head-node-ip>:8265` (or the port specified in your cluster config).
  - **If using AWS SSO login and SSM:**
    - You can port forward the dashboard using AWS SSM:
      ```bash
      aws ssm start-session \
        --target <instance-id> \
        --document-name AWS-StartPortForwardingSession \
        --parameters '{"portNumber":["8265"], "localPortNumber":["8265"]}'
      ```
    - Replace `<instance-id>` with your Ray head node's EC2 instance ID. This will forward the remote dashboard port 8265 to your local port 8265.
  - **If using SSH:**
    - You can use SSH port forwarding if needed:
      ```bash
      ssh -L 8265:localhost:8265 <ssh-username>@<head-node-ip>
      ```
- **Command-line:**
  - Check cluster status:
    ```bash
    ray status
    ```
  - List running jobs:
    ```bash
    ray job list
    ```

### 4. Stopping the Ray Cluster

```bash
ray down <path-to-ray-cluster-yaml>
```

---

## Distributed Training and Evaluation Scripts

### 1. Ray-based Evaluation (`ray_evaluator.py`)

This script performs distributed evaluation of a pre-trained Masked Language Model (MLM) using Ray.

**Example usage:**

```bash
python scripts/ray_evaluator.py \
  --raw_dataset_dir <RAW_DATASET_DIR> \
  --model_name <MODEL_NAME_OR_PATH> \
  --batch_size 128 \
  --mlm_prob 0.15 \
  --wandb_project <WANDB_PROJECT_NAME>
```

- `--raw_dataset_dir`: Path to the raw HuggingFace dataset directory.
- `--model_name`: Model name or path (e.g., 'Rostlab/prot_bert_bfd').
- `--batch_size`: Batch size for evaluation.
- `--mlm_prob`: Masked LM probability.
- `--wandb_project`: (Optional) W&B project name for logging.
- `--schema_path`: (Optional) Path to a JSON file with masking schema.
- `--test`: (Optional) Run on a small subset for testing.

### 2. Ray-based Fine-tuning (`ray_fine_tune.py`)

This script performs distributed fine-tuning of a pre-trained MLM or custom model using Ray and HuggingFace Trainer.

**Example usage:**

```bash
python scripts/ray_fine_tune.py \
  --config <CONFIG_JSON_PATH> \
  --num_workers 8 \
  --s3_output_path <S3_OUTPUT_PATH>
```

- `--config`: Path to a JSON config file (see template below).
- `--num_workers`: Number of Ray workers (GPUs).
- `--s3_output_path`: (Optional) S3 path to sync the best model after training.
- `--test`: (Optional) Run in test mode with a smaller dataset.

### 3. Ray-based Training for Custom Models (`ray_train.py`)

This script trains custom LSTM/Transformer models using Ray for distributed training.

**Example usage:**

```bash
python scripts/ray_train.py \
  --dataset <DATASET_PATH> \
  --model <MODEL_TYPE> \
  --tokenizer <TOKENIZER_PATH> \
  --use-ray \
  --num_workers 4
```

- `--dataset`: Path to the HuggingFace dataset.
- `--model`: Model type ('lstm', 'bilstm', 'transformer').
- `--tokenizer`: Path to the tokenizer file.
- `--use-ray`: Enable Ray distributed training.
- `--num_workers`: Number of Ray workers (GPUs).

### 4. Ray-based Data Processing (`ray_datawriter.py`)

This script processes raw Parquet files and creates tokenized datasets for training. It supports both HuggingFace models (BERT, ProtBERT, ESM, LLaMA) and custom models (LSTM, Transformer).

**Example usage (local):**

```bash
# Process data for HuggingFace models (BERT, ProtBERT, ESM, LLaMA)
python scripts/ray_datawriter.py \
  --path "data/*.parquet" \
  --model-name protbert \
  --output-raw "processed_data" \
  --max-len 1024 \
  --use-ray \
  --num-workers 4

# Process data for custom models (LSTM, Transformer)
python scripts/ray_datawriter.py \
  --path "data/*.parquet" \
  --model-name lstm \
  --output-raw "processed_data" \
  --max-len 1024 \
  --bpe-vocab 200 \
  --truncate-long \
  --use-ray \
  --num-workers 4
```

**Example usage (AWS with Ray cluster):**

```bash
# Submit to Ray cluster
ray submit <path-to-ray-cluster-yaml> scripts/ray_datawriter.py -- \
  --path "s3://your-bucket/data/*.parquet" \
  --model-name protbert \
  --output-raw "s3://your-bucket/processed_data" \
  --max-len 1024 \
  --use-ray \
  --num-workers 8 \
  --s3-key <your-aws-key> \
  --s3-secret <your-aws-secret>

# Or using ray job submit
ray job submit --address='ray://<head-node-ip>:10001' -- \
  python scripts/ray_datawriter.py \
  --path "s3://your-bucket/data/*.parquet" \
  --model-name lstm \
  --output-raw "s3://your-bucket/processed_data" \
  --max-len 1024 \
  --bpe-vocab 200 \
  --truncate-long \
  --use-ray \
  --num-workers 8 \
  --s3-key <your-aws-key> \
  --s3-secret <your-aws-secret>
```

**Key parameters:**
- `--path`: Path to Parquet files (supports glob patterns and S3 paths)
- `--model-name`: Model type ('bert', 'protbert', 'esm', 'llama', 'lstm', 'transformer')
- `--output-raw`: Output directory for processed dataset
- `--max-len`: Maximum sequence length (default: 1024)
- `--bpe-vocab`: Vocabulary size for BPE tokenizer (custom models only, default: 200)
- `--truncate-long`: Truncate long sequences instead of padding
- `--use-ray`: Enable Ray distributed processing
- `--num-workers`: Number of Ray workers
- `--s3-key`, `--s3-secret`: AWS credentials for S3 access

### 5. Running Locally with Ray (Non-Distributed)

You can use Ray for orchestration on a single machine without launching a distributed cluster. This is useful for debugging or running on a single GPU/CPU, but still benefits from Ray's job management and logging.

**Example: Run with a single Ray worker (local, non-distributed):**

```bash
python scripts/ray_train.py \
  --dataset <DATASET_PATH> \
  --model <MODEL_TYPE> \
  --tokenizer <TOKENIZER_PATH> \
  --use-ray \
  --num_workers 1
```

Or for fine-tuning:

```bash
python scripts/ray_fine_tune.py \
  --config <CONFIG_JSON_PATH> \
  --num_workers 1
```

Or for data processing:

```bash
python scripts/ray_datawriter.py \
  --path "data/*.parquet" \
  --model-name protbert \
  --output-raw "processed_data" \
  --max-len 1024 \
  --use-ray \
  --num-workers 1
```

- The `--use-ray` and `--num_workers 1` flags ensure Ray is used, but only a single worker/process is launched.
- No Ray cluster or YAML config is needed for this mode; Ray will run locally.

---

## Ray Cluster YAML Template

Below is a template for a Ray cluster configuration YAML file for AWS, based on a real working example. Replace all placeholder values (in angle brackets) with your actual configuration. **Do not use real secrets or tokens in your config.**

```yaml
# ray_cluster_template.yaml
cluster_name: <your-cluster-name>

provider:
  type: aws
  region: <aws-region>
  aws_profile: <aws-profile-name>
  use_internal_ips: true

auth:
  ssh_user: <ssh-username>
  ssh_proxy_command: |
    aws ssm start-session --target `aws ec2 describe-instances --filters 'Name=tag:Name,Values=<head-node-tag>' 'Name=instance-state-name,Values=running' --query 'Reservations[*].Instances[*].InstanceId' --output text | head -n1` --document-name AWS-StartSSHSession --parameters 'portNumber=22'

available_node_types:
  ray.head.default:
    node_config:
      InstanceType: <head-instance-type>
      ImageId: <ami-id>
      SubnetId: <subnet-id>
      KeyName: <key-name>
      IamInstanceProfile:
        Arn: <iam-instance-profile-arn>
      InstanceMarketOptions:
        MarketType: spot
      BlockDeviceMappings:
        - DeviceName: /dev/sda1
          Ebs:
            VolumeSize: 100
            VolumeType: gp3
            DeleteOnTermination: true
  ray.worker.default:
    min_workers: <min-workers>
    max_workers: <max-workers>
    node_config:
      InstanceType: <worker-instance-type>
      ImageId: <ami-id>
      SubnetId: <subnet-id>
      KeyName: <key-name>
      IamInstanceProfile:
        Arn: <iam-instance-profile-arn>
      InstanceMarketOptions:
        MarketType: spot
      BlockDeviceMappings:
        - DeviceName: /dev/sda1
          Ebs:
            VolumeSize: 100
            VolumeType: gp3
            DeleteOnTermination: true

head_node_type: ray.head.default
max_workers: <max-workers>

setup_commands:
  - |
    if ! command -v python &> /dev/null; then
      sudo ln -s $(which python3) /usr/bin/python
    fi
  # Format and mount NVMe to /scratch if not already formatted
  - |
    sudo chown <ssh-username>:<ssh-username> /opt/dlami/nvme
    export HF_CACHE=/opt/dlami/nvme/hf_cache
    export PATH=/home/<ssh-username>/.local/bin:$PATH
  # Install git if needed
  - |
    if ! command -v git &> /dev/null; then
      sudo apt update
      sudo apt install -y git
    fi
  # Clone your repo if not already present
  - |
    if [ ! -d "/home/<ssh-username>/<repo-name>" ]; then
      git clone <your-repo-url> /home/<ssh-username>/<repo-name>
    fi
  - |
    cd /home/<ssh-username>/<repo-name>
    ln -s /opt/dlami/nvme data
  # Install AWS CLI if needed
  - |
    if ! command -v aws &> /dev/null; then
      sudo apt update
      sudo apt install -y awscli
    fi
  # Copy data from S3 to /scratch (optional)
  # - aws s3 cp --recursive <s3-bucket-path> /home/<ssh-username>/<repo-name>/data/<dataset-dir> --quiet
  - pip install -q --upgrade pip
  - pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  - pip install -q transformers datasets accelerate wandb evaluate tqdm scikit-learn s3fs tokenizers "ray[data,train,tune,serve]" peft
  # wandb login (optional)
  # - wandb login <your-wandb-api-key>

file_mounts: {}
initialization_commands: []
```

---

## Fine-tuning Config JSON Template

Below is a template for the fine-tuning configuration JSON file used by `ray_fine_tune.py`. Replace all placeholder values (in angle brackets) with your actual configuration.

```json
{
  "model_key": "protbert",
  "dataset_path": "<path-to-hf-dataset>",
  "checkpoint_path": "<output-checkpoint-dir>",
  "num_epochs": 3,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "mlm_prob": 0.15,
  "wandb_project": "<wandb-project-name>",
  "tokenizer_path": "<path-to-tokenizer>",
  "gradient_checkpointing": true,
  "use_lora": false,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_target_modules": ["query", "value"],
  "lora_dropout": 0.05,
  "early_stop": true,
  "early_stopping_patience": 3,
  "s3_output_path": "<s3-bucket-path>"
}
```

--- 