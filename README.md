# Quantitative Understanding of Epitope Specificity in T-cells (QUEST)

## Setup:

To run the data loaders, and training scripts please follow the setup mentioned below:

```{bash}
cd quest
ln -s <path to tcr_llm folder> data/tcr-llm
ln -s <path to temp directory for checkpoints> temp
mkdir wandb # Store wandb logs
mamba env create -f env/environment.yaml
```

Additional pre-requisites include setting up your `wandb`. We are using `wandb` to log training metrics and at this time, the scripts will not work without having `wandb` configured on your profile.

## Run LSTM:

`run.sh` is a wrapper around `generate_dataset.py` and `run_lstm.py`. To run the training script, the minimal input required is the path to the parquet sequence table found in this path `data/tcr_llm/parquet`.


The example run will create a toy dataset containing 0.2% of VDJdb. Configure a LSTM with the parameters mentioned below and execute the training across 4 GPUs using `torchrun`

```
batch_size=128
seq_length=6
embedding_dim=16
hidden_dim=128
num_layers=1
learning_rate=0.001
num_epochs=20
```

```{sh}
./run_lstm.sh -i data/tcr_llm/parquet/vdjdb_tcr_sequence.parquet -p 0.2
```

**Note**: You will need to setup the following environment variables to get the torchrun working `MASTER_ADDR, MASTER_PORT, NCCL_SOCKET_IFNAME, NCCL_DEBUG, TORCH_NCCL_BLOCKING_WAIT, NCCL_P2P_LEVEL, TORCH_DISTRIBUTED_DEBUG`