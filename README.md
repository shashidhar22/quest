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

## Training models:

The script `scripts/train.py` can be used to train multiple models. The model configurations for each type of model is defined in `quest/config.py`. Currently, `train.py` can be used to train LSTM and Bidirectional LSTMs. 

The example show how to create a toy dataset and train an LSTM. 

```{python}
# Generate toy dataset
python scripts/generate_dataset.py -i <path_to_input_database> -o <output_path> -t <temp_path> -p <percentage of input data>
# Train LSTM
PTHONPATH=$(pwd) torchrun --nproc_per_node=1 --nnodes=1 scripts/train.py -d <path to dataset> -m lstm
```

To run a hyperparameter sweep use the following command, as always the configuration for the hyperparameter sweep can be found in `scripts/config.py`
```{python}
# Generate toy dataset
python scripts/generate_dataset.py -i <path_to_input_database> -o <output_path> -t <temp_path> -p <percentage of input data>
# Run hypterparameter sweep 
PTHONPATH=$(pwd) python scripts/train.py -d <path to dataset> -m lstm -s 
```

**Note**: Here we use WandDB to perform the hyperparameter sweeps. It is recommended that this performed on a single node without torchrun.

**Note**: You will need to setup the following environment variables to get the torchrun working `MASTER_ADDR, MASTER_PORT, NCCL_SOCKET_IFNAME, NCCL_DEBUG, TORCH_NCCL_BLOCKING_WAIT, NCCL_P2P_LEVEL, TORCH_DISTRIBUTED_DEBUG`. Add all the environment variables to a `.env` file in the quest project folder and `source` it when launching the script the first time 