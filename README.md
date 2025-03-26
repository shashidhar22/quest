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

## Format sequencing data and public databases:

The current training dataset includes sequencing data generated from various platforms across 32 published studies and 3 internal studies. Additionally, we have included the following public databases: VDJdb, Tcrdb, McPAS-TCR, iReceptor, IEDB, and CEDAR. There is no one standard format for the sequencing data and public databases. Therefore, we have provided scripts to format the data into a standard format. This is a one time pre-processing task and will be deprecated in the future (once we have released the dataset). The script below generates two standard formats, MRI (minimal required information) and Sequence format. 

The MRI table contains the followig fields:
`'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene', 'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'peptide', 'mhc_one', 'mhc_two', 'sequence', 'repertoire_id', 'study_id', 'category', 'molecule_type', 'host_organism', 'source'`

The Sequence table contains the following fields:
`'source', 'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene', 'trb', 'trbd_gene', 'trbj_gene','trbv_gene', 'peptide', 'mhc_one', 'mhc_two', 'sequence'`

The script `format_dataset.py` can be used to format the data. The script requires the following arguments:
- `--config_path`: Path to a YAML file that difnes paths for the sequencind data and public databases. 

A configuration file is provided in the `config` folder. The configuration file defines the paths to the sequencing data and public databases. 

The `header_config.yaml` file defines the column names for the sequencing data. 

To format the data, run the following command:

```{bash}
python format_dataset.py --config_path config/header_config.yaml
```

**Note**: The current dataset includes over 27000 sequencing runs and 6 public databases. The formatting script will take a long time to run. And it is recommended to run it on a cluster with adequate storage and memory.

## Generate HuggingFace datasets:

We use the `datasets` library from HuggingFace to load the data. The script `dataloader.py` can be used to generate the HuggingFace datasets. The script requires the following arguments:

- `-i`: Path to the formatted data in `.parquet` format. Multiple paths can be provided by separating them with a space.
- `-o`: Path to the output directory where the HuggingFace datasets will be stored.
- `-p`: Percentage of the dataset that will be used to generate the HuggingFace dataset. This is useful for testing the script on a smaller dataset.
- `-m`: Model type. The script will generate the HuggingFace dataset based on the model type. The model types include `rnn`, `transformer`.
- `-v`: Vocabulary size parameter for BPE tokenization. This parameter defines the vocabulary size for the BPE tokenization.
- `-s`: Sequence length parameter for RNN models. This parameter defines the maximum sequence length for the RNN models.
- `--build_tokenizer`: If this flag is set, the script will build a tokenizer and save it in the output directory. The tokenizer is used to tokenize the sequences before training the models.

Other optional arguments can be found by running the following command:

```{bash}
python dataloader.py -h
```

To generate a HuggingFace dataset for the LSTM model, run the following command:

```{bash}
python dataloader.py -i <path to formatted data> -o <output directory> -p 0.01 -m rnn -s 100 -v 100 --build_tokenizer
```

**Note**: While training BERT it is recommended to use the default BERT tokenizer. The script will generate the HuggingFace dataset using the default BERT tokenizer. However, `dataloader.py` can be used to add additional tokens to the BERT tokenizer. The script requires the following arguments


## Training models:

The script `scripts/train.py` can be used to train multiple models. The model configurations for each type of model is defined in `quest/config.py`. Currently, `train.py` can be used to train LSTM, Bidirectional LSTMs, Transformers, and BERT models. The script requires the following arguments:
 - `-d`: Path to the HuggingFace dataset.
 - `-m`: Model type. The model types include `lstm`, `bilstm`, `transformer`, `bert`.
 - `-t`: Path to the tokenizer. The tokenizer is used to tokenize the sequences before training the models.
 - `-c`: Path to create model checkpoints.
 - `-s`: If this flag is set, the script will perform a hyperparameter sweep. The hyperparameter sweep configurations are defined in `quest/config.py`.
 - `-i`: Wandb Sweep ID. This is used to resume a hyperparameter sweep.
 - `-r`: Resume training. If this flag is set, the script will resume training from the last checkpoint.

The below command can be used to train an LSTM model on a single node with 4 GPUs. The output and error logs are stored in `output.log` and `error.log` respectively:

```{bash}
PYTHONPATH=$(pwd) torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  scripts/train.py \
  -d <dataset path> \
  -m lstm \
  -c <checkpoint path> \
  --tokenizer <tokenizer path> > output.log 2> error.log
```

**Note**: Here we use WandDB to perform the hyperparameter sweeps. The script `create_sweep.py` can be used to create a sweep. The script requires the following arguments:
- `-m`: Model type. The model types include `lstm`, `bilstm`, `transformer`, `bert`.
- `-t`: Path to the tokenizer. The tokenizer is used to tokenize the sequences before training the models.
- `-d`: Path to the HuggingFace dataset.

The script will create a sweep and print the sweep ID. The sweep ID can be used to resume the sweep using the `train.py` script.

```{bash}
PYTHONPATH=$(pwd) torchrun --nproc_per_node=4 --nnodes=1 scripts/train.py -m lstm -i $SWEEP_ID -s
```

**Note**: You will need to setup the following environment variables to get the torchrun working `MASTER_ADDR, MASTER_PORT, NCCL_SOCKET_IFNAME, NCCL_DEBUG, TORCH_NCCL_BLOCKING_WAIT, NCCL_P2P_LEVEL, TORCH_DISTRIBUTED_DEBUG, WANDB_API_KEY, CUDA_VISIBLE_DEVICES`. Add all the environment variables to a `.env` file in the quest project folder and `source` it when launching the script the first time 