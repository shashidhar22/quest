#!/bin/bash

# === Configuration ===
# Environment variables for distributed training
# Load the .env file
set -a            # Automatically export all variables
source path/to/.env
set +a            # Turn off automatic export



# === Usage and Default Values ===
# Default values for optional arguments
output="data/model"
temp_path="temp"
percentage=100
batch_size=128
seq_length=6
embedding_dim=16
hidden_dim=128
num_layers=1
learning_rate=0.001
num_epochs=20

# Display usage instructions
usage() {
    echo "Usage: $0 -i <input_dataset> -o <output_folder> [options]"
    echo "Options:"
    echo "  -p <percentage>          Percentage of sequences to use (default: $percentage)"
    echo "  -b <batch_size>          Batch size for training (default: $batch_size)"
    echo "  -s <seq_length>          Length of input sequences (default: $seq_length)"
    echo "  -m <embedding_dim>       Dimension of embedding layer (default: $embedding_dim)"
    echo "  -n <hidden_dim>          Hidden layer size (default: $hidden_dim)"
    echo "  -l <num_layers>          Number of LSTM layers (default: $num_layers)"
    echo "  -t <learning_rate>       Learning rate for training (default: $learning_rate)"
    echo "  -e <num_epochs>          Number of training epochs (default: $num_epochs)"
    exit 1
}

# === Argument Parsing ===
if [ "$#" -lt 2 ]; then
    usage
fi

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input) input="$2"; shift ;;
        -o|--output) output="$2"; shift ;;
        -t|--temp_path) temp_path="$2"; shift ;;
        -p|--percentage) percentage="$2"; shift ;;
        -b|--batch_size) batch_size="$2"; shift ;;
        -s|--seq_length) seq_length="$2"; shift ;;
        -m|--embedding_dim) embedding_dim="$2"; shift ;;
        -n|--hidden_dim) hidden_dim="$2"; shift ;;
        -l|--num_layers) num_layers="$2"; shift ;;
        -r|--learning_rate) learning_rate="$2"; shift ;;
        -e|--num_epochs) num_epochs="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check required arguments
if [ -z "$input" ]; then
    echo "Error: Input dataset (-i) is required."
    usage
fi

# === Dataset Preparation ===
echo "Generating dataset from $input..."

python scripts/generate_dataset.py \
    -i "$input" \
    -o "$output" \
    -t "$temp_path" \
    -p "$percentage" \
    -b "$batch_size" \
    -s "$seq_length" \
    -m "$embedding_dim" \
    -n "$hidden_dim" \
    -l "$num_layers" \
    -r "$learning_rate" \
    -e "$num_epochs"

# Construct output path
base_name="${input##*/}"    # Extract file name
base_name="${base_name%.*}" # Remove file extension
dataset_path="${temp_path}/${base_name}_${percentage}_P.pkl"


# === Run Training ===
echo "Starting LSTM training with dataset: $dataset_path..."

torchrun --nproc_per_node=4 --nnodes=1 scripts/run_lstm.py -d "$dataset_path" 

echo "LSTM training script executed successfully."
