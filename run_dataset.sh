#!/bin/bash
# Multi-dataset runner script for TabSynth
# Usage: ./run_dataset.sh <dataset_name> [--train] [--sample] [--eval]
# Example: ./run_dataset.sh adult --train --sample --eval

set -e

# Activate virtual environment
source .venv/bin/activate

DATASET=${1:-adult}
shift
FLAGS=${@:---train --sample --eval}

CONFIG_PATH="src/tabsynth/exp/$DATASET/config.toml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found for dataset '$DATASET'"
    echo "Available datasets:"
    ls -d src/tabsynth/exp/*/ | xargs -n1 basename
    exit 1
fi

echo "=========================================="
echo "Running pipeline for dataset: $DATASET"
echo "Config: $CONFIG_PATH"
echo "Flags: $FLAGS"
echo "=========================================="

python src/tabsynth/scripts/pipeline.py --config "$CONFIG_PATH" $FLAGS

echo "=========================================="
echo "Pipeline completed for dataset: $DATASET"
echo "Results saved to: outputs/src/tabsynth/exp/$DATASET/check/"
echo "=========================================="
