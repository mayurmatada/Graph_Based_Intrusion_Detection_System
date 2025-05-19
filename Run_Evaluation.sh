#!/bin/bash

set -e  # Exit on error

# -------------------------
# CONFIG PATHS
# -------------------------
CHECKPOINT_DIR="Parameter_Databases/Checkpoints/"
CHECKPOINT_FILE="$CHECKPOINT_DIR/trial_0.pt"

# -------------------------
# CHECK FOR REQUIRED FILES
# -------------------------
echo "Checking for evaluation prerequisites..."

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "Error: Checkpoint files not found at $CHECKPOINT_FILE"
    echo "Please train a model before running evaluation."
    exit 1
fi

# -------------------------
# RUN EVALUATION
# -------------------------
echo "Running evaluation script..."
python -m src.util.eval

echo "Evaluation completed."
