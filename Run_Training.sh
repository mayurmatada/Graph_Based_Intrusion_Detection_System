#!/bin/bash

set -e  # Exit on error

# -------------------------
# CONFIG PATHS
# -------------------------
OPTUNA_DB="Parameter_Databases/Optuna/optuna_study.db"
TB_LOG_DIR="Parameter_Databases/Tensorboard/"
CHECKPOINT_DIR="Parameter_Databases/Checkpoints/"

# -------------------------
# CLEANUP
# -------------------------
echo "Cleaning old logs..."

if [ -f "$OPTUNA_DB" ]; then
    rm "$OPTUNA_DB"
    echo "Deleted old Optuna DB"
fi

if [ -d "$TB_LOG_DIR" ]; then
    rm -rf "${TB_LOG_DIR:?}"/*
    echo "Deleted contents of TensorBoard log directory"
else
    mkdir -p "$TB_LOG_DIR"
    echo "Created TensorBoard log directory"
fi

if [ -d "$CHECKPOINT_DIR" ]; then
    rm -rf "${CHECKPOINT_DIR:?}"/*
    echo "Deleted contents of Checkpoints directory"
else
    mkdir -p "$CHECKPOINT_DIR"
    echo "Created Checkpoints directory"
fi

# -------------------------
# RUN EXPERIMENT IN BACKGROUND
# -------------------------
echo "Running training pipeline in background..."
python src/models/GNN_script.py 

TRAIN_PID=$!
echo "Training started with PID $TRAIN_PID"

# -------------------------
# WAIT FOR OPTUNA DB AND TB LOG DIR TO APPEAR
# -------------------------
echo "Waiting for Optuna DB and TensorBoard logs to appear..."

while true; do
    if [ -f "$OPTUNA_DB" ] && [ -d "$TB_LOG_DIR" ] && [ "$(ls -A $TB_LOG_DIR)" ]; then
        echo "Detected Optuna DB and TensorBoard logs."
        break
    else
        echo "Waiting... (Optuna DB exists? $( [ -f "$OPTUNA_DB" ] && echo yes || echo no ), TB log dir ready? $( [ -d "$TB_LOG_DIR" ] && [ "$(ls -A $TB_LOG_DIR)" ] && echo yes || echo no ))"
        sleep 2
    fi
done


# -------------------------
# LAUNCH BOARDS
# -------------------------
# Clean up previous TensorBoard instances
if lsof -i:6006 -t >/dev/null ; then
    echo "Killing existing TensorBoard process on port 6006"
    kill -9 $(lsof -i:6006 -t)
fi
echo "Launching TensorBoard..."
tensorboard --logdir "$TB_LOG_DIR" --port 6006 --reload_interval 5 &

echo "Launching Optuna Dashboard..."
optuna-dashboard sqlite:///$OPTUNA_DB --port 8000 &

# -------------------------
# WAIT FOR TRAINING TO FINISH
# -------------------------~
wait $TRAIN_PID
echo "Training completed."
