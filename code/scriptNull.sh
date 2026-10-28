#!/bin/bash

# ==========================================
# Paths and Directories
# ==========================================
# Paths and Directories
# ==========================================
# Update these paths as needed
#SUBMISSIONS_DIR="/teo_fs_fast/projects/GENOVA/CaloChallenge/Dataset_2/submissions"
REFERENCE_FILE="/teo_fs_fast/projects/GENOVA/CaloChallenge/Dataset_1_pions/data/dataset_1_pions_1.hdf5"
OUTPUT_DIR="/teo_fs_fast/projects/GENOVA/CaloChallenge/Dataset_1_pions/train_null"
LOG_DIR="/teo_fs_fast/projects/GENOVA/CaloChallenge/Dataset_1_pions/logs"

# ==========================================
# Python Script Parameters
# ==========================================
PYTHON_SCRIPT="evaluate_hypothesis.py" 
PARTICLE_TYPE="pion"
DATASET="1"
MODE="all"
# Using arrays for arguments that take multiple values
METRICS=("ks" "swd" "sks" "mmd")
BATCHSIZES=("100" "500" "1000" "5000" "10000" "20000")
NITER="1000"
NSLICES="200"
MIN_ENERGY="10"
SMALL_SAMPLE="1"
CPU="/CPU:2"  # 0,1,2:teogpu01,02,03
GPU="/device:GPU:2"  # 0,1,2:teogpu01,02,03
# ==========================================
# Setup Environment & Logging
# ==========================================
if [[ "$CONDA_DEFAULT_ENV" != "genova_falkon" ]]; then
    echo "Activating conda environment: genova_falkon"
    eval "$(conda shell.bash hook)"
    conda activate genova_falkon
fi

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/evaluation_log_$(date +%Y%m%d_%H%M%S).log"

echo "Starting evaluation at $(date)" | tee -a "$LOG_FILE"
#echo "Reading submissions from: $SUBMISSIONS_FILE" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "------------------------------------------------" | tee -a "$LOG_FILE"


# Extract filename without the .hdf5 extension to use as folder name
MODEL_NAME=$(basename "$REFERENCE_FILE" .hdf5)

# Define and create the model-specific output directory
MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME+"Null"}"
mkdir -p "$MODEL_OUTPUT_DIR"

echo "Running null evaluation on: $MODEL_NAME" | tee -a "$LOG_FILE"
echo "Saving results to: $MODEL_OUTPUT_DIR" | tee -a "$LOG_FILE"



python "$PYTHON_SCRIPT" \
    --reference_file "$REFERENCE_FILE" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --particle_type "$PARTICLE_TYPE" \
    --dataset "$DATASET" \
    --mode "$MODE" \
    --metrics "${METRICS[@]}" \
    --batchsizes "${BATCHSIZES[@]}" \
    --niter "$NITER" \
    --overwrite \
    --cpu "$CPU" \
    --gpu "$GPU" \
    --nslices "$NSLICES" \
    --min_energy "$MIN_ENERGY" \
    --small_sample "$SMALL_SAMPLE" \
    2>&1 | tee -a "$LOG_FILE" \
    
echo "Finished: $MODEL_NAME" | tee -a "$LOG_FILE"
echo "------------------------------------------------" | tee -a "$LOG_FILE"


echo "All evaluations completed at $(date)" | tee -a "$LOG_FILE"