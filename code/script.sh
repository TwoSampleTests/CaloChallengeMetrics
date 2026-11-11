#!/bin/bash

# ==========================================
# Paths and Directories
# ==========================================
# Update these paths as needed
SUBMISSIONS_DIR="../results/Dataset_3/submissions"
REFERENCE_FILE="../results/Dataset_3/data/dataset_3_2.hdf5"  #file_1 is for training, file _2 is for metric evaluation for ds3 file _full is the merge of 3 and 4 (split of eval data)
OUTPUT_DIR="../results/Dataset_3/alternative_hypotheses"
LOG_DIR="../results/Dataset_3/logs"

# ==========================================
# Python Script Parameters
# ==========================================
PYTHON_SCRIPT="evaluate_hypothesis.py" 
PARTICLE_TYPE="electron"  #pion, photon, electron
DATASET="3"  # 1, 2, or 3
MODE="all"
# Using arrays for arguments that take multiple values
METRICS=("ks" "swd" "sks")  #"ks" "swd" "sks" "mmd"
BATCHSIZES=("20000")   #"100" "500" "1000" "5000" "10000" "20000"
NITER="1000"
NSLICES="200"  # 200
MIN_ENERGY="0.01515"     # minimal readout per voxel, ds1: 10, ds2/3: 0.5 keV / 0.033 = 0.01515 scaling factor
SMALL_SAMPLE="1"
CPU="/CPU:0"  # choose CPU from available 
GPU="/device:GPU:0"  # choose GPU from available

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
echo "Reading submissions from: $SUBMISSIONS_DIR" | tee -a "$LOG_FILE"
echo "Reading reference from: $REFERENCE_FILE" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "------------------------------------------------" | tee -a "$LOG_FILE"

# ==========================================
# Execution Loop
# ==========================================
# Iterate over all .hdf5 files in the specified submissions directory
for SUBMISSION_FILE in "$SUBMISSIONS_DIR"/*.hdf5; do
    # Safety check in case the directory is empty
    if [[ ! -f "$SUBMISSION_FILE" ]]; then
        echo "No .hdf5 files found in $SUBMISSIONS_DIR" | tee -a "$LOG_FILE"
        break
    fi

    # Extract filename without the .hdf5 extension to use as folder name
    MODEL_NAME=$(basename "$SUBMISSION_FILE" .hdf5)
    
    # Define and create the model-specific output directory
    MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME}"
    mkdir -p "$MODEL_OUTPUT_DIR"

    echo "Running evaluation on: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "Saving results to: $MODEL_OUTPUT_DIR" | tee -a "$LOG_FILE"
#TF_FORCE_GPU_ALLOW_GROWTH=true\           #to better allocate memory and avoid OOM issues 
    TF_FORCE_GPU_ALLOW_GROWTH=true python "$PYTHON_SCRIPT" \
        --reference_file "$REFERENCE_FILE" \
        --submission_file "$SUBMISSION_FILE" \
        --output_dir "$MODEL_OUTPUT_DIR" \
        --particle_type "$PARTICLE_TYPE" \
        --dataset "$DATASET" \
        --mode "$MODE" \
        --cpu "$CPU" \
        --gpu "$GPU" \
        --overwrite \
        --metrics "${METRICS[@]}" \
        --batchsizes "${BATCHSIZES[@]}" \
        --niter "$NITER" \
        --nslices "$NSLICES" \
        --min_energy "$MIN_ENERGY" \
        --small_sample "$SMALL_SAMPLE" 2>&1 | tee -a "$LOG_FILE"
        
    echo "Finished: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "------------------------------------------------" | tee -a "$LOG_FILE"
done

echo "All evaluations completed at $(date)" | tee -a "$LOG_FILE"