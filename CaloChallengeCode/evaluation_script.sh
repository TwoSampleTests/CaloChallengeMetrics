echo "Activating conda environment: genova_falkon"
eval "$(conda shell.bash hook)"
conda activate genova_falkon


for DATASET in "1-pions" "1-photons" "2" "3"; do 
    
    SUBMISSIONS_DIR="/teo_fs_fast/projects/GENOVA/CaloChallenge/CaloChallengeMetrics/CaloChallengeCode/dataset_copy/${DATASET}/submissions"
    
    if [[ "$DATASET" == "1-pions" ]]; then        
        REFERENCE_FILE="/teo_fs_fast/projects/GENOVA/CaloChallenge/CaloChallengeMetrics/CaloChallengeCode/dataset_copy/${DATASET}/data/dataset_1_pions_2.hdf5"
    fi
    if [[ "$DATASET" == "1-photons" ]]; then
        REFERENCE_FILE="/teo_fs_fast/projects/GENOVA/CaloChallenge/CaloChallengeMetrics/CaloChallengeCode/dataset_copy/${DATASET}/data/dataset_1_photons_2.hdf5"
    fi
    if [[ "$DATASET" == "2" ]]; then
        REFERENCE_FILE="/teo_fs_fast/projects/GENOVA/CaloChallenge/CaloChallengeMetrics/CaloChallengeCode/dataset_copy/${DATASET}/data/dataset_2_2.hdf5"
    fi
    if [[ "$DATASET" == "3" ]]; then
        REFERENCE_FILE="/teo_fs_fast/projects/GENOVA/CaloChallenge/CaloChallengeMetrics/CaloChallengeCode/dataset_copy/${DATASET}/data/dataset_3_2.hdf5"
    fi

    for SUBMISSION_FILE in "$SUBMISSIONS_DIR"/*.hdf5; do
        # Safety check in case the directory is empty
        if [[ ! -f "$SUBMISSION_FILE" ]]; then
            echo "No .hdf5 files found in $SUBMISSIONS_DIR"
            break
        fi

        python evaluate.py\
            -i "$SUBMISSION_FILE"\
            -r "$REFERENCE_FILE"\
            -m "all"\
            -d "${DATASET}"\
            --output_dir "/teo_fs_fast/projects/GENOVA/CaloChallenge/CaloChallengeMetrics/CaloChallengeCode/evaluation_results"
    done
done

echo "All evaluations completed at $(date)"