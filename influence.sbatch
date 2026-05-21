#!/bin/bash
#SBATCH --job-name="inf1"
#SBATCH --container-image=kalinchukd/inf_env:latest
#SBATCH --container-mount-home
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24  
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:00:00 
#SBATCH --container-workdir=/srv/home/users/kalinchukd23cs/InfEstimation_benchmark

python3 --version

IFS=',' read -r MODEL_NAME VAL_FLAG MAX_LENGTH BATCH_SIZE LOGGING_STEP EPOCHS LORA_R LORA_ALPHA TARGET_LAYER LOAD_8BIT < finetune_params.txt

echo "MODEL_NAME = $MODEL_NAME"
echo "MAX_LENGTH = $MAX_LENGTH"
echo "EPOCHS = $EPOCHS"


while read -r HVP_METHOD; do
    [ -z "$HVP_METHOD" ] && continue

    while read -r DATASET_NAME; do
        [ -z "$DATASET_NAME" ] && continue

        echo "DATASET_NAME = $DATASET_NAME"
        echo "HVP_METHOD = $HVP_METHOD"

        python3 influence.py \
            --model "$MODEL_NAME" \
            --dataset "$DATASET_NAME" \
            --epochs "$EPOCHS" \
            --hvp_cal "$HVP_METHOD" \
            --inf_args "$INF_ARGS" \
            --max_length "$MAX_LENGTH"

    done < dataset_settings.txt

done < hvp_settings.txt


python3 -c "from postprocess_utils import plot_all_acc_cov; plot_all_acc_cov()"