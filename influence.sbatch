#!/bin/bash
#SBATCH --job-name="inf1"
#SBATCH --container-image=kalinchukd/inf_env3:latest
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


while read -r INF_METHOD; do
    [ -z "$INF_METHOD" ] && continue

    while read -r DATASET_NAME; do
        [ -z "$DATASET_NAME" ] && continue

        echo "DATASET_NAME = $DATASET_NAME"
        echo "INF_METHOD = $INF_METHOD"

        python3 influence.py \
            --model "$MODEL_NAME" \
            --dataset "$DATASET_NAME" \
            --epochs "$EPOCHS" \
            --inf_method "$INF_METHOD" \
            --inf_args "$INF_ARGS" \
            --max_length "$MAX_LENGTH"

    done < dataset_settings.txt

done < methods_to_run.txt


python3 -c "from postprocess_utils import generate_table_metrics; generate_table_metrics()"