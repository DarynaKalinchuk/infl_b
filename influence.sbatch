#!/bin/bash
#SBATCH --job-name="inf1"
#SBATCH --container-image=kalinchukd/inf_env3:latest
#SBATCH --container-mount-home
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24
#SBATCH --time=0-23:00:00 
#SBATCH --container-workdir=/srv/home/users/kalinchukd23cs/InfEstimation_benchmark
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --gres=gpu:h100:1

python3 --version

IFS=',' read -r MODEL_NAME VAL_FLAG MAX_LENGTH BATCH_SIZE LOGGING_STEP EPOCHS LORA_R LORA_ALPHA TARGET_LAYER LOAD_8BIT < finetune_params.txt

echo "MODEL_NAME = $MODEL_NAME"
echo "VAL_FLAG = $VAL_FLAG"
echo "MAX_LENGTH = $MAX_LENGTH"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "LOGGING_STEP = $LOGGING_STEP"
echo "EPOCHS = $EPOCHS"
echo "LORA_R = $LORA_R"
echo "LORA_ALPHA = $LORA_ALPHA"
echo "TARGET_LAYER = $TARGET_LAYER"
echo "LOAD_8BIT = $LOAD_8BIT"


while read -r DATASET_NAME; do
    [ -z "$DATASET_NAME" ] && continue

    while read -r INF_METHOD; do
        [ -z "$INF_METHOD" ] && continue

        echo "DATASET_NAME = $DATASET_NAME"
        echo "INF_METHOD = $INF_METHOD"

        torchrun --nproc_per_node=1 influence.py \
            --model "$MODEL_NAME" \
            --dataset "$DATASET_NAME" \
            --epochs "$EPOCHS" \
            --inf_method "$INF_METHOD" \
            --inf_args "$INF_ARGS" \
            --max_length "$MAX_LENGTH"

    done < methods_to_run.txt

    python3 -c "from postprocess_utils import generate_kde_plots; generate_kde_plots(name_begin='${DATASET_NAME}_${EPOCHS}', cache_dir='cache/${MODEL_NAME}')"
done < dataset_settings.txt


python3 -c "from postprocess_utils import generate_table_metrics; generate_table_metrics()"

