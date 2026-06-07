#!/bin/bash
#SBATCH --job-name="inf1"
#SBATCH --container-image=kalinchukd/inf_env3:latest
#SBATCH --container-mount-home
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24
#SBATCH --time=0-23:00:00 
#SBATCH --container-workdir=/srv/home/users/kalinchukd23cs/infl_b
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --gres=gpu:h100:1

python3 --version

IFS=',' read -r MODEL_NAME MAX_LENGTH BATCH_SIZE LOGGING_STEP EPOCHS LORA_R LORA_ALPHA LOAD_8BIT < settings_txt/finetune_params.txt

echo "MODEL_NAME = $MODEL_NAME"
echo "MAX_LENGTH = $MAX_LENGTH"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "EPOCHS = $EPOCHS"


while read -r DATASET_NAME; do
    [ -z "$DATASET_NAME" ] && continue

    while read -r INF_METHOD; do
        [ -z "$INF_METHOD" ] && continue

        echo "DATASET_NAME = $DATASET_NAME"
        echo "INF_METHOD = $INF_METHOD"

        torchrun --standalone --nproc_per_node=1 influence.py \
            --model "$MODEL_NAME" \
            --dataset "$DATASET_NAME" \
            --epochs "$EPOCHS" \
            --inf_method "$INF_METHOD" \
            --inf_args "$INF_ARGS" \
            --max_length "$MAX_LENGTH"

    done < settings_txt/methods_to_run.txt

    python3 -c "from postprocess_utils import generate_kde_plots; generate_kde_plots(name_begin='${DATASET_NAME}_${EPOCHS}', cache_dir='cache/${MODEL_NAME}')"
done < settings_txt/dataset_settings.txt


python3 -c "from postprocess_utils import generate_table_metrics; generate_table_metrics()"

