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

        METRICS_PATH="results/json/${DATASET_NAME}_${MODEL_NAME}_${INF_METHOD}_metrics_results.json"
        SCORES_PATH="scores/${MODEL_NAME}/${DATASET_NAME}_${EPOCHS}${INF_METHOD}.csv"
        RUNTIME_PATH="runtime_stats/${MODEL_NAME}/${DATASET_NAME}_${EPOCHS}${INF_METHOD}.txt"

        torchrun --standalone --nproc_per_node=1 influence.py \
            --model "$MODEL_NAME" \
            --dataset "$DATASET_NAME" \
            --epochs "$EPOCHS" \
            --inf_method "$INF_METHOD" \
            --inf_args "$INF_ARGS" \
            --max_length "$MAX_LENGTH" \
            --metrics_path "$METRICS_PATH" \
            --scores_path "$SCORES_PATH" \
            --runtime_path "$RUNTIME_PATH"
        
        python3 -c "from postprocess_utils import run_benchmark_measures; run_benchmark_measures(metrics_path='$METRICS_PATH', method='$INF_METHOD', scores_path='$SCORES_PATH', runtime_path='$RUNTIME_PATH', dataset_path='datasets/$DATASET_NAME')"


    done < settings_txt/methods_to_run.txt

    python3 -c "from postprocess_utils import generate_combined_plots; generate_combined_plots(name_begin='${DATASET_NAME}_${EPOCHS}', results_dir='results/distr_plots', model_n='${MODEL_NAME}')"

done < settings_txt/dataset_settings.txt


python3 -c "from postprocess_utils import generate_table_metrics; generate_table_metrics()"

