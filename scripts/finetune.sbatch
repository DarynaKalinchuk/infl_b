#!/bin/bash
#SBATCH --job-name="fntune1"
#SBATCH --container-image=kalinchukd/inf_env3:latest
#SBATCH --container-mount-home
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24  
#SBATCH --time=0-1:00:00 
#SBATCH --container-workdir=/srv/home/users/kalinchukd23cs/InfEstimation_benchmark
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --gres=gpu:h100:1

python3 --version

IFS=',' read -r MODEL_NAME MAX_LENGTH BATCH_SIZE LOGGING_STEP EPOCHS LORA_R LORA_ALPHA LOAD_8BIT < settings_txt/finetune_params.txt


LOAD8BIT_ARG=""
if [ "$LOAD_8BIT" == "True" ]; then
    LOAD8BIT_ARG="--load_in_8bit"
fi

echo "MODEL_NAME = $MODEL_NAME"
echo "MAX_LENGTH = $MAX_LENGTH"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "LOGGING_STEP = $LOGGING_STEP"
echo "EPOCHS = $EPOCHS"
echo "LORA_R = $LORA_R"
echo "LORA_ALPHA = $LORA_ALPHA"
echo "LOAD_8BIT = $LOAD_8BIT"

while read -r DATASET_NAME; do
    echo "DATASET_NAME = $DATASET_NAME"

    python3 finetune.py \
        --model $MODEL_NAME \
        $LOAD8BIT_ARG \
        --dataset $DATASET_NAME \
        --max_length $MAX_LENGTH \
        --batch_size $BATCH_SIZE \
        --logging_step $LOGGING_STEP \
        --epochs $EPOCHS \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA
        
done < settings_txt/dataset_settings.txt