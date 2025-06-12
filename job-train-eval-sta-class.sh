#!/bin/bash

#SBATCH --job-name=train-eval-sta-class
#SBATCH --output=output-train-eval-sta-class-.txt

#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --account=desinformation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

##SBATCH --mail-user=sofiane.elguendouze@inria.fr
##SBATCH --mail-type=BEGIN,END,FAIL

module purge

module load miniconda

conda activate /home/selguendouze/.conda/envs/amtm_env

##---------------------------------------------------------------

set -ex

DATASET_NAME=combinedBcause-Touche-PE
TRAIN_FILE=./data/statements/classification/$DATASET_NAME-train.tsv
VALIDATION_FILE=./data/statements/classification/$DATASET_NAME-validation.tsv
##TEST_FILE=./data/statements/classification/touche-PE-test.tsv
TEST_FILE=./data/statements/classification/$DATASET_NAME-test.tsv

OUTPUT_DIR=./output
TASK_TYPE=sta-class
MODEL=bert #roberta #deberta-v3
LABELS="Position Evidence" #"Position Attack Support"
RELEVANT_LABELS="Position Evidence" #"Position Attack Support"
#EXPERIMENT_NAME=
#RUN_NAME=
EPOCHS=5
EARLY_STOP=2
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION=1
MAX_GRAD=1
MAX_SEQ_LENGTH=128
LEARNING_RATE=2e-5
WEIGHT_DECAY=0
WARMUP_STEPS=0
LOG_STEPS=90
SAVE_STEPS=180
RANDOM_SEED=42

python ./scripts/train.py \
  --train-data $TRAIN_FILE \
  --validation-data $VALIDATION_FILE \
  --output-dir $OUTPUT_DIR \
  --task-type $TASK_TYPE \
  --model $MODEL \
  --dataset-name $DATASET_NAME\
  --labels $LABELS \
  --num-devices 1 \
  --num-workers 10 \
  --epochs $EPOCHS \
  --early-stopping $EARLY_STOP \
  --batch-size $TRAIN_BATCH_SIZE \
  --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
  --max-grad-norm $MAX_GRAD \
  --max-seq-length $MAX_SEQ_LENGTH \
  --learning-rate $LEARNING_RATE \
  --weight-decay $WEIGHT_DECAY \
  --warmup-steps $WARMUP_STEPS \
  --weighted-loss \
  --log-every-n-steps $LOG_STEPS \
  --save-every-n-steps $SAVE_STEPS \
  --random-seed $RANDOM_SEED


python ./scripts/eval.py \
  --test-data $TEST_FILE \
  --output-dir $OUTPUT_DIR \
  --task-type $TASK_TYPE \
  --model $MODEL \
  --dataset-name $DATASET_NAME \
  --eval-all-checkpoints \
  --labels $LABELS \
  --relevant-labels $RELEVANT_LABELS \
  --num-workers -1 \
  --batch-size $EVAL_BATCH_SIZE \
  --max-seq-length $MAX_SEQ_LENGTH \
  --weighted-loss \
  --random-seed $RANDOM_SEED