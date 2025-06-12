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

DATASET_NAME=combinedBcause
#TRAIN_FILE=./data/statements/classification/$DATASET_NAME-train.tsv
#VALIDATION_FILE=./data/statements/classification/$DATASET_NAME-validation.tsv
#TEST_FILE=./data/statements/classification/$DATASET_NAME-test.tsv

TEST_FILE=./data/statements/classification/newest-3-2-25-bcause-test.tsv

OUTPUT_DIR=./output
TASK_TYPE=sta-class
MODEL=deberta-v3
LABELS="Position Attack Support"
RELEVANT_LABELS="Position Attack Support"
#EXPERIMENT_NAME=
#RUN_NAME=CBTonLatestB



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