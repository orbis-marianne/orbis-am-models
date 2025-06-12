#!/bin/bash

DATASET_NAME=disputool
TRAIN_FILE=./data/relation/$DATASET_NAME-train.tsv
VALIDATION_FILE=./data/relation/$DATASET_NAME-validation.tsv
TEST_FILE=./data/relation/$DATASET_NAME-test.tsv

OUTPUT_DIR=./output
TASK_TYPE=rel-class
MODEL=deberta-v3 #roberta #

LABELS="noRel Attack Support"
RELEVANT_LABELS="Attack Support"
#EXPERIMENT_NAME=
#RUN_NAME=
EPOCHS=10
EARLY_STOP=2
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=64
GRADIENT_ACCUMULATION=1
MAX_GRAD=1
MAX_SEQ_LENGTH=64
LEARNING_RATE=1e-4
WEIGHT_DECAY=0
WARMUP_STEPS=0
LOG_STEPS=415
SAVE_STEPS=930
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
  --log-every-n-steps $LOG_STEPS \
  --save-every-n-steps $SAVE_STEPS \
  --random-seed $RANDOM_SEED\
  --add-prefix-space


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
  --random-seed $RANDOM_SEED\
  --add-prefix-space