#!/bin/bash

PROJECT_NAME=easyr1_seed_r1_grpo
EXPERIMENT_NAME=qwen3_vl_2b_default_e5_val1
MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct

TRAIN_FILES=${TRAIN_FILES:-"data/train_val_parquet/train.parquet"}
VAL_FILES=${VAL_FILES:-"data/train_val_parquet/val_L1.parquet"}
IMAGE_DIR=${IMAGE_DIR:-"data"}
CKPT_PATH=${CKPT_PATH:-"outputs"}

# Run the training using Apptainer
python3 -m verl.trainer.main \
  config=examples/config_seed_r1.yaml \
  data.train_files=$TRAIN_FILES \
  data.val_files=$VAL_FILES \
  data.image_dir=$IMAGE_DIR \
  data.seed=42 \
  data.video_fps=2.0 \
  data.nframes=null \
  worker.actor.global_batch_size=8 \
  worker.actor.model.model_path=$MODEL_PATH \
  worker.rollout.tensor_parallel_size=8 \
  trainer.n_gpus_per_node=8 \
  trainer.save_checkpoint_path=$CKPT_PATH \
  trainer.val_before_train=true \
  trainer.total_epochs=5 \
  trainer.project_name=$PROJECT_NAME \
  trainer.experiment_name=$EXPERIMENT_NAME
