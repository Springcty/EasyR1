#!/bin/bash
#SBATCH --job-name=easyr1-grpo-prog-r05
#SBATCH --error=logs/easyr1-grpo-prog-r05-qwen3vl-2b-seedr1.%j.err
#SBATCH --output=logs/easyr1-grpo-prog-r05-qwen3vl-2b-seedr1.%j.out
#SBATCH --gres=gpu:8
#SBATCH --constraint="A6000|6000Ada|L40|L40S"
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=32
#SBATCH --mem=512GB
#SBATCH -t 2-00:00:00              # time limit: 2 days (D-HH:MM:SS)

# Disable P2P for better stability
export NCCL_P2P_DISABLE=1

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the training using Apptainer
apptainer exec --nv --cleanenv \
  --bind /home/tianyuca:/home/tianyuca \
  --bind /data/user_data/tianyuca:/data/user_data/tianyuca \
  ~/easyr1.sif \
  python3 -m verl.trainer.main \
  config=examples/config_seed_r1.yaml \
  data.train_files=/data/user_data/tianyuca/RL_sabotage/SEED-Bench-R1/RL/train_w_task_progress.parquet \
  data.val_files=/data/user_data/tianyuca/RL_sabotage/SEED-Bench-R1/RL/val_L1.parquet \
  data.seed=42 \
  data.video_fps=2.0 \
  data.nframes=null \
  data.sabotage_key=sabotage_key \
  data.sabotage_ratio=0.5 \
  worker.actor.global_batch_size=8 \
  worker.actor.model.model_path=Qwen/Qwen3-VL-2B-Instruct \
  worker.rollout.tensor_parallel_size=8 \
  trainer.n_gpus_per_node=8 \
  trainer.save_checkpoint_path=/data/user_data/tianyuca/RL_sabotage/results \
  trainer.val_before_train=true \
  trainer.project_name=easyr1_seed_r1_grpo \
  trainer.experiment_name=qwen3_vl_2b_prog_r05_r1_val1
