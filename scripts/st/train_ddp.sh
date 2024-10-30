#! /bin/bash
#SBATCH --partition=SC-A800
#SBATCH --job-name="humanise"
#SBATCH --gres=gpu:4
#SBATCH --qos=plus
#SBATCH --cpus-per-task=48
#SBATCH --time 24:00:00

EXP_NAME=${1:-"stfsq_ddp4"}

HYDRA_FULL_ERROR=1 /mnt/lustre/home/jingze/anaconda3/envs/afford/bin/torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train_stfsq_ddp.py \
            hydra/job_logging=none hydra/hydra_logging=none \
            exp_name=${EXP_NAME} \
            output_dir=outputs/stage1/ \
            platform=TensorBoard \
            task=motion_quantizer \
            task.train.batch_size=128 \
            task.train.phase=train \
            task.dataset.sigma=0.8 \
            model=sthvqvae \
