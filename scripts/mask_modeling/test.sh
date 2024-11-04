#! /bin/bash
#SBATCH --partition=SC-L40S
#SBATCH --job-name="humanise"
#SBATCH --gres=gpu:1
#SBATCH --qos=plus
#SBATCH --cpus-per-task=10
#SBATCH --time 24:00:00

EXP_DIR=$1
CONT=$2
SEED=$3

if [ -z "$SEED" ]
then
    SEED=2023
fi

HYDRA_FULL_ERROR=1 /mnt/lustre/home/jingze/anaconda3/envs/afford/bin/python test_mask_trans.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_dir=${EXP_DIR} \
            seed=${SEED} \
            output_dir=outputs/stage2/ \
            task=contact_motion_gen \
            model=mask_transformer \
            task.dataset.sigma=0.8 \
            task.dataset.max_horizon=132 \
            task.dataset.sets=["HUMANISE"] \
            task.evaluator.k_samples=0 \
            task.evaluator.eval_nbatch=32 \
            task.evaluator.num_k_samples=320 \
            task.test.contact_folder=${CONT} \
            model.transformer.num_layers=6 \
            model.transformer.embed_dim=512 \
            
