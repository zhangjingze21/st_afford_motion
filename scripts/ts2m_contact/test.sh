#! /bin/bash
#SBATCH --partition=SC-A800
#SBATCH --job-name="humanise"
#SBATCH --gres=gpu:1
#SBATCH --qos=plus
#SBATCH --cpus-per-task=48
#SBATCH --time 24:00:00

EXP_DIR=$1
SEED=$2

if [ -z "$SEED" ]
then
    SEED=2023
fi

/mnt/lustre/home/jingze/anaconda3/envs/afford/bin/python test.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_dir=${EXP_DIR} \
            seed=${SEED} \
            output_dir=outputs \
            diffusion.steps=10 \
            task=contact_gen \
            model=cdm \
            model.arch=Perceiver \
            task.dataset.sigma=0.8 \
            task.dataset.sets=["HUMANISE"] \
            task.evaluator.k_samples=0 \
            task.evaluator.eval_nbatch=32 \
            task.evaluator.num_k_samples=320
