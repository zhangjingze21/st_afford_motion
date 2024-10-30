EXP_NAME=${1:-"debug"}

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train_stfsq_ddp.py \
            hydra/job_logging=none hydra/hydra_logging=none \
            exp_name=${EXP_NAME} \
            output_dir=outputs \
            platform=TensorBoard \
            task=motion_quantizer \
            task.train.batch_size=256 \
            task.train.phase=train \
            task.dataset.sigma=0.8 \
            task.dataset.train_transforms=['RandomRotation','ApplyTransformCMDM','RandomMaskLang','NumpyToTensor'] \
            model=sthvqvae \
