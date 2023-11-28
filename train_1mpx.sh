DATA_DIR=/media/HDD2/MAE/datasets/RVT/gen4
MDL_CFG=base
GPU_IDS=[0,2,3]
BATCH_SIZE_PER_GPU=4
TRAIN_WORKERS_PER_GPU=2
EVAL_WORKERS_PER_GPU=1
python train.py model=rnndet dataset=gen4 dataset.path=${DATA_DIR} wandb.project_name=RVT \
wandb.group_name=1mpx +experiment/gen4="${MDL_CFG}.yaml" hardware.gpus=${GPU_IDS} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}