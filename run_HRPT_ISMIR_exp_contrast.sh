
# Training High Resolution Piano Transcription (HRPT)

# Modify to your workspace

WORKSPACE="/home/yuehpo/coding/violin-mamba"

# Notice:
# 1. The checkpoints will be saved in this directory as `./checkpoints` under WORKSPACE
# 2. Place dataset path under WORKSPACE, it should be {WORKSPACE}/hdf5s/mosa/...

# Tensorboard log directory
TB="/home/yuehpo/coding/violin-mamba/tb_0724"

# Pretrained model path
PRETRAIN_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/pretrained/high_resolution_MAESTRO_augmentations.pth"

cd piano_transcription
# --- Train from scratch w/o augmentation ---
python3 pytorch/main_contrast.py train \
    --workspace=$WORKSPACE \
    --logdir=$TB \
    --model_tag wo_aug_debug_from_scratch_contrast_weight_0.0_cosine_annealing_lr0724 \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --loss_type='regress_onset_offset_frame_velocity_bce' \
    --augmentation='none' \
    --max_note_shift=2 \
    --batch_size=5 \
    --learning_rate=5e-4 \
    --reduce_iteration=1000 \
    --resume_iteration=0 \
    --early_stop=10000 \
    --device 0 \
    --dataset mosa \
    --contrast_weight=0.0

# --- Train from scratch w/ augmentation ---
python3 pytorch/main_contrast.py train \
    --workspace=$WORKSPACE \
    --logdir=$TB \
    --model_tag w_aug_debug_from_scratch_contrast_weight_0.0_cosine_annealing_lr0724 \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --loss_type='regress_onset_offset_frame_velocity_bce' \
    --augmentation='aug' \
    --max_note_shift=2 \
    --batch_size=5 \
    --learning_rate=5e-4 \
    --reduce_iteration=1000 \
    --resume_iteration=0 \
    --early_stop=10000 \
    --device 0 \
    --dataset mosa \
    --contrast_weight=0.0

# --- Finetune w/o augmentation ---
python3 pytorch/main_contrast.py train \
    --workspace=$WORKSPACE \
    --logdir=$TB \
    --model_tag wo_aug_debug_finetune_contrast_weight_0.0_cosine_annealing_lr0724 \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --loss_type='regress_onset_offset_frame_velocity_bce' \
    --augmentation='none' \
    --max_note_shift=2 \
    --batch_size=5 \
    --learning_rate=5e-4 \
    --reduce_iteration=1000 \
    --resume_iteration=0 \
    --early_stop=10000 \
    --device 0 \
    --dataset mosa \
    --pretrain_path=$PRETRAIN_PATH \
    --contrast_weight=0.0

# --- Finetune w/ augmentation ---
python3 pytorch/main_contrast.py train \
    --workspace=$WORKSPACE \
    --logdir=$TB \
    --model_tag w_aug_debug_finetune_contrast_weight_0.0_cosine_annealing_lr0724 \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --loss_type='regress_onset_offset_frame_velocity_bce' \
    --augmentation='aug' \
    --max_note_shift=2 \
    --batch_size=5 \
    --learning_rate=5e-4 \
    --reduce_iteration=1000 \
    --resume_iteration=0 \
    --early_stop=10000 \
    --device 0 \
    --dataset mosa \
    --pretrain_path=$PRETRAIN_PATH \
    --contrast_weight=0.0