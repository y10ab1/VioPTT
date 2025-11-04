
# Training High Resolution Piano Transcription (HRPT)

# Modify to your workspace

WORKSPACE="/home/yuehpo/coding/violin-mamba"

# Notice:
# 1. The checkpoints will be saved in this directory as `./checkpoints` under WORKSPACE
# 2. Place dataset path under WORKSPACE, it should be {WORKSPACE}/hdf5s/mosa/...

# Tensorboard log directory
TB="/home/yuehpo/coding/violin-mamba/tb"

# Piano Pretrained model path
# PRETRAIN_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/pretrained/high_resolution_MAESTRO_augmentations.pth"
# Violin Pretrained model path
# PRETRAIN_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_debug_from_scratch_contrast_weight_0.0_cosine_annealing_lr0724/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/10000_iterations.pth"
# PRETRAIN_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_w_technique_annotation_mixed_dataset_new_ssv_local_technique_feature_10frames/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/8000_iterations.pth"
PRETRAIN_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_w_technique_annotation_mixed_dataset_mosapt_ssv_local_technique_feature_10frames_volume_normalized_per_class_acc_0906/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/17000_iterations.pth"

MODEL_TAG="finetune_rwc_tech_only_vad"
cd piano_transcription
# --- 1. Train note transcription system ---
# python3 pytorch/main_contrast.py train \
#     --workspace=$WORKSPACE \
#     --logdir=$TB \
#     --model_tag $MODEL_TAG \
#     --model_type='Regress_onset_offset_frame_velocity_CRNN' \
#     --loss_type='regress_onset_offset_frame_velocity_bce' \
#     --augmentation='aug' \
#     --max_note_shift=2 \
#     --batch_size=5 \
#     --learning_rate=5e-4 \
#     --reduce_iteration=1000 \
#     --resume_iteration=0 \
#     --early_stop=10000 \
#     --device 0 \
#     --dataset mixed \
#     --contrast_weight=0.0 \
#     --ctc_weight=0.0 \
#     --technique_weight=1.0 
#     # --pretrain_path=$PRETRAIN_PATH 

python3 pytorch/main_contrast.py train \
  --workspace $WORKSPACE \
  --model_type Regress_onset_offset_frame_velocity_CRNN \
  --loss_type none \
  --augmentation 'aug' \
  --max_note_shift 2 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --reduce_iteration 10000 \
  --resume_iteration 0 \
  --early_stop 10000 \
  --device 0 \
  --dataset rwc_tech \
  --rwc_h5_path ~/data/rwc_processed_data.h5 \
  --logdir $TB \
  --model_tag $MODEL_TAG \
  --pretrain_path $PRETRAIN_PATH \
  --contrast_weight 0.0 \
  --ctc_weight 0.0 \
  --technique_weight 1.0