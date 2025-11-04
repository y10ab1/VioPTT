#!/bin/bash


# Path to trained model checkpoint

# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_debug_contrast_onset_offset_binary_supcon_loss_cosine_annealing_lr/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/10000_iterations.pth"
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=8/10000_iterations.pth" # good
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_debug_contrast_onset_offset_binary_supcon_loss_contrast_weight_0.0/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/10000_iterations.pth" # good good

# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_debug_from_scratch_contrast_weight_0.0_cosine_annealing_lr0724/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/10000_iterations.pth"
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/wo_aug_debug_from_scratch_contrast_weight_0.0_cosine_annealing_lr0724/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=2/batch_size=5/10000_iterations.pth"
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/wo_aug_debug_finetune_contrast_weight_0.0_cosine_annealing_lr0724/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=2/batch_size=5/10000_iterations.pth"
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_debug_finetune_contrast_weight_0.0_cosine_annealing_lr0724/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/10000_iterations.pth"
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_debug_contrast_onset_offset_binary_supcon_loss_contrast_weight_0.0_cosine_annealing_warm_restarts/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/10000_iterations.pth"
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_debug_contrast_onset_offset_binary_supcon_loss_contrast_weight_0.0_ctc_weight_0.1/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/9000_iterations.pth"
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_debug_contrast_onset_offset_binary_supcon_loss_contrast_weight_0.0_ctc_weight_0.0_mosapt/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/10000_iterations.pth"
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_mosapt_train_from_scratch_w_technique_mixed_dataset/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/3000_iterations.pth"
CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_w_technique_annotation_mixed_dataset_mosapt_ssv_local_technique_feature_10frames_volume_normalized_per_class_acc_0906/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/15000_iterations.pth"

# Model type (adjust based on your model)
MODEL_TYPE="Regress_onset_offset_frame_velocity_CRNN"

# Post processor type
POST_PROCESSOR="regression"  # or "onsets_frames""

# Model tag
MODEL_TAG="w_aug_w_technique_annotation_mixed_dataset_mosapt_ssv_local_technique_feature_10frames_volume_normalized_per_class_acc_0906"


# Run evaluation for Bach10
python evaluate.py \
    --model_tag $MODEL_TAG \
    --dataset bach10 \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_type $MODEL_TYPE \
    --post_processor_type $POST_PROCESSOR \
    --device 0 \
    > logs/bach10_eval_${MODEL_TAG}.log 2>&1
echo "Bach10 evaluation completed, results saved in logs/bach10_eval_${MODEL_TAG}.log"

# Run evaluation for URMP
python evaluate.py \
    --model_tag $MODEL_TAG \
    --dataset urmp \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_type $MODEL_TYPE \
    --post_processor_type $POST_PROCESSOR \
    --device 0 \
    > logs/urmp_eval_${MODEL_TAG}.log 2>&1
echo "URMP evaluation completed, results saved in logs/urmp_eval_${MODEL_TAG}.log"