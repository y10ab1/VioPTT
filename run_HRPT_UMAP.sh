CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_note_technique/note_level_tech_no_transcription_features_fold_0_0915_3CV_with_transcription_features_all/720_iterations.pth"
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_note_technique/note_level_tech_no_transcription_features_fold_1_0915_3CV_with_transcription_features_all/880_iterations.pth"
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_note_technique/note_level_tech_no_transcription_features_fold_2_0915_3CV_with_transcription_features_all/660_iterations.pth"
TRANSCRIPTOR_CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_debug_from_scratch_contrast_weight_0.0_cosine_annealing_lr0724/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/10000_iterations.pth"


# python /home/yuehpo/coding/violin-mamba/piano_transcription/pytorch/visualize_umap_note_technique.py \
#   --checkpoint $CHECKPOINT_PATH \
#   --transcriptor_checkpoint $TRANSCRIPTOR_CHECKPOINT_PATH \
#   --rwc_notes_root /home/yuehpo/coding/violin-mamba/rwc_notes \
#   --split test \
#   --fold_id 0 \
#   --device 0 \
#   --out_png /home/yuehpo/coding/violin-mamba/umap_note_technique.png \
#   --out_npz /home/yuehpo/coding/violin-mamba/umap_note_technique.npz \
#   --n_neighbors 10 --min_dist 0.05 \
#   --use_trans_features \
#   --trans_features_list reg_onset_output reg_offset_output frame_output velocity_output \
#   --plot_confusion \
#   --confusion_png /home/yuehpo/coding/violin-mamba/confusion_note_technique.png \
#   --ignore_confusion_class no_technique



python /home/yuehpo/coding/violin-mamba/piano_transcription/pytorch/visualize_umap_note_technique_3fold.py \
  --rwc_notes_root /home/yuehpo/coding/violin-mamba/rwc_notes \
  --checkpoints \
    /home/yuehpo/coding/violin-mamba/checkpoints/main_note_technique/note_level_tech_no_transcription_features_fold_0_0915_3CV_with_transcription_features_all/720_iterations.pth \
    /home/yuehpo/coding/violin-mamba/checkpoints/main_note_technique/note_level_tech_no_transcription_features_fold_1_0915_3CV_with_transcription_features_all/880_iterations.pth \
    /home/yuehpo/coding/violin-mamba/checkpoints/main_note_technique/note_level_tech_no_transcription_features_fold_2_0915_3CV_with_transcription_features_all/660_iterations.pth \
  --fold_ids 0 1 2 \
  --split test \
  --transcriptor_checkpoint /home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_debug_from_scratch_contrast_weight_0.0_cosine_annealing_lr0724/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/10000_iterations.pth \
  --use_trans_features \
  --plot_confusion \
  --ignore_confusion_class no_technique \
  --n_neighbors 10 --min_dist 0.05 \
  --out_dir /home/yuehpo/coding/violin-mamba