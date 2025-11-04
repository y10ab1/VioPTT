
# Training High Resolution Piano Transcription (HRPT)

# Modify to your workspace

WORKSPACE="/home/yuehpo/coding/VioPTT"

# Notice:
# 1. The checkpoints will be saved in this directory as `./checkpoints` under WORKSPACE
# 2. Place dataset path under WORKSPACE, it should be {WORKSPACE}/hdf5s/mosa/...

# Tensorboard log directory
TB="${WORKSPACE}/tb"

# Piano Pretrained model path
# PRETRAIN_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/pretrained/high_resolution_MAESTRO_augmentations.pth"
# Violin Pretrained model path
# PRETRAIN_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_debug_from_scratch_contrast_weight_0.0_cosine_annealing_lr0724/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/10000_iterations.pth"
# PRETRAIN_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_w_technique_annotation_mixed_dataset_mosapt_ssv_local_technique_feature_10frames_volume_normalized_per_class_acc_0906/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/17000_iterations.pth"
PRETRAIN_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_debug_from_scratch_contrast_weight_0.0_cosine_annealing_lr0724/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/10000_iterations.pth"

# PRETRAIN_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_note_technique/note_level_tech_no_transcription_features_train_mosapt_test_rwc_0910/19000_iterations.pth"

cd piano_transcription

TODAY_DATE="0915"
EARLY_STOP=1000

## CV for RWC notes
for fold_id in {0..2}
do
  MODEL_TAG="note_level_tech_no_transcription_features_fold_${fold_id}_${TODAY_DATE}_3CV"
  
  python3 pytorch/main_note_technique.py \
    --workspace $WORKSPACE \
    --logdir $TB \
    --model_tag $MODEL_TAG \
    --augmentation 'aug' \
    --mosapt_hdf5s_dir ${WORKSPACE}/hdf5s/mosa/ \
    --device 0 \
    --batch_size 128 \
    --learning_rate 5e-4 \
    --early_stop $EARLY_STOP \
    --num_workers 4 \
    --dataset train_mosapt_valid_rwc_test_rwc \
    --fold_id $fold_id 
  echo "-----------------------------------------------------------------------------"
  echo "-------------Done for $MODEL_TAG fold ${fold_id}-----------------------------"
  echo "-----------------------------------------------------------------------------"
  echo ""
done
echo ""
echo "-----------------------------------------------------------------------------"
echo "-----------------------------Done for $MODEL_TAG-----------------------------"
echo "-----------------------------------------------------------------------------"
echo ""




## CV for RWC notes
for fold_id in {0..2}
do
  MODEL_TAG="note_level_tech_no_transcription_features_fold_${fold_id}_${TODAY_DATE}_3CV_with_transcription_features_all"
  
  python3 pytorch/main_note_technique.py \
    --workspace $WORKSPACE \
    --logdir $TB \
    --model_tag $MODEL_TAG \
    --augmentation 'aug' \
    --mosapt_hdf5s_dir ${WORKSPACE}/hdf5s/mosa/ \
    --device 0 \
    --batch_size 128 \
    --learning_rate 5e-4 \
    --early_stop $EARLY_STOP \
    --num_workers 4 \
    --dataset train_mosapt_valid_rwc_test_rwc \
    --fold_id $fold_id \
    --transcriptor_checkpoint $PRETRAIN_PATH \
    --use_trans_features \
    --trans_features_list reg_onset_output reg_offset_output frame_output velocity_output

  echo "-----------------------------------------------------------------------------"
  echo "-------------Done for $MODEL_TAG fold ${fold_id}-----------------------------"
  echo "-----------------------------------------------------------------------------"
  echo ""
done

echo ""
echo "-----------------------------------------------------------------------------"
echo "-----------------------------Done for $MODEL_TAG-----------------------------"
echo "-----------------------------------------------------------------------------"
echo ""





## CV for RWC notes
for fold_id in {0..2}
do
  MODEL_TAG="note_level_tech_no_transcription_features_fold_${fold_id}_${TODAY_DATE}_3CV_with_transcription_features_exclude_onset"
  
  python3 pytorch/main_note_technique.py \
    --workspace $WORKSPACE \
    --logdir $TB \
    --model_tag $MODEL_TAG \
    --augmentation 'aug' \
    --mosapt_hdf5s_dir ${WORKSPACE}/hdf5s/mosa/ \
    --device 0 \
    --batch_size 128 \
    --learning_rate 5e-4 \
    --early_stop $EARLY_STOP \
    --num_workers 4 \
    --dataset train_mosapt_valid_rwc_test_rwc \
    --fold_id $fold_id \
    --transcriptor_checkpoint $PRETRAIN_PATH \
    --use_trans_features \
    --trans_features_list reg_offset_output frame_output velocity_output

  echo "-----------------------------------------------------------------------------"
  echo "-------------Done for $MODEL_TAG fold ${fold_id}-----------------------------"
  echo "-----------------------------------------------------------------------------"
  echo ""
done

echo ""
echo "-----------------------------------------------------------------------------"
echo "-----------------------------Done for $MODEL_TAG-----------------------------"
echo "-----------------------------------------------------------------------------"
echo ""





## CV for RWC notes
for fold_id in {0..2}
do
  MODEL_TAG="note_level_tech_no_transcription_features_fold_${fold_id}_${TODAY_DATE}_3CV_with_transcription_features_exclude_offset"
  
  python3 pytorch/main_note_technique.py \
    --workspace $WORKSPACE \
    --logdir $TB \
    --model_tag $MODEL_TAG \
    --augmentation 'aug' \
    --mosapt_hdf5s_dir ${WORKSPACE}/hdf5s/mosa/ \
    --device 0 \
    --batch_size 128 \
    --learning_rate 5e-4 \
    --early_stop $EARLY_STOP \
    --num_workers 4 \
    --dataset train_mosapt_valid_rwc_test_rwc \
    --fold_id $fold_id \
    --transcriptor_checkpoint $PRETRAIN_PATH \
    --use_trans_features \
    --trans_features_list reg_onset_output frame_output velocity_output

  echo "-----------------------------------------------------------------------------"
  echo "-------------Done for $MODEL_TAG fold ${fold_id}-----------------------------"
  echo "-----------------------------------------------------------------------------"
  echo ""
done

echo ""
echo "-----------------------------------------------------------------------------"
echo "-----------------------------Done for $MODEL_TAG-----------------------------"
echo "-----------------------------------------------------------------------------"
echo ""







## CV for RWC notes
for fold_id in {0..2}
do
  MODEL_TAG="note_level_tech_no_transcription_features_fold_${fold_id}_${TODAY_DATE}_3CV_with_transcription_features_exclude_frame"
  
  python3 pytorch/main_note_technique.py \
    --workspace $WORKSPACE \
    --logdir $TB \
    --model_tag $MODEL_TAG \
    --augmentation 'aug' \
    --mosapt_hdf5s_dir ${WORKSPACE}/hdf5s/mosa/ \
    --device 0 \
    --batch_size 128 \
    --learning_rate 5e-4 \
    --early_stop $EARLY_STOP \
    --num_workers 4 \
    --dataset train_mosapt_valid_rwc_test_rwc \
    --fold_id $fold_id \
    --transcriptor_checkpoint $PRETRAIN_PATH \
    --use_trans_features \
    --trans_features_list reg_onset_output reg_offset_output velocity_output

  echo "-----------------------------------------------------------------------------"
  echo "-------------Done for $MODEL_TAG fold ${fold_id}-----------------------------"
  echo "-----------------------------------------------------------------------------"
  echo ""
done

echo ""
echo "-----------------------------------------------------------------------------"
echo "-----------------------------Done for $MODEL_TAG-----------------------------"
echo "-----------------------------------------------------------------------------"
echo ""






## CV for RWC notes
for fold_id in {0..2}
do
  MODEL_TAG="note_level_tech_no_transcription_features_fold_${fold_id}_${TODAY_DATE}_3CV_with_transcription_features_exclude_velocity"
  
  python3 pytorch/main_note_technique.py \
    --workspace $WORKSPACE \
    --logdir $TB \
    --model_tag $MODEL_TAG \
    --augmentation 'aug' \
    --mosapt_hdf5s_dir ${WORKSPACE}/hdf5s/mosa/ \
    --device 0 \
    --batch_size 128 \
    --learning_rate 5e-4 \
    --early_stop $EARLY_STOP \
    --num_workers 4 \
    --dataset train_mosapt_valid_rwc_test_rwc \
    --fold_id $fold_id \
    --transcriptor_checkpoint $PRETRAIN_PATH \
    --use_trans_features \
    --trans_features_list reg_onset_output reg_offset_output frame_output

  echo "-----------------------------------------------------------------------------"
  echo "-------------Done for $MODEL_TAG fold ${fold_id}-----------------------------"
  echo "-----------------------------------------------------------------------------"
  echo ""
done

echo ""
echo "-----------------------------------------------------------------------------"
echo "-----------------------------Done for $MODEL_TAG-----------------------------"
echo "-----------------------------------------------------------------------------"
echo ""


grep -E "final_test|Done" ${WORKSPACE}/run_HRPT_finetune_note_tech_0915_3CV.log > ${WORKSPACE}/final_test_0915_3CV.txt