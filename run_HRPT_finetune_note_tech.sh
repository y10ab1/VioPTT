
# Training High Resolution Piano Transcription (HRPT)

# Modify to your workspace

WORKSPACE="/home/yuehpo/coding/VioPTT"

# Notice:
# 1. The checkpoints will be saved in this directory as `./checkpoints` under WORKSPACE
# 2. Place dataset path under WORKSPACE, it should be {WORKSPACE}/hdf5s/mosa/...

# Tensorboard log directory
TB="${WORKSPACE}/tb"

PRETRAIN_PATH="${WORKSPACE}/checkpoints/transcriptor_model.pth"

cd piano_transcription

TODAY_DATE="0915"

MODEL_TAG="note_level_tech_no_transcription_features_train_mosapt_test_rwc_all_1000iterations_0914_with_transcription_features_all_max_pooling"

python3 pytorch/main_note_technique.py \
  --workspace $WORKSPACE \
  --logdir $TB \
  --model_tag $MODEL_TAG \
  --augmentation 'aug' \
  --rwc_notes_dir ${WORKSPACE}/rwc_notes/ \
  --device 0 \
  --batch_size 128 \
  --learning_rate 5e-4 \
  --early_stop 1000 \
  --num_workers 4 \
  --dataset train_mosapt_test_rwc \
  --fold_id -1 \
  --transcriptor_checkpoint $PRETRAIN_PATH \
  --use_trans_features \
  --trans_features_list reg_onset_output reg_offset_output frame_output velocity_output


echo "-----------------------------------------------------------------------------"
echo "-----------------------------Done for $MODEL_TAG-----------------------------"
echo "-----------------------------------------------------------------------------"

MODEL_TAG="note_level_tech_no_transcription_features_train_mosapt_test_rwc_all_1000iterations_${TODAY_DATE}_with_transcription_features_exclude_onset_max_pooling"

python3 pytorch/main_note_technique.py \
  --workspace $WORKSPACE \
  --logdir $TB \
  --model_tag $MODEL_TAG \
  --augmentation 'aug' \
  --rwc_notes_dir ${WORKSPACE}/rwc_notes/ \
  --device 0 \
  --batch_size 128 \
  --learning_rate 5e-4 \
  --early_stop 1000 \
  --num_workers 4 \
  --dataset train_mosapt_test_rwc \
  --fold_id -1 \
  --transcriptor_checkpoint $PRETRAIN_PATH \
  --use_trans_features \
  --trans_features_list reg_offset_output frame_output velocity_output

echo "-----------------------------------------------------------------------------"
echo "-----------------------------Done for $MODEL_TAG-----------------------------"
echo "-----------------------------------------------------------------------------"

MODEL_TAG="note_level_tech_no_transcription_features_train_mosapt_test_rwc_all_1000iterations_${TODAY_DATE}_with_transcription_features_exclude_offset_max_pooling"

python3 pytorch/main_note_technique.py \
  --workspace $WORKSPACE \
  --logdir $TB \
  --model_tag $MODEL_TAG \
  --augmentation 'aug' \
  --rwc_notes_dir ${WORKSPACE}/rwc_notes/ \
  --device 0 \
  --batch_size 128 \
  --learning_rate 5e-4 \
  --early_stop 1000 \
  --num_workers 4 \
  --dataset train_mosapt_test_rwc \
  --fold_id -1 \
  --transcriptor_checkpoint $PRETRAIN_PATH \
  --use_trans_features \
  --trans_features_list reg_onset_output frame_output velocity_output

echo "-----------------------------------------------------------------------------"
echo "-----------------------------Done for $MODEL_TAG-----------------------------"
echo "-----------------------------------------------------------------------------"

MODEL_TAG="note_level_tech_no_transcription_features_train_mosapt_test_rwc_all_1000iterations_${TODAY_DATE}_with_transcription_features_exclude_frame_max_pooling"

python3 pytorch/main_note_technique.py \
  --workspace $WORKSPACE \
  --logdir $TB \
  --model_tag $MODEL_TAG \
  --augmentation 'aug' \
  --rwc_notes_dir ${WORKSPACE}/rwc_notes/ \
  --device 0 \
  --batch_size 128 \
  --learning_rate 5e-4 \
  --early_stop 1000 \
  --num_workers 4 \
  --dataset train_mosapt_test_rwc \
  --fold_id -1 \
  --transcriptor_checkpoint $PRETRAIN_PATH \
  --use_trans_features \
  --trans_features_list reg_onset_output reg_offset_output velocity_output

echo "-----------------------------------------------------------------------------"
echo "-----------------------------Done for $MODEL_TAG-----------------------------"
echo "-----------------------------------------------------------------------------"

MODEL_TAG="note_level_tech_no_transcription_features_train_mosapt_test_rwc_all_1000iterations_${TODAY_DATE}_with_transcription_features_exclude_velocity_max_pooling"

python3 pytorch/main_note_technique.py \
  --workspace $WORKSPACE \
  --logdir $TB \
  --model_tag $MODEL_TAG \
  --augmentation 'aug' \
  --rwc_notes_dir ${WORKSPACE}/rwc_notes/ \
  --device 0 \
  --batch_size 128 \
  --learning_rate 5e-4 \
  --early_stop 1000 \
  --num_workers 4 \
  --dataset train_mosapt_test_rwc \
  --fold_id -1 \
  --transcriptor_checkpoint $PRETRAIN_PATH \
  --use_trans_features \
  --trans_features_list reg_onset_output reg_offset_output frame_output

echo "-----------------------------------------------------------------------------"
echo "-----------------------------Done for $MODEL_TAG-----------------------------"
echo "-----------------------------------------------------------------------------"


MODEL_TAG="note_level_tech_no_transcription_features_train_mosapt_test_rwc_all_1000iterations_${TODAY_DATE}_max_pooling"

python3 pytorch/main_note_technique.py \
  --workspace $WORKSPACE \
  --logdir $TB \
  --model_tag $MODEL_TAG \
  --augmentation 'aug' \
  --rwc_notes_dir ${WORKSPACE}/rwc_notes/ \
  --device 0 \
  --batch_size 128 \
  --learning_rate 5e-4 \
  --early_stop 1000 \
  --num_workers 4 \
  --dataset train_mosapt_test_rwc \
  --fold_id -1 
  # --transcriptor_checkpoint $PRETRAIN_PATH \
  # --use_trans_features 
  # --pretrained_note_model_checkpoint $PRETRAIN_PATH

echo "-----------------------------------------------------------------------------"
echo "-----------------------------Done for $MODEL_TAG-----------------------------"
echo "-----------------------------------------------------------------------------"

