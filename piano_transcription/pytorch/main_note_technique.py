import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torchvision.ops import sigmoid_focal_loss as focal_loss
from torch.utils.tensorboard import SummaryWriter

from utilities import (create_folder, get_filename, create_logging)
from data_generator import RWCTechNoteWavDataset, MOSAPTNoteDataset
from models_contrast import (
    Regress_onset_offset_frame_velocity_CRNN,
    NoteLevelTechniqueModel,
)
from pytorch_utils import move_data_to_device
import config
import random

seed = 42

# add seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def accuracy_from_logits(logits: torch.Tensor, one_hot: torch.Tensor) -> float:
    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1)  # (B,)
        targets = torch.argmax(one_hot, dim=-1)  # (B,)
        return (preds == targets).float().mean().item()


def build_transcriptor(device, frames_per_second, classes_num, checkpoint_path):
    model = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=frames_per_second,
        classes_num=classes_num,
        output_features=False,
        predict_technique=False,
    ).to(device)
    if checkpoint_path is None or (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f"Transcription checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    try:
        model.load_state_dict(ckpt['model'], strict=False)
    except TypeError:
        model.load_state_dict(ckpt['model'])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def train(args):
    workspace = args.workspace
    notes_root = os.path.expanduser(args.rwc_notes_dir)
    logdir_root = os.path.join(args.logdir, args.model_tag)
    create_folder(logdir_root)
    writer = SummaryWriter(log_dir=logdir_root)

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__), args.model_tag)
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    device = torch.device(f"cuda:{args.device}") if (args.device is not None and args.device >= 0 and torch.cuda.is_available()) else torch.device('cpu')
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    early_stop = args.early_stop
    frames_per_second = config.frames_per_second
    trans_feat_dim = 88*4 if args.trans_features_list is None else 88*len(args.trans_features_list)

    # Dataset / Loader
    # Augmentor consistent with CustomDataset
    from data_generator import Augmentor
    augmentor = Augmentor() if args.augmentation == 'aug' else None

    if args.dataset == 'rwc_note_wav':
        train_dataset = RWCTechNoteWavDataset(notes_root, frames_per_second=frames_per_second, split='train', augmentor=augmentor, fold_id=args.fold_id)
        test_dataset = RWCTechNoteWavDataset(notes_root, frames_per_second=frames_per_second, split='test', augmentor=None, fold_id=args.fold_id)
    elif args.dataset == 'mosapt_note_h5':
        # Build file-level split for MOSAPT
        from utilities import traverse_folder
        _, h5_paths = traverse_folder(args.mosapt_hdf5s_dir)
        h5_keys = sorted([os.path.splitext(os.path.basename(p))[0] for p in h5_paths])
        rng = np.random.RandomState(args.mosapt_split_seed)
        rng.shuffle(h5_keys)
        split_idx = int(len(h5_keys) * float(args.mosapt_train_ratio))
        mosapt_train = set(h5_keys[:split_idx])
        mosapt_test = set(h5_keys[split_idx:])
        train_dataset = MOSAPTNoteDataset(
            hdf5s_dir=args.mosapt_hdf5s_dir,
            frames_per_second=frames_per_second,
            fixed_seconds=args.mosapt_fixed_seconds,
            augmentor=augmentor,
            split='train',
            allowed_file_keys=mosapt_train,
            mini_data=False,
        )
        test_dataset = MOSAPTNoteDataset(
            hdf5s_dir=args.mosapt_hdf5s_dir,
            frames_per_second=frames_per_second,
            fixed_seconds=args.mosapt_fixed_seconds,
            augmentor=None,
            split='test',
            allowed_file_keys=mosapt_test,
            mini_data=False,
        )
    elif args.dataset == 'train_mosapt_test_rwc':
        from utilities import traverse_folder
        _, h5_paths = traverse_folder(args.mosapt_hdf5s_dir)
        mosapt_all = sorted([os.path.splitext(os.path.basename(p))[0] for p in h5_paths])
        rng = np.random.RandomState(args.mosapt_split_seed)
        rng.shuffle(mosapt_all)
        split_idx = int(len(mosapt_all) * float(args.mosapt_train_ratio))
        mosapt_train = set(mosapt_all[:split_idx])
        mosapt_test = set(mosapt_all[split_idx:])
        train_dataset = MOSAPTNoteDataset(
            hdf5s_dir=args.mosapt_hdf5s_dir,
            frames_per_second=frames_per_second,
            fixed_seconds=args.mosapt_fixed_seconds,
            augmentor=augmentor,
            split='train',
            allowed_file_keys=mosapt_all,
            mini_data=False,
        )
        test_dataset = RWCTechNoteWavDataset(notes_root, frames_per_second=frames_per_second, split='test', augmentor=None, fold_id=args.fold_id)
    elif args.dataset == 'train_mosapt_valid_rwc_test_rwc':
        from utilities import traverse_folder
        _, h5_paths = traverse_folder(args.mosapt_hdf5s_dir)
        mosapt_all = sorted([os.path.splitext(os.path.basename(p))[0] for p in h5_paths])
        rng = np.random.RandomState(args.mosapt_split_seed)
        rng.shuffle(mosapt_all)
        train_dataset = MOSAPTNoteDataset(
            hdf5s_dir=args.mosapt_hdf5s_dir,
            frames_per_second=frames_per_second,
            fixed_seconds=args.mosapt_fixed_seconds,
            augmentor=augmentor,
            split='train',
            allowed_file_keys=mosapt_all,
            mini_data=False,
        )
        valid_dataset = RWCTechNoteWavDataset(notes_root, frames_per_second=frames_per_second, split='train', augmentor=None, fold_id=args.fold_id)
        test_dataset = RWCTechNoteWavDataset(notes_root, frames_per_second=frames_per_second, split='test', augmentor=None, fold_id=args.fold_id)
        
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.dataset == 'train_mosapt_valid_rwc_test_rwc':
        eval_loader = valid_loader
    else:
        eval_loader = test_loader

    # Models
    note_model = NoteLevelTechniqueModel(classes_num=5, sample_rate=config.sample_rate, frames_per_second=frames_per_second, 
                    trans_feat_dim=trans_feat_dim, use_trans_features=args.use_trans_features).to(device)
    if args.pretrained_note_model_checkpoint is not None:
        note_model.load_state_dict(torch.load(args.pretrained_note_model_checkpoint)['note_model'])
    transcriptor = None
    if args.use_trans_features:
        transcriptor = build_transcriptor(device, frames_per_second, config.classes_num, args.transcriptor_checkpoint)

    optimizer = optim.Adam(note_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, early_stop), eta_min=0.05 * learning_rate)
    ce = nn.CrossEntropyLoss(reduction='mean')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', get_filename(__file__), args.model_tag)
    create_folder(checkpoints_dir)

    iteration = 0
    train_bgn_time = time.time()
    best_test_acc = 0.0
    best_test_acc_iter = -1
    best_balanced_acc = 0.0
    best_balanced_acc_iter = -1
    while iteration < early_stop:
        for batch in train_loader:
            if iteration >= early_stop:
                break

            # Move
            technique_oh = move_data_to_device(batch['technique'], device)  # (B, 1, C)
            waveform = move_data_to_device(batch['waveform'], device)  # (B, L)

            # Optional transcription features
            trans_feats = None
            if args.use_trans_features and transcriptor is not None:
                with torch.no_grad():
                    out = transcriptor(waveform.float())
                    onset = out.get('reg_onset_output')
                    offset = out.get('reg_offset_output')
                    frame = out.get('frame_output')
                    velocity = out.get('velocity_output')
                    if args.trans_features_list is None:
                        tf = torch.cat([onset, offset, frame, velocity], dim=-1)  # (B, Tc, 88*4)
                    else:
                        tf = torch.cat([out.get(feat) for feat in args.trans_features_list], dim=-1)  # (B, Tc, 88*n_features)
                    # time align will be handled after encoding; estimate T via spec hop
                    trans_feats = tf

            # Forward
            logits = note_model(waveform.float(), trans_features=trans_feats)  # (B, C)
            if args.use_focal_loss:
                loss = focal_loss(logits, technique_oh[:, 0, :], reduction='mean')
            else:
                targets = technique_oh[:, 0, :].argmax(dim=-1)  # (B,)
                loss = ce(logits, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Metrics
            acc = accuracy_from_logits(logits, technique_oh[:, 0, :])
            writer.add_scalar('train/loss', loss.item(), iteration)
            writer.add_scalar('train/acc', acc, iteration)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], iteration)

            if iteration % 10 == 0:
                logging.info("--------------------------------")
                logging.info(f"iter {iteration} | loss={loss.item():.4f} acc={acc:.4f}")
                # Quick test set eval
                note_model.eval()
                with torch.no_grad():
                    total = 0
                    correct = 0
                    for tb in eval_loader:
                        t_wave = move_data_to_device(tb['waveform'], device)
                        t_onehot = move_data_to_device(tb['technique'], device)
                        t_feats = None
                        if args.use_trans_features and transcriptor is not None:
                            o = transcriptor(t_wave.float())
                            t_on = o.get('reg_onset_output'); t_off = o.get('reg_offset_output'); t_fr = o.get('frame_output'); t_vel = o.get('velocity_output')
                            if args.trans_features_list is None:
                                t_feats = torch.cat([t_on, t_off, t_fr, t_vel], dim=-1)  # (B, Tc, 88*4)
                            else:
                                t_feats = torch.cat([o.get(feat) for feat in args.trans_features_list], dim=-1)  # (B, Tc, 88*n_features)
                        t_logits = note_model(t_wave.float(), t_feats)
                        preds = torch.argmax(t_logits, dim=-1)
                        target = torch.argmax(t_onehot[:, 0, :], dim=-1)
                        correct += (preds == target).sum().item()
                        total += preds.numel()
                    test_acc = correct / max(1, total)
                note_model.train()
                writer.add_scalar('test/acc', test_acc, iteration)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_test_acc_iter = iteration
                    logging.info(f"New best raw test acc={best_test_acc:.4f} at iter {best_test_acc_iter}")

                # Per-class accuracy
                try:
                    class_names = getattr(train_dataset, 'technique_classes', ['flageolet', 'normal', 'pizzicato', 'spiccato', 'no_technique'])
                    num_classes = len(class_names)
                    correct_cls = [0 for _ in range(num_classes)]
                    total_cls = [0 for _ in range(num_classes)]
                    note_model.eval()
                    with torch.no_grad():
                        for tb in eval_loader:
                            t_wave = move_data_to_device(tb['waveform'], device)
                            t_onehot = move_data_to_device(tb['technique'], device)
                            t_feats = None
                            if args.use_trans_features and transcriptor is not None:
                                o = transcriptor(t_wave.float())
                                t_on = o.get('reg_onset_output'); t_off = o.get('reg_offset_output'); t_fr = o.get('frame_output'); t_vel = o.get('velocity_output')
                                if args.trans_features_list is None:
                                    t_feats = torch.cat([t_on, t_off, t_fr, t_vel], dim=-1)  # (B, Tc, 88*4)
                                else:
                                    t_feats = torch.cat([o.get(feat) for feat in args.trans_features_list], dim=-1)  # (B, Tc, 88*n_features)
                            t_logits = note_model(t_wave.float(), t_feats)
                            preds = torch.argmax(t_logits, dim=-1)  # (B,)
                            target = torch.argmax(t_onehot[:, 0, :], dim=-1)  # (B,)
                            for c in range(num_classes):
                                mask = (target == c)
                                total_cls[c] += int(mask.sum().item())
                                if mask.any():
                                    correct_cls[c] += int((preds[mask] == target[mask]).sum().item())
                    for c in range(num_classes):
                        acc_c = (correct_cls[c] / total_cls[c]) if total_cls[c] > 0 else 0.0
                        writer.add_scalar(f'test/acc_{class_names[c]}', acc_c, iteration)
                        logging.info(f"test_acc_{class_names[c]}={acc_c:.4f} ({correct_cls[c]}/{total_cls[c]})")
                    # Macro balanced accuracy across classes present in this batch
                    valid_classes = [c for c in range(num_classes) if total_cls[c] > 0]
                    if len(valid_classes) > 0:
                        balanced_acc = float(np.mean([(correct_cls[c] / total_cls[c]) for c in valid_classes]))
                    else:
                        balanced_acc = 0.0
                    writer.add_scalar('test/acc_balanced', balanced_acc, iteration)
                    if balanced_acc > best_balanced_acc:
                        best_balanced_acc = balanced_acc
                        best_balanced_acc_iter = iteration
                        logging.info(f"New best balanced test acc={best_balanced_acc:.4f} at iter {best_balanced_acc_iter}")
                        # save checkpoint
                        ckpt = {
                            'iteration': iteration,
                            'note_model': note_model.state_dict(),
                        }
                        ckpt_path = os.path.join(checkpoints_dir, f"{best_balanced_acc_iter}_iterations.pth")
                        torch.save(ckpt, ckpt_path)
                        logging.info(f"Saved checkpoint to {ckpt_path}")
                except Exception as e:
                    logging.warning(f"Per-class accuracy logging failed: {e}")
                
                logging.info(f"test_acc={test_acc:.4f}")

            if iteration % 1000 == 0:
                ckpt = {
                    'iteration': iteration,
                    'note_model': note_model.state_dict(),
                }
                ckpt_path = os.path.join(checkpoints_dir, f"{iteration}_iterations.pth")
                torch.save(ckpt, ckpt_path)
                logging.info(f"Saved checkpoint to {ckpt_path}")

            iteration += 1

    writer.close()
    logging.info('Training completed.')
    logging.info(f"Best raw test acc={best_test_acc:.4f} at iter {best_test_acc_iter}")
    logging.info(f"Best balanced test acc={best_balanced_acc:.4f} at iter {best_balanced_acc_iter}")

    if args.dataset == 'train_mosapt_valid_rwc_test_rwc':
        # load best balanced test acc model
        ckpt_path = os.path.join(checkpoints_dir, f"{best_balanced_acc_iter}_iterations.pth")
        ckpt = torch.load(ckpt_path)
        note_model.load_state_dict(ckpt['note_model'])
        logging.info(f"----------------Final test acc for fold {args.fold_id}, model tag: {args.model_tag}----------------")
        logging.info(f"Loaded best balanced test acc model from {ckpt_path}")

        # test on test set (same metrics as training)
        note_model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for tb in test_loader:
                t_wave = move_data_to_device(tb['waveform'], device)
                t_onehot = move_data_to_device(tb['technique'], device)
                t_feats = None
                if args.use_trans_features and transcriptor is not None:
                    o = transcriptor(t_wave.float())
                    t_on = o.get('reg_onset_output'); t_off = o.get('reg_offset_output'); t_fr = o.get('frame_output'); t_vel = o.get('velocity_output')
                    if args.trans_features_list is None:
                        t_feats = torch.cat([t_on, t_off, t_fr, t_vel], dim=-1)  # (B, Tc, 88*4)
                    else:
                        t_feats = torch.cat([o.get(feat) for feat in args.trans_features_list], dim=-1)  # (B, Tc, 88*n_features)
                t_logits = note_model(t_wave.float(), t_feats)
                preds = torch.argmax(t_logits, dim=-1)
                target = torch.argmax(t_onehot[:, 0, :], dim=-1)
                correct += (preds == target).sum().item()
                total += preds.numel()
            final_test_acc = correct / max(1, total)
            logging.info(f"final_test_acc={final_test_acc:.4f}")

            # Per-class accuracy and balanced accuracy
            try:
                class_names = getattr(train_dataset, 'technique_classes', ['flageolet', 'normal', 'pizzicato', 'spiccato', 'no_technique'])
                num_classes = len(class_names)
                correct_cls = [0 for _ in range(num_classes)]
                total_cls = [0 for _ in range(num_classes)]
                for tb in test_loader:
                    t_wave = move_data_to_device(tb['waveform'], device)
                    t_onehot = move_data_to_device(tb['technique'], device)
                    t_feats = None
                    if args.use_trans_features and transcriptor is not None:
                        o = transcriptor(t_wave.float())
                        t_on = o.get('reg_onset_output'); t_off = o.get('reg_offset_output'); t_fr = o.get('frame_output'); t_vel = o.get('velocity_output')
                        if args.trans_features_list is None:
                            t_feats = torch.cat([t_on, t_off, t_fr, t_vel], dim=-1)
                        else:
                            t_feats = torch.cat([o.get(feat) for feat in args.trans_features_list], dim=-1)
                    t_logits = note_model(t_wave.float(), t_feats)
                    preds = torch.argmax(t_logits, dim=-1)
                    target = torch.argmax(t_onehot[:, 0, :], dim=-1)
                    for c in range(num_classes):
                        mask = (target == c)
                        total_cls[c] += int(mask.sum().item())
                        if mask.any():
                            correct_cls[c] += int((preds[mask] == target[mask]).sum().item())
                for c in range(num_classes):
                    acc_c = (correct_cls[c] / total_cls[c]) if total_cls[c] > 0 else 0.0
                    logging.info(f"final_test_acc_{class_names[c]}={acc_c:.4f} ({correct_cls[c]}/{total_cls[c]})")
                valid_classes = [c for c in range(num_classes) if total_cls[c] > 0]
                if len(valid_classes) > 0:
                    balanced_acc = float(np.mean([(correct_cls[c] / total_cls[c]) for c in valid_classes]))
                else:
                    balanced_acc = 0.0
                logging.info(f"final_test_acc_balanced={balanced_acc:.4f}")
            except Exception as e:
                logging.warning(f"Final per-class accuracy logging failed: {e}")

        # # log results
        # logging.info(f"----------------Final test acc for fold {args.fold_id}, model tag: {args.model_tag}----------------")
        # for c in range(num_classes):
        #     acc_c = (correct_cls[c] / total_cls[c]) if total_cls[c] > 0 else 0.0
        #     logging.info(f"Final test {class_names[c]}={acc_c:.4f} ({correct_cls[c]}/{total_cls[c]})")
        # logging.info(f"Final test acc={final_test_acc:.4f}")
        # logging.info(f"Final test acc balanced={balanced_acc:.4f}")
        logging.info("--------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train note-level technique model')
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--model_tag', type=str, required=True)
    parser.add_argument('--augmentation', type=str, choices=['aug', 'no_aug'], default='aug')
    parser.add_argument('--rwc_notes_dir', type=str, default='PATH_TO_RWC_NOTES_DIR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--early_stop', type=int, default=20000, help='Number of optimization steps')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_trans_features', action='store_true', default=False)
    parser.add_argument('--trans_features_list', nargs='+', default=None)
    parser.add_argument('--transcriptor_checkpoint', type=str, default=None)
    parser.add_argument('--pretrained_note_model_checkpoint', type=str, default=None)
    parser.add_argument('--fold_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, choices=['rwc_note_wav', 'mosapt_note_h5', 'train_mosapt_test_rwc', 'train_mosapt_valid_rwc_test_rwc'], default='rwc_note_wav')
    parser.add_argument('--mosapt_hdf5s_dir', type=str, default='PATH_TO_MOSAPT_HDF5S')
    parser.add_argument('--mosapt_train_ratio', type=float, default=0.9)
    parser.add_argument('--mosapt_split_seed', type=int, default=42)
    parser.add_argument('--mosapt_fixed_seconds', type=float, default=2.0)
    parser.add_argument('--use_focal_loss', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)


