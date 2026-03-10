import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from utilities import (create_folder, get_filename, create_logging, 
    StatisticsContainer, RegressionPostProcessor) 
from data_generator import MaestroDataset, Augmentor, Sampler, TestSampler, collate_fn
# from data_generator import MosaDataset, MosaSampler, MosaTestSampler
from data_generator import CustomDataset, CustomSampler,CustomTestSampler
from data_generator import RWCTechniqueDataset, RWCTechniqueSampler, RWCTechniqueTestSampler
from data_generator import RWCTechNoteWavDataset
from models_contrast import Regress_onset_offset_frame_velocity_CRNN, Regress_pedal_CRNN, AcousticModelCRnn8Dropout, NoteLevelTechniqueModel
from pytorch_utils import move_data_to_device
from losses import (
    get_loss_func,
    supcon_loss,
    all_pitch_supcon_loss,
    onset_binary_supcon_loss,
    offset_binary_supcon_loss,
    per_pitch_onoff_ctc_loss,
    technique_frame_ce_loss,
)
from evaluate import SegmentEvaluator
import config

from pytorch_metric_learning import losses, reducers, miners
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

def train(args):
    """Train a piano transcription system.

    Args:
      workspace: str, directory of your workspace
      model_type: str, e.g. 'Regressonset_regressoffset_frame_velocity_CRNN'
      loss_type: str, e.g. 'regress_onset_offset_frame_velocity_bce'
      augmentation: str, e.g. 'none'
      batch_size: int
      learning_rate: float
      reduce_iteration: int
      resume_iteration: int
      early_stop: int
      device: 'cuda' | 'cpu'
      mini_data: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    max_note_shift = args.max_note_shift
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    num_workers = args.num_workers

    device = torch.device(f"cuda:{args.device}") if args.device >= 0 and torch.cuda.is_available() else torch.device('cpu')
    mini_data = args.mini_data
    filename = args.filename
    dataset = args.dataset
    logdir = args.logdir
    model_tag = args.model_tag
    contrast_weight = args.contrast_weight
    ctc_weight = args.ctc_weight
    use_cosine_annealing_warm_restarts = args.use_cosine_annealing_warm_restarts
    use_aux_contrast_model = args.use_aux_contrast_model
    print('contrast weight: ', contrast_weight)
    print('use cosine annealing warm restarts: ', use_cosine_annealing_warm_restarts)
    print('use aux contrast model: ', use_aux_contrast_model)
    print('ctc weight: ', ctc_weight)
    technique_weight = getattr(args, 'technique_weight', 0.0)
    print('technique weight: ', technique_weight)

    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    # num_workers = 8

    # Loss function
    loss_func = get_loss_func(loss_type)

    # # Supervised contrastive loss
    # supcon_loss = losses.SupConLoss(temperature=0.07) # input: features, labels
    
    # TensorBoard writer
    logdir = os.path.join(logdir, args.model_tag)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    logging.info(f'TensorBoard logs will be saved to: {logdir}')

    # Paths
    hdf5s_dir = os.path.join(workspace, 'hdf5s', dataset)
    # Special handling for single-file RWC dataset: symlink the file into workspace/hdf5s/rwc
    if dataset == 'rwc':
        rwc_src = os.path.expanduser(getattr(args, 'rwc_h5_path', '~/data/rwc_processed_data.h5'))
        if not os.path.exists(rwc_src):
            raise FileNotFoundError(f"RWC h5 file not found at {rwc_src}. Set --rwc_h5_path correctly.")
        create_folder(hdf5s_dir)
        link_name = os.path.join(hdf5s_dir, os.path.basename(rwc_src))
        try:
            if os.path.islink(link_name):
                # Update link if it points elsewhere
                current_target = os.readlink(link_name)
                if current_target != rwc_src:
                    os.remove(link_name)
                    os.symlink(rwc_src, link_name)
            elif not os.path.exists(link_name):
                os.symlink(rwc_src, link_name)
        except OSError:
            # Fall back: copy if symlink not permitted
            if not os.path.exists(link_name):
                import shutil
                shutil.copyfile(rwc_src, link_name)


    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, model_tag,
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift),
        'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename, model_tag,
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size), 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, model_tag,
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size))
    create_folder(logs_dir)

    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # Model
    Model = eval(model_type)
    model = Model(frames_per_second=frames_per_second, classes_num=classes_num, output_features=True, predict_technique=(technique_weight > 0))
    aux_contrast_onset_model = nn.Linear(512, 128)
    aux_contrast_offset_model = nn.Linear(512, 128)
    # aux_contrast_model = nn.Sequential(
    #     nn.Linear(512, 128),
    #     nn.BatchNorm1d(128),
    #     nn.ReLU(),
    # )
    # Using model's built-in technique head; remove auxiliary technique models


    if augmentation == 'none':
        augmentor = None
    elif augmentation == 'aug':
        augmentor = Augmentor()
    else:
        raise Exception('Incorrect argumentation!')
    
    # Dataset & Samplers
    if dataset in ['mosa', 'mosapt', 'mixed', 'rwc', 'rwc_tech', 'rwc_tech_note_wav', 'viotech']:
        if dataset == 'mixed':
            # Build MOSA and MOSAPT datasets
            hdf5s_dir_mosa = os.path.join(workspace, 'hdf5s', 'mosa')
            hdf5s_dir_mosapt = os.path.join(workspace, 'hdf5s', 'mosapt')

            train_dataset_mosa = CustomDataset(
                hdf5s_dir=hdf5s_dir_mosa,
                segment_seconds=segment_seconds,
                frames_per_second=frames_per_second,
                max_note_shift=max_note_shift,
                augmentor=augmentor,
                include_technique_label=False,
            )
            train_dataset_mosapt = CustomDataset(
                hdf5s_dir=hdf5s_dir_mosapt,
                segment_seconds=segment_seconds,
                frames_per_second=frames_per_second,
                max_note_shift=max_note_shift,
                augmentor=augmentor,
                include_technique_label=True,
            )

            # Independent samplers for underlying datasets
            train_sampler_mosa = CustomSampler(
                hdf5s_dir=hdf5s_dir_mosa,
                split='train',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data,
            )
            train_sampler_mosapt = CustomSampler(
                hdf5s_dir=hdf5s_dir_mosapt,
                split='train',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data,
            )
            # Mixed dataset wrapper dispatching metas to the correct base dataset
            class MixedDataset(object):
                def __init__(self, mosa_ds, mosapt_ds):
                    self.mosa_ds = mosa_ds
                    self.mosapt_ds = mosapt_ds

                def __getitem__(self, tagged_meta):
                    src, meta = tagged_meta  # src in {'mosa', 'mosapt'}
                    if src == 'mosa':
                        return self.mosa_ds[meta]
                    else:
                        return self.mosapt_ds[meta]

            train_dataset = MixedDataset(train_dataset_mosa, train_dataset_mosapt)

            # Batch sampler that yields tagged metas
            class MixedBatchSampler(object):
                def __init__(self, sampler_a, sampler_b, mosa_ratio=0.5):
                    self.sampler_a = sampler_a
                    self.sampler_b = sampler_b
                    self.mosa_ratio = mosa_ratio

                def __iter__(self):
                    iter_a = iter(self.sampler_a)
                    iter_b = iter(self.sampler_b)
                    while True:
                        use_b = (np.random.rand() < self.mosa_ratio)
                        if use_b:
                            try:
                                batch = next(iter_b)
                                yield [('mosapt', m) for m in batch]
                                continue
                            except StopIteration:
                                iter_b = iter(self.sampler_b)
                        try:
                            batch = next(iter_a)
                            yield [('mosa', m) for m in batch]
                        except StopIteration:
                            iter_a = iter(self.sampler_a)

                def __len__(self):
                    return -1

                def state_dict(self):
                    state = {}
                    if hasattr(self.sampler_a, 'state_dict'):
                        state['sampler_a'] = self.sampler_a.state_dict()
                    if hasattr(self.sampler_b, 'state_dict'):
                        state['sampler_b'] = self.sampler_b.state_dict()
                    return state

                def load_state_dict(self, state):
                    if 'sampler_a' in state and hasattr(self.sampler_a, 'load_state_dict'):
                        self.sampler_a.load_state_dict(state['sampler_a'])
                    if 'sampler_b' in state and hasattr(self.sampler_b, 'load_state_dict'):
                        self.sampler_b.load_state_dict(state['sampler_b'])

            train_sampler = MixedBatchSampler(train_sampler_mosa, train_sampler_mosapt, mosa_ratio=args.mosa_ratio)

            # Evaluation on both MOSA and MOSAPT
            evaluate_dataset_mosa = CustomDataset(
                hdf5s_dir=hdf5s_dir_mosa,
                segment_seconds=segment_seconds,
                frames_per_second=frames_per_second,
                max_note_shift=0,
                include_technique_label=False,
            )
            evaluate_dataset_mosapt = CustomDataset(
                hdf5s_dir=hdf5s_dir_mosapt,
                segment_seconds=segment_seconds,
                frames_per_second=frames_per_second,
                max_note_shift=0,
                include_technique_label=True,
            )
            evaluate_dataset = MixedDataset(evaluate_dataset_mosa, evaluate_dataset_mosapt)

            evaluate_train_sampler_mosa = CustomTestSampler(
                hdf5s_dir=hdf5s_dir_mosa,
                split='train',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data,
            )
            evaluate_train_sampler_mosapt = CustomTestSampler(
                hdf5s_dir=hdf5s_dir_mosapt,
                split='train',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data,
            )
            # Finite mixed sampler for evaluation (no iterator reset)
            class MixedEvalBatchSampler(object):
                def __init__(self, sampler_a, sampler_b, mosa_ratio=0.5):
                    self.sampler_a = sampler_a
                    self.sampler_b = sampler_b
                    self.mosa_ratio = mosa_ratio


                def __iter__(self):
                    iter_a = iter(self.sampler_a)
                    iter_b = iter(self.sampler_b)

                    a_exhausted = False
                    b_exhausted = False

                    while True:
                        use_b = (np.random.rand() < self.mosa_ratio)
                        if use_b and not b_exhausted:
                            try:
                                batch = next(iter_b)
                                yield [('mosapt', m) for m in batch]
                                continue
                            except StopIteration:
                                b_exhausted = True
                        if not a_exhausted:
                            try:
                                batch = next(iter_a)
                                yield [('mosa', m) for m in batch]
                            except StopIteration:
                                a_exhausted = True

                        if a_exhausted and b_exhausted:
                            break

                def __len__(self):
                    return -1

            evaluate_train_sampler = MixedEvalBatchSampler(
                evaluate_train_sampler_mosa, evaluate_train_sampler_mosapt, mosa_ratio=0.5
            )

            evaluate_test_sampler_mosa = CustomTestSampler(
                hdf5s_dir=hdf5s_dir_mosa,
                split='test',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data,
            )
            evaluate_test_sampler_mosapt = CustomTestSampler(
                hdf5s_dir=hdf5s_dir_mosapt,
                split='test',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data,
            )
            evaluate_test_sampler = MixedEvalBatchSampler(
                evaluate_test_sampler_mosa, evaluate_test_sampler_mosapt, mosa_ratio=0.5
            )
        else:
            if dataset == 'rwc_tech':
                # Technique-only training using RWC single H5
                rwc_src = os.path.expanduser(getattr(args, 'rwc_h5_path', '~/data/rwc_processed_data.h5'))
                if not os.path.exists(rwc_src):
                    raise FileNotFoundError(f"RWC h5 file not found at {rwc_src}. Set --rwc_h5_path correctly.")

                # Build deterministic file-level split
                import h5py
                with h5py.File(rwc_src, 'r') as f:
                    all_files = sorted([k for k in f.keys() if k.startswith('file_')])
                rng = np.random.RandomState(getattr(args, 'rwc_split_seed', 1234))
                rng.shuffle(all_files)
                max_files = getattr(args, 'rwc_max_files', 0)
                if max_files and max_files > 0:
                    all_files = all_files[:max_files]
                train_ratio = float(getattr(args, 'rwc_train_ratio', 0.9))
                split_idx = int(len(all_files) * train_ratio)
                rwc_train_files = all_files[:split_idx]
                rwc_test_files = all_files[split_idx:]

                # Datasets
                train_dataset = RWCTechniqueDataset(
                    rwc_h5_path=rwc_src,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=max_note_shift,
                    augmentor=augmentor,
                )
                evaluate_dataset = RWCTechniqueDataset(
                    rwc_h5_path=rwc_src,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=0,
                    augmentor=None,
                )

                # Samplers
                train_sampler = RWCTechniqueSampler(
                    rwc_h5_path=rwc_src,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                    max_files=getattr(args, 'rwc_max_files', 0),
                    allowed_file_keys=rwc_train_files,
                )

                # Evaluation samplers (file-level split)
                evaluate_train_sampler = RWCTechniqueTestSampler(
                    rwc_h5_path=rwc_src,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                    max_files=getattr(args, 'rwc_max_files', 0),
                    allowed_file_keys=rwc_train_files,
                )
                evaluate_test_sampler = RWCTechniqueTestSampler(
                    rwc_h5_path=rwc_src,
                    split='test',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                    max_files=getattr(args, 'rwc_max_files', 0),
                    allowed_file_keys=rwc_test_files,
                )
            elif dataset == 'rwc_tech_note_wav':
                # Note-level technique dataset from exported WAVs
                notes_root = os.path.expanduser(getattr(args, 'rwc_notes_root', './rwc_notes'))
                train_dataset = RWCTechNoteWavDataset(notes_root, frames_per_second=frames_per_second, return_logmel=True)
                evaluate_dataset = RWCTechNoteWavDataset(notes_root, frames_per_second=frames_per_second, return_logmel=True)
                # Simple sequential sampler using DataLoader default sampler (no batch_sampler)
                train_sampler = None
                evaluate_train_sampler = None
                evaluate_test_sampler = None
            elif dataset == 'viotech':
                # Similar to mosa/mosapt but for viotech
                train_dataset = CustomDataset(
                    hdf5s_dir=hdf5s_dir,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=max_note_shift,
                    augmentor=augmentor,
                    include_technique_label=False, # Use False for now as per "mimic mosa"
                )
                evaluate_dataset = CustomDataset(
                    hdf5s_dir=hdf5s_dir,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=0,
                    include_technique_label=False,
                )

                # Sampler for training
                train_sampler = CustomSampler(
                    hdf5s_dir=hdf5s_dir,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )

                # Samplers for evaluation (single dataset case only)
                evaluate_train_sampler = CustomTestSampler(
                    hdf5s_dir=hdf5s_dir,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )
                evaluate_test_sampler = CustomTestSampler(
                    hdf5s_dir=hdf5s_dir,
                    split='test',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )
            else:
                train_dataset = CustomDataset(
                    hdf5s_dir=hdf5s_dir,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=max_note_shift,
                    augmentor=augmentor,
                    include_technique_label=(dataset == 'mosapt'),
                )
                evaluate_dataset = CustomDataset(
                    hdf5s_dir=hdf5s_dir,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=0,
                    include_technique_label=(dataset == 'mosapt'),
                )

                # Sampler for training
                train_sampler = CustomSampler(
                    hdf5s_dir=hdf5s_dir,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )

                # Samplers for evaluation (single dataset case only)
                evaluate_train_sampler = CustomTestSampler(
                    hdf5s_dir=hdf5s_dir,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )
                evaluate_test_sampler = CustomTestSampler(
                    hdf5s_dir=hdf5s_dir,
                    split='test',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )
    else:
        train_dataset = MaestroDataset(
            hdf5s_dir=hdf5s_dir,
            segment_seconds=segment_seconds,
            frames_per_second=frames_per_second,
            max_note_shift=max_note_shift,
            augmentor=augmentor,
        )
        evaluate_dataset = MaestroDataset(
            hdf5s_dir=hdf5s_dir,
            segment_seconds=segment_seconds,
            frames_per_second=frames_per_second,
            max_note_shift=0,
        )

        # Sampler for training
        train_sampler = Sampler(
            hdf5s_dir=hdf5s_dir,
            split='train',
            segment_seconds=segment_seconds,
            hop_seconds=hop_seconds,
            batch_size=batch_size,
            mini_data=mini_data,
        )

        # Samplers for evaluation
        evaluate_train_sampler = TestSampler(
            hdf5s_dir=hdf5s_dir,
            split='train',
            segment_seconds=segment_seconds,
            hop_seconds=hop_seconds,
            batch_size=batch_size,
            mini_data=mini_data,
        )
        evaluate_test_sampler = TestSampler(
            hdf5s_dir=hdf5s_dir,
            split='test',
            segment_seconds=segment_seconds,
            hop_seconds=hop_seconds,
            batch_size=batch_size,
            mini_data=mini_data,
        )


    # Dataloader
    if dataset == 'rwc_tech_note_wav':
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        evaluate_train_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset,
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = evaluate_train_loader
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
            batch_sampler=train_sampler, collate_fn=collate_fn, 
            num_workers=num_workers, pin_memory=True)

        evaluate_train_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
            batch_sampler=evaluate_train_sampler, collate_fn=collate_fn, 
            num_workers=num_workers, pin_memory=True)
            
        test_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
            batch_sampler=evaluate_test_sampler, collate_fn=collate_fn, 
            num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = SegmentEvaluator(model, batch_size)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    # Optimize params: for rwc_tech with technique-only training, freeze non-technique branches
    if dataset == 'rwc_tech' and technique_weight > 0:
        for name, p in model.named_parameters():
            is_tech = ('technique_model' in name) or ('technique_fc' in name) or ('bn0' in name) or ('logmel_extractor' in name) or ('spectrogram_extractor' in name)
            # Keep front-end (spec/logmel) trainable if you want full finetune; here we freeze them
            if is_tech:
                p.requires_grad = True
            else:
                p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    if use_aux_contrast_model:
        params += list(aux_contrast_onset_model.parameters())
        params += list(aux_contrast_offset_model.parameters())

    optimizer = optim.Adam(params, lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    if use_cosine_annealing_warm_restarts:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=early_stop//2, T_mult=2, eta_min=0.05*learning_rate)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=early_stop, eta_min=0.05*learning_rate)

    # Load pretrained checkpoint if available
    if args.pretrain_path is not None and os.path.exists(args.pretrain_path):
        logging.info('Loading pretrained checkpoint from {}'.format(args.pretrain_path))
        checkpoint = torch.load(args.pretrain_path, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model'], strict=False)
        except TypeError:
            # For older torch versions without strict kw-only
            model.load_state_dict(checkpoint['model'])
        logging.info('Pretrained checkpoint loaded successfully')

    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            model_type, 'loss_type={}'.format(loss_type), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
                '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']

    else:
        iteration = 0
    
    # model.to("cuda")

    if 'cuda' in str(device):
        model.to(device)
        if use_aux_contrast_model:
            aux_contrast_onset_model.to(device)
            aux_contrast_offset_model.to(device)
    else:
        print('Using CPU.....')

    # Parallel
    # print('GPU number: {}'.format(torch.cuda.device_count()))
    # model = torch.nn.DataParallel(model)
    

    train_bgn_time = time.time()

    for batch_data_dict in train_loader:

        # Evaluation 
        if iteration % 10 == 0:# and iteration > 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()

            logging.info('Evaluating on train set...')
            evaluate_train_statistics = evaluator.evaluate(evaluate_train_loader)
            logging.info('Evaluating on test set...')
            test_statistics = evaluator.evaluate(test_loader)

            logging.info('    Train statistics: {}'.format(evaluate_train_statistics))
            # logging.info('    Validation statistics: {}'.format(validate_statistics))
            logging.info('    Test statistics: {}'.format(test_statistics))

            # Log evaluation metrics to TensorBoard
            if evaluate_train_statistics:
                for key, value in evaluate_train_statistics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'train/{key}', value, iteration)
            
            if test_statistics:
                for key, value in test_statistics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'test/{key}', value, iteration)

            statistics_container.append(iteration, evaluate_train_statistics, data_type='train')
            # statistics_container.append(iteration, validate_statistics, data_type='validation')
            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()
        
        # Save model
        if iteration % 1000 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.state_dict(), 
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        # Reduce learning rate
        # if iteration % reduce_iteration == 0 and iteration > 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.9

        # Per PyTorch docs, call optimizer.step() before scheduler.step()
        # We log LR after stepping scheduler to reflect the updated value
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], iteration)
        scheduler.step()
        
        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
         
        model.train()
        if dataset == 'rwc_tech_note_wav':
            # Build a forward dict compatible with technique loss: logits (B,1,C) and labels (B,1,C)
            # Prepare inputs
            if 'logmel' in batch_data_dict:
                logmel = move_data_to_device(batch_data_dict['logmel'], device)
                # Ensure shape (B, 1, T, F)
                if logmel.ndim == 3:
                    logmel = logmel[:, None, :, :]
            else:
                # If logmel not provided, compute from waveform (fallback)
                logmel = None
            # Build (or cache) note model on first iter
            if not hasattr(train, '_note_model'):
                train._note_model = NoteLevelTechniqueModel(classes_num=5, sample_rate=sample_rate, 
                                                            frames_per_second=frames_per_second, use_trans_features=use_trans_features).to(device)
            note_model = train._note_model
            logits = note_model(logmel)
            probs = torch.softmax(logits, dim=-1)
            B = probs.shape[0]
            batch_output_dict = {'technique_output': probs[:, None, :]}  # (B,1,C)
        else:
            batch_output_dict = model(batch_data_dict['waveform'])
        if use_aux_contrast_model:
            aux_contrast_onset_model.train()
            aux_contrast_offset_model.train()
            batch_data_dict['reg_onset_features'] = aux_contrast_onset_model(batch_output_dict['reg_onset_features'])
            batch_data_dict['reg_offset_features'] = aux_contrast_offset_model(batch_output_dict['reg_offset_features'])


        # Technique classification loss only when batch has technique labels
        if technique_weight > 0 and ('technique' in batch_data_dict):
            loss_technique = technique_frame_ce_loss(
                batch_output_dict,
                batch_data_dict,
                device=device,
            )
        else:
            loss_technique = torch.tensor(0.0, device=device)

        if contrast_weight > 0:
            loss_contrast = onset_binary_supcon_loss(batch_output_dict, batch_data_dict) + offset_binary_supcon_loss(batch_output_dict, batch_data_dict)
        else:
            loss_contrast = torch.tensor(0.0, device=device)

        if ctc_weight > 0:
            loss_ctc = per_pitch_onoff_ctc_loss(model, batch_output_dict, batch_data_dict)
        else:
            loss_ctc = torch.tensor(0.0, device=device)

        base_loss = loss_func(model, batch_output_dict, batch_data_dict)
        if ctc_weight > 0:
            loss_ctc = per_pitch_onoff_ctc_loss(model, batch_output_dict, batch_data_dict)
        else:
            loss_ctc = torch.tensor(0.0, device=device)

        loss = base_loss + contrast_weight * loss_contrast + ctc_weight * loss_ctc + technique_weight * loss_technique

        # Log loss to TensorBoard
        writer.add_scalar('train/loss', loss.item(), iteration)
        writer.add_scalar('train/loss_base', base_loss.item(), iteration)

        if contrast_weight > 0:
            writer.add_scalar('train/loss_contrast', loss_contrast.item(), iteration)
        if ctc_weight > 0:
            writer.add_scalar('train/loss_ctc', loss_ctc.item(), iteration)
        if technique_weight > 0:
            writer.add_scalar('train/loss_technique', loss_technique.item(), iteration)

        preview_loss = {
            'loss': loss.item(),
            'base_loss': base_loss.item(),
        }

        if ctc_weight > 0:
            # print('iteration:', iteration, 'loss:', loss.item(), 'base_loss:', base_loss.item(), 'loss_contrast:', loss_contrast.item(), 'loss_ctc:', loss_ctc.item())
            preview_loss['loss_ctc'] = loss_ctc.item()
        if technique_weight > 0:
            # print('iteration:', iteration, 'loss:', loss.item(), 'base_loss:', base_loss.item(), 'loss_contrast:', loss_contrast.item(), 'loss_technique:', loss_technique.item())
            preview_loss['loss_technique'] = loss_technique.item()
        if contrast_weight > 0:
            # print('iteration:', iteration, 'loss:', loss.item(), 'base_loss:', base_loss.item(), 'loss_contrast:', loss_contrast.item())
            preview_loss['loss_contrast'] = loss_contrast.item()
        
        # flatten the string of preview_loss
        preview_loss_str = f'{iteration} | ' + '| '.join([f'{k}: {v:.6f}' for k, v in preview_loss.items()])
        print(preview_loss_str)

        # Backward
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1
    
    # Close TensorBoard writer
    writer.close()
    logging.info('Training completed. TensorBoard writer closed.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, required=True, choices=['none', 'aug'])
    parser_train.add_argument('--max_note_shift', type=int, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--reduce_iteration', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--device', type=int, default=None, help='-1 for CPU, 0 for GPU 0, 1 for GPU 1, etc.')
    parser_train.add_argument('--pretrain_path', type=str, default=None)
    parser_train.add_argument('--dataset', type=str, choices=['mosa', 'maestro', 'mosapt', 'mixed', 'rwc', 'rwc_tech', 'rwc_tech_note_wav', 'viotech'], default='mosa')
    parser_train.add_argument('--logdir', type=str, required=True)
    parser_train.add_argument('--model_tag', type=str, required=True)
    parser_train.add_argument('--contrast_weight', type=float, default=0.0)
    parser_train.add_argument('--ctc_weight', type=float, default=0.0)
    parser_train.add_argument('--use_aux_contrast_model', action='store_true', default=False)
    parser_train.add_argument('--use_cosine_annealing_warm_restarts', action='store_true', default=False)
    parser_train.add_argument('--technique_weight', type=float, default=0.0)
    parser_train.add_argument('--mosa_ratio', type=float, default=0.5)
    parser_train.add_argument('--num_workers', type=int, default=8)
    parser_train.add_argument('--rwc_h5_path', type=str, default='~/data/rwc_processed_data.h5')
    parser_train.add_argument('--rwc_train_ratio', type=float, default=0.9)
    parser_train.add_argument('--rwc_split_seed', type=int, default=1234)
    parser_train.add_argument('--rwc_max_files', type=int, default=0)
    parser_train.add_argument('--rwc_notes_root', type=str, default='./rwc_notes')
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')