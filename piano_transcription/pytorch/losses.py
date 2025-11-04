import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_metric_learning import losses, reducers, miners
from einops import rearrange

def technique_frame_ce_loss(output_dict, target_dict, device=None, use_active_mask=True):
    """Frame-level technique classification loss using model technique head.

    Args:
      output_dict: must contain 'technique_output' of shape (B, T, C), C=5.
      target_dict: must contain 'technique' one-hot tensor (B, T, C), C=5.
      device: optional torch.device for creating zero tensors.

    Returns:
      CrossEntropy loss over frames.
    """
    if ('technique_output' not in output_dict) or ('technique' not in target_dict):
        return torch.tensor(0.0, device=device if device is not None else None)

    probs = output_dict['technique_output']  # (B, T_pred, C)
    eps = 1e-6
    probs = probs.clamp(min=eps, max=1.0 - eps)
    logits = torch.log(probs) - torch.log(1.0 - probs)  # (B, T_pred, C)

    technique_labels = target_dict['technique']  # (B, T_tgt, C) one-hot

    # Align time dimensions if they differ
    if logits.shape[1] != technique_labels.shape[1]:
        min_T = min(logits.shape[1], technique_labels.shape[1])
        logits = logits[:, :min_T, :]
        technique_labels = technique_labels[:, :min_T, :]

    target_indices = torch.argmax(technique_labels, dim=-1)  # (B, T)

    logits_2d = logits.reshape(-1, logits.shape[-1])  # (B*T, C)
    targets_1d = target_indices.reshape(-1)  # (B*T,)

    # Optionally ignore silent frames using frame-wise activity mask
    if use_active_mask and ('frame_roll' in target_dict):
        # Active if any pitch is on
        frame_activity = (target_dict['frame_roll'].sum(dim=-1) > 0)  # (B, T_fr) bool
        # Align mask time dimension as well
        if frame_activity.shape[1] != logits.shape[1]:
            min_Tm = min(frame_activity.shape[1], logits.shape[1])
            frame_activity = frame_activity[:, :min_Tm]
            logits_2d = logits[:, :min_Tm, :].reshape(-1, logits.shape[-1])
            targets_1d = target_indices[:, :min_Tm].reshape(-1)
        active_mask = frame_activity.reshape(-1)
        if active_mask.any():
            logits_2d = logits_2d[active_mask]
            targets_1d = targets_1d[active_mask]
        else:
            return torch.tensor(0.0, device=logits.device if hasattr(logits, 'device') else device)
    ce = nn.CrossEntropyLoss(reduction='mean')
    loss = ce(logits_2d, targets_1d)

    return loss

def technique_aux_frame_ce_loss(aux_technique_model, output_dict, target_dict, device=None):
    """Auxiliary frame-level technique classification loss.

    Args:
      aux_technique_model: nn.Module mapping (B, T, 512) -> (B, T, 4)
      output_dict: must contain 'frame_features' tensor of shape (B, T, 512)
      target_dict: must contain 'technique' one-hot tensor (B, T, C), where C=5 incl. 'no_technique'
      device: optional torch.device for creating zero tensors

    Returns:
      Scalar tensor loss (CrossEntropy over all frames using provided one-hot labels).
    """
    if ('frame_features' not in output_dict) or ('technique' not in target_dict):
        return torch.tensor(0.0, device=device if device is not None else None)

    technique_logits = aux_technique_model(output_dict['frame_features'])  # (B, T, C)
    technique_labels = target_dict['technique']  # (B, T, C) one-hot including 'no_technique'

    target_indices = torch.argmax(technique_labels, dim=-1)  # (B, T)
    logits_2d = technique_logits.reshape(-1, technique_logits.shape[-1])  # (B*T, C)
    targets_1d = target_indices.reshape(-1)  # (B*T,)
    ce = nn.CrossEntropyLoss(reduction='mean')
    loss = ce(logits_2d, targets_1d)

    return loss

def supcon_loss(output_dict, target_dict):
    """Supervised contrastive loss.
    """
    loss_func = losses.SupConLoss(temperature=0.07)
    features = output_dict['reg_onset_features']   # (B, T, D)
    labels = target_dict['reg_onset_roll']         # (B, T, 88)

    features = rearrange(features, 'b t d -> (b t) d')      # (B*T, D)
    labels = rearrange(labels, 'b t p -> (b t) p')          # (B*T, 88)

    features = F.normalize(features, dim=1)

    total_loss = 0.0
    count = 0

    for pitch_class in range(labels.shape[1]):
        pitch_label = labels[:, pitch_class]   # (B*T,) binary
        mask = pitch_label > 0

        if mask.sum() < 2:
            print(pitch_class, 'pitch_class has less than 2 positive')
            continue  # 至少要有 2 個 positive

        selected_features = features[mask]  # shape (N, D)
        selected_labels = torch.zeros_like(pitch_label[mask], dtype=torch.long)

        # 所有 positive 給同一類標籤（class 0）
        loss = loss_func(selected_features, selected_labels)
        total_loss += loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)
    print(total_loss / count, 'total loss')
    return total_loss / count

def all_pitch_supcon_loss(output_dict, target_dict):
    loss_func = losses.SupConLoss(temperature=0.07)
    features = output_dict['reg_onset_features']  # (B, T, D)
    labels = target_dict['reg_onset_roll']        # (B, T, 88)

    features = rearrange(features, 'b t d -> (b t) d')   # (B*T, D)
    labels = rearrange(labels, 'b t p -> (b t) p')       # (B*T, 88)

    features = F.normalize(features, dim=1)

    # 準備 feature 與 class label
    selected_features = []
    selected_labels = []

    for pitch in range(labels.shape[1]):
        mask = labels[:, pitch] > 0
        if mask.sum() < 2:
            continue
        selected_features.append(features[mask])
        selected_labels.append(torch.full((mask.sum(),), pitch, dtype=torch.long, device=features.device))

    if len(selected_features) == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    all_feats = torch.cat(selected_features, dim=0)
    all_labs = torch.cat(selected_labels, dim=0)

    loss = loss_func(all_feats, all_labs)
    print(loss, 'all pitch supcon loss')
    return loss

def onset_binary_supcon_loss(output_dict, target_dict):
    loss_func = losses.SupConLoss(temperature=0.07)
    features = output_dict['reg_onset_features']  # (B, T, D)
    labels = target_dict['reg_onset_roll']        # (B, T, 88)

    features = rearrange(features, 'b t d -> (b t) d')  # (B*T, D)
    labels = rearrange(labels, 'b t p -> (b t) p')      # (B*T, 88)

    # 合併為 binary onset label：任一 pitch 為 1 即視為 onset
    binary_labels = (labels.sum(dim=1) > 0).long()  # shape: (B*T,)

    if binary_labels.sum() < 2 or (binary_labels == 0).sum() < 2:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    # 計算 positive 與 negative 的數量, 並隨機dropout掉數量較多的那一類（通常是negative）的labels與features，確保 positive 與 negative 的數量平衡
    positive_mask = binary_labels == 1
    negative_mask = binary_labels == 0
    
    num_positive = positive_mask.sum()
    num_negative = negative_mask.sum()
    
    # 確保平衡：取較小數量作為目標
    target_count = min(num_positive, num_negative)
    
    # 隨機選取 positive samples
    positive_indices = torch.where(positive_mask)[0]
    if len(positive_indices) > target_count:
        selected_positive_indices = positive_indices[torch.randperm(len(positive_indices))[:target_count]]
    else:
        selected_positive_indices = positive_indices
    
    # 隨機選取 negative samples
    negative_indices = torch.where(negative_mask)[0]
    if len(negative_indices) > target_count:
        selected_negative_indices = negative_indices[torch.randperm(len(negative_indices))[:target_count]]
    else:
        selected_negative_indices = negative_indices
    
    # 合併選取的 indices
    selected_indices = torch.cat([selected_positive_indices, selected_negative_indices])
    
    # 選取平衡後的 features 和 labels
    balanced_features = features[selected_indices]
    balanced_labels = binary_labels[selected_indices]

    balanced_features = F.normalize(balanced_features, dim=1)
    loss = loss_func(balanced_features, balanced_labels)
    print(loss, 'onset binary supcon loss')
    return loss

def offset_binary_supcon_loss(output_dict, target_dict):
    loss_func = losses.SupConLoss(temperature=0.07)
    features = output_dict['reg_offset_features']  # (B, T, D)
    labels = target_dict['reg_offset_roll']        # (B, T, 88)
    
    features = rearrange(features, 'b t d -> (b t) d')  # (B*T, D)
    labels = rearrange(labels, 'b t p -> (b t) p')      # (B*T, 88)
    
    # 合併為 binary offset label：任一 pitch 為 1 即視為 offset
    binary_labels = (labels.sum(dim=1) > 0).long()  # shape: (B*T,)

    if binary_labels.sum() < 2 or (binary_labels == 0).sum() < 2:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    positive_mask = binary_labels == 1
    negative_mask = binary_labels == 0
    
    num_positive = positive_mask.sum()
    num_negative = negative_mask.sum()
    
    target_count = min(num_positive, num_negative)

    positive_indices = torch.where(positive_mask)[0]
    if len(positive_indices) > target_count:
        selected_positive_indices = positive_indices[torch.randperm(len(positive_indices))[:target_count]]
    else:
        selected_positive_indices = positive_indices

    negative_indices = torch.where(negative_mask)[0]
    if len(negative_indices) > target_count:
        selected_negative_indices = negative_indices[torch.randperm(len(negative_indices))[:target_count]]
    else:
        selected_negative_indices = negative_indices

    selected_indices = torch.cat([selected_positive_indices, selected_negative_indices])

    balanced_features = features[selected_indices]
    balanced_labels = binary_labels[selected_indices]

    balanced_features = F.normalize(balanced_features, dim=1)
    loss = loss_func(balanced_features, balanced_labels)
    print(loss, 'offset binary supcon loss')
    return loss

def onset_offset_binary_supcon_loss(output_dict, target_dict):
    loss_func = losses.SupConLoss(temperature=0.07)
    onset_features = output_dict['reg_onset_features']
    offset_features = output_dict['reg_offset_features']
    onset_labels = target_dict['reg_onset_roll']
    offset_labels = target_dict['reg_offset_roll']

    onset_features = rearrange(onset_features, 'b t d -> (b t) d')
    offset_features = rearrange(offset_features, 'b t d -> (b t) d')

    onset_labels = rearrange(onset_labels, 'b t p -> (b t) p')
    offset_labels = rearrange(offset_labels, 'b t p -> (b t) p')

    onset_features = F.normalize(onset_features, dim=1)
    offset_features = F.normalize(offset_features, dim=1)
    # 
def bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be 
    deactivated when calculation BCE.
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return torch.sum(matrix * mask) / torch.sum(mask)

############ CTC losses ############
def per_pitch_onoff_ctc_loss(model, output_dict, target_dict):
    """Per-pitch CTC using existing onset/offset predictions.

    Each pitch is modeled as a 3-class CTC stream: {blank(0), onset(1), offset(2)}.

    Requires in output_dict:
      - 'reg_onset_output': (B, T, 88)
      - 'reg_offset_output': (B, T, 88)

    Requires in target_dict:
      - 'onset_roll': (B, T, 88)
      - 'offset_roll': (B, T, 88)
    """
    if ('reg_onset_output' not in output_dict) or ('reg_offset_output' not in output_dict):
        raise KeyError('per_pitch_onoff_ctc_loss requires reg_onset_output and reg_offset_output in output_dict')
    if ('onset_roll' not in target_dict) or ('offset_roll' not in target_dict):
        raise KeyError('per_pitch_onoff_ctc_loss requires onset_roll and offset_roll in target_dict')

    onset_prob = output_dict['reg_onset_output']  # (B, T, 88)
    offset_prob = output_dict['reg_offset_output']  # (B, T, 88)

    B, T, P = onset_prob.shape
    device = onset_prob.device

    eps = 1e-6
    onset_p = torch.clamp(onset_prob, eps, 1.0 - eps)
    offset_p = torch.clamp(offset_prob, eps, 1.0 - eps)
    blank_p = torch.clamp((1.0 - onset_p) * (1.0 - offset_p), eps, 1.0 - eps)

    # (B, T, P, 3) -> normalized log-probs
    log_probs = torch.stack((blank_p.log(), onset_p.log(), offset_p.log()), dim=-1)
    log_probs = torch.log_softmax(log_probs, dim=-1)
    # (T, B*P, 3)
    log_probs = log_probs.permute(1, 0, 2, 3).reshape(T, B * P, 3)

    onset_gt = target_dict['onset_roll']  # (B, T, 88)
    offset_gt = target_dict['offset_roll']  # (B, T, 88)

    target_tensors = []
    target_lengths = []
    selected_columns = []

    col_index = 0
    for b in range(B):
        for p in range(P):
            seq = []
            on_seq = onset_gt[b, :, p]
            off_seq = offset_gt[b, :, p]
            for t in range(T):
                if on_seq[t] > 0.5:
                    seq.append(1)
                if off_seq[t] > 0.5:
                    seq.append(2)
            if len(seq) > 0:
                target_tensors.append(torch.tensor(seq, dtype=torch.long, device=device))
                target_lengths.append(len(seq))
                selected_columns.append(col_index)
            col_index += 1

    if len(selected_columns) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    log_probs = log_probs[:, selected_columns, :]
    input_lengths = torch.full((len(selected_columns),), T, dtype=torch.long, device=device)
    targets = torch.cat(target_tensors, dim=0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)

    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    return loss

############ High-resolution regression loss ############
def regress_onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression, velocity regression and frame-wise classification losses.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss


def regress_pedal_bce(model, output_dict, target_dict):
    """High-resolution piano pedal regression loss, including pedal onset 
    regression, pedal offset regression and pedal frame-wise classification losses.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['reg_pedal_onset_roll'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['reg_pedal_offset_roll'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss

############ Google's onsets and frames system loss ############
def google_onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    """Google's onsets and frames system piano note loss. Only used for comparison.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss


def google_pedal_bce(model, output_dict, target_dict):
    """Google's onsets and frames system piano pedal loss. Only used for comparison.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['pedal_onset_roll'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['pedal_offset_roll'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss


def get_loss_func(loss_type):
    if loss_type == 'none':
        return lambda model, output_dict, target_dict: torch.tensor(0.0, device=next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else None)
    if loss_type == 'regress_onset_offset_frame_velocity_bce':
        return regress_onset_offset_frame_velocity_bce

    elif loss_type == 'regress_pedal_bce':
        return regress_pedal_bce

    elif loss_type == 'google_onset_offset_frame_velocity_bce':
        return google_onset_offset_frame_velocity_bce

    elif loss_type == 'google_pedal_bce':
        return google_pedal_bce

    elif loss_type == 'per_pitch_onoff_ctc':
        return per_pitch_onoff_ctc_loss

    else:
        raise Exception('Incorrect loss_type!')