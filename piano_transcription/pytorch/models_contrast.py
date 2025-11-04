import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from pytorch_utils import move_data_to_device


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        
        return x


class AcousticModelCRnn8Dropout(nn.Module):
    def __init__(self, classes_num, midfeat, momentum, output_features=False):
        super(AcousticModelCRnn8Dropout, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=momentum)

        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)

        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()
        if output_features:
            self.output_features = True
        else:
            self.output_features = False

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, time_steps, classes_num)
        """

        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        
        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        output = torch.sigmoid(self.fc(x))

        if self.output_features:
            return output, x
        else:
            return output


class Regress_onset_offset_frame_velocity_CRNN(nn.Module):
    def __init__(self, frames_per_second, classes_num, output_features=False, predict_technique=False):
        super(Regress_onset_offset_frame_velocity_CRNN, self).__init__()

        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        midfeat = 1792
        momentum = 0.01
        technique_classes_num = 5
        self.predict_technique = predict_technique

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)

        self.frame_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum, output_features)
        self.reg_onset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum, output_features)
        self.reg_offset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum, output_features)
        self.velocity_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum, output_features)
        if self.predict_technique:
            # Use local-context technique model that consumes precomputed logmel features (no global sequence context)
            self.technique_model = LocalTechniqueFeatureConvModel(
                classes_num=technique_classes_num,
                window_size=10,
                proj_size=256,
                hidden_size=128,
                momentum=momentum,
                output_features=output_features,
            )
            self.use_local_technique = True

        self.technique_gru = nn.GRU(input_size=88 * 3 + 5, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)
        if self.predict_technique:
            # self.technique_fc = nn.Linear(512, technique_classes_num, bias=True)
            self.technique_fc = nn.Linear(269, technique_classes_num, bias=True)

        self.frame_gru = nn.GRU(input_size=88 * 3, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()
        if output_features:
            self.output_features_dict = {}
            self.output_features = True
        else:
            self.output_features = False
        

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)
        if self.predict_technique:
            init_gru(self.technique_gru)
            init_layer(self.technique_fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.output_features:
            frame_output, frame_features = self.frame_model(x)
            reg_onset_output, reg_onset_features = self.reg_onset_model(x)
            reg_offset_output, reg_offset_features = self.reg_offset_model(x)
            velocity_output, velocity_features = self.velocity_model(x)
            if self.predict_technique:
                technique_output, technique_features = self.technique_model(x)
        else:
            frame_output = self.frame_model(x)  # (batch_size, time_steps, classes_num)
            reg_onset_output = self.reg_onset_model(x)  # (batch_size, time_steps, classes_num)
            reg_offset_output = self.reg_offset_model(x)    # (batch_size, time_steps, classes_num)
            velocity_output = self.velocity_model(x)    # (batch_size, time_steps, classes_num)
            if self.predict_technique:
                technique_output = self.technique_model(x)    # (batch_size, time_steps, classes_num)

        # Use velocities to condition onset regression
        x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()), dim=2)
        (x, _) = self.reg_onset_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))
        """(batch_size, time_steps, classes_num)"""

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        frame_output = torch.sigmoid(self.frame_fc(x))  # (batch_size, time_steps, classes_num)
        """(batch_size, time_steps, classes_num)"""

        # Use onset, offset, frame, technique features to condition technique classification
        if self.predict_technique:
            x = torch.cat((reg_onset_output.detach(), reg_offset_output.detach(), frame_output.detach(), technique_output), dim=2)
            # (x, _) = self.technique_gru(x)
            x = F.dropout(x, p=0.5, training=self.training, inplace=False)
            technique_output = torch.sigmoid(self.technique_fc(x))  # (batch_size, time_steps, classes_num)
        """(batch_size, time_steps, classes_num)"""


        output_dict = {
            'reg_onset_output': reg_onset_output, 
            'reg_offset_output': reg_offset_output, 
            'frame_output': frame_output, 
            'velocity_output': velocity_output,
            # 'frame_features': frame_features,
            # 'reg_onset_features': reg_onset_features,
            # 'reg_offset_features': reg_offset_features,
            # 'velocity_features': velocity_features,
        }
        if self.output_features:
            output_dict['frame_features'] = frame_features
            output_dict['reg_onset_features'] = reg_onset_features
            output_dict['reg_offset_features'] = reg_offset_features
            output_dict['velocity_features'] = velocity_features
        
        if self.predict_technique:
            output_dict['technique_output'] = technique_output
        
        if self.predict_technique and self.output_features:
            output_dict['technique_features'] = technique_features

        # if self.output_features:
        #     self.output_features_dict['frame_features'] = frame_features
        #     self.output_features_dict['reg_onset_features'] = reg_onset_features
        #     self.output_features_dict['reg_offset_features'] = reg_offset_features
        #     self.output_features_dict['velocity_features'] = velocity_features

        #     return output_dict, self.output_features_dict

        return output_dict


class Regress_pedal_CRNN(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Regress_pedal_CRNN, self).__init__()

        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        midfeat = 1792
        momentum = 0.01

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)

        self.reg_pedal_onset_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        self.reg_pedal_offset_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        self.reg_pedal_frame_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        
    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        reg_pedal_onset_output = self.reg_pedal_onset_model(x)  # (batch_size, time_steps, classes_num)
        reg_pedal_offset_output = self.reg_pedal_offset_model(x)  # (batch_size, time_steps, classes_num)
        pedal_frame_output = self.reg_pedal_frame_model(x)  # (batch_size, time_steps, classes_num)
        
        output_dict = {
            'reg_pedal_onset_output': reg_pedal_onset_output, 
            'reg_pedal_offset_output': reg_pedal_offset_output,
            'pedal_frame_output': pedal_frame_output}

        return output_dict


# This model is not trained, but is combined from the trained note and pedal models.
class Note_pedal(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        """The combination of note and pedal model.
        """
        super(Note_pedal, self).__init__()

        self.note_model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second, classes_num)
        self.pedal_model = Regress_pedal_CRNN(frames_per_second, classes_num)

    def load_state_dict(self, m, strict=False):
        self.note_model.load_state_dict(m['note_model'], strict=strict)
        self.pedal_model.load_state_dict(m['pedal_model'], strict=strict)

    def forward(self, input):
        note_output_dict = self.note_model(input)
        pedal_output_dict = self.pedal_model(input)

        full_output_dict = {}
        full_output_dict.update(note_output_dict)
        full_output_dict.update(pedal_output_dict)
        return full_output_dict


class LocalTechniqueFeatureConvModel(nn.Module):
    def __init__(
        self,
        classes_num: int = 5,
        window_size: int = 10,
        proj_size: int = 256,
        hidden_size: int = 128,
        momentum: float = 0.01,
        output_features: bool = False,
        block_time_kernels: Optional[Sequence[int]] = [3, 3, 3, 3],
    ):
        """Technique model that takes precomputed logmel features (B, 1, T, F) and limits
        temporal context to ~window_size frames via Conv1d.

        This mirrors `AcousticModelCRnn8Dropout`'s channel progression but without temporal GRUs,
        and ensures strictly local temporal dependence using a temporal Conv1d head.
        """
        super(LocalTechniqueFeatureConvModel, self).__init__()

        # Configure temporal kernel sizes for the 4 front conv blocks.
        # By default uses [3, 3, 3, 3] which contributes (k-1) per block -> 8 frames total.
        if block_time_kernels is None:
            block_time_kernels = [3, 3, 3, 3]
        if len(block_time_kernels) != 4:
            raise ValueError("block_time_kernels must have length 4")
        for k in block_time_kernels:
            if k < 1:
                raise ValueError("Temporal kernel must be >= 1")
            if k % 2 == 0:
                raise ValueError("Temporal kernels must be odd to preserve length with symmetric padding")

        def freq_only_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_ch, momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_ch, momentum),
                nn.ReLU(inplace=True),
            )

        def freq_time_block(in_ch: int, out_ch: int, time_kernel: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=(time_kernel, 3),
                    padding=(time_kernel // 2, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch, momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_ch, momentum),
                nn.ReLU(inplace=True),
            )

        self.conv_block1 = freq_time_block(1, 48, block_time_kernels[0])
        self.conv_block2 = freq_time_block(48, 64, block_time_kernels[1])
        self.conv_block3 = freq_time_block(64, 96, block_time_kernels[2])
        self.conv_block4 = freq_time_block(96, 128, block_time_kernels[3])

        midfeat = 1792  # 128 channels * ~14 mel bins after pooling
        self.fc_proj = nn.Linear(midfeat, proj_size, bias=False)
        self.bn_proj = nn.BatchNorm1d(proj_size, momentum=momentum)

        # Keep total temporal receptive field within window_size.
        # Front-end temporal contribution is sum_over_blocks(k_i - 1).
        front_added = sum(max(0, k - 1) for k in block_time_kernels)
        # Head kernel covers the remaining context.
        head_kernel = max(1, window_size - front_added)
        left_pad = head_kernel // 2
        right_pad = head_kernel - 1 - left_pad
        self.temporal_pad = nn.ConstantPad1d((left_pad, right_pad), 0.0)
        self.temporal_conv = nn.Conv1d(
            in_channels=proj_size,
            out_channels=hidden_size,
            kernel_size=head_kernel,
            padding=0,
            bias=True,
        )
        self.head_bn = nn.BatchNorm1d(hidden_size, momentum=momentum)
        self.head_act = nn.ReLU(inplace=True)
        self.head_dropout = nn.Dropout(p=0.2)
        self.pointwise = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=classes_num,
            kernel_size=1,
            bias=True,
        )

        self.output_features = output_features

    def forward(self, x: torch.Tensor):
        """Forward.

        Args:
          x: precomputed logmel tensor (B, 1, T, F)
        Returns:
          probs: (B, T, classes_num) in [0, 1]
        """
        # Frequency-only conv blocks with freq pooling
        x = self.conv_block1(x)
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x)
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x)
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x)
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        x = F.dropout(x, p=0.2, training=self.training)

        # (B, C, T, F') -> (B, T, C*F')
        x = x.transpose(1, 2).flatten(2)

        # Frame-wise projection
        x = self.fc_proj(x)
        x = self.bn_proj(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)

        # Local temporal head
        x = x.transpose(1, 2)
        x = self.temporal_pad(x)
        x = self.temporal_conv(x)
        x = self.head_bn(x)
        x = self.head_act(x)
        x = self.head_dropout(x)
        x = self.pointwise(x)
        x = torch.sigmoid(x)
        probs = x.transpose(1, 2)

        if self.output_features:
            return probs, probs  # keep API parity (features not used); placeholder
        else:
            return probs


class NoteLevelTechniqueModel(nn.Module):
    def __init__(self, classes_num: int = 5, sample_rate: int = 16000, frames_per_second: int = 100,
                 trans_feat_dim: int = 88 * 4, mel_bins: int = 229, use_trans_features: bool = True,
                 output_features: bool = False):
        """Note-level technique classifier with 2D CNN over Logmel, like LocalTechniqueFeatureConvModel.

        Inputs:
          - waveform: (B, L) mono note audio (fixed length)
          - trans_features: (B, T, 88*4) optional transcription probs (onset/offset/frame/velocity)

        Output: (B, classes_num)
        """
        super(NoteLevelTechniqueModel, self).__init__()
        self.sample_rate = sample_rate
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.output_features = output_features

        # STFT + Logmel inside model
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
            hop_length=hop_size, win_length=window_size, window='hann',
            center=True, pad_mode='reflect', freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
            n_fft=window_size, n_mels=mel_bins, fmin=30, fmax=sample_rate // 2,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum=0.01)

        # 2D conv stack (reuse ConvBlock)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=0.01)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=0.01)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=0.01)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=0.01)

        # Audio projection after global pooling
        self.audio_fc = nn.Linear(128, 128)

        # Transcription features projection (onset/offset/frame concatenated along last dim)
        self.use_trans_features = use_trans_features
        self.trans_proj = nn.Sequential(
            nn.Linear(trans_feat_dim, 256), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
        )

        # Fusion and head
        self.fusion = nn.Sequential(
            nn.Linear(128 + (128 if self.use_trans_features else 0), 128), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(128, classes_num)
        )

    @torch.no_grad()
    def extract_transcription_features(self, transcriptor: nn.Module, waveform: torch.Tensor) -> torch.Tensor:
        """Run frozen transcription model to get time-aligned (onset, offset, frame) probs.
        Returns (B, T, 88*4)."""
        transcriptor.eval()
        out = transcriptor(waveform)
        onset = out.get('reg_onset_output')
        offset = out.get('reg_offset_output')
        frame = out.get('frame_output')
        velocity = out.get('velocity_output')
        feats = torch.cat([onset, offset, frame, velocity], dim=-1)
        return feats

    def forward(self, waveform: torch.Tensor, trans_features: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Compute logmel from waveform
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # 2D conv with pooling
        x = self.conv_block1(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # Global average pooling over time and freq -> (B, C)
        x = x.mean(dim=[2, 3])
        x = F.relu(self.audio_fc(x))

        # Transcription features path
        if self.use_trans_features and trans_features is not None:
            tp = self.trans_proj(trans_features).mean(dim=1)
            # max pooling with amax
            # tp = self.trans_proj(trans_features.amax(dim=1))
            z = torch.cat([x, tp], dim=-1)
        else:
            z = x

        logits = self.fusion(z)
        if self.output_features:
            return logits, z
        else:
            return logits