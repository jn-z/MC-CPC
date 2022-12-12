import numpy as np
import math
import torch
from torch.utils import data
import h5py
import random
from scipy.io import wavfile
from collections import defaultdict
from random import randint
import pdb
from scipy.fftpack import fft,ifft


class RawDatasetSpkClass(data.Dataset):
    def __init__(self, raw_file, all_file, list_file, index_file, audio_window, frame_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            index_file: spk2idx
            audio_window: 20480
        """
        self.raw_file  = raw_file
        self.audio_window = audio_window
        self.frame_window = frame_window
        self.utts = []
        self.length = []
        self.bits = 2
        self.snr_range = (15, 30)
        self.clip_factors = [0.8, 0.9, 0.7]
        with open(list_file) as f:
            temp = f.readlines()
        with open(all_file) as f:
            all_temp = f.readlines()
        self.h5f = h5py.File(self.raw_file, 'r')
        import pdb
        for i in temp: # sanity check
            utt_len = self.h5f[i.strip()].shape[0]
            if utt_len > 64:
                self.utts.append(i.strip())
        for j in all_temp: # sanity check
            max_len = self.h5f[j.strip()].shape[0]
            self.length.append(max_len)

        with open(index_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.spk2idx = {}
        for i in content:
            spk = i.split(' ')[0]
            idx = int(i.split(' ')[1])
            self.spk2idx[spk] = idx

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        max_len = 1280
        name_list = []
        yushu = max_len % self.audio_window
        if yushu == 0:
            re_length = max_len
        else:
            re_length = max_len + self.audio_window - yushu
        utt_id = self.utts[index].strip()
        feature_data = self.h5f[utt_id]
        data_range = np.max(feature_data) - np.min(feature_data)
        norm_data = (feature_data - np.min(feature_data)) / data_range
        avg_data = np.mean(norm_data, axis=0)
        sigma = np.std(norm_data, axis=0)
        feature_data2 = (norm_data - avg_data) / sigma
        feature_data2 = np.resize(feature_data2, re_length) # add repeatedly from beginning to end
        feature_data2 = feature_data2[np.newaxis, :]
        wav = torch.from_numpy(feature_data2).float()
        if random.random() < 0:
             len_wav = wav.shape[1]
             noise = torch.randn(wav.shape)
             norm_constant = 2.0 ** (self.bits - 1)
             norm_wave = wav / norm_constant
             norm_noise = noise / norm_constant
             signal_power = torch.sum(norm_wave ** 2) / len_wav
             noise_power = torch.sum(norm_noise ** 2) / len_wav
             snr = np.random.randint(self.snr_range[0], self.snr_range[1])
             covariance = torch.sqrt((signal_power / noise_power) * 10 ** (- snr / 10))
             wav = wav + covariance * noise
        if random.random() < 0:
             wav_length = wav.shape[1]
             cf = random.choice(self.clip_factors)
             clipped_wav = torch.clamp(wav.view(-1), cf * torch.min(wav), cf * torch.max(wav))
             wav = clipped_wav.view(1, -1)
        feature_data2 = np.squeeze(wav)
        speaker = utt_id.split(' ')[0]
        label = torch.tensor(self.spk2idx[speaker])
        speaker_name = speaker.strip()
        name_list.append(speaker_name)

        return feature_data2, label.repeat(self.frame_window), name_list
