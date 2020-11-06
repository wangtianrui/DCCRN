# coding: utf-8
# Author：WangTianRui
# Date ：2020/10/15 18:47
import json
import os

import numpy as np
import soundfile as sf
import torch
from torch.utils import data


class DNSDataset(data.Dataset):
    dataset_name = "DNS"

    def __init__(self, json_dir, data_home, only_two=True):
        super(DNSDataset, self).__init__()
        self.json_dir = json_dir
        self.data_home = data_home
        with open(json_dir, "r") as f:
            self.mix_infos = json.load(f)
        self.wav_ids = list(self.mix_infos.keys())
        self.only_two = only_two

    def __len__(self):
        return len(self.wav_ids)

    def __getitem__(self, item):
        utt_info = self.mix_infos[self.wav_ids[item]]
        # print("loader:", utt_info)
        # print("data home", self.data_home)
        temp = os.path.join(self.data_home, utt_info["mix"])
        if os.path.exists(temp):
            noisy = torch.from_numpy(sf.read(os.path.join(self.data_home, utt_info["mix"]), dtype="float32")[0])
            clean = torch.from_numpy(sf.read(os.path.join(self.data_home, utt_info["clean"]), dtype="float32")[0])
        else:
            print("path error", os.path.join(self.data_home, utt_info["mix"]))
            return
            # print("loader:", noisy.size())
        # print("loader:", clean.size())
        if self.only_two:
            return noisy, clean
        else:
            noise = torch.from_numpy(sf.read(utt_info["noise"], dtype="float32")[0])
            return noisy, clean, noise


def load_hop_wav(path, frame_dur, hop_dur, sr=16000):
    # signal, _ = lib.load(path, sr=sr)
    signal = sf.read(path, dtype="float32")[0]
    win = int(frame_dur * sr)
    hop = int(hop_dur * sr)
    rest = (len(signal) - win) % hop
    signal = np.pad(signal, (0, hop - rest), "constant")
    n_frames = int((len(signal) - win) // hop)
    strides = signal.itemsize * np.array([hop, 1])
    return torch.tensor(np.lib.stride_tricks.as_strided(signal, shape=(n_frames, win), strides=strides))


class WavHopDataset(data.Dataset):
    def __init__(self, json_dir, frame_dur, hop_dur, data_home, loader=load_hop_wav):
        self.json_dir = json_dir
        self.data_home = data_home
        with open(json_dir, "r") as f:
            self.mix_infos = json.load(f)
        self.wav_ids = list(self.mix_infos.keys())

        self.loader = loader
        self.frame_dur = frame_dur
        self.hop_dur = hop_dur

    def __getitem__(self, item):
        utt_info = self.mix_infos[self.wav_ids[item]]
        return self.loader(os.path.join(self.data_home, utt_info["mix"]), self.frame_dur, self.hop_dur), \
               self.loader(os.path.join(self.data_home, utt_info["clean"]), self.frame_dur, self.hop_dur)

    def __len__(self):
        return len(self.wav_ids)
