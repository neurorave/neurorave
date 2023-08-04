import os
import json
import torch
import librosa
import numpy as np
import torch.nn.functional
from torch.utils.data import Dataset


class NSynthDataset(Dataset):
    def __init__(self,
                 audio_dir,
                 nsynth_json_path,
                 sampling_rate=16000,
                 transform=None):
        super(NSynthDataset, self).__init__()
        with open(nsynth_json_path, "r") as f:
            self.audio_labels = json.load(f)
        self.audio_name_list = list(self.audio_labels.keys())
        self.audio_dir = audio_dir
        self.transform = transform
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, index):
        audio_file = os.path.join(self.audio_dir,
                                  f"{self.audio_name_list[index]}.wav")
        # sr, audio = scipy.io.wavfile.read(audio_file)
        signal, _ = librosa.load(audio_file, self.sampling_rate)
        audio = signal.astype(np.float32)
        audio = torch.tensor(audio.copy()).float()
        if self.transform:
            audio = self.transform(audio)
        pitch = self.audio_labels[self.audio_name_list[index]]['pitch']
        # pad the audio signal with zeros to get a length equals to power of two
        # here sampling rate is 16000 Hz and we have 4 seconds samples so the
        # audio length is 64000. The nearest power of two is 2**16=65536
        zeros_length = 2**16 - audio.shape[-1]
        audio_pad = torch.nn.functional.pad(input=audio,
                                            pad=(0, zeros_length),
                                            mode='constant',
                                            value=0.0)
        return audio_pad, pitch
