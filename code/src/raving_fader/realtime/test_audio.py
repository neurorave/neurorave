import torch
import librosa
import numpy as np
from audio_descriptors.features import compute_all
import soundfile as sf

model_name = "FRAVE_DK"

model_path = "../models/" + model_name + "/"
config_path = model_path + "train_config.yaml"

model = torch.jit.load(model_path + "rave_32.ts")

audio, sr = librosa.load("../audio/darbouka1.wav", sr=model.sr)

torch.set_grad_enabled(False)

x = torch.tensor(audio, requires_grad=False).unsqueeze(0).unsqueeze(0)

z = model.encode(x)

features = compute_all(audio,
                       sr=model.sr,
                       descriptors=model.descriptors,
                       mean=False,
                       resample=z.shape[-1])

features = {descr: features[descr] for descr in model.descriptors}
attr = np.array(list(features.values())).astype(np.float32)

attr = torch.tensor(attr).unsqueeze(0)

attr = model.normalize_all(attr)

zc = torch.cat((z, attr), dim=1)
audio_rec = model.decode(zc).squeeze().detach().numpy()

sf.write("../audio/darbouka1_rec.wav", audio_rec, samplerate=model.sr)