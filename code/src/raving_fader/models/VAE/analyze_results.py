import torch
import librosa
import numpy as np
import os
from os import path
from pathlib import Path
import glob
from effortless_config import Config
from init import init
import yaml
from fader_loader import spectral_features_spectro
from tqdm import tqdm


def test(loader, config):
    model_path = []
    print("YO")
    for (path, dirnames, filenames) in os.walk(config["data"]["trained_models_path"]):
        model_path.extend(os.path.join(path, name) for name in filenames)
    scaler = torch.nn.Conv2d(1, 1, (3, 3), stride=2).to(config["train"]["device"])
    scaler.weight.data.fill_(-1)
    scaler.weight.data[0, 0, 1, 1] = 1
    for m in model_path:
        print("TESTING")
        print(m)
        with torch.no_grad():
            model = torch.load(m, map_location=config["train"]["device"])
            model.eval()
            recon_feats = []
            target_feats = []
            mel = []
            m_stft = []
            jnd = []
            mel_cycle = []
            for batch_x, batch_y in tqdm(iter(loader), total=5000):
                # Take the input & attributes
                batch_x = batch_x.float().to(config["train"]["device"], non_blocking=True)
                batch_y = batch_y.float().to(config["train"]["device"], non_blocking=True)  # B x A x 1
                # Create random attributes
                y_random = batch_y.clone()
                for i in range(config["data"]["num_attributes"]):
                    y_random[:, i] = (torch.rand(batch_y.shape[0])).float()
                    # print(y_random)
                target_feats.append(y_random)
                # Reconstructions
                _, x_reconstruct_orig, _ = model(batch_x, batch_y)
                _, x_reconstruct, _ = model(batch_x, y_random)
                _, x_reconstruct_cycle, _ = model(x_reconstruct, batch_y)
                # Compute spectral feats from reconstruction
                feats = []
                for i in range(batch_x.shape[0]):
                    denorm_spec = torch.exp(x_reconstruct[i].detach()) - 0.1
                    feats.append(
                        np.mean(spectral_features_spectro(denorm_spec.cpu().numpy()
                                                          , config["preprocess"]["sample_rate"]),
                                axis=1))
                mel.append(torch.mean(torch.sqrt((x_reconstruct_orig - x_reconstruct) ** 2), axis=(-1, -2)))
                mel_cycle.append(torch.mean(torch.sqrt((x_reconstruct_orig - x_reconstruct_cycle) ** 2), axis=(-1, -2)))
                cur_stft = 0
                for r in range(5):
                    cur_stft += torch.mean(torch.sqrt((x_reconstruct_orig - x_reconstruct) ** 2), axis=(-1, -2))
                    x_reconstruct_orig = scaler(x_reconstruct_orig.unsqueeze(1)).squeeze()
                    x_reconstruct = scaler(x_reconstruct.unsqueeze(1)).squeeze()
                m_stft.append(cur_stft)
                jnd.append((mel[-1] + m_stft[-1]) * .1)
                print()
                feats = np.stack(feats, axis=0)
                recon_feats.append(feats)
            # Store difference between reconstruct attributes and ground truth
            # for i in range(len(tests_list)):
            #    x = cur_feats  # tests_list[i].cpu()
            #    y = recon_feats[i]
            torch.save([target_feats, recon_feats, mel, m_stft, jnd, mel_cycle],
                       m + '_results.pth')
