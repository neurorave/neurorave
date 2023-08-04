import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from ddsp.model import DDSP
from effortless_config import Config
from os import path
from preprocess import Dataset
from tqdm import tqdm
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness, quantify
import soundfile as sf
from einops import rearrange
from ddsp.utils import get_scheduler
import numpy as np
import torchaudio
import torch.nn.functional as F

from raving_fader.datasets.data_loaders import rave_data_loaders
from preprocess import FaderDataset
from raving_fader.models.fader.networks.latent_discriminator import LatentDiscriminator


class args(Config):
    CONFIG = "/data/nils/raving-fader/src/ddsp/config.yaml"
    NAME = "debug_fddsp"
    ROOT = "runs"
    STEPS = 500000
    BATCH = 16
    START_LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 400000
    LAMBDA_DIS = 0.01


### PARAMETERS

lr_latdis = 0.0001

##########  CONFIG #######
args.parse_args()
print(args)
with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##########  MODELS #######
config["model"].update({"n_descriptors": 5})
# config_dict =
model = DDSP(**config["model"]).to(device)

latent_discriminator = LatentDiscriminator(
    latent_size=config["model"]["n_latent"],
    num_attributes=5,
    num_classes=16,
    num_layers=2).to(device)

out_dir = config["preprocess"]["out_dir"]
dataset = FaderDataset(out_dir)
dataloader, validloader = rave_data_loaders(args.BATCH, dataset, num_workers=8)

mean_loudness, std_loudness = mean_std_loudness(
    torch.from_numpy(dataset.loudness))
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

writer = SummaryWriter(path.join(args.ROOT, args.NAME), flush_secs=20)

with open(path.join(args.ROOT, args.NAME, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)

lat_dis_opt = torch.optim.Adam(latent_discriminator.parameters(), lr=lr_latdis)

schedule = get_scheduler(
    len(dataloader),
    args.START_LR,
    args.STOP_LR,
    args.DECAY_OVER,
)

# scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

best_loss = float("inf")
mean_loss = 0
n_element = 0
step = 0
epochs = int(np.ceil(args.STEPS / len(dataloader)))

wave2mfcc = torchaudio.transforms.MFCC(sample_rate=16000,
                                       n_mfcc=30,
                                       melkwargs={
                                           "n_fft":
                                           1024,
                                           "hop_length":
                                           config["preprocess"]["block_size"],
                                           "f_min":
                                           20,
                                           "f_max":
                                           int(16000 / 2),
                                           "n_mels":
                                           128,
                                           "center":
                                           True
                                       }).to(device)

for e in tqdm(range(epochs)):
    for s, p, l, attr in dataloader:

        # GET THE DATA
        s = s.to(device)
        mfcc = wave2mfcc(s)[:, :, :-1]
        mfcc = mfcc.permute(0, 2, 1)
        p = p.unsqueeze(-1).to(device)
        l = l.unsqueeze(-1).to(device)
        l = (l - mean_loudness) / std_loudness

        attr = attr.to(device)
        ########  STEP DDSP ########

        # Freeze latent discriminator and activate encoder
        latent_discriminator.eval()
        model.train()

        # Encode
        z = model.encoder(mfcc)

        # Compute attribute classes with quantification
        attr_cls = quantify(
            attr,
            torch.tensor(dataset.fader_dataset.bin_values).to(device)).long()
        attr = dataset.fader_dataset.normalize_all(attr)

        # Get latent discriminator prediction
        attr_cls_pred = latent_discriminator(torch.permute(z, (0, 2, 1)))
        lat_dis_loss = -F.cross_entropy(attr_cls_pred, attr_cls).mean()

        with torch.no_grad():
            accuracy = (attr_cls_pred.argmax(dim=1) == attr_cls).sum(dim=(0,
                                                                          2))
            accuracy = accuracy / (attr_cls.shape[0] *
                                   attr_cls.shape[-1]) * 100
        # Add conditionning to z

        z_c = torch.cat((z, torch.permute(attr, (0, 2, 1))), dim=2)

        # Forward
        y = model.decode(z_c, p, l).squeeze(-1)

        # Loss rec
        ori_stft = multiscale_fft(
            s,
            config["train"]["scales"],
            config["train"]["overlap"],
        )
        rec_stft = multiscale_fft(
            y,
            config["train"]["scales"],
            config["train"]["overlap"],
        )

        loss = 0
        for s_x, s_y in zip(ori_stft, rec_stft):
            lin_loss = (s_x - s_y).abs().mean()
            log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss = loss + lin_loss + log_loss

        losstot = loss + args.LAMBDA_DIS * lat_dis_loss

        opt.zero_grad()
        losstot.backward()
        opt.step()

        ########  STEP LAT DIS ########
        model.eval()
        latent_discriminator.train()

        with torch.no_grad():
            z = model.encoder(mfcc)

        # Get latent discriminator prediction
        attr_cls_pred = latent_discriminator(torch.permute(z, (0, 2, 1)))
        lat_dis_loss_disc = F.cross_entropy(attr_cls_pred, attr_cls).mean()

        lat_dis_opt.zero_grad()
        lat_dis_loss_disc.backward()
        lat_dis_opt.step()

        ########  LOGS ########
        writer.add_scalar("loss", loss.item(), step)
        writer.add_scalar("latent_loss_encstep", lat_dis_loss.item(), step)
        writer.add_scalar("latent_loss_discstep", lat_dis_loss_disc.item(),
                          step)

        accuracy = accuracy.cpu()
        for i, descr in enumerate(dataset.fader_dataset.descriptors):
            writer.add_scalar("accuracy/" + descr, accuracy[i], step)

        step += 1

        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element

    if not e % 10:
        writer.add_scalar("lr", schedule(e), e)
        writer.add_scalar("reverb_decay", model.reverb.decay.item(), e)
        writer.add_scalar("reverb_wet", model.reverb.wet.item(), e)
        # scheduler.step()
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                model.state_dict(),
                path.join(args.ROOT, args.NAME, "state.pth"),
            )

        mean_loss = 0
        n_element = 0

        audio = torch.cat([s, y], -1).reshape(-1).detach().cpu().numpy()

        sf.write(
            path.join(args.ROOT, args.NAME, f"eval_{e:06d}.wav"),
            audio,
            config["preprocess"]["sampling_rate"],
        )
