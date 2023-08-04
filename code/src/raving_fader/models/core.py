import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from abc import abstractmethod
from einops import rearrange
from sklearn.decomposition import PCA

from raving_fader.models.rave.networks import Encoder, StackDiscriminators
from raving_fader.models.rave.pqmf import CachedPQMF,PQMF
from raving_fader.models.rave.core import multiscale_stft, Loudness
from raving_fader.helpers.data_viz import plot_metrics


class BaseRAVE(nn.Module):
    def __init__(self,
                 data_size,
                 capacity,
                 latent_size,
                 ratios,
                 bias,
                 d_capacity,
                 d_multiplier,
                 d_n_layers,
                 warmup,
                 mode,
                 no_latency=False,
                 min_kl=1e-1,
                 max_kl=1e-1,
                 cropped_latent_size=0,
                 feature_match=True,
                 sr=24000,
                 device="cpu"):
        super().__init__()
        self.device = device

        if data_size == 1:
            self.pqmf = None
        else:
            self.pqmf = CachedPQMF(70 if no_latency else 100, data_size)

        self.loudness = Loudness(sr, 512)

        encoder_out_size = cropped_latent_size if cropped_latent_size else latent_size

        self.encoder = Encoder(
            data_size,
            capacity,
            encoder_out_size,
            ratios,
            "causal" if no_latency else "centered",
            bias,
        )

        # Initialize decoder in sub class
        self.decoder = None

        self.discriminator = StackDiscriminators(
            3,
            in_size=1,
            capacity=d_capacity,
            multiplier=d_multiplier,
            n_layers=d_n_layers,
        )

        self.idx = 0

        # self.register_buffer("latent_pca", torch.eye(encoder_out_size))
        # self.register_buffer("latent_mean", torch.zeros(encoder_out_size))
        # self.register_buffer("fidelity", torch.zeros(encoder_out_size))

        self.latent_size = latent_size

        self.warmup = warmup
        self.warmed_up = False
        self.sr = sr
        self.mode = mode
        self.step = 0

        self.min_kl = min_kl
        self.max_kl = max_kl
        self.cropped_latent_size = cropped_latent_size

        self.feature_match = feature_match

        # TODO : initialize weights ?

    @abstractmethod
    def _init_optimizer(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode(self, *args):
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch, step):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch):
        raise NotImplementedError

    def lin_distance(self, x, y):
        return torch.norm(x - y) / torch.norm(x)

    def log_distance(self, x, y):
        return abs(torch.log(x + 1e-7) - torch.log(y + 1e-7)).mean()

    def distance(self, x, y):
        scales = [2048, 1024, 512, 256, 128]
        x = multiscale_stft(x, scales, .75)
        y = multiscale_stft(y, scales, .75)

        lin = sum(list(map(self.lin_distance, x, y)))
        log = sum(list(map(self.log_distance, x, y)))

        return lin + log

    def reparametrize(self, mean, scale):
        std = nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        if self.cropped_latent_size:
            noise = torch.randn(
                z.shape[0],
                self.latent_size - self.cropped_latent_size,
                z.shape[-1],
            ).to(z.device)
            z = torch.cat([z, noise], 1)
        return z, kl

    def adversarial_combine(self, score_real, score_fake, mode="hinge"):
        if mode == "hinge":
            loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
            loss_dis = loss_dis.mean()
            loss_gen = -score_fake.mean()
        elif mode == "square":
            loss_dis = (score_real - 1).pow(2) + score_fake.pow(2)
            loss_dis = loss_dis.mean()
            loss_gen = (score_fake - 1).pow(2).mean()
        elif mode == "nonsaturating":
            score_real = torch.clamp(torch.sigmoid(score_real), 1e-7, 1 - 1e-7)
            score_fake = torch.clamp(torch.sigmoid(score_fake), 1e-7, 1 - 1e-7)
            loss_dis = -(torch.log(score_real) +
                         torch.log(1 - score_fake)).mean()
            loss_gen = -torch.log(score_fake).mean()
        else:
            raise NotImplementedError
        return loss_dis, loss_gen

    def encode(self, x):
        if self.pqmf is not None:
            x = self.pqmf(x)

        mean, scale = self.encoder(x)
        z, _ = self.reparametrize(mean, scale)
        return z

    def train(self,
              train_loader,
              val_loader,
              max_steps,
              models_dir,
              model_filename,
              ckpt=None,
              it_ckpt=None,
              display_step=1000):
        start = time.time()

        # TENSORBOARD
        writer = SummaryWriter(
            os.path.join(
                models_dir,
                f"runs/exp__{model_filename}_{time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime())}",
            ))

        # Initialize optimizer
        self._init_optimizer()

        # resume training from checkpoint if specify
        start_epoch = 0
        start_it = 0
        if ckpt:
            print(ckpt)
            checkpoint = torch.load(ckpt)
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
            self.gen_opt.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            start_it = checkpoint["step"]
            print(
                f">>> Resume training from : {ckpt} \n"
                f"- epoch : {start_epoch} - step : {start_it} - train loss : {checkpoint['loss']} - valid loss : {checkpoint['val_loss']}"
            )
        is_best = False
        min_loss = 0

        train_losses = {
            "it": [],
            "epoch": [],
            "loss_dis": [],
            "loss_gen": [],
            "loud_dist": [],
            "regularization": [],
            "pred_true": [],
            "pred_fake": [],
            "distance": [],
            "beta": [],
            "feature_matching": [],
        }

        check_nepoch = True
        if len(train_loader) >= 10000:
            val_check = 10000
            check_nepoch = False
            print(f">>>>>>>>> valcheck nstep : {val_check}")
        else:
            val_check = 10000 // len(train_loader)
            print(f">>>>>>>>> valcheck nepoch : {val_check}")

        valid_loss = []

        max_epochs = max_steps // len(train_loader)
        n_epochs = max_epochs if max_epochs <= 100000 else 100000
        for epoch in range(n_epochs):
            # Start with the representation learning training stage 1 (VAE)
            # Once the number of 'warmup' iterations is reached, move to the
            # 2nd adversarial training stage (freeze encoder + add discriminator)
            step = len(train_loader) * epoch
            self.step = step
            if self.step > self.warmup:
                self.warmed_up = True

            # keep track of the current batch iteration per epoch
            batch_idx = 0
            # store the losses values for display
            losses_display = np.zeros(len(train_losses.keys()) - 2)

            for x in tqdm(train_loader):
                # current training iteration
                step = len(train_loader) * epoch + batch_idx
                self.step = step

                # apply a training step on the current batch x
                x = x.to(self.device)
                cur_losses = self.training_step(x, step)

                # keep track of the loss
                losses_display += np.asarray(list(cur_losses.values()))

                # Display training information
                if self.step % display_step == 0 or (epoch == n_epochs - 1
                                                     and batch_idx
                                                     == len(train_loader) - 1):
                    train_losses['it'].append(self.step)
                    train_losses['epoch'].append(epoch)
                    for k, l in zip(list(cur_losses.keys()), losses_display):
                        train_losses[k].append(l / display_step)
                        writer.add_scalar(f"training loss : {k}",
                                          l / display_step, self.step)

                    print(
                        f"\nEpoch: [{epoch}/{n_epochs}] \tStep: [{self.step}/{max_steps}]"
                        f"\tTime: {time.time() - start} (s) \tTotal_gen_loss: {train_losses['loss_gen'][-1]}"
                    )
                    losses_display = np.zeros(len(train_losses.keys()) - 2)

                # evaluate on the validation set
                if (not check_nepoch and self.step % val_check == 0) \
                        or (check_nepoch and epoch % val_check == 0 and batch_idx == 0):
                    with torch.no_grad():
                        x_eval = []
                        y_eval = []
                        n_examples = 8
                        val_loss = 0
                        for x_val in tqdm(val_loader):
                            x_val = x_val.to(self.device)
                            y_val, dist = self.validation_step(x_val)
                            if len(x_eval) < n_examples:
                                x_eval.append(
                                    np.squeeze(
                                        x_val[0].detach().cpu().numpy()))
                                y_eval.append(
                                    np.squeeze(
                                        y_val[0].detach().cpu().numpy()))
                            val_loss += dist
                        print(
                            f"\nEpoch: [{epoch}/{n_epochs}] \t Validation loss: {val_loss / len(val_loader)}"
                        )
                        writer.add_scalar(f"Validation loss :",
                                          val_loss / len(val_loader),
                                          self.step)
                        # add audio to tensorboard
                        for j in range(len(x_eval)):
                            fig_stft = plot_metrics(signal_in=x_eval[j],
                                                    signal_out=y_eval[j],
                                                    sr=self.sr)
                            writer.add_audio(
                                f"ground_truth_sound/{j}",
                                x_eval[j],
                                global_step=self.step,
                                sample_rate=self.sr,
                            )
                            writer.add_audio(
                                f"generated_sound/{j}",
                                y_eval[j],
                                global_step=self.step,
                                sample_rate=self.sr,
                            )
                            writer.add_figure(
                                f"Output Images/{j}",
                                fig_stft,
                                global_step=self.step,
                            )
                        valid_loss.append(val_loss / len(val_loader))
                        if len(valid_loss) == 1:
                            min_loss = valid_loss[-1]
                        else:
                            is_best = (valid_loss[-1] < min_loss)
                            min_loss = valid_loss[-1] if is_best else min_loss

                # save model checkpoints
                if self.step % display_step == 0 \
                        or (it_ckpt and self.step in it_ckpt) \
                        or (epoch == n_epochs - 1 and batch_idx == len(train_loader) - 1):
                    vae_checkpoint = {
                        "step": start_it + self.step,
                        "epoch": start_epoch + epoch,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "decoder_state_dict": self.decoder.state_dict(),
                        "optimizer_state_dict": self.gen_opt.state_dict(),
                        "loss": train_losses['loss_gen'][-1],
                        "val_loss": valid_loss[-1]
                    }
                    if it_ckpt and self.step in it_ckpt:
                        torch.save(
                            vae_checkpoint,
                            os.path.join(models_dir,
                                         f"{model_filename}__vae_{self.step}.ckpt"))
                    elif self.warmed_up:
                        torch.save(
                            vae_checkpoint,
                            os.path.join(models_dir,
                                         f"{model_filename}__vae_stage2.ckpt"))
                        torch.save(
                            {
                                "step":
                                self.step,
                                "epoch":
                                epoch,
                                "discriminator_state_dict":
                                self.discriminator.state_dict(),
                                "optimizer_state_dict":
                                self.dis_opt.state_dict(),
                                "loss":
                                train_losses['loss_dis'][-1]
                            },
                            os.path.join(
                                models_dir,
                                f"{model_filename}__discriminator.ckpt"))
                    else:
                        torch.save(
                            vae_checkpoint,
                            os.path.join(models_dir,
                                         f"{model_filename}__vae.ckpt"))
                        if is_best:
                            torch.save(
                                vae_checkpoint,
                                os.path.join(
                                    models_dir,
                                    f"{model_filename}__vae_best.ckpt"))

                # update number of batch iteration
                batch_idx += 1
