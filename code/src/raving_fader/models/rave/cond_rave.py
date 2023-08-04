from raving_fader.models.rave.rave import RAVE
from raving_fader.models.rave.networks import Generator
import torch
from raving_fader.models.rave.core import multiscale_stft, get_beta_kl_cyclic_annealed, Loudness
import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from einops import rearrange
from sklearn.decomposition import PCA
from raving_fader.helpers.data_viz import plot_metrics


class CRAVE(RAVE):
    def __init__(self,
                 data_size,
                 capacity,
                 latent_size,
                 ratios,
                 bias,
                 loud_stride,
                 use_noise,
                 noise_ratios,
                 noise_bands,
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
                 device="cpu",
                 descriptors=[None]):
        super().__init__(data_size,
                         capacity,
                         latent_size,
                         ratios,
                         bias,
                         loud_stride,
                         use_noise,
                         noise_ratios,
                         noise_bands,
                         d_capacity,
                         d_multiplier,
                         d_n_layers,
                         warmup,
                         mode,
                         no_latency=False,
                         min_kl=min_kl,
                         max_kl=max_kl,
                         cropped_latent_size=cropped_latent_size,
                         feature_match=feature_match,
                         sr=sr,
                         device=device)

        self.descriptors = descriptors

        new_latent_size = latent_size + len(self.descriptors)
        self.decoder = Generator(
            new_latent_size,
            capacity,
            data_size,
            ratios,
            loud_stride,
            use_noise,
            noise_ratios,
            noise_bands,
            "causal" if no_latency else "centered",
            bias,
        )

        self.latent_size = new_latent_size

    def training_step(self, batch, step):
        audio, features = batch
        x = audio.unsqueeze(1)

        # STEP 1: VAE FOR REPRESENTATION LEARNING

        # 1. MULTIBAND DECOMPOSITION
        if self.pqmf is not None:
            x = self.pqmf(x)

        # 2. ENCODE INPUT
        # if train stage 1 repr learning encoder.train()
        # else (train stage 2 adversarial) freeze encoder encoder.eval()
        if self.warmed_up:  # EVAL ENCODER
            self.encoder.eval()
        else:
            self.encoder.train()
        self.decoder.train()
        # get latent space samples
        # and compute regularization loss KL div
        z, kl = self.reparametrize(*self.encoder(x))

        if self.warmed_up:  # FREEZE ENCODER
            z = z.detach()
            kl = kl.detach()

        # 3. DECODE LATENT
        # Decode latent space samples add noise to the output
        # only during adversarial training stage 2
        # ADD CONDITIONNING
        z_c = torch.cat((z, features), dim=1)
        # print(z_c.shape)
        y = self.decoder(z_c, add_noise=self.warmed_up)

        # 4. DISTANCE BETWEEN INPUT AND OUTPUT
        # compute reconstruction loss ie. multiscale spectral distance
        # between each input and reconstructed signal PQMF bands
        # + between the input signal and the reconstructed signal with
        # full band recomposition apply the inverse PQMF
        distance = self.distance(x, y)

        # inverse multi band decomposition (pqmf -1) --> recomposition
        if self.pqmf is not None:  # FULL BAND RECOMPOSITION
            x = self.pqmf.inverse(x)
            y = self.pqmf.inverse(y)
            distance = distance + self.distance(x, y)

        # 5. Compute the loudness on the input and reconstructed signal
        # Add the Loudness MSE to the VAE reconstruction distance loss
        loud_x = self.loudness(x)
        loud_y = self.loudness(y)
        loud_dist = (loud_x - loud_y).pow(2).mean()
        distance = distance + loud_dist

        # STEP 2: GAN ADVERSARIAL TRAINING
        # Apply once warmed_up = True
        feature_matching_distance = 0.
        if self.warmed_up:  # DISCRIMINATION
            self.discriminator.train()
            # 1. compute discriminator predictions on fake and real data
            feature_true = self.discriminator(x)
            feature_fake = self.discriminator(y)

            loss_dis = 0
            loss_adv = 0

            pred_true = 0
            pred_fake = 0

            for scale_true, scale_fake in zip(feature_true, feature_fake):
                # 2. Compute Feature matching distance
                feature_matching_distance = feature_matching_distance + 10 * sum(
                    map(
                        lambda x, y: abs(x - y).mean(),
                        scale_true,
                        scale_fake,
                    )) / len(scale_true)

                # 3. Compute Hinge loss
                _dis, _adv = self.adversarial_combine(
                    scale_true[-1],
                    scale_fake[-1],
                    mode=self.mode,
                )
                pred_true = pred_true + scale_true[-1].mean()
                pred_fake = pred_fake + scale_fake[-1].mean()

                loss_dis = loss_dis + _dis
                loss_adv = loss_adv + _adv

        else:
            pred_true = torch.tensor(0.).to(x)
            pred_fake = torch.tensor(0.).to(x)
            loss_dis = torch.tensor(0.).to(x)
            loss_adv = torch.tensor(0.).to(x)

        # COMPOSE GEN LOSS
        beta = get_beta_kl_cyclic_annealed(
            step=step,
            cycle_size=5e4,
            warmup=self.warmup // 2,
            min_beta=self.min_kl,
            max_beta=self.max_kl,
        )
        # Compute total VAE decoder/generator loss
        loss_gen = distance + loss_adv + beta * kl
        if self.feature_match:
            loss_gen = loss_gen + feature_matching_distance

        # OPTIMIZATION
        # optimizer steps:
        # Before the backward pass, zero all of the network gradients
        # self.optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        # loss.backward()
        # Calling the step function to update the parameters
        # self.optimizer.step()
        if step % 2 and self.warmed_up:
            self.dis_opt.zero_grad()
            loss_dis.backward(retain_graph=True)
            self.dis_opt.step()
        else:
            self.gen_opt.zero_grad()
            loss_gen.backward(retain_graph=True)
            self.gen_opt.step()

        losses = {
            "loss_dis": loss_dis.detach().cpu(),
            "loss_gen": loss_gen.detach().cpu(),
            "loud_dist": loud_dist.detach().cpu(),
            "regularization": kl.detach().cpu(),
            "pred_true": pred_true.mean().cpu(),
            "pred_fake": pred_fake.mean().cpu(),
            "distance": distance.detach().cpu(),
            "beta": beta,
            "feature_matching": feature_matching_distance,
        }
        return losses

    def validation_step(self, batch):
        # evaluate VAE representation learning on validation set
        audio, features = batch
        x = audio.unsqueeze(1)

        self.encoder.eval()
        self.decoder.eval()

        # 1. multiband decomposition PQMF
        if self.pqmf is not None:
            x = self.pqmf(x)

        # 2. Encode data
        mean, scale = self.encoder(x)
        # 3. Get latent space samples
        z, _ = self.reparametrize(mean, scale)
        # 4. Decode latent space samples
        z_c = torch.cat((z, features), dim=1)
        # print(features.shape)
        # print(z.shape)
        y = self.decoder(z_c, add_noise=self.warmed_up)
        # 5. inverse multi band decomposition (pqmf -1) --> recomposition
        if self.pqmf is not None:
            x = self.pqmf.inverse(x)
            y = self.pqmf.inverse(y)

        # 6. compute reconstruction loss ie. multiscale spectral distance
        distance = self.distance(x, y)

        # return torch.cat([x, y], -1), mean
        return y, distance

    def train(self,
              train_loader,
              val_loader,
              max_steps,
              models_dir,
              model_filename,
              ckpt=None,
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
                x = tuple(tens.to(self.device) for tens in x)

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
                            x_val = tuple(
                                tens.to(self.device) for tens in x_val)

                            # x_val = x_val.to(self.device)
                            y_val, dist = self.validation_step(x_val)
                            x_val, features = x_val
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
                if self.step % display_step == 0 or (epoch == n_epochs - 1
                                                     and batch_idx
                                                     == len(train_loader) - 1):
                    vae_checkpoint = {
                        "step": start_it + self.step,
                        "epoch": start_epoch + epoch,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "decoder_state_dict": self.decoder.state_dict(),
                        "optimizer_state_dict": self.gen_opt.state_dict(),
                        "loss": train_losses['loss_gen'][-1],
                        "val_loss": valid_loss[-1]
                    }
                    if self.warmed_up:
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
                                         f"{model_filename}_{step}_vae.ckpt"))
                        if is_best:
                            torch.save(
                                vae_checkpoint,
                                os.path.join(
                                    models_dir,
                                    f"{model_filename}__vae_best.ckpt"))

                # update number of batch iteration
                batch_idx += 1
