import torch

import time
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from raving_fader.helpers.data_viz import plot_metrics

from raving_fader.models.core import BaseRAVE
from raving_fader.models.rave.networks.generator import Generator
from raving_fader.models.fader.networks.latent_discriminator import LatentDiscriminator


class FadeRAVE(BaseRAVE):

    def __init__(self,
                 data_size,
                 descriptors,
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
                 num_classes=16,
                 num_lat_dis_layers=2,
                 device="cpu",
                 **kwargs):
        super().__init__(data_size, capacity, latent_size, ratios, bias,
                         d_capacity, d_multiplier, d_n_layers, warmup, mode,
                         no_latency, min_kl, max_kl, cropped_latent_size,
                         feature_match, sr, device)

        self.descriptors = descriptors
        self.num_classes = num_classes
        self.num_attributes = len(self.descriptors)
        # Set RAVE generator with latent and attributes as inputs
        new_latent_size = latent_size + self.num_attributes
        # self.latent_size = new_latent_size
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

        # TODO: implement
        self.latent_discriminator = LatentDiscriminator(
            latent_size=latent_size,
            num_attributes=self.num_attributes,
            num_classes=num_classes,
            num_layers=num_lat_dis_layers,
        )

    def _init_optimizer(self, learning_rate=1e-4, beta_1=0.5, beta_2=0.9):
        param_vae = list(self.encoder.parameters()) + list(
            self.decoder.parameters())
        self.gen_opt = torch.optim.Adam(param_vae,
                                        lr=learning_rate,
                                        betas=(beta_1, beta_2))
        self.dis_opt = torch.optim.Adam(self.discriminator.parameters(),
                                        lr=learning_rate,
                                        betas=(beta_1, beta_2))
        self.lat_dis_opt = torch.optim.Adam(
            self.latent_discriminator.parameters(),
            lr=learning_rate,
            weight_decay=1e-4)  # TODO: change betas

    def decode(self, z, attr):
        # TODO: update with attributes
        y = self.decoder(z, add_noise=True)
        if self.pqmf is not None:
            y = self.pqmf.inverse(y)
        return y

    # TODO: beta warmup vae rave

    def get_attr_loss(self, output, attributes_cls):
        """
        Compute attributes loss.
        """
        loss = F.cross_entropy(output, attributes_cls)
        # k += n_cat
        return loss

    def get_beta(self, step, beta_inf, beta_delay):
        if step < beta_delay:
            beta = 0
        else:
            beta = min(beta_inf, beta_inf * (step - beta_delay) / beta_delay)
        return (beta)

    # TODO: lambda warmup
    def get_lambda(self, step, lambda_dis, lambda_delay):
        if step < lambda_delay:
            return (0)
        else:
            return (min(lambda_dis,
                        lambda_dis * (step - lambda_delay) / lambda_delay))

    def mu_law(self, attr, num_classes):
        attr = torch.clip(attr, -1, 1)
        # Manual mu-law companding and mu-bits quantization
        mu = torch.tensor([num_classes - 1]).to(attr, non_blocking=True)
        # mu = mu.to(config["train"]["device"])
        magnitude = torch.log1p(mu * torch.abs(attr)) / torch.log1p(mu)
        attr = torch.sign(attr) * magnitude
        # Map signal from [-1, +1] to [0, mu-1]
        attr = (attr + 1) / 2 * mu + 0.5

        return attr.long()

    def quantify(self, allarr, bins):
        nz = allarr.shape[-1]
        allarr_cls = torch.zeros_like(allarr)

        for i in range(allarr.shape[1]):
            data = allarr[:, i, :].flatten()
            data_cls = torch.bucketize(data, bins[i, 1:], right=False)
            allarr_cls[:, i, :] = data_cls.reshape(-1, nz)

        return allarr_cls

    # TODO: update
    def lat_dis_step(self, x, attr_cls):
        """
        Train the latent discriminator.
        """
        x = x.unsqueeze(1)
        self.encoder.eval()
        self.decoder.eval()
        self.latent_discriminator.train()
        # batch / encode / discriminate
        with torch.no_grad():
            z = self.encode(x)
        attr_cls_pred = self.latent_discriminator(z)
        # loss / optimize
        latent_dis = self.get_attr_loss(attr_cls_pred, attr_cls)
        # print('Breakpoint discriminator  step')
        # breakpoint()
        self.lat_dis_opt.zero_grad()
        latent_dis.backward()
        #if config["discriminator"]["clip_grad_norm"]:
        #    clip_grad_norm(self.lat_dis.parameters(),
        #                   config["discriminator"]["clip_grad_norm"])
        self.lat_dis_opt.step()
        return latent_dis.item()

    def training_step(self, x, attr, attr_cls, beta_inf, beta_delay,
                      lambda_inf, lambda_delay):
        x = x.unsqueeze(1)
        # STEP 1: VAE FOR REPRESENTATION LEARNING

        # 1. MULTIBAND DECOMPOSITION
        if self.pqmf is not None:
            x = self.pqmf(x)

        # 2. ENCODE INPUT
        # if train stage 1 repr learning encoder.train()
        # else (train stage 2 adversarial) freeze encoder encoder.eval()
        self.latent_discriminator.eval()
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

        # 2BIS. GET DISCRIMINATOR OUTPUT
        # TODO: lat_dis_loss
        attr_cls_pred = self.latent_discriminator(z)
        lat_dis_loss = -self.get_attr_loss(attr_cls_pred, attr_cls)
        # print('Breakpoint vae training step')
        # breakpoint()
        # 3. DECODE LATENT
        # Decode latent space samples add noise to the output
        # only during adversarial training stage 2
        z_c = torch.cat((z, attr), dim=1)
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
            feature_matching_distance = torch.tensor(0.).to(x)
        # COMPOSE GEN LOSS
        # beta = get_beta_kl_cyclic_annealed(
        #     step=step,
        #     cycle_size=5e4,
        #     warmup=self.warmup // 2,
        #     min_beta=self.min_kl,
        #     max_beta=self.max_kl,
        # )
        lambda_cur = self.get_lambda(self.step, lambda_inf, lambda_delay)
        beta = self.get_beta(self.step, beta_inf, beta_delay)

        # TODO: lambda * lat_dis
        # Compute total VAE decoder/generator loss
        loss_gen = distance + loss_adv + beta * kl + lambda_cur * lat_dis_loss

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
        if self.step % 2 and self.warmed_up:
            self.dis_opt.zero_grad()
            loss_dis.backward(retain_graph=True)
            self.dis_opt.step()

        else:
            self.gen_opt.zero_grad()
            loss_gen.backward(retain_graph=True)
            # TODO: clip_grad_norm
            self.gen_opt.step()
            #scheduler.step

        losses = {
            "loss_dis": loss_dis.detach().cpu(),
            "loss_gen": loss_gen.detach().cpu(),
            "loud_dist": loud_dist.detach().cpu(),
            "regularization": kl.detach().cpu(),
            "pred_true": pred_true.mean().detach().cpu(),
            "pred_fake": pred_fake.mean().detach().cpu(),
            "distance": distance.detach().cpu(),
            "beta": beta,
            "lambda": lambda_cur,
            "feature_matching": feature_matching_distance.detach().cpu(),
            "latent_dis_loss_gen": lat_dis_loss.detach().cpu(),
        }
        return losses

    def validation_step(self, x, attr):
        # evaluate VAE representation learning on validation set
        x = x.unsqueeze(1)

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
        z_c = torch.cat((z, attr), dim=1)
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
              bin_values=None,
              it_ckpt=None,
              display_step=1000,
              n_lat_dis=1,
              beta_inf=0.1,
              beta_delay=30000,
              lambda_inf=1,
              lambda_delay=100000,
              val_check=None,
              rave_mode=False,
              dataset=None,
              **kwargs):
        start = time.time()
        # TENSORBOARD
        writer = SummaryWriter(
            os.path.join(
                models_dir,
                f"runs/exp__{model_filename}_{time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime())}",
            ))

        # Initialize optimizer
        self._init_optimizer()

        # CF. Ninon scheduler on VAE optimizer # TODO: check scheduler.step()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.gen_opt,
            mode='min',
            factor=0.5,
            patience=6,
            verbose=False,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=1e-07,
            eps=1e-08)

        # resume training from checkpoint if specify
        start_epoch = 0
        start_it = 0
        if ckpt:
            print(ckpt)
            checkpoint = torch.load(ckpt)
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"],
                                         strict=False)
            self.decoder.load_state_dict(checkpoint["decoder_state_dict"],
                                         strict=False)
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
            "lambda": [],
            "feature_matching": [],
            "latent_dis_loss_gen": [],
            "latent_dis_loss_dis": []
        }

        check_nepoch = True
        if val_check == None:
            if len(train_loader) >= 10000:
                val_check = 10000
                check_nepoch = False
                print(f">>>>>>>>> valcheck nstep : {val_check}")
            else:
                check_nepoch = True
                val_check = 10000 // len(train_loader)
                print(f">>>>>>>>> valcheck nepoch : {val_check}")
        else:
            check_nepoch = True
            val_check = val_check
        valid_loss = []

        max_epochs = max_steps // len(train_loader)
        n_epochs = max_epochs if max_epochs <= 100000 else 100000
        cur_display_step = 0
        losses_display = np.zeros(len(train_losses.keys()) - 2)
        for epoch in range(start_epoch, n_epochs):
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

            for x, attr in tqdm(train_loader):
                # TODO: get mulaw attr
                # current training iteration
                step = len(train_loader) * epoch + batch_idx
                self.step = step

                # TODO: alternate lat_dis_step() with train_step()

                # apply a training step on the current batch x
                x = x.to(self.device)
                attr = attr.to(self.device)

                attr_cls = self.quantify(
                    attr,
                    torch.tensor(bin_values).to(self.device)).long()
                attr = dataset.normalize_all(attr)

                if rave_mode == True:
                    attr = torch.zeros_like(attr)

                cur_losses = self.training_step(x, attr, attr_cls, beta_inf,
                                                beta_delay, lambda_inf,
                                                lambda_delay)

                cur_losses_dis = 0

                if self.warmed_up == False:
                    for k in range(n_lat_dis):
                        cur_losses_dis += self.lat_dis_step(
                            x, attr_cls) / n_lat_dis

                # keep track of the loss
                cur_losses["latent_dis_loss_dis"] = cur_losses_dis
                losses_display += np.asarray(list(cur_losses.values()))
                cur_display_step += 1

                # Display training information
                if self.step % display_step == 0 and self.step > 0:
                    train_losses['it'].append(self.step)
                    train_losses['epoch'].append(epoch)
                    for k, l in zip(list(cur_losses.keys()), losses_display):
                        train_losses[k].append(l / cur_display_step)
                        writer.add_scalar(f"training loss : {k}",
                                          l / display_step, self.step)

                    print(
                        f"\nEpoch: [{epoch}/{n_epochs}] \tStep: [{self.step}/{max_steps}]"
                        f"\tTime: {time.time() - start} (s) \tTotal_gen_loss: {train_losses['loss_gen'][-1]}"
                    )
                    losses_display = np.zeros(len(train_losses.keys()) - 2)
                    cur_display_step = 0
                # evaluate on the validation set
                if (not check_nepoch and self.step % val_check == 0) \
                        or (check_nepoch and epoch % val_check == 0 and batch_idx == 0):
                    with torch.no_grad():
                        x_eval = []
                        y_eval = []
                        n_examples = 8
                        val_loss = 0
                        for x_val, attr_val in tqdm(val_loader):
                            x_val = x_val.to(self.device)
                            attr_val = attr_val.to(self.device)
                            attr_val = dataset.normalize_all(attr_val)
                            if rave_mode == True:
                                attr_val = torch.zeros_like(attr_val)
                            y_val, dist = self.validation_step(x_val, attr_val)
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
                if ((self.step % 10000 == 0 and self.step>0) \
                        or (it_ckpt and self.step in it_ckpt) \
                        or (epoch == n_epochs - 1 and batch_idx == len(train_loader) - 1))and(len(valid_loss)>0):
                    print(self.step)
                    print(display_step)
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
                            os.path.join(
                                models_dir,
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
