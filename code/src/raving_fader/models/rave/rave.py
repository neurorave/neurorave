import torch

from raving_fader.models.core import BaseRAVE
from raving_fader.models.rave.networks import Generator
from raving_fader.models.rave.core import get_beta_kl_cyclic_annealed


class RAVE(BaseRAVE):
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
                 device="cpu"):
        super().__init__(
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
            no_latency,
            min_kl,
            max_kl,
            cropped_latent_size,
            feature_match,
            sr,
            device
        )

        self.decoder = Generator(
            latent_size,
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

    def _init_optimizer(self, learning_rate=1e-4, beta_1=0.5, beta_2=0.9):
        param_vae = list(self.encoder.parameters()) + list(
            self.decoder.parameters())
        self.gen_opt = torch.optim.Adam(param_vae,
                                        lr=learning_rate,
                                        betas=(beta_1, beta_2))
        self.dis_opt = torch.optim.Adam(self.discriminator.parameters(),
                                        lr=learning_rate,
                                        betas=(beta_1, beta_2))

    def decode(self, z):
        y = self.decoder(z, add_noise=True)
        if self.pqmf is not None:
            y = self.pqmf.inverse(y)
        return y

    def training_step(self, batch, step):
        x = batch.unsqueeze(1)

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
        y = self.decoder(z, add_noise=self.warmed_up)

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
        # beta = get_beta_kl_cyclic_annealed(
        #     step=step,
        #     cycle_size=5e4,
        #     warmup=self.warmup // 2,
        #     min_beta=self.min_kl,
        #     max_beta=self.max_kl,
        # )
        beta = 0.1

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
        x = batch.unsqueeze(1)

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
        y = self.decoder(z, add_noise=self.warmed_up)

        # 5. inverse multi band decomposition (pqmf -1) --> recomposition
        if self.pqmf is not None:
            x = self.pqmf.inverse(x)
            y = self.pqmf.inverse(y)

        # 6. compute reconstruction loss ie. multiscale spectral distance
        distance = self.distance(x, y)

        # return torch.cat([x, y], -1), mean
        return y, distance
