from torch.utils.tensorboard import SummaryWriter

import torch
from torch.autograd import Variable
from model import get_attr_loss
from utils import clip_grad_norm, get_lambda
import os


class Trainer:
    def __init__(self, model, lat_dis, optimizer, lat_dis_optimizer, log, config):
        self.config = config

        self.model = model
        self.lat_dis = lat_dis
        self.log = log

        # optimizers
        self.model_optimizer = optimizer
        self.lat_dis_optimizer = lat_dis_optimizer

        # global loss
        self.global_loss = torch.zeros(1).to(config["train"]["device"])

        # loss for latent discriminator
        self.latent_dis_loss = torch.zeros(1).to(config["train"]["device"])

        # losses for vae
        self.vae_loss = torch.zeros(1).to(config["train"]["device"])
        self.recon_loss_mean = torch.zeros(1).to(config["train"]["device"])
        self.kl_div_mean = torch.zeros(1).to(config["train"]["device"])

        # best reconstruction loss / best accuracy
        self.best_loss = 1e12
        self.best_accu = -1e12
        self.config["model"]["n_total_iter"] = 0

    def lat_dis_step(self, config, batch_x, batch_y):
        """
        Train the latent discriminator.
        """
        self.model.eval()
        self.lat_dis.train()
        # batch / encode / discriminate
        batch_x = batch_x.to(config["train"]["device"], non_blocking=True)
        with torch.no_grad():
            enc_outputs = self.model.encode(Variable(batch_x))
        preds = self.lat_dis(Variable(enc_outputs[0].data))
        # loss / optimize
        latent_dis = get_attr_loss(preds, batch_y, False, config)
        self.latent_dis_loss += latent_dis.detach()
        self.lat_dis_optimizer.zero_grad()
        latent_dis.backward()
        if config["discriminator"]["clip_grad_norm"]:
            clip_grad_norm(self.lat_dis.parameters(),
                           config["discriminator"]["clip_grad_norm"])
        self.lat_dis_optimizer.step()

    def vae_step(self, config, batch_x, batch_y, criterion, beta, lambda_dis):
        """
        Train the vae with kl div loss.
        Train the encoder with discriminator loss.
        """
        self.model.train()
        if config["model"]["fader"]:
            self.lat_dis.eval()
        # batch / encode / decode
        batch_x, batch_y = batch_x.to(config["train"]["device"]), batch_y.to(
            config["train"]["device"], non_blocking=True)
        if not config["model"]["fader"]:
            batch_y = torch.zeros_like(batch_y).to(config["train"]["device"], non_blocking=True)
        enc_outputs, dec_outputs, z_loss = self.model(batch_x, batch_y)
        # Compute reconstruction loss
        recon_loss = criterion(batch_x, dec_outputs).mean(dim=(-1, -2)).sum()
        self.recon_loss_mean += recon_loss.detach()
        if config["debug"]:
            print('Batch')
            print(batch_x.min())
            print(dec_outputs.min())
            print(batch_x.max())
            print(dec_outputs.max())
            print(batch_x.mean())
            print(dec_outputs.mean())
        # compute ELBO loss
        self.kl_div_mean += z_loss.detach()
        loss = recon_loss + beta * z_loss
        self.vae_loss += loss.detach()

        # encoder loss from the latent discriminator
        if config["model"]["fader"]:
            lat_dis_preds = self.lat_dis(enc_outputs.data)
            lat_dis_loss = get_attr_loss(lat_dis_preds, batch_y, True, config)
            loss = loss + lambda_dis * lat_dis_loss
        # check NaN
        self.global_loss += loss.detach()
        if (loss != loss).data.any():
            print("NaN detected")
            exit()
        # optimize
        self.model_optimizer.zero_grad()
        loss.backward()

        if config["discriminator"]["clip_grad_norm"]:
            clip_grad_norm(self.model.parameters(),
                           config["discriminator"]["clip_grad_norm"])
        self.model_optimizer.step()

    def step(self, step):
        """
        End training iteration / save training statistics.
        """

        # Get summary writer
        self.log.write_scalar('global_loss', self.global_loss, step)
        self.log.write_scalar('latent_dis_loss', self.latent_dis_loss, step)
        self.log.write_scalar('vae_loss/train', self.vae_loss, step)
        self.log.write_scalar('reconstruction_loss/train', self.recon_loss_mean, step)
        self.log.write_scalar('kldiv_loss/train', self.kl_div_mean, step)
        self.global_loss.fill_(0)
        self.latent_dis_loss.fill_(0)
        self.vae_loss.fill_(0)
        self.recon_loss_mean.fill_(0)
        self.kl_div_mean.fill_(0)

        return self.global_loss, self.latent_dis_loss, self.vae_loss, self.recon_loss_mean, self.kl_div_mean

    def save_model(self, variant, config):
        """
        Save the model
        """
        def save(model, name):
            if len(config["data"]["model_path"]) == 0:
                model_variants = [config["data"]["dataset"]]
                config["data"]["model_path"] = config["data"]["output_path"]
                for m in model_variants:
                    config["data"]["model_path"] += str(m) + '_'
                config["data"][
                    "model_path"] = config["data"]["model_path"][:-1] + '/'
            path = os.path.join(config["data"]["model_path"],
                                '%s_%s.pth' % (name, variant))
            torch.save(model, path)

        save(self.model, 'model')
        save(self.lat_dis, 'lat_dis')

    def save_best_periodic(self, to_log, step, early_stop, config):
        """
        Save the best models / periodically save the models.
        """

        if to_log['vae_loss'] < self.best_loss:
            self.best_loss = to_log['vae_loss']
            self.log.write_scalar('global_loss', self.global_loss, step)
            self.save_model('full', config)
            early_stop = 0

        if to_log['n_epoch'] % 5 == 0 and to_log['n_epoch'] > 0:
            self.save_model('periodic-%i' % to_log['n_epoch'], config)
        return early_stop


########################
# Associate functions #
########################

# def save(model, config, variant):
#     # Save entire model
#     if not os.path.exists(config["data"]["model_path"]):
#         os.makedirs(config["data"]["model_path"])
#     torch.save(model, config["data"]["model_path"] + '_' + str(variant) + '.pth')
#
#
# def resume_training(config, epoch):
#     # Specify the wishing epoch resuming here
#     model = torch.load(config["data"]["model_path"] + '_epoch_' + str(epoch) + '.pth')
#     model.eval()
