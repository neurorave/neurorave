import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from abc import abstractmethod
from torch.autograd import Variable


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *input):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(
            BaseModel,
            self).__str__() + '\nTrainable parameters: {}'.format(params)


##
# -----------------------------------------------------------
#
# Basic Auto encoder without regularization
#
# -----------------------------------------------------------


class AE(BaseModel):
    def __init__(self, encoder, decoder, config):
        super(AE, self).__init__()
        self.sample = None
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.iteration = 0
        self.input_size = config["data"]["input_size"][0]
        self.map_latent = nn.Linear(config["model"]["enc_hidden_size"],
                                    config["model"]["latent_size"])
        self.z_loss = torch.Tensor(1).zero_().to(config["train"]["device"])

    def encode(self, x):
        x = self.encoder(x)
        x = self.map_latent(x)
        return x, x, x

    def decode(self, z):
        recon = self.decoder(z)
        return recon

    def forward(self, x):
        self.sample = x
        if self.training:
            self.decoder.sample = self.sample
            self.decoder.iteration += 1
        z, _, _ = self.encode(x)
        recon = self.decode(z)
        return recon, z, self.z_loss


##
# -----------------------------------------------------------
#
# Variational auto-encoder (Kullback-Leibler regularization)
# With latent discriminator on decoder
#
# -----------------------------------------------------------


class VAEFader(BaseModel):
    def __init__(self, encoder, decoder, config):
        super(VAEFader, self).__init__()
        self.config = config
        self.sample = None
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.iteration = 0
        self.input_size = config["data"]["input_size"]
        self.linear_mu = nn.Linear(config["model"]["enc_hidden_size"],
                                   config["model"]["latent_size"])
        self.linear_var = nn.Linear(config["model"]["enc_hidden_size"],
                                    config["model"]["latent_size"])

    # generate spectrogram from latent space
    def generate(self, z):
        generated_bar = self.decoder(z)
        return generated_bar

    def encode(self, x):
        x = x.float()
        out = self.encoder(x, self.config)
        mu = self.linear_mu(out)
        var = F.softplus(self.linear_var(out))
        distribution = Normal(mu, var)
        z = distribution.rsample()
        return z, mu, var

    def decode(self, z, y):
        recon = self.decoder(z, y)
        return recon

    def forward(self, x, y):
        if self.training:
            self.decoder.sample = self.sample
            self.decoder.iteration += 1
        # Encode latent input
        z, mu, var = self.encode(x)
        # Regularize latent
        z_loss = regularize(z, mu, var)
        # Decode
        recon = self.decode(z, y)
        return z, recon, z_loss  # Return enc_outputs, dec_outputs


########################
# VAE helper functions #
########################


def regularize(z, mu, var):
    n_batch = z.shape[0]
    # KL Div
    kl_div = -0.5 * torch.mean(1 + torch.log(var) - mu.pow(2) - var, axis=(-1))
    # Normalize by size of batch
    #kl_div = kl_div / n_batch
    return kl_div.sum() / n_batch


class LatentDiscriminator(nn.Module):
    def __init__(self, config):
        super(LatentDiscriminator, self).__init__()
        activation_f = nn.ReLU
        z_dim = config["model"]["latent_size"]
        n_attr = config["data"]["num_attributes"]
        n_classes = config["model"]["num_classes"]
        # if activation == 'tanh':
        #     activation_f = nn.Tanh
        # elif activation == 'relu':
        #     activation_f = nn.ReLU

        assert config["discriminator"]["num_layers"] >= 2
        layers = []
        for _ in range(config["model"]["num_layers"] - 1):
            layers.append(nn.Linear(z_dim, z_dim // 2))
            layers.append(activation_f())
            z_dim = z_dim // 2
        layers.append(nn.Linear(z_dim, n_attr * n_classes))
        # layers.append(nn.Sigmoid())
        self.layers = nn.ModuleList(layers)

    def forward(self, lv):
        for i in range(len(self.layers)):
            lv = self.layers[i](lv)
        return lv


def get_attr_loss(output, attributes, flip, config):
    """
    Compute attributes loss.
    """
    assert type(flip) is bool
    k = 0
    loss = 0
    n_cat = config["model"]["num_classes"]
    for i in range(config["data"]["num_attributes"]):
        # categorical
        x = output[:, k:k + n_cat].contiguous()
        y = attributes[:, i]
        y = y.view(-1)
        # generate different categories
        shift = torch.LongTensor(y.size()).random_(n_cat - 1) + 1
        y = (y + Variable(shift.to(config["train"]["device"], non_blocking=True))) % n_cat
        loss += F.cross_entropy(x, y)
        k += n_cat
    return loss


def update_predictions(all_preds, preds, attributes, config):
    """
    Update discriminator / classifier predictions.
    """
    assert len(all_preds) == config["data"]["num_attributes"]
    k = 0
    n_cat = config["model"]["num_classes"]
    for j, i in enumerate(range(config["data"]["num_attributes"])):
        _preds = preds[:, k:k + n_cat].max(1)[1]
        y = attributes[:, i]
        y = y.view(-1)
        # generate different categories
        shift = torch.LongTensor(y.size()).random_(n_cat - 1) + 1
        y = (y + Variable(shift.cuda())) % n_cat
        all_preds[j].extend((_preds == y).tolist())
        k += n_cat


def get_mappings(params):
    """
    Create a mapping between attributes and their associated IDs.
    """
    if not hasattr(params, 'mappings'):
        mappings = []
        k = 0
        for (_, n_cat) in params.attr:
            assert n_cat >= 2
            mappings.append((k, k + n_cat))
            k += n_cat
        assert k == params.n_attr
        params.mappings = mappings
    return params.mappings

#
# def flip_attributes(attributes, params, attribute_id, new_value=None):
#     """
#     Randomly flip a set of attributes.
#     """
#     assert attributes.size(1) == params.n_attr
#     mappings = get_mappings(params)
#     attributes = attributes.data.clone().cpu()
#
#     def flip_attribute(attribute_id, new_value=None):
#         bs = attributes.size(0)
#         i, j = mappings[attribute_id]
#         attributes[:, i:j].zero_()
#         if new_value is None:
#             y = torch.LongTensor(bs).random_(j - i)
#         else:
#             assert new_value in range(j - i)
#             y = torch.LongTensor(bs).fill_(new_value)
#         attributes[:, i:j].scatter_(1, y.unsqueeze(1), 1)
#
#     if attribute_id == 'all':
#         assert new_value is None
#         for attribute_id in range(len(params.attr)):
#             flip_attribute(attribute_id)
#     else:
#         assert type(new_value) is int
#         flip_attribute(attribute_id, new_value)
#
#     return Variable(attributes.cuda())
