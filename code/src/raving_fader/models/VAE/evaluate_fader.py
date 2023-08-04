from torch.utils.tensorboard import SummaryWriter

import json
import numpy as np

from model import update_predictions
from utils import print_accuracies


class Evaluator(object):
    def __init__(self, model, lat_dis, criterion, log, config):
        """
        Evaluator initialization.
        """
        # parameters
        self.config = config

        # modules
        self.model = model
        self.lat_dis = lat_dis
        self.criterion = criterion
        self.log = log

    def eval_reconstruction_loss(self, batch_x, batch_y):
        """
        Compute the vae reconstruction perplexity.
        """
        self.model.eval()
        _, dec_outputs, _ = self.model(batch_x, batch_y)
        costs = self.criterion(dec_outputs, batch_x).mean(dim=(-1, -2)).sum()
        # costs = (rec_cost.sum(dim=0))

        return costs

    def eval_lat_dis_accuracy(self, batch_x, batch_y):
        """
        Compute the latent discriminator prediction accuracy.
        """
        config = self.config
        self.model.eval()
        self.lat_dis.eval()

        all_preds = [[] for _ in range(config["data"]["num_attributes"])]
        enc_outputs = self.model.encode(batch_x)
        preds = self.lat_dis(enc_outputs[0])
        update_predictions(all_preds, preds, batch_y, config)

        return [np.mean(x) for x in all_preds]

    def evaluate(self, eval_x, eval_y, n_epoch):
        """
        Evaluate all models / log evaluation results.
        """
        config = self.config

        # reconstruction loss
        vae_loss = self.eval_reconstruction_loss(eval_x, eval_y)

        # latent discriminator accuracy
        log_lat_dis = {}
        if config["discriminator"]["n_lat_dis"]:
            lat_dis_accu = self.eval_lat_dis_accuracy(eval_x, eval_y)
            log_lat_dis['lat_dis_accu'] = np.mean(lat_dis_accu)
            for accu, name in zip(lat_dis_accu, [
                    "rms", "rolloff", "bandwidth",
                    "centroid"
            ]):
                log_lat_dis['lat_dis_accu_%s' % name] = accu
            self.log.write_scalars("attributes", log_lat_dis, n_epoch)
        # log autoencoder loss
        # JSON log
        to_log = dict([('n_epoch', n_epoch),
                       ('vae_loss', vae_loss)])  # .update(log_lat_dis)
        to_log.update(log_lat_dis)
        return to_log


def compute_accuracy(classifier, data, params):
    """
    Compute the classifier prediction accuracy.
    """
    classifier.eval()
    bs = params.batch_size

    all_preds = [[] for _ in range(len(classifier.attr))]
    for i in range(0, len(data), bs):
        batch_x, batch_y = data.eval_batch(i, i + bs)
        preds = classifier(batch_x).data.cpu()
        update_predictions(all_preds, preds, batch_y.data.cpu(), params)

    return [np.mean(x) for x in all_preds]
