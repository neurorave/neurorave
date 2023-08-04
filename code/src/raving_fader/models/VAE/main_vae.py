from torch.utils.tensorboard import SummaryWriter  # /!\ Keep this one first

import shutil
from time import time
import subprocess
from datetime import datetime
from texttable import Texttable

import torch.nn as nn

# Import Analyzers
from stats import *
from latent import *
# from figures import evaluate_dimensions
from analyze import sampling, test_faders
from signal_features import compute_signal_features
# Import models
from fader_loader import import_fader_dataset, import_random
from train_fader import Trainer
from evaluate_fader import Evaluator
from modules import EncoderCNN, DecoderCNNFader
from model import VAEFader, LatentDiscriminator
from utils import mu_law_encode
from logger import Logger
from analyze_results import test

# analyse
from analyze import reconstruction

from torchvision import models
from torchsummary import summary

# Import utils
from init import init
import yaml
from effortless_config import Config
from tqdm import tqdm

## INIT ##
GIT_ID = "NO_GIT_ID"
try:
    GIT_ID = subprocess.check_output(["git", "describe",
                                      "--always"]).strip().decode("utf-8")
except:
    pass
DATE = datetime.now().strftime("%d_%m_%Y__%H:%M:%S")


class Args(Config):
    CONFIG = f"config.yaml"
    NAME = f"{DATE}_faders_vae_{GIT_ID}"
    # ROOT = "runs"


with open(Args.CONFIG, "r") as config:
    config = yaml.safe_load(config)
args = init(config)

##
# -----------------------------------------------------------
#
# Data Importation
#
# -----------------------------------------------------------
# Data importing
print('[Importing dataset]')
# Load dataset and data loaders
if os.path.exists(config["data"]["loaders_path"] + '.th'):
    print('Found ' + config["data"]["loaders_path"] + '.th')
    train_loader, valid_loader, test_loader = torch.load(config["data"]["loaders_path"] + '.th')
else:
    train_loader, valid_loader, test_loader, config = import_fader_dataset(config)

# Define size of spectro
batch_x, batch_y = next(iter(train_loader))
print("batch_x size (inputs): ")
print(batch_x.shape)
config["data"]["input_size"] = batch_x.shape[
                               1:]  # [batch *] 80 * 87: spectro

# Define size of attributes
print("batch_y size (attributes): ")
print(batch_y.shape)
config["data"]["attributes_size"] = batch_y.shape[1:]  # [batch *] 7 * 431

# Define numbers of attributes
print("number of attr: ")
print(batch_y.shape[1])
config["data"]["num_attributes"] = batch_y.shape[1]
print("*" * 100)

# Valid batch
# valid_x, valid_y = next(iter(valid_loader))

##
# -----------------------------------------------------------
#
# Model and layer creation
#
# -----------------------------------------------------------

print('[Creating encoder and decoder]')
config["model"]["type_mod"] = 'normal'
encoder = EncoderCNN(config)
config["model"]["cnn_size"] = encoder.cnn_size
decoder = DecoderCNNFader(config)

print('[Creating model]')
if config["model"]["model"] == 'vae_fader':
    model = VAEFader(encoder, decoder,
                     config).float().to(config["train"]["device"])
    lat_dis = LatentDiscriminator(config).to(config["train"]["device"])
else:
    model = None
    lat_dis = None
    print("Oh no, unknown model " + config["model"]["model"] + ".\n")
    exit()
# Print model
# print(model)

# Initialize the model weights
# print('[Initializing weights]')
# if args.initialize:
#     model.apply(init_classic)

##
# -----------------------------------------------------------
#
# Optimizer
#
# -----------------------------------------------------------
print('[Creating optimizer]')
# Optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config["train"]["lr"],
                             weight_decay=1e-4)
# if config["model"]["fader"]:
lat_dis_optimizer = torch.optim.Adam(lat_dis.parameters(),
                                     lr=config["train"]["lr"],
                                     weight_decay=1e-4)

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.5,
                                                       patience=6,
                                                       verbose=False,
                                                       threshold=0.0001,
                                                       threshold_mode='rel',
                                                       cooldown=0,
                                                       min_lr=1e-07,
                                                       eps=1e-08)

##
# -----------------------------------------------------------
#
# Losses functions
#
# -----------------------------------------------------------
print('[Creating criterion]')
# Losses
criterion = nn.MSELoss(reduction="none")

##
# -----------------------------------------------------------
#
# Training loop
#
# -----------------------------------------------------------
# Set time
time0 = time()
# Initial test
print('[Initial evaluation]')
# learn.test(model, args, epoch=0)  # First test on randomly initialized data

# Comment if you don't need the summary of your model (architecture, nb parameters)
# summary(model, input_size=(1, 80, 87)
print('[Starting main training]')
log = Logger(config)
trainer = Trainer(model, lat_dis, optimizer, lat_dis_optimizer, log, config)
evaluator = Evaluator(model, lat_dis, criterion, log, config)

# Start epochs
n_iter = 0
early_stop = 0
beta = torch.zeros(1).to(config["train"]["device"])
lambda_dis = torch.zeros(1).to(config["train"]["device"])
iter_train = 1  # Update beta for ELBO
for n_epoch in range(config["train"]["epochs"]):
    print(f"Starting epoch: {n_epoch}")
    # start training
    print(f'train pass on: {config["train"]["device"]}')

    # if config["debug"]:
    #     train_loader, valid_loader, test_loader = import_random(config)

    for batch_x, batch_y in tqdm(iter(train_loader), total=len(train_loader)):
        batch_x = batch_x.to(config["train"]["device"], non_blocking=True)
        batch_y = batch_y.to(config["train"]["device"], non_blocking=True)
        batch_y = mu_law_encode(batch_y, config["model"]["num_classes"],
                                config)
        if config["model"]["test_trained"]:
            test(train_loader, config)
            exit()
        # latent discriminator training
        for _ in range(config["discriminator"]["n_lat_dis"]):
            trainer.lat_dis_step(config, batch_x, batch_y)

        # autoencoder training
        trainer.vae_step(config, batch_x, batch_y, criterion, beta, lambda_dis)

    # writing results
    if config["debug"]:
        # Get the graph of the model in tensorboard
        log.write_model_graph(model, [batch_x, batch_y])
    for name, weight in model.named_parameters():
        log.write_histogram(name, weight, n_epoch)
        log.write_histogram(f'{name}.grad', weight.grad, n_epoch)
    log.write_scalar('lambda_dis/train', lambda_dis, n_epoch)
    log.write_scalar('beta/train', beta, n_epoch)

    # Update beta and lambda for regularization
    if iter_train > config["model"][
        "beta_delay"] and beta < config["model"]["beta"]:
        beta += (config["model"]["beta"] / config["train"]["epochs"])
    if config["model"]["fader"]:
        if iter_train > config["discriminator"][
            "lambda_delay"] and lambda_dis < config[
            "discriminator"]["lambda_lat_dis"]:
            lambda_dis += (config["discriminator"]["lambda_lat_dis"] /
                           config["train"]["epochs"])

    # Track on stuffs
    iter_train += 1
    print("*******" * 10)
    print('* Useful & incredible tracking:')
    t = Texttable()
    t.add_rows([[
        'Name', 'global_loss', 'latent_dis_loss', 'vae_loss/train',
        'recon_loss', 'kl_div'
    ],
        [
            'Train', trainer.global_loss, trainer.latent_dis_loss, trainer.vae_loss,
            trainer.recon_loss_mean, trainer.kl_div_mean
        ]])
    print(t.draw())
    print(10 * '*******')
    # print training statistics
    trainer.step(n_epoch)

    # Start evaluation
    print(f'validation pass on: {config["train"]["device"]}')
    for eval_x, eval_y in tqdm(iter(valid_loader), total=len(valid_loader)):
        eval_x = eval_x.float().to(config["train"]["device"], non_blocking=True)
        eval_y = eval_y.float().to(config["train"]["device"], non_blocking=True)
        # run all evaluations / save best or periodic model
        to_log = evaluator.evaluate(eval_x, eval_y, n_epoch)

    early_stop = trainer.save_best_periodic(to_log, n_epoch, early_stop,
                                            config)
    # early stopping
    if config["train"]["early_stop"] > 0:
        early_stop += 1
        if early_stop > config["train"]["early_stop"]:
            print('[Model stopped early]')
            break

    # Compare input data and reconstruction
    #if n_epoch % 1 == 0:
    #    reconstruction(config, model, n_epoch, test_loader, log)

    # Test faders
    print(f'test pass on: {config["train"]["device"]}')
    # if n_epoch % 1 == 0:
    #     test_faders(config, model, test_loader, n_epoch, log)

print('\nTraining Time in minutes =', (time() - time0) / 60)

test_faders(config, model, train_loader, 200, log)

# %% ---------------------------------------------------------
#
# Load selected model
#
# -----------------------------------------------------------
print('[Importing model]')
# Infer correct model path if absent
if len(config["data"]["model_path"]) == 0:
    model_variants = [config["data"]["dataset"]]
    config["data"]["model_path"] = config["data"]["output_path"]
    for m in model_variants:
        config["data"]["model_path"] += str(m) + '_'
    config["data"]["model_path"] = config["data"]["model_path"][:-1] + '/'
# Reload best performing model
model = torch.load(config["data"]["model_path"] + 'model_full.pth',
                   map_location=config["train"]["device"])
latent_dis = torch.load(config["data"]["model_path"] + 'lat_dis_full.pth',
                        map_location=config["train"]["device"])
##
# -----------------------------------------------------------
#
# Evaluate stuffs when model trained
#
# -----------------------------------------------------------
print('[Evaluation]')
model = model.eval()
# Sample random point from latent space
print('   - Sampling.')
sampling(config, model, log)
# Statistics
print('   - Statistics.')
# stats_model(model, test_loader, config)

print("*" * 50)
print("OVER")
print("*" * 50)
