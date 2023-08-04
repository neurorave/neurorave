import os
import click

from raving_fader.config import settings, DataConfig, RaveConfig, FaderConfig, TrainConfig, BaseConfig
from raving_fader.helpers.core import load_config, write_config
# from raving_fader.helpers.eval import *
from raving_fader.models import models, FadeRAVE
from raving_fader.pipelines import pipelines

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from raving_fader.models.prior.model_pl import Model
from raving_fader.datasets.attr_prior_dataset import get_dataset_attr
# from raving_fader.datasets. import get_dataset_latent
from raving_fader.datasets.rave_dataset import get_dataset
from raving_fader.datasets.data_loaders import rave_data_loaders
from raving_fader.helpers.core import search_for_run
########################
#     RAVE TRAINING    #
########################


@click.argument("model")
@click.option(
    "--data_dir",
    default=settings.DATA_DIR,
    help=
    "Absolute path to data audio directory which contains an audio directory "
    "with the .wav files, default is the $DATA_DIR environment variable",
)
@click.option(
    "--models_dir",
    default=settings.MODELS_DIR,
    help=
    "Absolute path to the models directory to store checkpoints and training "
    "configurations, default is the $MODELS_DIR environment variable",
)
@click.option(
    "--config_dir",
    default=settings.CONFIG_DIR,  # ../config
    help=
    "Absolute path to the configuration directory with pre-filled train config "
    "yaml files, default is the $MODELS_DIR environment variable")
@click.option(
    "--config_file",
    "-f",
    default="fader_config.yaml",
    help=
    "Name of the model's yaml configuration file to use for train, default is "
    "rave_config.yaml")
@click.option(
    "--name",
    "-n",
    default=None,
    help="Name of the model can also be specify in the yaml configuration file"
    "if not specified (None) the name would be set to 'rave_{timestamp}'")
@click.option("--data_name",
              default=None,
              help="Name of the folder containing the pre-processed data")
@click.option(
    "--ckpt",
    default=None,
    help=
    "Filepath to model checkpoint to resume training from, default in yaml config file"
)
@click.option("--sr",
              default=None,
              help="Audio data sampling rate, default in yaml config file")
@click.option(
    "--batch",
    default=None,
    help="batch size, default in yaml config file, default in yaml config file"
)
@click.option(
    "--max_steps",
    default=None,
    help="number of total training steps, default in yaml config file")
@click.option(
    "--warmup",
    default=None,
    help="number of training steps for the first stage representation learning, "
    "default in yaml config file")
@click.option("--lambda_inf",
              default=None,
              help="Lambda for latent discriminator loss")
@click.option("--rave_mode",
              default=None,
              help="Disable conditionning if rave_mode=True")
@click.option("--save_step", "-i", default=100000)
def train(model, data_dir, models_dir, config_dir, config_file, name,
          data_name, ckpt, sr, batch, max_steps, warmup, save_step, lambda_inf,
          rave_mode):
    """
    Available models :
        - rave
        - ravelight
        - crave
        - faderave
    """
    if model not in list(models.keys()):
        raise ValueError(
            f"Model named {model} not implemented. Try models in : {list(models.keys())}"
        )
    # -----------------
    #    Load config
    # -----------------
    config = load_config(os.path.join(config_dir, config_file))

    # >>> DATA
    # if name:
    #     config["data"]["data_name"] = data_name

    data_config = DataConfig(**config["data"])
    # create results directory
    os.makedirs(os.path.join(models_dir, name), exist_ok=True)

    # add / update values
    data_config.wav = os.path.join(data_dir, "audio")
    data_config.preprocessed = data_dir
    if sr:
        data_config.sr = int(sr)

    # >>> MODEL
    rave_config = RaveConfig(**config["rave"])
    # rave
    if warmup:
        rave_config.warmup = warmup
    fader_config = None
    if model == "faderave":
        fader_config = FaderConfig(rave=rave_config, **config["fader"])

    # >>> TRAIN
    if name:
        config["train"]["name"] = name
    train_config = TrainConfig(**config["train"])
    name = train_config.name
    # train
    train_config.models_dir = os.path.join(models_dir, name)
    if batch:
        train_config.batch = batch
    if max_steps:
        train_config.max_steps = max_steps
    if ckpt:
        train_config.ckpt = ckpt
    if lambda_inf:
        train_config.lambda_inf = lambda_inf
    if rave_mode:
        train_config.rave_mode = rave_mode

    # save config
    config = BaseConfig(data=data_config,
                        rave=rave_config,
                        fader=fader_config,
                        train=train_config)

    write_config(os.path.join(models_dir, name, "train_config.yaml"),
                 config.dict())

    # -----------------
    #    Train Model
    # -----------------
    # instantiate pipeline
    pipeline = pipelines[model](config)

    # train
    if save_step:
        it_ckpt = [
            int(i * save_step)
            for i in range(1, config.train.max_steps // save_step + 1)
        ]
    pipeline.train(ckpt=ckpt, it_ckpt=it_ckpt)


@click.argument("attr_pth")
@click.option("--name", "-n", default=None)
@click.option("--batch", "-b", default=8)
@click.option("--max_steps", "-s", default=1000000)
@click.option("--n_workers", "-w", default=2)
@click.option('--use_gpu', default=1)
def train_prior_attr(attr_pth, name, batch, use_gpu, n_workers, max_steps):
    print(
        f">>> use_gpu: {use_gpu} -- CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    # DATA
    descriptors = ["centroid", "rms", "bandwidth", "sharpness", "booming"]
    dataset = get_dataset_attr(attr_pth, ref_descriptors=descriptors)
    train_set, val_set = rave_data_loaders(batch, dataset)
    # MODEL
    model = Model(resolution=64,
                  res_size=512,
                  skp_size=256,
                  kernel_size=3,
                  cycle_size=4,
                  n_layers=10,
                  descriptors=descriptors,
                  latent=None)

    #TRAIN
    #CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="validation",
        filename="best",
    )
    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")

    val_check = {}
    if len(train_set) >= 10000:
        val_check["val_check_interval"] = 10000
    else:
        nepoch = 10000 // len(train_set)
        val_check["check_val_every_n_epoch"] = nepoch
    ckpt = None
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(os.path.join("runs", name),
                                            name="prior_attr"),
        gpus=use_gpu,
        callbacks=[validation_checkpoint, last_checkpoint],
        resume_from_checkpoint=search_for_run(ckpt),
        max_epochs=100000,
        max_steps=max_steps,
        **val_check,
    )
    trainer.fit(model, train_set, val_set)


##########################
# ---------------------- #
#    Train RAVE Prior    #
# ---------------------- #
##########################
@click.argument("model_pth")
@click.argument("chkpt")
@click.option("--data_dir", default=settings.DATA_DIR)
@click.option("--name", "-n", default=None)
@click.option("--batch", "-b", default=8)
@click.option("--max_steps", "-s", default=1000000)
@click.option("--n_workers", "-w", default=2)
@click.option('--use_gpu', default=1)
@click.argument("latent_pth")
@click.option("--name", "-n", default=None)
@click.option("--batch", "-b", default=8)
@click.option("--max_steps", "-s", default=1000000)
@click.option("--n_workers", "-w", default=2)
@click.option('--use_gpu', default=1)
def train_prior_attr(latent_pth, name, batch, use_gpu, n_workers, max_steps):
    print(
        f">>> use_gpu: {use_gpu} -- CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    # DATA
    dataset = get_dataset_latent(latent_pth)
    train_set, val_set = rave_data_loaders(batch, dataset)
    # MODEL
    model = Model(resolution=64,
                  res_size=512,
                  skp_size=256,
                  kernel_size=3,
                  cycle_size=4,
                  n_layers=10,
                  descriptors=None,
                  latent=latent)

    #TRAIN
    #CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="validation",
        filename="best",
    )
    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")

    val_check = {}
    if len(train_set) >= 10000:
        val_check["val_check_interval"] = 10000
    else:
        nepoch = 10000 // len(train_set)
        val_check["check_val_every_n_epoch"] = nepoch
    ckpt = None
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(os.path.join("runs", name),
                                            name="prior_attr"),
        gpus=use_gpu,
        callbacks=[validation_checkpoint, last_checkpoint],
        resume_from_checkpoint=search_for_run(ckpt),
        max_epochs=100000,
        max_steps=max_steps,
        **val_check,
    )
    trainer.fit(model, train_set, val_set)