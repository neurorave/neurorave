import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from typing import Any, List


class Logger:
    """
    Summary class for interacting with TensorBoard. We mainly use this class
    to perform some online debugging of our toolbox. However, this can also
    be used to generate nice plots of how the metrics behave.
    """

    def __init__(self, config):
        self.config = config
        self.summary_writer = SummaryWriter(self.config["data"]["output_path"] + "/tensorboard_logs/" +
                                            self.config["data"]["representation"] + "/" +
                                            str(self.config["train"]["device"]))
        pass

    def write_generic(self,
                      write_type: str,
                      name: str,
                      value: Any,
                      step: int = 0):
        # Get summary writer
        writer = self.summary_writer or SummaryWriter(self.args.output + "/" + self.args.model_save)
        getattr(writer, write_type)(name, value, step)
        writer.flush()
        self.summary_writer = writer

    def write_scalar(self, name: str, value: float, step: int = 0):
        """ Add a single scalar (value) in a given plot (name) at time step (step)"""
        self.write_generic("add_scalar", name, value, step)

    def write_scalars(self, name: str, values: dict, step: int = 0):
        """ Add multiple scalars (values) through a dict in a given plot (name) at time step (step)"""
        self.write_generic("add_scalars", name, values, step)

    def write_histogram(self, name: str, values: torch.Tensor, index: int = 0):
        """ Create an histogram from a Tensor (values) in a given plot (name) at vertical index (index)"""
        self.write_generic("add_histogram", name, values, index)

    def write_image(self, name: str, image: np.ndarray, step: int = 0):
        """ Add a np.ndarray image of shape (C, H, W) """
        self.write_generic("add_image", name, image, step)

    def write_images(self, name: str, images: np.ndarray, step: int = 0):
        """ Add multiple np.ndarray images of shape (N, C, H, W) """
        self.write_generic("add_images", name, images, step)

    def write_figure(self, name: str, figure: plt.figure, step: int = 0):
        """ Directly add a Matplotlib figure (handle pointer)  """
        self.write_generic("add_figure", name, figure, step)

    def write_graph(self, name: str, figure: plt.figure, step: int = 0):
        """ Directly add a Matplotlib figure (handle pointer)  """
        self.write_generic("add_figure", name, figure, step)

    def write_embedding(self,
                        features: torch.Tensor,
                        classes: torch.Tensor,
                        label_imgs: torch.Tensor):
        """
        Add an embedding projector to the graph
        """
        writer = self.summary_writer or SummaryWriter(self.args.output + "/" + self.args.model_save)
        writer.add_embedding(features, metadata=classes, label_imgs=label_imgs)
        writer.flush()
        self.summary_writer = writer

    def write_model_graph(self, model: torch.nn.Module, input_t: torch.Tensor):
        """
        Add a model graph to the board
        """
        writer = self.summary_writer or SummaryWriter(self.args.output + "/" + self.args.model_save)
        writer.add_graph(model, input_t)
        writer.flush()
        self.summary_writer = writer
