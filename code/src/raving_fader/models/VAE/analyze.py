from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import distributions
import librosa
from fader_loader import spectral_features, spectral_features_spectro
from utils import griffin_lim, mel_to_stft
import soundfile as sf
from matplotlib.patches import Polygon


def reconstruction(config, model, epoch, loader, log):
    model = model.eval()
    if config["data"]["representation"] in ['spectrogram', 'melspectrogram']:
        cur_input, cur_attribute = next(iter(loader))
        cur_input, cur_attribute = cur_input.float().to(
            config["train"]["device"]), cur_attribute.float().to(
                config["train"]["device"])
        # Plot setting
        n_rows = 2  # array of sub-plots
        fig_size = np.array([8, 20])  # figure size, inches
        # create figure (fig), and array of axes (ax)
        fig, ax = plt.subplots(nrows=n_rows, figsize=fig_size)
        ax[0].imshow(librosa.power_to_db(cur_input[0].cpu()),
                     origin='lower',
                     aspect='auto')
        # Reconstruction
        _, x_reconstruct, _ = model(cur_input, cur_attribute)
        x_reconstruct = x_reconstruct[0].detach()

        # print(librosa.power_to_db(x_reconstruct.cpu()).shape)
        x_reconstruct = x_reconstruct.squeeze(0)
        ax[1].imshow(librosa.power_to_db(x_reconstruct.cpu()),
                     origin='lower',
                     aspect='auto')
        # write row/col indices as axes' title for identification
        plt.tight_layout(True)
        plt.savefig(config["data"]["figures_path"] + 'epoch_' + str(epoch))
        log.write_figure("reconstruction_" + str(epoch), plt.gcf())
        plt.close()

        fig, ax = plt.subplots(nrows=n_rows, figsize=fig_size)
        ax[0].plot((cur_input[0, :, 10].cpu()))
        ax[1].plot((x_reconstruct[:, 10].cpu()))
        #plt.tight_layout(True)
        plt.savefig(config["data"]["figures_path"] + 'epoch_slice_' +
                    str(epoch))
        log.write_figure("figures/reconstruction_slicing_" + str(epoch), plt.gcf())
        plt.close()

        # # generate .wav from sampling
        # if config["data"]["representation"] == 'melspectrogram':
        #     x_reconstruct = mel_to_stft(x_reconstruct, config)
        # x_reconstruct = torch.exp(x_reconstruct) - 0.1
        # latent_wav = griffin_lim(x_reconstruct, config)
        # sf.write(config["data"]["wav_results_path"] + "reconstruction_epoch_" + str(epoch) + ".wav", latent_wav.cpu(),
        #          config["preprocess"]["sample_rate"])
    else:
        print("Oh no, unknown representation :" +
              config["data"]["representation"] + ".\n")
        exit()


def sampling(config, model, log):
    # Create normal distribution representing latent space
    latent = distributions.normal.Normal(torch.tensor([0], dtype=torch.float),
                                         torch.tensor([1], dtype=torch.float))
    # Sampling random from latent space
    z = latent.sample(sample_shape=torch.Size([
        config["reconstruction"]["nb_samples"], config["model"]["latent_size"]
    ])).squeeze(2)
    z = z.to(config["train"]["device"])
    attributes = torch.rand(config["reconstruction"]["nb_samples"],
                            config["data"]["num_attributes"],
                            device=config["train"]["device"])
    # Pass through the decoder
    generated_spec = model.decode(z, attributes)
    # generate figure from sampling
    generated_spec = generated_spec.detach()
    generated_spec = generated_spec.squeeze(1)
    for i in range(config["reconstruction"]["nb_samples"]):
        if config["data"]["representation"] in [
                'spectrogram', 'melspectrogram'
        ]:
            plt.imshow(librosa.power_to_db(generated_spec[i].cpu()),
                       origin='lower',
                       aspect='auto')
            plt.title("Sampling from latent space")
            plt.savefig(config["data"]["figures_path"] + 'sampling' + str(i) +
                        '.png')
            log.write_figure('figures/sampling', plt.gcf())
            plt.close()
            # generate .wav from sampling
            # if config["data"]["representation"] == 'melspectrogram':
            #     generated_spec = mel_to_stft(generated_spec, config)
            # generated_spec = torch.exp(generated_spec) - 0.1
            # latent_wav = griffin_lim(generated_spec, config)
            # sf.write(file=config["data"]["wav_results_path"] + "sampling.wav",
            #          data=latent_wav.squeeze(0).cpu(),
            #          samplerate=config["preprocess"]["sample_rate"])
        # elif config["data"]["representation"] == 'wav':
        #     latent_wav = model.decode(z).cpu()
        #     sf.write(config["data"]["wav_results_path"] + "sampling.wav",
        #              librosa.effects.trim(latent_wav),
        #              config["preprocess"]["sample_rate"])
        else:
            print("Oh no, unknown representation " + config["model"]["model"] +
                  ".\n")
            exit()


def write_features_compare(config, x, y, idx, n_epoch, log):
    data = np.empty((x.shape[0], x.shape[1] + y.shape[1]))
    data[:, ::2] = x
    data[:, 1::2] = y
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bp = plt.boxplot(data,
                     labels=[
                         "roll", "roll_r", "flat", "flat_r", "band", "band_r",
                         "centro", "centro_r"
                     ])
    # Now fill the boxes with desired colors
    box_colors = ['pink', 'royalblue']
    num_boxes = (data.shape[1])
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()),
                 np.average(data[i]),
                 color='w',
                 marker='*',
                 markeredgecolor='k')
    plt.savefig(config["data"]["figures_path"] + 'attributes_reconstruction' +
                str(i) + "_epoch_" + str(n_epoch) + '.png')
    log.write_figure('figures/attributes_' + str(idx) + "_epoch_" + str(n_epoch), plt.gcf())
    plt.close()


def test_faders(config, model, loader, n_epoch, log):
    """
    Takes test set, pass it into encoder and try to decode with other attributes.
    Check if faders is able to slide generation attributes
    :param config:
    :param model:
    :return:
    """
    model = model.eval()
    if config["data"]["representation"] in ['spectrogram', 'melspectrogram']:
        cur_input, cur_attribute = next(iter(loader))
        cur_input, cur_attribute = cur_input.float().to(
            config["train"]["device"]), cur_attribute.float().to(
                config["train"]["device"])
        cur_feats = []
        for i in range(cur_input.shape[0]):
            cur_feats.append(
                np.mean(spectral_features_spectro(
                    cur_input[i].detach().cpu().numpy() + 3),
                        axis=1))
        cur_feats = np.stack(cur_feats, axis=0)
        # Reconstruction with slided attribute
        tests_list = []
        tests_list.append(cur_attribute.clone())
        # One by one
        for i in range(config["data"]["num_attributes"]):
            orig_clone = cur_attribute.clone()
            orig_clone[:, i].fill_(1)
            tests_list.append(orig_clone)
        # Half value of the attribute
        half_attr = torch.div(cur_attribute, 2)
        # print(half_attr)
        tests_list.append(half_attr)
        # One by one
        eye = torch.eye(7).unsqueeze(0).repeat(config["train"]["batch_size"],
                                               1, 1)
        for i in range(config["data"]["num_attributes"]):
            tests_list.append(eye[:, i, :])
        # print(tests_list)
        # Reconstructions
        recon_feats = []
        for i in range(len(tests_list)):
            # print(tests_list[i])
            tests_list[i] = tests_list[i].float().to(config["train"]["device"])
            _, x_reconstruct, _ = model(cur_input, tests_list[i])
            # Take first element of BS
            # x_reconstruct = x_reconstruct[0].detach().cpu().numpy()
            # Compute spectral feats from reconstruction
            feats = []
            for i in range(cur_input.shape[0]):
                feats.append(
                    np.mean(spectral_features_spectro(
                        x_reconstruct[i].detach().cpu().numpy() + 3),
                            axis=1))
            feats = np.stack(feats, axis=0)
            recon_feats.append(feats)
        # Store difference between reconstruct attributes and ground truth
        for i in range(len(tests_list)):
            x = cur_feats  # tests_list[i].cpu()
            y = recon_feats[i]
            write_features_compare(config, x, y, i, n_epoch, log)

        # x_reconstruct = x_reconstruct.squeeze(0)
        # ax[1].imshow(librosa.power_to_db(x_reconstruct.cpu()), origin='lower', aspect='auto')
        # # write row/col indices as axes' title for identification
        # plt.tight_layout(True)
        # plt.savefig(config["data"]["figures_path"] + 'epoch_' + str(epoch))
        # plt.close()


if __name__ == "__main__":
    attr = torch.tensor(
        [0.1709, -0.0961, -0.2020, -0.1514, -0.2104, -0.1312, -0.2395],
        device='cuda:0')
    # print(attr)
    tests_list = []
    # Half value of the attribute
    half_attr = torch.div(attr, 2)
    # print(half_attr)
    tests_list.append(half_attr)
    # One by one
    empty = torch.zeros(7)
    for i in range(7):
        empty[i] = torch.Tensor([1])
        tests_list.append(empty)
        empty = torch.zeros(7)
    # Eye matrix
    eye = torch.eye(7)
    tests_list.append(eye)
    print(tests_list)
