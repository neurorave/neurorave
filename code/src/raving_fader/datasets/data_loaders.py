import torch
from torch.utils.data import DataLoader, random_split
from raving_fader.config import settings
from raving_fader.datasets.nsynth_dataset import NSynthDataset


def rave_data_loaders(batch_size, dataset, num_workers=8):
    val = max(batch_size, (5 * len(dataset)) // 100)
    train = len(dataset) - val
    train, val = random_split(
        dataset,
        [train, val],
        generator=torch.Generator().manual_seed(42),
    )

    train = DataLoader(train,
                       batch_size,
                       True,
                       drop_last=True,
                       num_workers=num_workers,
                       pin_memory=False)
    val = DataLoader(val,
                     batch_size,
                     False,
                     num_workers=num_workers,
                     pin_memory=False)
    return train, val


def nsynth_data_loader(batch_size,
                       nsynth_json_path,
                       audio_dir=settings.DATA_DIR,
                       valid_ratio=0.2,
                       num_threads=0):
    # Load the dataset for the training/validation sets
    train_valid_set = NSynthDataset(
        audio_dir=audio_dir,
        nsynth_json_path=nsynth_json_path,
        transform=None  # transforms.ToTensor()
    )

    # Split it into training and validation sets
    # if valid_ratio = 0.2 : 80%/20% split for train/valid
    nb_train = round((1.0 - valid_ratio) * len(train_valid_set))
    nb_valid = round(valid_ratio * len(train_valid_set))
    train_set, valid_set = torch.utils.data.dataset.random_split(
        train_valid_set, [nb_train, nb_valid])

    # Define DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_threads)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_threads)
    return train_loader, valid_loader
