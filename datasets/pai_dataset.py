import os
import pprint
import torch
import pandas as pd
import numpy as np
import time
import cv2
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from datasets.transforms import RandomCrop, ToTensor, SqueezeAndFlip, Permutation, RandomPace
from datasets.transforms import DeterministicCrop, FlipSequence, DropOutFrames, DropOutJoints, JointNoise, PointNoise
from datasets.samplers import ContrastiveSampler, TwoViewsSampler

from .cache import DatasetCache
from multiprocessing import Manager

from .uwg_features import get_features

PATH = 'validation/annotations.csv'

class PAIDataset(Dataset):
    DATASET_PATH = PATH

    def __init__(self, args, image_transforms = None, cache = None):
        self.args = args
        self.image_transforms = image_transforms

        self.annotations = pd.read_csv(os.path.join(self.args.environment['data-dir'], PATH))
        self.annotations.track_id = self.annotations.track_id.astype('category').cat.codes
        self.cache = cache

        print(self.__class__.__name__, len(self.annotations.track_id.unique()))

    @property
    def num_classes(self):
        return len(self.annotations.track_id.unique())

    def __len__(self):
        return len(self.annotations.index)

    def on_epoch_end(self):
        self.annotations = self.annotations.sample(frac = 1.).reset_index(drop = True)

    @classmethod
    def val_dataloader(cls, args):
        composed = transforms.Compose([
            DeterministicCrop(period_length = args.period_length),
            ToTensor()
        ])

        return DataLoader(
            cls(args = args, image_transforms = composed),
            batch_size = args.batch_size,
            shuffle = False
        )

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]

        image = np.load(os.path.join(self.args.environment['data-dir'], sample['file_name']))

        assert not np.any(np.isnan(image))
        image[:, :, 1] = - image[:, :, 1]

        instantiated_sample = {
            'image': image,
            'track_id': np.array(sample['track_id']),
        }

        for f in get_features(self.args):
            attribute_feature = sample[f]
            if self.args.round_attributes:
                attribute_feature = np.round(attribute_feature)

            instantiated_sample[f] = np.array(attribute_feature)

        if self.image_transforms:
            instantiated_sample = self.image_transforms(instantiated_sample)

        inputs = np.expand_dims(instantiated_sample['image'], -1)
        instantiated_sample['image'] = inputs.astype(np.float32)

        return instantiated_sample
