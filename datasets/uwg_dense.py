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
from datasets.transforms import FlipSequence, DropOutFrames, DropOutJoints, JointNoise, PointNoise
from datasets.samplers import ContrastiveSampler, TwoViewsSampler


from .uwg_features import get_features

PATH = 'data-2.02/annotations.csv'


class UWGDense(Dataset):
    DATASET_PATH = PATH

    def __init__(self, args, image_transforms = None):
        self.args = args
        self.image_transforms = image_transforms

        self.annotations = pd.read_csv(os.path.join(args.environment['data-dir'], PATH))

        if 'fraction' in self.args and self.args.fraction != None:
            ids = np.arange(len(self.annotations['track_id'].unique()))
            ids = np.random.choice(ids, size = int(self.args.fraction * len(ids)), replace = False)
            self.annotations = self.annotations[self.annotations['track_id'].isin(ids)]

        self.annotations.track_id = self.annotations.track_id.astype('category').cat.codes

        print(len(self.annotations.track_id.unique()))

    @property
    def num_classes(self):
        return len(self.annotations.track_id.unique())

    def __len__(self):
        return len(self.annotations.index)

    def on_epoch_end(self):
        self.annotations = self.annotations.sample(frac = 1.).reset_index(drop = True)

    @classmethod
    def train_dataloader(cls, args):
        composed = transforms.Compose([
            RandomCrop(period_length = max(args.paces) * args.period_length),
            RandomPace(paces = args.paces, period_length = args.period_length),
            SqueezeAndFlip(amount = 0.1, flip_prob = 0.5),
            FlipSequence(probability = 0.5),
            JointNoise(std = 0.0005),
            PointNoise(std = 0.0005),
            Permutation(do_apply = args.permutation, permutation_size = args.permutation_size, prob = 0.50),
            # DropOutJoints(prob = 0.5, dropout_rate_range = 0.05),
            ToTensor()
        ])

        dataset = cls(args = args, image_transforms = composed)
        sampler = TwoViewsSampler(args, dataset)

        return DataLoader(
            dataset,
            batch_sampler = sampler,
            num_workers = 15,
            pin_memory = True,
        )

    @classmethod
    def val_dataloader(cls, args):
        composed = transforms.Compose([
            RandomCrop(period_length = args.period_length),
            ToTensor()
        ])

        return DataLoader(
            cls(args = args, image_transforms = composed),
            batch_size = args.batch_size,
            shuffle = False
        )

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]

        if self.args.use_cache and self.cache.is_cached(sample['file_name']):
            image = self.cache.get(sample['file_name'])
        else:
            image = np.load(os.path.join(self.args.environment['data-dir'], sample['file_name']))

            if self.args.use_cache:
                self.cache.put(key = sample['file_name'], value = image)

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
