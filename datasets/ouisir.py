import os
import torch
import pandas as pd
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from datasets.transforms import RandomCrop, ToTensor, SqueezeAndFlip, Permutation, RandomPace, DeterministicCrop
from datasets.transforms import FlipSequence, DropOutFrames, DropOutJoints, JointNoise, PointNoise

from datasets.samplers import ContrastiveSampler, TwoViewsSampler

isir2casia = np.array([
    [0, 0],
    [1, 4],
    [2, 6],
    [3, 10],
    [4, 13],
    [5, 7],
    [6, 11],
    [7, 15],
    [8, 12],
    [9, 16],
    [10, 2],
    [11, 14],
    [12, 1],
    [13, 3],
    [14, 17],
    [15, 5],
    [16, 7],
    [17, 9],
])

PATH = 'annotations.csv'

def get_isir(args, kind):
    df = pd.read_csv(os.path.join(args.environment['ouisir-dir'], PATH))
    df['track_id'] = df['track_id'].astype('category').cat.codes
    df['camera_id'] = df['camera_id'].astype('category').cat.codes
    df['seq_id'] = df['seq_id'].astype('category').cat.codes

    train_df = df[df['track_id'].apply(lambda x: int(x) % 2 == 1)].reset_index(drop = True)
    val_df = df[df['track_id'].apply(lambda x: int(x) % 2 == 0)].reset_index(drop = True)

    train_df = df
    # train_df['track_id'] = np.arange(len(train_df.index))


    if kind == 'train':
        return train_df
    elif kind in ['test', 'val']:
        return val_df

    return train_df, val_df

class OUISIRDataset(Dataset):
    DATASET_PATH = PATH

    def __init__(self, args, annotations = None, kind = 'train', image_transforms = None):
        self.args = args

        if annotations is None:
            self.annotations = get_isir(args = args, kind = kind)
        else:
            self.annotations = annotations

        if kind == 'train' and 'fraction' in self.args and self.args.fraction is not None:
            self.annotations = self.annotations.groupby('track_id').apply(lambda x: x.sample(frac = args.fraction))
            self.annotations = self.annotations.reset_index(drop = True)

        if kind == 'train' and 'runs' in self.args and self.args.runs is not None:
            self.annotations = self.annotations.groupby(['camera_id', 'track_id']).apply(lambda x: x.sample(n = self.args.runs))
            self.annotations = self.annotations.reset_index(drop = True)

        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.annotations.index)

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
            ToTensor()
        ])

        dataset = cls(args = args, image_transforms = composed)

        if 'num_batch_ids' in args and 'num_views' in args:
            raise Exception('Choose either num_batch_ids or num_views!!!!!!!1')

        # sampler = ContrastiveSampler(args, dataset)
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
            DeterministicCrop(period_length = args.period_length),
            ToTensor()
        ])

        return DataLoader(
            cls(args = args, image_transforms = composed, kind = 'val'),
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = 15,
            pin_memory = True,
        )

    def on_epoch_end(self):
        self.annotations = self.annotations.sample(frac = 1.0).reset_index(drop = True)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.annotations.iloc[idx]
        image = np.load(os.path.join(self.args.environment['ouisir-dir'], 'poses', sample['file_name']))

        image[:, :, 1] = - image[:, :, 1]
        image[:, isir2casia[:, 1]] = image[:, isir2casia[:, 0]]

        instantiated_sample = {
            'image': image,
            'track_id': sample['track_id'].reshape((1, )),
            'camera_id': sample['camera_id'].reshape((1, )),
            'seq_id': sample['seq_id'].reshape((1, )),
        }

        if self.image_transforms:
            instantiated_sample = self.image_transforms(instantiated_sample)

        if type(instantiated_sample) is list:
            for l in instantiated_sample:
                l['image'] = np.expand_dims(l['image'], -1).astype(np.float32)

        else:
            inputs = np.expand_dims(instantiated_sample['image'], -1)
            instantiated_sample['image'] = inputs.astype(np.float32)

        return instantiated_sample
