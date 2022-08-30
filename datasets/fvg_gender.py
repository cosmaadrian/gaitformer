import os
import torch
import pandas as pd
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.utils import class_weight

from datasets.transforms import RandomCrop, ToTensor, SqueezeAndFlip, Permutation, RandomPace, DeterministicCrop
from datasets.transforms import FlipSequence, DropOutFrames, DropOutJoints, JointNoise, PointNoise

from datasets.samplers import ContrastiveSampler, TwoViewsSampler

dir_path = os.path.dirname(os.path.realpath(__file__))
PATH = f'{dir_path}/annotations/fvg_gender_fixed.csv'

def get_fvg(kind, train_split = 136):
    if train_split is None:
        train_split = 180

    df = pd.read_csv(PATH)

    df['track_id'] = df['track_id'].astype('category').cat.codes
    df['run_id'] = df['run_id'].astype('category').cat.codes
    df['session_id'] = df['session_id'].astype('category').cat.codes

    train_ids = np.arange(0, train_split)
    test_ids = np.arange(train_split, 225)

    train_df = df[df['track_id'].isin(train_ids)]
    val_df = df[df['track_id'].isin(test_ids)]

    max_age = train_df['age'].max()
    min_age = train_df['age'].min()

    train_df['age'] = (train_df['age'] - min_age) / (max_age - min_age)
    val_df['age'] = (val_df['age'] - min_age) / (max_age - min_age)

    if kind == 'train':
        return train_df
    elif kind in ['test', 'val']:
        return val_df

    return train_df, val_df

class FVGGenderDataset(Dataset):
    DATASET_PATH = PATH

    def __init__(self, args, annotations = None, kind = 'train', image_transforms = None):
        self.args = args

        if annotations is None:
            self.annotations = get_fvg(kind = kind)
        else:
            self.annotations = annotations

        self.image_transforms = image_transforms

        if kind == 'train' and 'fraction' in self.args and self.args.fraction is not None:
            self.annotations = self.annotations.sample(frac = args.fraction)
            self.annotations = self.annotations.reset_index(drop = True)

        if self.image_transforms is not None:
            return

        if kind == 'train':
            self.image_transforms = transforms.Compose([
                RandomCrop(period_length = max(args.paces) * args.period_length),
                RandomPace(paces = args.paces, period_length = args.period_length),
                SqueezeAndFlip(amount = 0.1, flip_prob = 0.5),
                FlipSequence(probability = 0.5),
                JointNoise(std = 0.0005),
                PointNoise(std = 0.0005),
                Permutation(do_apply = args.permutation, permutation_size = args.permutation_size, prob = 0.25),
                ToTensor()
            ])
        else:
            self.image_transforms = composed = transforms.Compose([
                DeterministicCrop(period_length = args.period_length),
                ToTensor()
            ])

    @classmethod
    def train_dataloader(cls, args):
        composed = transforms.Compose([
                RandomCrop(period_length = max(args.paces) * args.period_length),
                RandomPace(paces = args.paces, period_length = args.period_length),
                SqueezeAndFlip(amount = 0.15, flip_prob = 0.5),
                FlipSequence(probability = 0.5),
                JointNoise(std = 0.0005),
                PointNoise(std = 0.0005),
                Permutation(do_apply = args.permutation, permutation_size = args.permutation_size, prob = 0.50),
                ToTensor()
            ])
        dataset = cls(args = args, image_transforms = composed)

        sampler = ContrastiveSampler(args, dataset)

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

    def __len__(self):
        return len(self.annotations.index)

    def on_epoch_end(self):
        self.annotations = self.annotations.sample(frac = 1.0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.annotations.iloc[idx]
        image = np.load(os.path.join(self.args.environment['eval-data-dir'], sample['file_name']))

        instantiated_sample = {
            'image': image,
            'track_id': sample['track_id'].reshape((1, )),
            'session_id': sample['session_id'].reshape((1, )),
            'run_id': sample['run_id'].reshape((1, )),
            'age': sample['age'].reshape((1, )),
            'gender': sample['gender'].reshape((1, )),
        }

        if self.image_transforms:
            instantiated_sample = self.image_transforms(instantiated_sample)

        inputs = np.expand_dims(instantiated_sample['image'], -1)
        instantiated_sample['image'] = inputs.astype(np.float32)

        return instantiated_sample
