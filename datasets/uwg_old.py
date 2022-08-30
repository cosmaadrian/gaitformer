import os
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

from multiprocessing import Manager
from datasets.uwg_dense import UWGDense

PATH = 'uwg-old/uwg-big-plain-2-andy/poses/annotations_clean.csv'

class UWGOld(UWGDense):
    DATASET_PATH = PATH

    def __init__(self, args, image_transforms = None, cache = None):
        self.args = args
        self.image_transforms = image_transforms

        self.annotations = pd.read_csv(os.path.join(self.args.environment['data-dir'], PATH))

        if 'fraction' in self.args and self.args.fraction is not None:
            ids = np.arange(len(self.annotations['track_id'].unique()))
            ids = np.random.choice(ids, size = int(self.args.fraction * len(ids)), replace = False)
            self.annotations = self.annotations[self.annotations['track_id'].isin(ids)]

        self.annotations.track_id = self.annotations.track_id.astype('category').cat.codes

        print(len(self.annotations.track_id.unique()))

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]

        file_path = sample['file_name'].replace('/media/ssd/gait-recognition/data/', 'uwg-old/')
        image = np.load(os.path.join(self.args.environment['data-dir'], file_path))

        assert not np.any(np.isnan(image))
        image[:, :, 1] = - image[:, :, 1]

        instantiated_sample = {
            'image': image,
            'track_id': np.array(sample['track_id']),
        }

        if self.image_transforms:
            instantiated_sample = self.image_transforms(instantiated_sample)

        inputs = np.expand_dims(instantiated_sample['image'], -1)
        instantiated_sample['image'] = inputs.astype(np.float32)

        return instantiated_sample
