import numpy as np
import torch
import cv2
from scipy.signal import resample

class DropOutFrames(object):
    def __init__(self, probability=0.1):
        self.probability = probability

    def __call__(self, sample):
        image = sample['image']

        mask = np.random.random(size = image.shape[0]) < self.probability
        mask_indices = np.argwhere(mask).ravel()
        image = np.take(image, mask_indices, axis = 0)

        sample.update({'image': image})
        return sample

class DropOutJoints(object):
    def __init__(self, prob=1, dropout_rate_range=0.1):
        self.dropout_rate_range = dropout_rate_range
        self.prob = prob

    def __call__(self, sample):
        if np.random.binomial(1, self.prob, 1) != 1:
            return sample

        data = sample['image']

        dropout_rate = np.random.uniform(0, self.dropout_rate_range, size = data.shape)
        zero_indices = 1 - np.random.binomial(1, dropout_rate, size = data.shape)
        data = data * zero_indices

        sample.update({'image': data})
        return sample

class RandomPace(object):
    def __init__(self, paces, period_length = 72):
        self.paces = paces
        self.unique_paces = np.unique(self.paces)
        self.period_length = period_length

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]

        pace_idx = np.random.choice(np.arange(len(self.paces)))
        pace = self.paces[pace_idx]

        if pace < 1:
            image = np.repeat(image, repeats = int(1 / pace), axis = 0)

            if h - self.period_length <= 0:
                # print(h, self.period_length)
                pass

            top = np.random.randint(0, h + 1 - self.period_length)
            image = image[top: top + self.period_length, :]

        if pace >= 1:
            pace = int(pace)
            if h - self.period_length <= 0:
                # print(h, self.period_length)
                pass
            top = np.random.randint(0, h + 1 - pace * self.period_length)
            image = image[top: top + pace * self.period_length, :]
            image = image[::pace]

        sample.update({'image': image})
        # sample['pace'] = np.array(pace_idx)
        sample['pace'] = np.argwhere(self.unique_paces == pace).ravel()

        return sample

class FlipSequence(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        if np.random.random() <= self.probability:
            return sample

        image = sample['image']

        image = np.flip(image, axis=0).copy()
        sample.update({'image': image})
        return sample


class SqueezeAndFlip(object):
    def __init__(self, amount = None, flip_prob = 0.25):
        """
            amount should be either None to simply randomly flip the image
            or some value between 0 and 0.5
        """
        self.amount = 0 if amount is None else amount
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image = sample['image']

        squeeze_amount = 1 - self.amount * np.random.random()

        chance = np.random.random()
        if chance < self.flip_prob:
            squeeze_amount *= -1

        image[:, 0] = squeeze_amount * image[:, 0]
        sample.update({'image': image})
        return sample

class PointNoise(object):
    def __init__(self, std=0.15):
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        noise = np.random.normal(0, self.std, image.shape).astype(np.float32)
        image = image + noise
        sample.update({'image': image})
        return sample

class JointNoise(object):
    def __init__(self, std=0.5):
        self.std = std

    def __call__(self, sample):
        data = sample['image']
        noise = np.hstack((
            np.random.normal(0, self.std, (data.shape[1], 2)),
            np.zeros((data.shape[1], 1))
        )).astype(np.float32)

        data =  data + np.repeat(noise[np.newaxis, ...], data.shape[0], axis=0)
        sample.update({'image': data})
        return sample

class Permutation(object):
    def __init__(self, do_apply = True, permutation_size = 12, prob = 0.2):
        self.permutation_size = permutation_size
        self.prob = prob
        self.apply = do_apply

    def __call__(self, sample):
        if not self.apply:
            return sample

        chance = np.random.random()
        if chance > self.prob:
            sample['permutation'] = np.array([0])
            return sample

        image = sample['image']
        h, w = image.shape[:2]

        top = np.random.randint(self.permutation_size + 1, h - self.permutation_size)
        bottom = np.random.randint(0, top - self.permutation_size)

        permutation1 = image[top: top + self.permutation_size].copy()
        permutation2 = image[bottom: bottom + self.permutation_size].copy()

        image[bottom: bottom + self.permutation_size] = permutation1
        image[top: top + self.permutation_size] = permutation2

        sample['permutation'] = np.array([1])
        sample.update({'image': image})

        return sample
