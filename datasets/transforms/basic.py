import numpy as np
import torch
import cv2
import random


class RandomCrop(object):
    """Crop randomly the image in a sample."""

    def __init__(self, period_length):
        self.period_length = period_length

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]

        if type(self.period_length) == list:
            period = random.randrange(*self.period_length)
        else:
            period = int(self.period_length)

        try:
            if h - period > 0:
                top = np.random.randint(0, h - period)
                image = image[top: top + period, :]
            elif h - period < 0:
                image = np.tile(image, reps = (int(period / h + 2), 1, 1))
                image = image[:period, :]
        except:
            print("RandomCrop:", image.shape)
            exit()

        if image.shape[0] < period:
            print(h, period, h - period, image.shape)
            exit()

        sample.update({'image': image})
        return sample

class DeterministicCrop(object):
    def __init__(self, period_length):
        self.period_length = period_length

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        period = int(self.period_length)

        middle = h // 2

        original = image.copy()
        try:
            if middle - period // 2 >= 0:
                image = image[middle - period // 2: middle + period // 2]
                # print("larger: ", original.shape, image.shape)
            elif h - period < 0:
                if image.shape[0] % 2 == 1:
                    extra_left = image[:period // 2 - middle + 1].copy()
                else:
                    extra_left = image[:period // 2 - middle].copy()

                extra_right = image[-(period // 2 - middle):].copy()
                image = np.vstack((extra_left, image, extra_right))
                image = image[:period]
                # print("smaller: ", original.shape, image.shape)
        except:
            print(original.shape, image.shape)
            exit()

        # if original.shape[0] == 1:

        if image.shape[0] < period:
            image = np.tile(image, reps = (int(period / h + 2), 1, 1))
            image = image[:period, :]

        if image.shape[0] < period:
            print(h, period, original.shape, image.shape)
            exit()

        sample.update({'image': image})
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, from_sample = True):
        self.from_sample = from_sample

    def __call__(self, sample):
        if self.from_sample:
            tensor_sample = dict()

            for key, value in sample.items():
                if 'track_id' in key:
                    tensor_sample[key] = torch.from_numpy(value).long()
                else:
                    if 'image' in key:
                        tensor_sample[key] = torch.from_numpy(value).float().permute((2, 0, 1))
                    else:
                        tensor_sample[key] = torch.from_numpy(value).float()


            return tensor_sample

        return torch.from_numpy(sample).float()