from torch.utils.data.sampler import Sampler
import pandas as pd
import numpy as np

class TwoViewsSampler(Sampler):
    def __init__(self, args, dataset, *argz, **kwargs):
        super(TwoViewsSampler, self).__init__(dataset, *argz, **kwargs)
        self.dataset = dataset
        self.args = args

        assert self.args.batch_size % self.args.num_views == 0

    def __iter__(self):

        for _ in range(len(self)):
            replace = False
            if len(self.dataset.annotations.index) < self.args.batch_size // self.args.num_views:
                replace = True

            inputs = self.dataset.annotations.sample(n = self.args.batch_size // self.args.num_views, replace = replace)
            df = inputs.iloc[np.repeat(np.arange(len(inputs.index)), self.args.num_views)]
            yield df.index

    def __len__(self):
        return int(np.ceil(len(self.dataset.annotations.index) / self.args.batch_size))
