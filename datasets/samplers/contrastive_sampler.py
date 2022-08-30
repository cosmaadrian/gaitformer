from torch.utils.data.sampler import Sampler
import pandas as pd
import numpy as np

class ContrastiveSampler(Sampler):
    def __init__(self, args, dataset, *argz, **kwargs):
        super(ContrastiveSampler, self).__init__(dataset, *argz, **kwargs)
        self.dataset = dataset
        self.args = args
        self.drop_last = True

    def __iter__(self):
        self.dataset.on_epoch_end()

        # This is just a normal sampler, not contrastive i guess?????
        for batch_idx in range(len(self)):
            first_part = self.dataset.annotations.iloc[self.args.batch_size * batch_idx: self.args.batch_size * (batch_idx + 1)]
            yield first_part.index

    def __len__(self):
        return int(np.ceil(len(self.dataset.annotations.index) / self.args.batch_size))
