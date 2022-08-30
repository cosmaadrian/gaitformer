import torch
import torch.nn as nn

from datasets.uwg_features import get_features


class AttributeEmbedding(nn.Module):
    def __init__(self, args):
        super(AttributeEmbedding, self).__init__()
        self.args = args

        self.embedding = torch.nn.Parameter(
            torch.randn(1, 2 * len(get_features(self.args)), self.args.model_args['attribute_embedding_size'])
        )

    def forward(self, appearance_probs):
        probs_tiled = torch.tile(appearance_probs, (1, 1, 2))
        probs_tiled[:, 1] = 1 - probs_tiled[:, 1]
        probs_tiled = probs_tiled.reshape((appearance_probs.shape[0], -1, 1))

        attributed_embeddings = self.embedding * probs_tiled
        attributed_embeddings = attributed_embeddings.reshape((-1, len(get_features(self.args)), 2, self.args.model_args['attribute_embedding_size']))
        output = attributed_embeddings.mean(2)

        return output