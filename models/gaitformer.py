import torch
import math

from datasets.uwg_features import get_features

class GaitFormer(torch.nn.Module):
    def __init__(self, args):
        super(GaitFormer, self).__init__()
        self.args = args

        self.positional_embedding = torch.nn.Parameter(
            torch.randn(1, self.args.period_length, self.args.model_args['embedding_size'])
        )

        self.skeleton_encoding = torch.nn.Linear(
            in_features = 54,
            out_features = self.args.model_args['embedding_size'],
        )

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                self.args.model_args['embedding_size'],
                self.args.model_args['n_heads'],
                activation = 'gelu',
                batch_first = True,
                norm_first = True,
                dropout = self.args.model_args['dropout_prob'],
                dim_feedforward = 2 * self.args.model_args['embedding_size']
            ),
            self.args.model_args['n_layers']
        )

        self.out_embedding = torch.nn.Linear(
            self.args.model_args['embedding_size'],
            self.args.model_args['embedding_size'],
            bias = False
        )

        self.projection = torch.nn.Linear(
            self.args.model_args['embedding_size'],
            self.args.model_args['projection_size']
        )

        self.out_appearance = torch.nn.Linear(
            in_features = self.args.model_args['embedding_size'],
            out_features = len(get_features(self.args)),
            bias = True
        )

    def forward(self, image, attribute_probs = None):
        image = image.permute((0, 2, 3, 1, 4)).reshape((image.shape[0], self.args.period_length, -1))
        skel_proj = self.skeleton_encoding(image)
        proj_emb = self.positional_embedding + skel_proj

        post_transformer = self.encoder(proj_emb)

        feats = post_transformer[:, 0]

        embedding = self.out_embedding(feats)
        emb = torch.nn.functional.normalize(
            embedding
        )

        appearance = torch.sigmoid(self.out_appearance(feats))
        projection = torch.nn.functional.normalize(self.projection(torch.nn.functional.gelu(embedding)))

        return {
            'appearance': appearance,
            'representation': emb,
            'projection': projection
        }
