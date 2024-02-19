from pytorch_metric_learning import losses
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from rdkit import Chem
import rdkit
from rdkit.Chem import Descriptors
import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
import torch
from transformers import EsmForMaskedLM
from transformers import EsmTokenizer
import protein_search
from protein_search.distributed_inference import *
import sys
from tqdm import tqdm
from datasets import Dataset
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from pytorch_metric_learning.losses import SelfSupervisedLoss,TripletMarginLoss
loss_fn = SelfSupervisedLoss(TripletMarginLoss(
                                    distance = CosineSimilarity(),
                                    reducer = ThresholdReducer(high=0.3),
                                    embedding_regularizer = LpRegularizer()
                                ))



class MeanPooler(nn.Module):
    """Reduces the sequence embeddings (batch_size, seq_length, hidden_size)
    to a single embedding (batch_size, hidden_size) by averaging."""

    def __init__(self, config) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The average over sequence length gives even weighting to each sequence position
        return x.mean(dim=1)


class FirstPooler(EsmPooler):
    """Reduces the sequence embeddings (batch_size, seq_length, hidden_size)
    to a single embedding (batch_size, hidden_size) by taking the first hidden state."""


POOLER_DISPATCH = {"mean": MeanPooler, "first": FirstPooler}


class EsmContrastiveProjectionHead(nn.Module):
    def __init__(self, config: ContrastiveEsmConfig) -> None:
        super().__init__()
   
        # We use a different projection head for both modes
        # since, by default, the embeddings fall into different subspaces.
        self.projection_1 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
        )
        self.projection_2 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
        )

        loss_fn = SelfSupervisedLoss(TripletMarginLoss(
                                    distance = CosineSimilarity(),
                                    reducer = ThresholdReducer(high=0.3),
                                    embedding_regularizer = LpRegularizer()
                                    ))

        self.pooler = POOLER_DISPATCH[config.contrastive_pooler](config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes that the first modality embeddings are the first half of the tensor
        # and the second modality embeddings are the second half.

        # Pool the sequence embeddings to get a single embedding per sequence
        x = self.pooler(x)  # (batch_size, hidden_size)

        # Collect the modality  embeddings separately
        # These have shape (batch_size // 2, hidden_size)
        half_batch_size = x.shape[0] // 2
        modality_1_embed = x[:half_batch_size]
        modality_2_embed = x[half_batch_size:]

        # Project the embeddings into a lower dimensional space
        # These have shape (batch_size // 2, projection_size)
        z_modality_1 = self.projection_1(modality_1_embed)
        z_modality_2 = self.projection_2(modality_2_embed)

        # Concatenate the modality embeddings
        # This has shape (batch_size, projection_size)
        z = torch.cat([z_modality_1, z_modality_2], dim=0)

        # Compute the contrastive loss following SimCLR
        return self.loss_fn(z)








class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float) -> None:
        """Contrastive loss for SimCLR.

        Reference: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html#SimCLR

        Parameters
        ----------
        temperature: float
            Determines how peaked the distribution. Since many similarity
            metrics are bounded, the temperature parameter allows us to
            balance the influence of many dissimilar image patches versus
            one similar patch.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # NOTE: z.shape == (batch_size, hidden_size)
        # TODO: Can we cache the pos_mask calculation with lru_cache?
        batch_size = z.shape[0]
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        cos_sim.masked_fill_(self_mask, -65504)
        # Find positive example -> batch_size // 2 away from the original example
        pos_mask = self_mask.roll(shifts=batch_size // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        return nll

