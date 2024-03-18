import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
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
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses
import torch.nn.functional as F
import os
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L
from pytorch_lightning.loggers import WandbLogger

torch.autograd.set_detect_anomaly(True)

'''
Define classes to create projection layers
'''

class EsmMolProjectionHead(L.LightningModule):
    def __init__(self, projection_1, projection_2, projection_3, loss_fn, base_lr):
        super().__init__()
        # We use a different projection head for both modes
        # since the embeddings fall into different subspaces.
        self.projection_1 = projection_1
        self.projection_2 = projection_2
        self.projection_3 = projection_3
        self.loss_fn = loss_fn
        self.lr = base_lr 
    def training_step(self, batch, batch_idx):
        # Project the embeddings into a lower dimensional space
        # These have shape (batch_size, projection_size)
        x, y, y2 = batch
        anchor = self.projection_1(x)
        positive = self.projection_2(y)
        negative = self.projection_3(y2)
        loss = self.loss_fn(anchor, positive, negative)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # Compute the metric loss following pytorch-metric-learning
        return loss 
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y, y2 = batch
        anchor = self.projection_1(x)
        positive = self.projection_2(y)
        negative = self.projection_3(y2)
        val_loss = self.loss_fn(anchor, positive, negative)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return val_loss



# load checkpoint
checkpoint = "./metric_learning/secjwdwn/checkpoints/epoch=29-step=1290.ckpt"
autoencoder = EsmMolProjectionHead.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# embed 4 fake images!
fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
