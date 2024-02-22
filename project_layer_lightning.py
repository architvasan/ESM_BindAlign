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

def dataload_fn(data1, data2, train_prop=0.8, BATCH=64):
    input_tensor1 = torch.tensor(data1).to(torch.float32)
    input_tensor2 = torch.tensor(data2).to(torch.float32)
    input_tensor3 = torch.tensor(data2).roll(BATCH, 0)
    dataset = TensorDataset(input_tensor1, input_tensor2, input_tensor3)
    train_size = int(train_prop * len(dataset))
    test_size = int(len(dataset) - train_size)
    training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(training_data, batch_size=BATCH, shuffle=True, num_workers=39)
    test_dataloader = DataLoader(test_data, batch_size=BATCH, shuffle=False, num_workers=39)
    return train_dataloader, test_dataloader


def load_emb_data_together(esmloc, molloc, datasets, esmpattern, molpattern, train_prop=0.8):
    data1 = np.load(f'{esmloc}/{datasets[0]}_train{esmpattern}')
    data2 = np.load(f'{molloc}/{datasets[0]}_train{molpattern}')
    for d in datasets:
        for t in ['val', 'test']:
            if d!=datasets[0] and t!='train':
                data1 = np.concatenate((data1, np.load(f'{esmloc}/{d}_{t}{esmpattern}')))
                data2 =  np.concatenate((data2, np.load(f'{molloc}/{d}_{t}{molpattern}')))
    trainloader, testloader = dataload_fn(data1, data2)
    return trainloader, testloader

'''
Test training of projection layers
'''
def train_esm_mol():
    from argparse import ArgumentParser, SUPPRESS
    from pathlib import Path
    '''
    Set all arguments
    '''
    parser = ArgumentParser()#add_help=False)
    parser.add_argument(
        "-r", "--hsize1", type=int, required=True, help="hidden size for esm embeddings"
    )
    parser.add_argument(
        "-s", "--hsize2", type=int, required=True, help="hidden size for molformer embeddings"
    )
    parser.add_argument(
        "-e", "--esmloc", type=Path, required=True, help="esm embeddings location"
    )
    parser.add_argument(
        "-m", "--molloc", type=Path, required=True, help="molformer embeddings location"
    )
    parser.add_argument(
        "-p", "--proj", type=int, required=True, help="projection size"
    )
    parser.add_argument(
        "-o", "--out", type=Path, required=True, help="output directory"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="dataset: BindingDB, BIOSNAP, DAVIS"
    )
    parser.add_argument(
        "-t", "--trainp", type=float, required=False, help="training proportion", default=0.8
    )
    parser.add_argument(
        "-c", "--epoch", type=int, required=False, help="number of training epochs", default=50
    )
    parser.add_argument(
        "-B", "--batch", type=int, required=False, help="batch size", default=64
    )
    parser.add_argument(
        "-l", "--lr", type=float, required=False, help="learning rate", default=0.0001
    )
    parser.add_argument(
        "-E", "--early", type=int, required=False, help="early stop", default=100
    )
    args = parser.parse_args()
    '''
    Instantiate model, loss, optimizer
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    Init EsmMolProjectionHead
    '''
    ### projection for protein seqs
    projection_1 = nn.Sequential(
                nn.Linear(args.hsize1, args.proj*4),
                nn.LeakyReLU(0.2),
                nn.Linear(args.proj*4, args.proj*2),
                nn.LeakyReLU(0.2),
                nn.Linear(args.proj*2, args.proj)
                            )
    ### projection for positive smiles
    projection_2 = nn.Sequential(
                    nn.Linear(args.hsize2, args.proj*2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(args.proj*2, args.proj),
                                )
    ### projection for negative smiles
    projection_3 = nn.Sequential(
                    nn.Linear(args.hsize2, args.proj*2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(args.proj*2, args.proj),
                                )
    ### loss function is Triplet Margin loss
    lossfn = nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')

    projhead = EsmMolProjectionHead(projection_1, projection_2, projection_3, lossfn, args.lr)
   
    '''
    Load data
    '''
    esm_pattern ='_prot.dat-embeddings.npy'
    mol_pattern = '.smi-embeddings.npy'
    datasets = ['BindingDB', 'DAVIS']#'BIOSNAP', 'DAVIS']
    #'BindingDB', 
    trainloader, testloader = load_emb_data_together(args.esmloc, args.molloc, datasets, esm_pattern, mol_pattern, train_prop=args.trainp)

    '''
    Do training
    '''

    wandb_logger = WandbLogger(project="metric_learning")
    trainer = L.Trainer(limit_train_batches=args.batch,
                            max_epochs=args.epoch,
                            logger=wandb_logger,
                            log_every_n_steps=30)

    trainer.fit(projhead, trainloader, testloader)

    return 


if __name__ == "__main__":
    train_esm_mol()


