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
import json

def ParamsJson(json_file):
    with open(json_file) as f:
       params = json.load(f)
    return params


torch.autograd.set_detect_anomaly(True)

'''
Define classes to create projection layers
'''

class EsmMolProjectionHead(L.LightningModule):
    def __init__(self, projection_1, projection_2, loss_fn, base_lr):
        super().__init__()
        # We use a different projection head for both modes
        # since the embeddings fall into different subspaces.
        self.projection_1 = projection_1
        self.projection_2 = projection_2
        #self.projection_3 = projection_3
        self.loss_fn = loss_fn
        self.lr = base_lr 
    def training_step(self, batch, batch_idx):
        # Project the embeddings into a lower dimensional space
        # These have shape (batch_size, projection_size)
        x, y, y2 = batch
        anchor = self.projection_1(x)
        positive = self.projection_2(y)
        negative = self.projection_2(y2)
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
        negative = self.projection_2(y2)
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


def load_emb_data_together(esmloc, molloc, datasets, esmpattern, molpattern, train_prop=0.8, batch=64):
    data1 = np.load(f'{esmloc}/{datasets[0]}_train{esmpattern}')
    data2 = np.load(f'{molloc}/{datasets[0]}_train{molpattern}')
    for d in datasets:
        for t in ['val', 'test']:
            if d!=datasets[0] and t!='train':
                data1 = np.concatenate((data1, np.load(f'{esmloc}/{d}_{t}{esmpattern}')))
                data2 =  np.concatenate((data2, np.load(f'{molloc}/{d}_{t}{molpattern}')))
    print(data1.shape)
    print(data2.shape)
    trainloader, testloader = dataload_fn(data1, data2, train_prop, batch)
    return trainloader, testloader, torch.tensor(data1).to(torch.float32), torch.tensor(data2).to(torch.float32)

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
        "-c", "--config", type=Path, required=True, help="config.json"
    )

    args = parser.parse_args()
    '''
    Instantiate model, loss, optimizer
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = ParamsJson(args.config)
    '''
    Init EsmMolProjectionHead
    '''
    ### projection for protein seqs
    projection_1 = nn.Sequential(
                nn.Linear(config['hsize1'], config['proj']*4),
                nn.LeakyReLU(0.2),
                nn.Linear(config['proj']*4, config['proj']*2),
                nn.LeakyReLU(0.2),
                nn.Linear(config['proj']*2, config['proj'])
                            )
    ### projection for positive smiles
    projection_2 = nn.Sequential(
                    nn.Linear(config['hsize2'], config['proj']*2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(config['proj']*2, config['proj']),
                                )
    ### projection for negative smiles
    if False:
        projection_3 = nn.Sequential(
                        nn.Linear(config['hsize2'], config['proj']*2),
                        nn.LeakyReLU(0.2),
                        nn.Linear(config['proj']*2, config['proj']),
                                    )

    ### loss function is Triplet Margin loss
    lossfn = nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
  
    '''
    Load data
    '''
    esm_pattern ='_prot.dat-embeddings.npy'
    mol_pattern = '.smi-embeddings.npy'
    datasets = ['BindingDB', 'DAVIS']#'BIOSNAP', 'DAVIS']
    #'BindingDB', 
    trainloader, testloader, protdat, smidat = load_emb_data_together(config['esmloc'], config['molloc'], datasets, esm_pattern, mol_pattern, train_prop=config['trainp'], batch = config['batch'])
    if config['Train']:
        print("Training")
        '''
        Do training
        '''
        projhead = EsmMolProjectionHead(projection_1, projection_2, lossfn, config['lr'])
        wandb_logger = WandbLogger(project=config["wandbproj"])
        trainer = L.Trainer(
                                max_epochs=config['epoch'],
                                logger=wandb_logger,
                                log_every_n_steps=1)

        #limit_train_batches=config['batch'],
        trainer.fit(projhead, trainloader, testloader)
    else:
        print("no Training")

    if config['Infer']==True:
        '''
        Do inference
        '''
        projhead = EsmMolProjectionHead.load_from_checkpoint(config['Check'], projection_1=projection_1, projection_2=projection_2, loss_fn=lossfn, base_lr=config['lr'])
        print(projhead.projection_1)
        projhead.to('cuda')
        projhead.projection_1.eval()
        prot_proj = projhead.projection_1(torch.tensor(protdat).to('cuda'))
        mol_proj = projhead.projection_2(torch.tensor(smidat).to('cuda'))
        print(prot_proj.size())
        print(mol_proj.size())
        out_patt = config['InfOut']
        torch.save(prot_proj, f'prot_{out_patt}.pt')
        torch.save(mol_proj, 'mol_{out_patt}.pt')
        #print(projhead.projection_1(torch.tensor(protdat).to('cuda')))

    return 


if __name__ == "__main__":
    train_esm_mol()


