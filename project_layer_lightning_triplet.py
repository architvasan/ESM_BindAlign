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
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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

def dataload_realtrip(data1, data2, data3, train_prop=0.8, BATCH=64):
    input_tensor1 = torch.tensor(data1).to(torch.float32)
    input_tensor2 = torch.tensor(data2).to(torch.float32)
    input_tensor3 = torch.tensor(data3).to(torch.float32)
    dataset = TensorDataset(input_tensor1, input_tensor2, input_tensor3)
    train_size = int(train_prop * len(dataset))
    test_size = int(len(dataset) - train_size)
    training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(training_data, batch_size=BATCH, shuffle=True, num_workers=39)
    test_dataloader = DataLoader(test_data, batch_size=BATCH, shuffle=False, num_workers=39)
    print(training_data)
    print(test_data)
    return train_dataloader, test_dataloader


def load_real_triplet(real_triplet_file,
                    raw_pwd,
                    t_emb_pwd,
                    p_emb_pwd,
                    n_emb_pwd,
                    train_prop=0.8, 
                    batch=64):
    '''
    -1. load embedding data for protein, pos, negative into embedding dict
    0. create empty data_t, data_p, nd data_n
    1. for loop over triplet_df
    2. get target sequence
        a. match to target in raw target dataframe, get dataset + train/test/val
        b. go to corresponding raw data location ---> extract embedding_index
        b. determine every time target is in triplet_df --> trip_t_inds 
            1. only do following steps if data_t[indices[0]]==[0,0,...,0]
        c. pull embedding from p_emb_pwd/dataset_train_test_val... using embedding_index
        d. for each t_ind in trip_t_inds: data_t[t_ind]=emb_targ[embedding_index] 
    3. repeat above for positive smiles
    4. repeat above for negative smiles
    return (data_t, data_p, data_n)
    '''

    triplet_df = pd.read_csv(real_triplet_file)#[:10000]
    #triplet_df = triplet_df.sample(10000)
    data_t = np.zeros((len(triplet_df), 2560))
    data_p = np.zeros((len(triplet_df), 768))
    data_n = np.zeros((len(triplet_df), 768))

    '''
    Step -1, 0: load embeddings + create raw dataframe
    '''
    t_emb_patt = '_prot.dat-embeddings.npy'
    p_emb_patt = '.npy'
    n_emb_patt = '_neg.npy'
    t_emb = {}
    p_emb = {}
    n_emb = {}

    t_raw = {}
    p_raw = {}
    n_raw = {}

    for d in ['BindingDB', 'BIOSNAP', 'DAVIS']:
        t_emb[d] = {}
        p_emb[d] = {}
        n_emb[d] = {}

        t_raw[d] = {}
        p_raw[d] = {}
        n_raw[d] = {}
        for t in ['train', 'val', 'test']:
            t_raw[d][t] = pd.read_csv(f'{raw_pwd}/{d}_{t}_prot.dat')['Target Sequence'].tolist() 
            p_raw[d][t] = pd.read_csv(f'{raw_pwd}/{d}_{t}.smi')['SMILES'].tolist()
            n_raw[d][t] = pd.read_csv(f'{raw_pwd}/{d}_{t}_neg.smi')['SMILES'].tolist()
            t_emb[d][t] = np.load(f'{t_emb_pwd}/{d}_{t}{t_emb_patt}')
            p_emb[d][t] =  np.load(f'{p_emb_pwd}/{d}_{t}{p_emb_patt}')
            n_emb[d][t] =  np.load(f'{n_emb_pwd}/{d}_{t}{n_emb_patt}')
            

    triplet_df = triplet_df.reset_index()

    from tqdm import tqdm
    #for index, row in tqdm(triplet_df.iterrows()):
    for index in tqdm(range(len(triplet_df))):
        row = triplet_df.iloc[index]
        tar_it = row['target']
        pos_it = row['positive']
        neg_it = row['negative']

        pos_ds = row['positive_dataset']
        pos_dt = row['positive_datatype']
        neg_ds = row['negative_dataset']
        neg_dt = row['negative_datatype']
        
        #tar_ind = np.where(np.array(t_raw[pos_ds][pos_dt])==tar_it)[0]
        #pos_ind = np.where(np.array(p_raw[pos_ds][pos_dt])==pos_it)[0]
        #neg_ind = np.where(np.array(n_raw[neg_ds][neg_dt])==neg_it)[0]

        tar_ind = t_raw[pos_ds][pos_dt].index(tar_it)
        pos_ind = p_raw[pos_ds][pos_dt].index(pos_it)
        neg_ind = n_raw[neg_ds][neg_dt].index(neg_it)
        #print(tar_ind)
        #if len(tar_ind > 1):
        #    tar_ind = tar_ind[0]
        #                         
        #if len(pos_ind > 1):
        #    pos_ind = pos_ind[0]
        #
        #if len(neg_ind > 1):
        #    neg_ind = neg_ind[0]
        data_t[index] = t_emb[pos_ds][pos_dt][tar_ind]
        data_p[index] = p_emb[pos_ds][pos_dt][pos_ind]
        data_n[index] = n_emb[neg_ds][neg_dt][neg_ind]
        
    print(data_t)
    print(data_p)
    print(data_n)
    trainloader, testloader = dataload_realtrip(data_t, data_p, data_n, train_prop=train_prop, BATCH=batch)

    return trainloader, testloader


def load_emb_data_together(esmloc, molloc, datasets, esmpattern, molpattern, train_prop=0.8, batch=64):
    data1 = np.load(f'{esmloc}/{datasets[0]}_train{esmpattern}')
    data2 = np.load(f'{molloc}/{datasets[0]}_train{molpattern}')
    for d in datasets:
        for t in ['train', 'val', 'test']:
            if d!=datasets[0] or t!='train':
                data1 = np.concatenate((data1, np.load(f'{esmloc}/{d}_{t}{esmpattern}')))
                data2 =  np.concatenate((data2, np.load(f'{molloc}/{d}_{t}{molpattern}')))
    print(data1.shape)
    print(data2.shape)
    trainloader, testloader = dataload_fn(data1, data2, train_prop, batch)
    return trainloader, testloader, torch.tensor(data1).to(torch.float32), torch.tensor(data2).to(torch.float32)

def load_inference_emb(esmloc, molloc, datasets, esmpattern, molpattern):
    data1 = np.load(f'{esmloc}/{datasets[0]}_train{esmpattern}')
    data2 = np.load(f'{molloc}/{datasets[0]}_train{molpattern}')
    for d in datasets:
        for t in ['train', 'val', 'test']:
            if d!=datasets[0] or t!='train':
                data1 = np.concatenate((data1, np.load(f'{esmloc}/{d}_{t}{esmpattern}')))
                data2 =  np.concatenate((data2, np.load(f'{molloc}/{d}_{t}{molpattern}')))
    return torch.tensor(data1).to(torch.float32), torch.tensor(data2).to(torch.float32)


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
  
    if config['Train']:
        print("Training")
        '''
        Do training
        '''

        '''
        Load data
        '''

        esm_pattern ='_prot.dat-embeddings.npy'
        mol_pattern = '.npy'
        datasets = ['BindingDB', 'DAVIS']#'BIOSNAP', 'DAVIS']
        #'BindingDB', 
        real_triplet_file = 'dataset/triplet_dataframe.csv' 
        raw_pwd = 'dataset'
        t_emb_pwd = 'dataset/protein_embeddings'
        p_emb_pwd = 'dataset/smi_embeddings_test/embeddings'
        n_emb_pwd = 'dataset/smi_embeddings_test/embeddings'
        trainloader, testloader= load_real_triplet(real_triplet_file,
                                                    raw_pwd,
                                                    t_emb_pwd,
                                                    p_emb_pwd,
                                                    n_emb_pwd,
                                                    train_prop=0.8, 
                                                    batch=64)
        print(trainloader, testloader)
        #import sys
        #sys.exit()

        projhead = EsmMolProjectionHead(projection_1, projection_2, lossfn, config['lr'])
        wandb_logger = WandbLogger(project=config["wandbproj"])
        trainer = L.Trainer(
                                max_epochs=config['epoch'],
                                logger=wandb_logger,
                                log_every_n_steps=1,
                                callbacks=[EarlyStopping(monitor="val_loss",
                                                        mode="min", 
                                                        patience = config['early'])])

        trainer.fit(projhead, trainloader, testloader)

    else:
        print("no Training")

    if config['Infer']==True:
        print("Inference")
        '''
        Do inference
        '''
        
        '''
        Load data
        '''
        datasets = config['datasets'] 
        esm_pattern ='_prot.dat-embeddings.npy'
        mol_pattern = '.smi-embeddings.npy'
        protdat, smidat = load_inference_emb(config['esmloc'], 
                                                config['molloc'],
                                                datasets,
                                                esm_pattern,
                                                mol_pattern)

        projhead = EsmMolProjectionHead.load_from_checkpoint(config['Check'], projection_1=projection_1, projection_2=projection_2, loss_fn=lossfn, base_lr=config['lr'])
        print(projhead.projection_1)
        projhead.to('cuda')
        projhead.projection_1.eval()
        prot_proj = projhead.projection_1(torch.tensor(protdat).to('cuda'))
        mol_proj = projhead.projection_2(torch.tensor(smidat).to('cuda'))
        print(prot_proj.size())
        print(mol_proj.size())
        out_patt = config['InfOut']
        torch.save(prot_proj, f'prot_{out_patt}')
        torch.save(mol_proj, f'mol_{out_patt}')
    return 

if __name__ == "__main__":
    train_esm_mol()


