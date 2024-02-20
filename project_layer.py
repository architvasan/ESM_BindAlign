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

torch.autograd.set_detect_anomaly(True)

'''
Define classes to create projection layers
'''

class EsmMolProjectionHead(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, proj_size) -> None:
        super().__init__()
   
        # We use a different projection head for both modes
        # since the embeddings fall into different subspaces.
        self.projection_1 = nn.Sequential(
            nn.Linear(hidden_size1, proj_size*4),
            nn.LeakyReLU(0.2),
            nn.Linear(proj_size*4, proj_size*2),
            nn.LeakyReLU(0.2),
            nn.Linear(proj_size*2, proj_size)
        )
        self.projection_2 = nn.Sequential(
            nn.Linear(hidden_size2, proj_size*2),
            nn.LeakyReLU(0.2),
            nn.Linear(proj_size*2, proj_size),
        )
        
    def forward(self, x1: torch.Tensor, x2:torch.Tensor) -> torch.Tensor:
        # Project the embeddings into a lower dimensional space
        # These have shape (batch_size, projection_size)
        z_data = self.projection_1(x1)
        z_data = self.projection_2(x2)

        # Compute the metric loss following pytorch-metric-learning
        return z_data


'''
Test training of projection layers
'''

def dataload_fn(data, train_prop=0.8):
    input_tensor = torch.tensor(data).to(torch.float32)
    if False:
        means = input_tensor.mean(dim=1, keepdim=True)
        stds = input_tensor.std(dim=1, keepdim=True)
        input_tensor = (input_tensor - means) / stds
        outmap_min, _ = torch.min(input_tensor, dim=1, keepdim=True)
        outmap_max, _ = torch.max(input_tensor, dim=1, keepdim=True)
        input_tensor = (input_tensor - outmap_min) / (outmap_max - outmap_min) # Broadcasting rules apply
    #input_tensor = torch.add(input_tensor, 2)
    #input_tensor = F.normalize(input_tensor)#, dim = 0)
    print(input_tensor)
    train_size = int(train_prop * len(input_tensor))
    test_size = int(len(input_tensor) - train_size)
    training_data, test_data = torch.utils.data.random_split(input_tensor, [train_size, test_size])
    train_dataloader = DataLoader(training_data, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=512, shuffle=True)

    return train_dataloader, test_dataloader


def load_emb_data(dataloc, datasets, pattern, train_prop=0.8):
    data = np.load(f'{dataloc}/{datasets}_train{pattern}')
    for t in ['val', 'test']:
        data = np.concatenate((data, np.load(f'{dataloc}/{datasets}_{t}{pattern}')))
    
    dataloader = dataload_fn(data)
    return dataloader


def train_esm_mol():
    from argparse import ArgumentParser, SUPPRESS
    from pathlib import Path
    '''
    Set all arguments
    '''
    parser = ArgumentParser()#add_help=False)

    #parser.add_argument('-h', '--help', action='help', default=SUPPRESS,
    #                help='Show this help message and exit.')

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
        "-l", "--lr", type=float, required=False, help="learning rate", default=0.0001
    )
    parser.add_argument(
        "-b", "--early", type=int, required=False, help="early stop", default=100
    )
    args = parser.parse_args()

    '''
    Instantiate model, loss, optimizer
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model =  EsmMolProjectionHead(args.hsize1, args.hsize2, args.proj).to(device)
    loss_fn = SelfSupervisedLoss(TripletMarginLoss(
                                    distance = CosineSimilarity(),
                                    reducer = ThresholdReducer(high=0.3),
                                    embedding_regularizer = LpRegularizer()
                                    ))
    optimizer = torch.optim.Adam(
                        [{"params": model.projection_1.parameters()}, {"params": model.projection_2.parameters()}],
                        lr=args.lr,
                                )

    '''
    Load data
    '''
    esm_pattern ='_prot.dat-embeddings.npy'
    mol_pattern = '.smi-embeddings.npy'
    #datasets = ['BindingDB']#'BIOSNAP', 'DAVIS']
    #'BindingDB', 
    esm_train, esm_test = load_emb_data(args.esmloc, args.dataset, esm_pattern, train_prop=args.trainp)
    mol_train, mol_test = load_emb_data(args.molloc, args.dataset, mol_pattern, train_prop=args.trainp)

    '''
    Begin training loop
    '''
    loss_history = []
    loss_test_history = []
    notmin=0
    for i in tqdm(range(args.epoch)):
        loss_train_i = []
        for j, (batch_e, batch_m) in enumerate(zip(esm_train, mol_train)):
            '''
            Training
            '''
            loss = loss_fn(
                        model.projection_1(batch_e.to(device)),
                        model.projection_2(batch_m.to(device))
                        )

            loss_train_i.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_history.append(np.min(loss_train_i))
        
        '''
        Validation
        '''
        loss_test = SelfSupervisedLoss(TripletMarginLoss())
        loss_test_i = []
        for k, (batch_et, batch_mt) in enumerate(zip(esm_test, mol_test)):
            loss_t = loss_test(
                        model.projection_1(batch_et.to(device)),
                        model.projection_2(batch_mt.to(device))
                        )
            loss_test_i.append(loss_t.item()) 
        loss_test_history.append(np.mean((loss_test_i)))

        if loss_test_history[-1]<=np.min(loss_test_history):
            notmin = 0
            print("min")
            torch.save({
                'epoch': i,
                'model_state_dict': model.projection_1.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f'{args.out}/esm_{args.dataset}proj.pt')

            torch.save({
                'epoch': i,
                'model_state_dict': model.projection_2.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,                                 
                }, f'{args.out}/mol_{args.dataset}proj.pt')
        else:
            notmin+=1
            if notmin>=args.early:
                return loss_history, loss_test
        if i%10==0:
            print(loss_test_history)
            print(loss_history)

    print(model.parameters)
    np.savetxt(f'{args.out}/loss_train_{args.dataset}.dat', loss_history)
    np.savetxt(f'{args.out}/loss_test_{args.dataset}.dat', loss_test_history)
    return loss_history, loss_test_history


if __name__ == "__main__":
    loss_history, loss_test = train_esm_mol()



















if False:
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

