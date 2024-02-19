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

#from torchsummary import summary
torch.autograd.set_detect_anomaly(True)

class ProjectModel(nn.Module):

    def __init__(self, in_dim, out_dim
                 ):
        super().__init__()
        self.model_type = 'Linear'

        self.embedding = nn.Embedding(in_dim, 2048)
        self.batchnorm = nn.BatchNorm1d(in_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(in_dim, 2048)
        self.act1 = nn.SELU()
        self.batchnorm2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(2048, 1024)
        self.act2 = nn.SELU()
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(1024, out_dim)
        self.act3 = nn.SELU()
        #print(self.linear1.weight.dtype) 
        
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        #self.embedding.weight.data.uniform_(-initrange, initrange)
        #self.batchnorm.bias.data.zero_()
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()
        self.linear3.bias.data.zero_()

        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear3.weight.data.uniform_(-initrange, initrange)
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        output = self.batchnorm(src)
        output = self.dropout1(output)
        #print(src.shape)
        output = self.linear1(output)
        output = self.act1(output)
        output = self.batchnorm2(output)
        output = self.dropout2(output)
        output = self.linear2(output)
        output = self.act2(output)
        output = self.batchnorm3(output)
        output = self.dropout3(output)
        output = self.linear3(output)
        output = self.act3(output)
        return output #torch.reshape(output, (-1,))

def training_data(features, labels):
    feature_tensor = torch.tensor(features).to(torch.float32)
    label_tensor = torch.tensor(labels).to(torch.float32)

    dataset = TensorDataset(feature_tensor, label_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = int(len(dataset) - train_size)
    training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(training_data, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=512, shuffle=True)

    return train_dataloader, test_dataloader


'''
Load protein embeddings:
'''
protein_data = np.load('dataset/protein_embeddings/BindingDB_train_prot.dat-embeddings.npy')
protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/BindingDB_val_prot.dat-embeddings.npy')))
protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/BindingDB_test_prot.dat-embeddings.npy')))
protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/BIOSNAP_train_prot.dat-embeddings.npy')))
protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/BIOSNAP_test_prot.dat-embeddings.npy')))
protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/BIOSNAP_val_prot.dat-embeddings.npy')))
protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/DAVIS_train_prot.dat-embeddings.npy')))
protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/DAVIS_test_prot.dat-embeddings.npy')))
protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/DAVIS_val_prot.dat-embeddings.npy')))

'''
Load SMILES embeddings
'''
smiles_data = np.load('dataset/smi_embeddings/BindingDB_train.smi-embeddings.npy')
smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/BindingDB_val.smi-embeddings.npy')))
smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/BindingDB_test.smi-embeddings.npy')))
smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/BIOSNAP_train.smi-embeddings.npy')))
smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/BIOSNAP_test.smi-embeddings.npy')))
smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/BIOSNAP_val.smi-embeddings.npy')))
smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/DAVIS_train.smi-embeddings.npy')))
smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/DAVIS_test.smi-embeddings.npy')))
smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/DAVIS_val.smi-embeddings.npy')))

print(protein_data.shape[1])
print(smiles_data.shape[1])

train_dataloader, test_dataloader = training_data(protein_data, smiles_data)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = ProjectModel(in_dim = protein_data.shape[1], out_dim = smiles_data.shape[1])
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay=1e-5)
from pytorch_metric_learning import losses
#loss_fn = losses.TripletMarginLoss()

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses

from pytorch_metric_learning.losses import SelfSupervisedLoss,TripletMarginLoss
loss_fn = SelfSupervisedLoss(TripletMarginLoss(
                                    distance = CosineSimilarity(),
                                    reducer = ThresholdReducer(high=0.3),
                                    embedding_regularizer = LpRegularizer()
                                ))
#loss_fn = SelfSupervisedLoss(TripletMarginLoss(
#                                    ))


loss_test_hist = []
loss_test_meanvals = []
for i in tqdm(range(50)):
    for j, (batch_X, batch_Y) in enumerate(train_dataloader):
        #print(batch_X.dtype)
        print(batch_X)
        print(model.parameters)
        preds = model(batch_X.to(device))
        #print(preds)
        #print(preds.shape)
        #print(batch_Y.shape)
        loss = loss_fn(preds, batch_Y.to(device))
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_test = SelfSupervisedLoss(TripletMarginLoss())
    loss_test_i = []
    for k, (batch_Xt, batch_yt) in enumerate(test_dataloader):
        y_hat = model(batch_Xt.to(device))
        y_grnd = batch_yt.to(device)
        loss_test_k = loss_test(y_hat, y_grnd)
        loss_test_hist.append({'epoch' : i, 'minibatch': k, 'trainloss': loss_test_k})
        loss_test_i.append(loss_test_k)
    print(loss_test_i)
    if False:
        loss_test_meanvals.append(np.mean(loss_test_i.to_numpy())) 
        if loss_test_meanvals[-1]==np.max(loss_test_meanvals):
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f'models/{base}.pt')
        
loss_test_hist_df = pd.DataFrame(loss_test_hist)
loss_test_hist_df.to_csv(f'test_loss_overtime.csv')
    
sys.exit()









###########################################################
###########################################################
if False:
    datasets = ['BindingDB', 'BIOSNAP', 'DAVIS']
    datatype = ['train', 'val', 'test']
    
    protein_seq = pd.read_csv('dataset/BindingDB_train_prot.dat')
    for dset in datasets:
        for dtype in datatype:
            if not (dset=='BindingDB' and dtype=='train'):
                protein_seq = pd.concat([protein_seq, pd.read_csv(f'dataset/{dset}_{dtype}_prot.dat')])
    protein_seq.to_csv('Combined_prot_seq.dat', index=False)
    
    
    smiles_seq = pd.read_csv('dataset/BindingDB_train.smi')
    for dset in datasets:
        for dtype in datatype:
            if not (dset=='BindingDB' and dtype=='train'):
                smiles_seq = pd.concat([smiles_seq, pd.read_csv(f'dataset/{dset}_{dtype}.smi')])
    smiles_seq.to_csv('Combined_smiles.smi', index=False)


f = open("Combined_prot_seq.dat", "r")
data = f.read().splitlines()[1:]
#data_tensor = torch.tensor([d for d in data])

smiles_seq = pd.read_csv('Combined_smiles.smi')
weights = [Descriptors.ExactMolWt(Chem.MolFromSmiles(smi)) for smi in smiles_seq['SMILES']]

df = {'text': data, 'labels': weights}
#print(df)
ds = Dataset.from_dict(df)
ds = ds.with_format("torch")
print(ds[0])
#print(ds['text'])
#print(df['text'])
#TD = CustomTextDataset(df['text'], df['labels'])
weights_tensor = torch.tensor(weights)

#ds = TensorDataset(data_tensor, weights_tensor)

batch_size = 64
num_data_workers = 1
#esm2_t36_3B_UR50D,facebook/esm1b_t33_650M_UR50S
esm_model, esm_tokenizer = get_esm_model('facebook/esm2_t6_8M_UR50D')
print(esm_model)
esm_model.to(device)

#summary(esm_model)
#esm_hidden = esm_model.hidden_states[-1]

#data = single_sequence_per_line_data_reader('Combined_prot_seq.dat')
dataloader = DataLoader(
    pin_memory=True,
    batch_size=batch_size,
    num_workers=num_data_workers,
    dataset=InMemoryDataset(data),
    collate_fn=DataCollator(esm_tokenizer),
    shuffle=False
)
optimizer = torch.optim.Adam(esm_model.parameters(), lr = 1e-5)

#print(dataloader)

#, labels
# training loop
from pytorch_metric_learning import losses
loss_func = losses.TripletMarginLoss()
for i, (batch, w) in tqdm(enumerate(zip(dataloader, weights_tensor))):
    #print(d)
    #print(batch)
    optimizer.zero_grad()
    #inputs = esm_tokenizer(
    #            batch,
    #            padding=True,
    #            truncation=True,
    #            return_tensors='pt')
    inputs = batch.to(device)
    print(inputs)
    # Get the model outputs with a forward pass
    outputs = esm_model(**inputs, output_hidden_states=True)

    print(outputs)
    # Get the last hidden states
    last_hidden_state = outputs.hidden_states[-1]
                                                                           
    # Compute the average pooled embeddings
    pooled_embeds = average_pool(last_hidden_state, inputs.attention_mask)
                                                                           
    # Get the batch size                                                  exit()
    batch_size = inputs.attention_mask.shape[0]
                                                                           
    # Store the pooled embeddings in the output buffer
    embeddings = pooled_embeds
    #    embeddings[idx : idx + batch_size, :] = pooled_embeds

    #embeddings = embeddings.numpy()
    print(embeddings.shape[0])

    loss = loss_func(embeddings, w)
    loss.backward()
    optimizer.step()














sys.exit()





embed_tensor = torch.tensor(protein_emb)
label_tensor = torch.tensor(weights)

dataset = TensorDataset(embed_tensor, label_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

print(dataset)
sys.exit()

#prot_emb_batches = group_list(protein_emb, group_size=64)
#for embeddings in prot_emb_batches:
#    print(embeddings)
##prot_smiles_df = pd.DataFrame(data = {'embed': protein_embed, 'labels': smiles_seq['SMILES']})
#smiles_weight_batches = group_list(smiles_seq['weight'], group_size=64)

loss_func = losses.TripletMarginLoss()

# your training loop
for j, (batchX, batchY) in enumerate(dataloader):
    loss = loss_func(batchX, batchY)
    loss.backward()


#for embeddings, labels in zip(prot_emb_batches, smiles_weight_batches):#range(len(prot_emb_batches)):#, (data, labels) in enumerate(dataloader):
#    #embeddings = prot_emb_batches[i]
#    #labels = smiles_weight_batches[i]
#    optimizer.zero_grad()
#	#embeddings = model(data)
#    loss = loss_func(embeddings, labels)
#    loss.backward()
#    optimizer.step()




if False:
    from pytorch_metric_learning import miners, losses
    miner = miners.MultiSimilarityMiner()
    loss_func = losses.TripletMarginLoss()
    
    # your training loop
    for i, (data, labels) in enumerate(dataloader):
    	optimizer.zero_grad()
    	embeddings = model(data)
    	hard_pairs = miner(embeddings, labels)
    	loss = loss_func(embeddings, labels, hard_pairs)
    	loss.backward()
    	optimizer.step()
    
    
