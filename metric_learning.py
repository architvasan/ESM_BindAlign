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
#from torchsummary import summary

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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    outputs = esm_model(**inputs)#, output_hidden_states=True)
                                                                           
    print(outputs)
    # Get the last hidden states
    last_hidden_state = outputs.hidden_states[-1]
                                                                           
    # Compute the average pooled embeddings
    pooled_embeds = average_pool(last_hidden_state, inputs.attention_mask)
                                                                           
    # Get the batch size                                                  exit()
    batch_size = inputs.attention_mask.shape[0]
                                                                           
    # Store the pooled embeddings in the output buffer
    embeddings[idx : idx + batch_size, :] = pooled_embeds
    #embeddings = embeddings.numpy()
    print(embeddings)

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
    
    
