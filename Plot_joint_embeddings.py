import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from rdkit import Chem
import rdkit
from rdkit.Chem import Descriptors
import sys
import torch
def filter(ar):
  return ar[np.isfinite(ar)]

def tsne_emb(data, out_path, n_comp=2):
    tsne_emb_data = TSNE(n_components=n_comp).fit_transform(data)
    np.save(out_path, tsne_emb_data)
    return tsne_emb_data

'''Load embeddings
'''
protein_emb = torch.load('prot_proj_test.pt').to('cpu')
smi_emb = torch.load('mol_proj_test.pt').to('cpu')
print(protein_emb)
print(smi_emb)
protein_emb = protein_emb.detach().numpy()
#protein_emb = np.nan_to_num(protein_emb, nan=0)[:]
#protein_emb = np.ma.masked_invalid(protein_emb)
#mask_prot_emb_nan = ~np.isnan(protein_emb).any(axis=1)
#protein_emb = protein_emb[mask_prot_emb_nan]
print(protein_emb)
#protein_emb = protein_emb[np.logical_not(np.isnan(protein_emb))]
#for pemb in protein_emb:
#    if np.nan in pemb: 
#        print(pemb)

print(protein_emb)
smi_emb = smi_emb.detach().numpy()
smi_emb = np.nan_to_num(smi_emb, nan=0)[:]
   
'''
tSNE data
'''
#prot_tsne = tsne_emb(protein_emb, 'tsne_prot_jnt.npy')
prot_tsne = np.load('tsne_prot_jnt.npy')
#smi_tsne = tsne_emb(smi_emb, 'tsne_smi_jnt.npy')
smi_tsne = np.load('tsne_smi_jnt.npy')
'''
Create joint proj
'''
indicator = [1 for it in range(len(protein_emb))]
indicator.extend([0 for it in range(len(smi_emb))])
total_emb = np.concatenate((protein_emb, smi_emb))
#total_tsne = tsne_emb(total_emb, 'tsne_protsmi_jnt.npy')
total_tsne = np.load('tsne_protsmi_jnt.npy')

'''
load all raw protein data to create embeddings
'''

import pandas as pd
protein_seq = pd.read_csv('dataset/BindingDB_train_prot.dat')
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/BindingDB_val_prot.dat')])
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/BindingDB_test_prot.dat')])
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/DAVIS_train_prot.dat')])
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/DAVIS_val_prot.dat')])
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/DAVIS_test_prot.dat')])
#protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/BIOSNAP_train_prot.dat')])
#protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/BIOSNAP_val_prot.dat')])
#protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/BIOSNAP_test_prot.dat')])

protein_seq = protein_seq[:]
protein_seq['length'] = [len(seq) for seq in protein_seq['Target Sequence']]
mask_prot = protein_seq['length'] < 1500

plt.scatter(x=prot_tsne[:,0], y=prot_tsne[:,1], s=2, c=protein_seq['length'], cmap='tab20c')
plt.colorbar(label="Protein length", orientation="horizontal")
plt.savefig('Images/ProteinJointEmbedded_length_New2.png', bbox_inches='tight')
plt.close()

'''
SMILES
'''

smiles_seq = pd.read_csv('dataset/BindingDB_train.smi')
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/BindingDB_val.smi')])
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/BindingDB_test.smi')])
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/DAVIS_train.smi')])
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/DAVIS_val.smi')])
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/DAVIS_test.smi')])
#smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/BIOSNAP_train.smi')])
#smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/BIOSNAP_val.smi')])
#smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/BIOSNAP_test.smi')])
smiles_seq['weight'] = [Descriptors.ExactMolWt(Chem.MolFromSmiles(smi)) for smi in smiles_seq['SMILES']]

smiles_seq = smiles_seq[:]
mask_smi = smiles_seq['weight'] < 1000
plt.scatter(x=smi_tsne[:,0], y=smi_tsne[:,1], s=2, c=smiles_seq['weight'], cmap='tab20c', vmin=0, vmax=1000)
plt.colorbar(label="Molecular Weight", orientation="horizontal")
plt.savefig('Images/SMILESJointEmbedded_Weight_New2.png', bbox_inches='tight')
plt.close()


'''
SMILES + prot length
'''
mask_prot = protein_seq['length'] < 800
plt.scatter(x=smi_tsne[:,0],
            y=smi_tsne[:,1],
            s=2,
            c=protein_seq['length'],
            cmap='tab20c',
            vmin=0,
            vmax=1000)
plt.colorbar(label="Protein Length", orientation="horizontal")

plt.savefig('Images/SMILESJointEmbedded_Length_New2.png', bbox_inches='tight')
plt.close()

'''
Protein + mol weight
'''

plt.scatter(x=prot_tsne[:,0],
            y=prot_tsne[:,1],
            s=2,
            c=smiles_seq['weight'],
            cmap='tab20c')
plt.colorbar(label="Molecular Weight", orientation="horizontal")
plt.savefig('Images/ProteinJointEmbedded_Weight_New2.png', bbox_inches='tight')
plt.close()

'''
Protein + smiles
'''

plt.scatter(x=total_tsne[:,0],
            y=total_tsne[:,1],
            s=2,
            c=indicator,
            cmap='tab20c')
plt.colorbar(label="Protein or SMILES", orientation="horizontal")
plt.savefig('Images/ProteinSMILESJointEmbedded_New2.png', bbox_inches='tight')
plt.close()

