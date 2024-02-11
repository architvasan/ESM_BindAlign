import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from rdkit import Chem
import rdkit
from rdkit.Chem import Descriptors

def filter(ar):
  return ar[np.isfinite(ar)]

if False:
    protein_data = np.load('dataset/protein_embeddings/BindingDB_train_prot.dat-embeddings.npy')
    protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/BindingDB_val_prot.dat-embeddings.npy')))
    protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/BindingDB_test_prot.dat-embeddings.npy')))
    protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/BIOSNAP_train_prot.dat-embeddings.npy')))
    protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/BIOSNAP_test_prot.dat-embeddings.npy')))
    protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/BIOSNAP_val_prot.dat-embeddings.npy')))
    protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/DAVIS_train_prot.dat-embeddings.npy')))
    protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/DAVIS_test_prot.dat-embeddings.npy')))
    protein_data = np.concatenate((protein_data, np.load('dataset/protein_embeddings/DAVIS_val_prot.dat-embeddings.npy')))
    
    smiles_data = np.load('dataset/smi_embeddings/BindingDB_train.smi-embeddings.npy')
    smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/BindingDB_val.smi-embeddings.npy')))
    smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/BindingDB_test.smi-embeddings.npy')))
    smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/BIOSNAP_train.smi-embeddings.npy')))
    smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/BIOSNAP_test.smi-embeddings.npy')))
    smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/BIOSNAP_val.smi-embeddings.npy')))
    smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/DAVIS_train.smi-embeddings.npy')))
    smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/DAVIS_test.smi-embeddings.npy')))
    smiles_data = np.concatenate((smiles_data, np.load('dataset/smi_embeddings/DAVIS_val.smi-embeddings.npy')))
    
    '''
    Protein
    '''
    protein_data[~np.isfinite(protein_data)] = 0#filter(protein_data)
    protein_data = np.ma.masked_equal(protein_data,0)
    print(protein_data)
    #X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    prot_embedded = TSNE(n_components=2).fit_transform(protein_data)
    np.save('tsne_prot.npy', prot_embedded)
    print(prot_embedded.shape)
    print(prot_embedded)
    
    plt.scatter(prot_embedded[:,0], prot_embedded[:,1], s=2, color='black')
    plt.savefig('ProteinEmbedded.png')
    plt.close()
    
    '''
    SMILES
    '''
    smiles_data[~np.isfinite(smiles_data)] = 0#filter(protein_data)
    smiles_data = np.ma.masked_equal(smiles_data,0)
    print(smiles_data)
    #X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    smi_embedded = TSNE(n_components=2).fit_transform(smiles_data)
    np.save('tsne_smi.npy', smi_embedded)
    print(smi_embedded.shape)
    print(smi_embedded)
    
    plt.scatter(smi_embedded[:,0], smi_embedded[:,1], s=2, color='black')
    plt.savefig('SmiEmbedded.png')
    plt.close()

'''
Color data
'''
'''
Protein
'''

import pandas as pd
prot_embedded = np.load('dataset/tsne_prot.npy')
protein_seq = pd.read_csv('dataset/BindingDB_train_prot.dat')
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/BindingDB_val_prot.dat')])
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/BindingDB_test_prot.dat')])
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/BIOSNAP_train_prot.dat')])
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/BIOSNAP_val_prot.dat')])
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/BIOSNAP_test_prot.dat')])
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/DAVIS_train_prot.dat')])
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/DAVIS_val_prot.dat')])
protein_seq = pd.concat([protein_seq, pd.read_csv('dataset/DAVIS_test_prot.dat')])

protein_seq['length'] = [len(seq) for seq in protein_seq['Target Sequence']]
#protein_seq['length']=(protein_seq['length']-protein_seq['length'].mean())/protein_seq['length'].std()
print(protein_seq)

plt.scatter(x=prot_embedded[:,0], y=prot_embedded[:,1], s=2, c=protein_seq['length'], cmap='tab20c')#, vmin=-1, vmax=1)
#plt.colorbar()
plt.colorbar(label="Sequence length", orientation="horizontal")

plt.savefig('ProteinEmbedded_length.png', bbox_inches='tight')
plt.close()

print(protein_seq)

'''
SMILES
'''

smi_embedded = np.load('dataset/tsne_smi.npy')
smiles_seq = pd.read_csv('dataset/BindingDB_train.smi')
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/BindingDB_val.smi')])
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/BindingDB_test.smi')])
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/BIOSNAP_train.smi')])
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/BIOSNAP_val.smi')])
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/BIOSNAP_test.smi')])
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/DAVIS_train.smi')])
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/DAVIS_val.smi')])
smiles_seq = pd.concat([smiles_seq, pd.read_csv('dataset/DAVIS_test.smi')])
smiles_seq['weight'] = [Descriptors.ExactMolWt(Chem.MolFromSmiles(smi)) for smi in smiles_seq['SMILES']]
print(smiles_seq)
plt.scatter(x=smi_embedded[:,0], y=smi_embedded[:,1], s=2, c=smiles_seq['weight'], cmap='tab20c')#, vmin=-1, vmax=1)
#plt.colorbar()
plt.colorbar(label="Molecular Weight", orientation="horizontal")

plt.savefig('SMILESEmbedded_length.png', bbox_inches='tight')
plt.close()

