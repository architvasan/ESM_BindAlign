import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def filter(ar):
  return ar[np.isfinite(ar)]

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
prot_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(protein_data)
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
smi_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(smiles_data)
print(smi_embedded.shape)
print(smi_embedded)

plt.scatter(smi_embedded[:,0], smi_embedded[:,1], s=2, color='black')
plt.savefig('SmiEmbedded.png')
plt.close()

