import pandas as pd
from tqdm import tqdm

'''
Obtain files of positive only points
'''
if False:
    for dataset in ['BindingDB', 'BIOSNAP', 'DAVIS']:
        for dt in ['train', 'test', 'val']:
            if dataset=='BIOSNAP':
                df = pd.read_csv(f'{dataset}/full_data/{dt}.csv')
            else:
                df = pd.read_csv(f'{dataset}/{dt}.csv')
            df_true = df[df['Label']==1]
            df_true['SMILES'].to_csv(f'{dataset}_{dt}.smi', index=False)
            df_true['Target Sequence'].to_csv(f'{dataset}_{dt}_prot.dat', index=False)

'''
Obtain files of negative only points
'''

if False:
    for dataset in ['BindingDB', 'BIOSNAP', 'DAVIS']:
        for dt in ['train', 'test', 'val']:
            if dataset=='BIOSNAP':
                df = pd.read_csv(f'{dataset}/full_data/{dt}.csv')
            else:
                df = pd.read_csv(f'{dataset}/{dt}.csv')
            df_true = df[df['Label']==0]
            df_true['SMILES'].to_csv(f'{dataset}_{dt}_neg.smi', index=False)
            df_true['Target Sequence'].to_csv(f'{dataset}_{dt}_prot_neg.dat', index=False)

'''
Create triplets
'''


df = pd.read_csv('BindingDB/train.csv', index_col=None)
print(df)
print(len(df))
df['dataset']=['BindingDB' for val in df['Target Sequence']]
df['datatype'] = ['train' for val in df['Target Sequence']]
for dataset in ['BindingDB', 'DAVIS']:
    for dt in ['train', 'test', 'val']:
        if dataset == 'BindingDB' and dt=='train':
            continue
        else:
            if dataset=='BIOSNAP':
                df_it = pd.read_csv(f'{dataset}/full_data/{dt}.csv')
                df_it['dataset'] = [dataset for val in df_it['Target Sequence']]
                df_it['datatype'] = [dt for val in df_it['Target Sequence']]
                df = pd.concat([df, df_it])
            else:
                df_it = pd.read_csv(f'{dataset}/{dt}.csv')
                df_it['dataset'] = [dataset for val in df_it['Target Sequence']]
                df_it['datatype'] = [dt for val in df_it['Target Sequence']]
                df = pd.concat([df, df_it])
                
df.to_csv('all_dataset.csv', index=False)

df_pos = df[df['Label']==1]
df_neg = df[df['Label']==0]

triplet_pro = []
#triplet_pro_dataset = []
#triplet_pro_datatype = []

triplet_pos = []
triplet_pos_dataset = []
triplet_pos_datatype = []

triplet_neg = []
triplet_neg_dataset = []
triplet_neg_datatype = []

for targ in tqdm(df_pos['Target Sequence'].unique()):
    pos = df_pos[df_pos['Target Sequence']==targ]['SMILES'].tolist()
    pos_dset = df_pos[df_pos['Target Sequence']==targ]['dataset'].tolist()
    pos_dt = df_pos[df_pos['Target Sequence']==targ]['datatype'].tolist()
    neg = df_neg[df_neg['Target Sequence']==targ]['SMILES'].tolist()
    neg_dset = df_neg[df_neg['Target Sequence']==targ]['dataset'].tolist()
    neg_dt = df_neg[df_neg['Target Sequence']==targ]['datatype'].tolist()

    if len(pos)>=1 and len(neg)>=1:
        for (p, p_dset, p_dt) in zip(pos, pos_dset, pos_dt):
            for (n, n_dset, n_dt) in zip(neg, neg_dset, neg_dt):
                triplet_pro.append(targ)
                
                triplet_pos.append(p)
                triplet_pos_dataset.append(p_dset)
                triplet_pos_datatype.append(p_dt)
                
                triplet_neg.append(n)
                triplet_neg_dataset.append(n_dset)
                triplet_neg_datatype.append(n_dt)
    else:
        continue

df_triplet = pd.DataFrame(
                    {
                   'target': triplet_pro,
                   'positive': triplet_pos,
                   'negative': triplet_neg,
                   'positive_dataset': triplet_pos_dataset,
                   'positive_datatype': triplet_pos_datatype,
                   'negative_dataset': triplet_neg_dataset,
                   'negative_datatype': triplet_neg_datatype
                }
                )

df_triplet.to_csv('triplet_dataframe.csv')
