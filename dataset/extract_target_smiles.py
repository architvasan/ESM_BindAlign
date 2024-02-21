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


df = pd.read_csv('BindingDB/train.csv')

for dataset in ['BindingDB', 'BIOSNAP', 'DAVIS']:
    for dt in ['train', 'test', 'val']:
        if dataset == 'BindingDB' and dt=='train':
            continue
        else:
            if dataset=='BIOSNAP':
                df = pd.concat([df, pd.read_csv(f'{dataset}/full_data/{dt}.csv')])
            else:
                df = pd.concat([df, pd.read_csv(f'{dataset}/{dt}.csv')])

df.to_csv('all_dataset.csv')

df_pos = df[df['Label']==1]
df_neg = df[df['Label']==0]

triplet_pro = []
triplet_pos = []
triplet_neg = []
for targ in tqdm(df_pos['Target Sequence'].unique()):
    pos = df_pos[df_pos['Target Sequence']==targ]['SMILES'].tolist()
    neg = df_neg[df_neg['Target Sequence']==targ]['SMILES'].tolist()
    if len(pos)>=1 and len(neg)>=1:
        for p in pos:
            for n in neg:
                triplet_pro.append(targ)
                triplet_pos.append(p)
                triplet_neg.append(n)
    else:
        continue

df_triplet = pd.DataFrame(
                    {
                   'target': triplet_pro,
                   'positive': triplet_pos,
                   'negative': triplet_neg
                }
                )

df_triplet.to_csv('triplet_dataframe.csv')
