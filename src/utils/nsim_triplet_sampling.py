import pandas as pd
import random
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from itertools import combinations
from pathlib import Path
import os

# Set seed
# Seed is used for the split but not for sampling positive and negative whereas the global seed has been used
# You find the csv in the repo if you want to reuse the same train/valid sets. Otherwise specify the random_state in the sample function.
SEED = 10
random.seed(SEED)

# Import dataframe
path = '/media/alergn/hdd/github/speech-degradator/degraded_data_visqol_scores.csv'
df = pd.read_csv(path)
df.drop(index=df.index[:2], inplace=True)
df.dropna(axis=1, inplace=True)
df['nsim'] = df.iloc[:,3:35].mean(axis=1)
df = df[['reference', 'degraded', 'nsim']]
df['degraded'] = ['/'.join(x.split('/')[-2:]) for x in df['degraded']]

# Split train and val based on the clean file (never overlap the same clean file between the partitions)
splitter = GroupShuffleSplit(test_size=.20, n_splits=1, random_state = SEED)
split = splitter.split(df, groups=df['reference'])
train_inds, test_inds = next(split)
train_df = df.iloc[train_inds]
val_df = df.iloc[test_inds]

def create_triplets(df, N=1, hard_sampling=True):
    margin = 0.05
    df.drop_duplicates(inplace=True)
    anchor_list, positive_list, negative_list = [], [], []
    positive_nsim, negative_nsim = [], []
    positive_dist, negative_dist = [], []
    
    # Create Triplets (Anchor, Positive, Negative)
    for row in df['reference'].unique():
        df_g = df[df['reference'] == row]
        parts = Path(df_g.iloc[0]['reference']).parts
        filepath_clean = os.path.join('CLEAN', '/'.join(str(Path(*parts[parts.index('train-clean-100-wav'):])).split('/')[1:]))
        df_clean = pd.DataFrame({'degraded': filepath_clean, 'nsim': [1.0000]})
        df_g = pd.concat([df_g, df_clean])
        df_g.drop('reference', axis=1, inplace=True)
        
        for _ in range(N):
            # Sample anchor
            anchor = df_g.sample(1)
            anchor['nsim_dist'] = 0.0
            positive = df_g.iloc[(df_g['nsim']-anchor['nsim'].values).abs().argsort()[:2]].iloc[1].to_frame().T
            positive['nsim_dist'] = abs(anchor['nsim'].values - positive['nsim'].values)

            # Find negative
            df_g['nsim_dist'] = abs(anchor['nsim'].values - df_g['nsim'])
            df_g.drop(positive.index, axis=0, inplace=True)
            df_g.drop(anchor.index, axis=0, inplace=True)

            # Hard sampling takes the value closest to the positive
            if not hard_sampling:
                negative = df_g[df_g['nsim_dist'] > positive['nsim_dist'].values[0] + margin].sample(1)
            else:
                negative = df_g[df_g['nsim_dist'] == df_g['nsim_dist'].min()]
            
            assert positive['nsim_dist'].values[0] < negative['nsim_dist'].values[0]
            
            # Save data
            anchor_list.append(anchor['degraded'].values[0])
            positive_list.append(positive['degraded'].values[0])
            negative_list.append(negative['degraded'].values[0])
            
            #anchor_nsim.append(anchor['nsim'].values[0])
            positive_nsim.append(positive['nsim'].values[0])
            negative_nsim.append(negative['nsim'].values[0])

            #anchor_dist.append(anchor['nsim_dist'].values[0])
            positive_dist.append(positive['nsim_dist'].values[0])
            negative_dist.append(negative['nsim_dist'].values[0])
        
    df_out = pd.DataFrame({'Anchor': anchor_list, 'Positive': positive_list, 'Negative': negative_list, 'anc_pos_dist': positive_dist, 'anc_neg_dist': negative_dist})
    return df_out

# Create datasets (Easy)
train_triplets_df = create_triplets(train_df, N=3, hard_sampling=False)
train_triplets_df.dropna(axis=0, inplace=True)
valid_triplets_df = create_triplets(val_df, N=3, hard_sampling=False)
valid_triplets_df.dropna(axis=0, inplace=True)

# Create datasets (Hard)
train_triplets_df = create_triplets(train_df, N=3, hard_sampling=True)
train_triplets_df.dropna(axis=0, inplace=True)
valid_triplets_df = create_triplets(val_df, N=3, hard_sampling=True)
valid_triplets_df.dropna(axis=0, inplace=True)