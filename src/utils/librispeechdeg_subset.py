import os
import pandas as pd
import shutil

# MOVE DEGRADED FILES
# df_train = pd.read_csv('data/train.csv')
# df_valid = pd.read_csv('data/valid.csv')
# df = pd.concat([df_train, df_valid], axis=0)

# root = '/media/alergn/hdd/datasets/audio/speech/LibriSpeechDeg'
# dst_dir = os.path.join(root, 'nomad-ls')

# for row, cols in df.iterrows():
#     for c in cols[['Anchor', 'Positive', 'Negative']]:
#         src = os.path.join(root, c)
#         dst = os.path.join(dst_dir, c)
#         shutil.copy(src, dst)



# MOVE CLEAN FILES
df_train = pd.read_csv('data/train_nsim.csv')
df_valid = pd.read_csv('data/valid_nsim.csv')
df = pd.concat([df_train, df_valid], axis=0)

root = '/media/alergn/hdd/datasets/audio/speech/LibriSpeechDeg'
dst_dir = os.path.join(root, 'nomad-ls', 'CLEAN')

for row, cols in df.iterrows():
    for c in cols[['reference']]:
        src = os.path.join(root, 'CLEAN', c)
        dst_sub = os.path.join(dst_dir, '/'.join(c.split('/')[:-1]))
        if not os.path.isdir(dst_sub):
            os.makedirs(dst_sub)
        dst = os.path.join(dst_sub, c.split('/')[-1])
        shutil.copy(src, dst)